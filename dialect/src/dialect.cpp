#include "bcir/dialect.hpp"

#include <cctype>
#include <regex>
#include <utility>

namespace bcir {
namespace {

bool is_identifier_start(char ch) {
  return std::isalpha(static_cast<unsigned char>(ch)) || ch == '_';
}

bool is_identifier_continue(char ch) {
  return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
}

class Parser {
 public:
  explicit Parser(std::string_view source)
      : tokens(tokenize_dialect(source, &diagnostics)) {}

  ParseResult parse() {
    ParseResult result;
    if (match_keyword("module")) {
      result.module.name = expect_identifier("expected module name");
    } else {
      add_diag(peek().location,
               "expected 'module' keyword at start of translation unit");
      result.module.name = "anonymous";
    }

    if (!consume(TokenKind::LBrace, "expected '{' after module declaration")) {
      result.diagnostics = std::move(diagnostics);
      return result;
    }

    while (!is_at_end() && !check(TokenKind::RBrace)) {
      if (match_keyword("macro")) {
        result.module.macros.push_back(parse_macro());
      } else if (match_keyword("fn")) {
        result.module.functions.push_back(parse_function());
      } else {
        add_diag(peek().location, "expected 'macro' or 'fn' declaration");
        synchronize();
      }
    }

    consume(TokenKind::RBrace, "expected '}' at end of module");
    result.diagnostics = std::move(diagnostics);
    return result;
  }

 private:
  std::vector<Token> tokens;
  std::vector<Diagnostic> diagnostics;
  std::size_t current = 0;

  MacroDefinitionNode parse_macro() {
    MacroDefinitionNode macro;
    macro.name = expect_identifier("expected macro name");
    macro.params = parse_params(/*for_macro=*/true);
    macro.body = parse_block();
    return macro;
  }

  FunctionNode parse_function() {
    FunctionNode function;
    function.name = expect_identifier("expected function name");
    function.params = parse_params(/*for_macro=*/false);
    function.body = parse_block();
    return function;
  }

  std::vector<std::string> parse_params(bool for_macro) {
    std::vector<std::string> params;
    if (!consume(TokenKind::LParen, "expected '(' before parameter list")) {
      return params;
    }

    while (!check(TokenKind::RParen) && !is_at_end()) {
      std::string param = expect_identifier("expected parameter name");
      if (param.empty()) {
        if (for_macro) {
          add_diag(previous().location, "malformed macro params: empty name");
        }
      } else if (for_macro) {
        for (const std::string& existing : params) {
          if (existing == param) {
            add_diag(previous().location,
                     "malformed macro params: duplicate parameter '" + param +
                         "'");
          }
        }
      }
      params.push_back(param);

      if (!match(TokenKind::Comma)) {
        break;
      }
    }

    consume(TokenKind::RParen, "expected ')' after parameters");
    return params;
  }

  BlockNode parse_block() {
    BlockNode block;
    if (!consume(TokenKind::LBrace, "expected '{' before block")) {
      return block;
    }

    while (!check(TokenKind::RBrace) && !is_at_end()) {
      if (match_keyword("ld") || match_keyword("st")) {
        const bool is_load = previous().lexeme == "ld";
        block.operations.push_back(parse_ld_st(is_load));
      } else if (match_keyword("bin")) {
        block.operations.push_back(parse_bin());
      } else if (match_keyword("lane")) {
        block.operations.push_back(parse_lane_op());
      } else if (match_keyword("phase")) {
        block.operations.push_back(parse_phase());
      } else if (match_keyword("barrier")) {
        block.operations.push_back(parse_barrier());
      } else if (match_keyword("expand")) {
        block.operations.push_back(parse_macro_expansion());
      } else {
        add_diag(peek().location, "unknown operation in block");
        synchronize_statement();
      }
    }

    consume(TokenKind::RBrace, "expected '}' to close block");
    return block;
  }

  std::unique_ptr<Operation> parse_ld_st(bool is_load) {
    auto operation = std::make_unique<LdStOperation>(is_load);

    consume_keyword("rid", "expected 'rid' in ld/st operation");
    consume(TokenKind::Equal, "expected '=' after rid");
    const Token rid_token = advance();
    operation->rid = rid_token.lexeme;
    static const std::regex rid_regex(R"(^r[0-9]+$)");
    if (rid_token.kind != TokenKind::Identifier ||
        !std::regex_match(rid_token.lexeme, rid_regex)) {
      add_diag(rid_token.location,
               "malformed RID token '" + rid_token.lexeme + "'");
    }

    consume_keyword("lane", "expected 'lane' in ld/st operation");
    consume(TokenKind::Equal, "expected '=' after lane");
    const Token lane_token = advance();
    operation->lane = parse_lane_token(lane_token, /*for_directive=*/true);

    if (is_load) {
      consume_keyword("from", "expected 'from' in ld operation");
    } else {
      consume_keyword("to", "expected 'to' in st operation");
    }

    operation->target = expect_identifier("expected load/store target");
    consume(TokenKind::Semicolon, "expected ';' after ld/st operation");
    return operation;
  }

  std::unique_ptr<Operation> parse_bin() {
    auto operation = std::make_unique<BinaryOpOperation>();
    operation->opcode = expect_identifier("expected binary opcode");
    operation->dst = expect_identifier("expected binary destination");
    consume(TokenKind::Comma, "expected ',' after destination");
    operation->lhs = expect_identifier("expected left-hand side operand");
    consume(TokenKind::Comma, "expected ',' after left-hand side operand");
    operation->rhs = expect_identifier("expected right-hand side operand");
    consume(TokenKind::Semicolon, "expected ';' after binary operation");
    return operation;
  }

  std::unique_ptr<Operation> parse_lane_op() {
    auto operation = std::make_unique<LaneOpOperation>();
    operation->opcode = expect_identifier("expected lane operation opcode");
    operation->rid = expect_identifier("expected RID operand for lane op");

    static const std::regex rid_regex(R"(^r[0-9]+$)");
    if (!std::regex_match(operation->rid, rid_regex)) {
      add_diag(previous().location, "malformed RID token '" + operation->rid +
                                      "'");
    }

    const Token lane_token = advance();
    operation->lane = parse_lane_token(lane_token, /*for_directive=*/true);
    consume(TokenKind::Semicolon, "expected ';' after lane operation");
    return operation;
  }

  std::unique_ptr<Operation> parse_phase() {
    auto operation = std::make_unique<PhaseOperation>();
    Token token = advance();
    if (token.kind != TokenKind::Number) {
      add_diag(token.location, "invalid phase directive: expected integer");
      operation->phase = -1;
    } else {
      operation->phase = std::stoi(token.lexeme);
      if (operation->phase < 0 || operation->phase > 3) {
        add_diag(token.location, "invalid phase directive: phase must be 0..3");
      }
    }

    consume(TokenKind::Semicolon, "expected ';' after phase directive");
    return operation;
  }

  std::unique_ptr<Operation> parse_barrier() {
    auto operation = std::make_unique<BarrierOperation>();
    if (!check(TokenKind::Semicolon)) {
      operation->scope = expect_identifier("expected optional barrier scope");
    }
    consume(TokenKind::Semicolon, "expected ';' after barrier directive");
    return operation;
  }

  std::unique_ptr<Operation> parse_macro_expansion() {
    auto operation = std::make_unique<MacroExpansionOperation>();
    operation->macroName = expect_identifier("expected macro name in expansion");

    if (!consume(TokenKind::LParen,
                 "expected '(' before macro expansion arguments")) {
      return operation;
    }

    while (!check(TokenKind::RParen) && !is_at_end()) {
      operation->args.push_back(expect_identifier("expected macro argument"));
      if (!match(TokenKind::Comma)) {
        break;
      }
    }

    consume(TokenKind::RParen, "expected ')' after macro arguments");
    consume(TokenKind::Semicolon, "expected ';' after macro expansion");
    return operation;
  }

  int parse_lane_token(const Token& token, bool for_directive) {
    if (token.kind != TokenKind::Identifier || token.lexeme.rfind("lane", 0) != 0) {
      add_diag(token.location, "malformed lane token '" + token.lexeme + "'");
      return -1;
    }

    const std::string lane_number = token.lexeme.substr(4);
    if (lane_number.empty()) {
      add_diag(token.location, "malformed lane token '" + token.lexeme + "'");
      return -1;
    }

    for (char ch : lane_number) {
      if (!std::isdigit(static_cast<unsigned char>(ch))) {
        add_diag(token.location,
                 "malformed lane token '" + token.lexeme + "'");
        return -1;
      }
    }

    const int lane_value = std::stoi(lane_number);
    if (for_directive && (lane_value < 0 || lane_value > 63)) {
      add_diag(token.location,
               "invalid lane directive: lane must be in range 0..63");
    }
    return lane_value;
  }

  void synchronize() {
    while (!is_at_end()) {
      if (check(TokenKind::Semicolon) || check(TokenKind::RBrace)) {
        return;
      }
      advance();
    }
  }

  void synchronize_statement() {
    while (!is_at_end()) {
      if (match(TokenKind::Semicolon)) {
        return;
      }
      if (check(TokenKind::RBrace)) {
        return;
      }
      advance();
    }
  }

  bool match_keyword(std::string_view keyword) {
    if (check(TokenKind::Identifier) && peek().lexeme == keyword) {
      advance();
      return true;
    }
    return false;
  }

  bool consume_keyword(std::string_view keyword, std::string_view message) {
    if (match_keyword(keyword)) {
      return true;
    }
    add_diag(peek().location, std::string(message));
    return false;
  }

  std::string expect_identifier(std::string_view message) {
    if (check(TokenKind::Identifier)) {
      return advance().lexeme;
    }

    add_diag(peek().location, std::string(message));
    if (!is_at_end()) {
      return advance().lexeme;
    }
    return {};
  }

  bool consume(TokenKind kind, std::string_view message) {
    if (check(kind)) {
      advance();
      return true;
    }

    add_diag(peek().location, std::string(message));
    return false;
  }

  bool match(TokenKind kind) {
    if (check(kind)) {
      advance();
      return true;
    }
    return false;
  }

  bool check(TokenKind kind) const {
    if (is_at_end()) {
      return kind == TokenKind::EndOfFile;
    }
    return peek().kind == kind;
  }

  bool is_at_end() const { return peek().kind == TokenKind::EndOfFile; }

  Token advance() {
    if (!is_at_end()) {
      ++current;
    }
    return previous();
  }

  const Token& peek() const { return tokens[current]; }

  const Token& previous() const { return tokens[current - 1]; }

  void add_diag(SourceLocation location, std::string message) {
    diagnostics.push_back(Diagnostic{location, std::move(message)});
  }
};

}  // namespace

std::string dialect_component_banner() {
  return "bcir-dialect: ROP/MAP IR + parser/printer + verifier";
}

std::vector<Token> tokenize_dialect(std::string_view source,
                                    std::vector<Diagnostic>* diagnostics) {
  std::vector<Token> tokens;

  std::size_t line = 1;
  std::size_t column = 1;
  std::size_t offset = 0;

  const auto emit = [&](TokenKind kind, std::string lexeme, SourceLocation loc,
                        std::vector<Token>* out_tokens) {
    out_tokens->push_back(Token{kind, std::move(lexeme), loc});
  };

  auto add_diag = [&](SourceLocation location, const std::string& message) {
    if (diagnostics != nullptr) {
      diagnostics->push_back(Diagnostic{location, message});
    }
  };

  while (offset < source.size()) {
    const char ch = source[offset];
    SourceLocation loc{line, column, offset};

    if (std::isspace(static_cast<unsigned char>(ch))) {
      if (ch == '\n') {
        ++line;
        column = 1;
      } else {
        ++column;
      }
      ++offset;
      continue;
    }

    if (is_identifier_start(ch)) {
      std::size_t end = offset + 1;
      while (end < source.size() && is_identifier_continue(source[end])) {
        ++end;
      }
      emit(TokenKind::Identifier, std::string(source.substr(offset, end - offset)),
           loc, &tokens);
      column += (end - offset);
      offset = end;
      continue;
    }

    if (std::isdigit(static_cast<unsigned char>(ch))) {
      std::size_t end = offset + 1;
      while (end < source.size() &&
             std::isdigit(static_cast<unsigned char>(source[end]))) {
        ++end;
      }
      emit(TokenKind::Number, std::string(source.substr(offset, end - offset)), loc,
           &tokens);
      column += (end - offset);
      offset = end;
      continue;
    }

    switch (ch) {
      case '{':
        emit(TokenKind::LBrace, "{", loc, &tokens);
        break;
      case '}':
        emit(TokenKind::RBrace, "}", loc, &tokens);
        break;
      case '(':
        emit(TokenKind::LParen, "(", loc, &tokens);
        break;
      case ')':
        emit(TokenKind::RParen, ")", loc, &tokens);
        break;
      case ',':
        emit(TokenKind::Comma, ",", loc, &tokens);
        break;
      case ';':
        emit(TokenKind::Semicolon, ";", loc, &tokens);
        break;
      case '=':
        emit(TokenKind::Equal, "=", loc, &tokens);
        break;
      default:
        add_diag(loc, "unexpected character '" + std::string(1, ch) + "'");
        break;
    }

    ++offset;
    ++column;
  }

  tokens.push_back(Token{TokenKind::EndOfFile, "",
                         SourceLocation{line, column, offset}});
  return tokens;
}

ParseResult parse_dialect(std::string_view source) {
  Parser parser(source);
  return parser.parse();
}

}  // namespace bcir
