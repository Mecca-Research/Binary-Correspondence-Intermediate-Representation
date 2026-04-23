#include "bcir/dialect.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <utility>

namespace bcir {
namespace {

Diagnostic make_diagnostic(SourceLocation location, std::string message,
                           std::string pass = {}, std::string code = {}) {
  Diagnostic diagnostic;
  diagnostic.location = location;
  diagnostic.pass = std::move(pass);
  diagnostic.code = std::move(code);
  diagnostic.message = std::move(message);
  return diagnostic;
}

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
  struct RegistryType {
    std::string laneEnum;
    int width = 0;
    std::optional<int> tileWidth;
    std::string dtype;
  };

  std::vector<Token> tokens;
  std::vector<Diagnostic> diagnostics;
  std::map<std::string, RegistryType> register_types;
  std::map<std::string, RegistryType> memory_types;
  std::size_t current = 0;

  MacroDefinitionNode parse_macro() {
    MacroDefinitionNode macro;
    macro.location = peek().location;
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
      } else if (match_keyword("map_load")) {
        block.operations.push_back(
            parse_map_surface(MapSurfaceOperation::SurfaceKind::Load));
      } else if (match_keyword("map_store")) {
        block.operations.push_back(
            parse_map_surface(MapSurfaceOperation::SurfaceKind::Store));
      } else if (match_keyword("map_atomic_add")) {
        block.operations.push_back(
            parse_map_surface(MapSurfaceOperation::SurfaceKind::AtomicAdd));
      } else if (match_keyword("map_atomic_sub")) {
        block.operations.push_back(
            parse_map_surface(MapSurfaceOperation::SurfaceKind::AtomicSub));
      } else if (match_keyword("map_atomic_xor")) {
        block.operations.push_back(
            parse_map_surface(MapSurfaceOperation::SurfaceKind::AtomicXor));
      } else {
        add_diag(peek().location, "unknown operation in block");
        synchronize_statement();
      }
    }

    consume(TokenKind::RBrace, "expected '}' to close block");
    return block;
  }

  std::unique_ptr<Operation> parse_ld_st(bool is_load) {
    auto operation = std::make_unique<LdStOperation>(is_load, previous().location);

    consume_keyword("rid", "expected 'rid' in ld/st operation");
    consume(TokenKind::Equal, "expected '=' after rid");
    const auto rid_with_type = parse_register_with_optional_type();
    operation->rid = rid_with_type.first;
    operation->registryType = rid_with_type.second;

    static const std::regex rid_regex(R"(^r[0-9]+$)");
    if (!std::regex_match(operation->rid, rid_regex)) {
      add_diag(previous().location,
               "malformed RID token '" + operation->rid + "'");
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
    validate_load_store_type_compatibility(*operation, previous().location);

    consume(TokenKind::Semicolon, "expected ';' after ld/st operation");
    return operation;
  }

  std::unique_ptr<Operation> parse_bin() {
    auto operation = std::make_unique<BinaryOpOperation>(previous().location);
    operation->opcode = expect_identifier("expected binary opcode");

    const auto dst_with_type = parse_register_with_optional_type();
    operation->dst = dst_with_type.first;
    operation->dstType = dst_with_type.second;

    consume(TokenKind::Comma, "expected ',' after destination");

    const auto lhs_with_type = parse_register_with_optional_type();
    operation->lhs = lhs_with_type.first;
    operation->lhsType = lhs_with_type.second;

    consume(TokenKind::Comma, "expected ',' after left-hand side operand");

    const auto rhs_with_type = parse_register_with_optional_type();
    operation->rhs = rhs_with_type.first;
    operation->rhsType = rhs_with_type.second;

    validate_binary_types(*operation, previous().location);
    consume(TokenKind::Semicolon, "expected ';' after binary operation");
    return operation;
  }

  std::unique_ptr<Operation> parse_lane_op() {
    auto operation = std::make_unique<LaneOpOperation>(previous().location);
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
    auto operation = std::make_unique<PhaseOperation>(previous().location);
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
    auto operation = std::make_unique<BarrierOperation>(previous().location);
    if (!check(TokenKind::Semicolon)) {
      operation->scope = expect_identifier("expected optional barrier scope");
    }
    consume(TokenKind::Semicolon, "expected ';' after barrier directive");
    return operation;
  }

  std::unique_ptr<Operation> parse_macro_expansion() {
    auto operation = std::make_unique<MacroExpansionOperation>(previous().location);
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

  std::pair<std::string, std::optional<std::string>> parse_register_with_optional_type() {
    const std::string rid = expect_identifier("expected register operand");
    std::optional<std::string> type_text;
    if (match(TokenKind::Less)) {
      type_text = parse_registry_type_literal(previous().location);
    }
    return {rid, type_text};
  }

  std::string parse_registry_type_literal(const SourceLocation& start) {
    std::vector<std::string> parts;
    while (!check(TokenKind::Greater) && !is_at_end()) {
      if (check(TokenKind::Identifier) || check(TokenKind::Number)) {
        parts.push_back(advance().lexeme);
      } else {
        add_diag(peek().location, "invalid token inside registry type literal");
        advance();
      }
    }

    consume(TokenKind::Greater, "expected '>' to terminate registry type");
    if (parts.empty()) {
      add_diag(start, "empty registry type literal");
      return {};
    }

    std::vector<std::string> compact;
    for (const std::string& token : parts) {
      if (token != "x") {
        compact.push_back(token);
      }
    }

    std::string type_text;
    for (std::size_t i = 0; i < compact.size(); ++i) {
      if (i > 0) {
        type_text += ":";
      }
      type_text += compact[i];
    }
    return type_text;
  }

  std::optional<RegistryType> parse_registry_type_string(
      const std::string& type_text, const SourceLocation& location) {
    std::istringstream stream(type_text);
    std::vector<std::string> parts;
    std::string part;
    while (std::getline(stream, part, ':')) {
      parts.push_back(part);
    }

    if (parts.size() != 3 && parts.size() != 4) {
      add_diag(location, "invalid registry type form '" + type_text +
                             "': expected <lane x N x dtype> or tile variant");
      return std::nullopt;
    }

    static const std::set<std::string> kLaneEnums = {"U", "UX", "T", "GGG", "A", "H"};
    if (kLaneEnums.find(parts[0]) == kLaneEnums.end()) {
      add_diag(location, "invalid lane enum '" + parts[0] +
                             "' in registry type; expected one of U/UX/T/GGG/A/H");
      return std::nullopt;
    }

    auto parse_width = [&](const std::string& width_text, const std::string& label)
        -> std::optional<int> {
      if (width_text.empty() ||
          !std::all_of(width_text.begin(), width_text.end(), [](char ch) {
            return std::isdigit(static_cast<unsigned char>(ch));
          })) {
        add_diag(location, "invalid width '" + width_text + "' for " + label +
                               " in registry type '" + type_text + "'");
        return std::nullopt;
      }
      const int width = std::stoi(width_text);
      if (width < 1 || width > 64) {
        add_diag(location, "invalid width constraint '" + width_text +
                               "' in registry type '" + type_text +
                               "': width must be in range 1..64");
        return std::nullopt;
      }
      return width;
    };

    const std::optional<int> width = parse_width(parts[1], "lane");
    if (!width.has_value()) {
      return std::nullopt;
    }

    std::optional<int> tile_width;
    std::string dtype = parts.back();
    if (parts.size() == 4) {
      tile_width = parse_width(parts[2], "tile");
      if (!tile_width.has_value()) {
        return std::nullopt;
      }
    }

    static const std::set<std::string> kDtypes = {"i8",  "i16", "i32", "i64", "u8",  "u16",
                                                  "u32", "u64", "f16", "f32", "f64", "bf16"};
    if (kDtypes.find(dtype) == kDtypes.end()) {
      add_diag(location, "invalid data type '" + dtype +
                             "' in registry type; expected integer/float scalar dtype");
      return std::nullopt;
    }

    return RegistryType{parts[0], *width, tile_width, dtype};
  }

  std::string registry_type_to_text(const RegistryType& type) const {
    if (type.tileWidth.has_value()) {
      return "<" + type.laneEnum + " x " + std::to_string(type.width) + " x " +
             std::to_string(*type.tileWidth) + " x " + type.dtype + ">";
    }
    return "<" + type.laneEnum + " x " + std::to_string(type.width) + " x " +
           type.dtype + ">";
  }

  std::optional<RegistryType> resolve_register_type(
      const std::string& rid, const std::optional<std::string>& explicit_type,
      const SourceLocation& location) {
    static const std::regex rid_regex(R"(^r[0-9]+$)");
    if (explicit_type.has_value()) {
      const std::optional<RegistryType> parsed =
          parse_registry_type_string(*explicit_type, location);
      if (parsed.has_value() && std::regex_match(rid, rid_regex)) {
        register_types[rid] = *parsed;
      }
      return parsed;
    }
    if (!std::regex_match(rid, rid_regex)) {
      return std::nullopt;
    }

    const auto it = register_types.find(rid);
    if (it == register_types.end()) {
      add_diag(location, "missing registry type for '" + rid +
                             "'; annotate register as <lane x N x dtype> or tile variant");
      return std::nullopt;
    }
    return it->second;
  }

  void validate_binary_types(const BinaryOpOperation& operation,
                             const SourceLocation& location) {
    static const std::regex rid_regex(R"(^r[0-9]+$)");
    const bool lhs_is_rid = std::regex_match(operation.lhs, rid_regex);
    const bool rhs_is_rid = std::regex_match(operation.rhs, rid_regex);
    const bool dst_is_rid = std::regex_match(operation.dst, rid_regex);
    if (!lhs_is_rid || !rhs_is_rid || !dst_is_rid) {
      return;
    }

    const std::optional<RegistryType> lhs_type =
        resolve_register_type(operation.lhs, operation.lhsType, location);
    const std::optional<RegistryType> rhs_type =
        resolve_register_type(operation.rhs, operation.rhsType, location);

    if (!lhs_type.has_value() || !rhs_type.has_value()) {
      return;
    }

    if (lhs_type->laneEnum != rhs_type->laneEnum) {
      add_diag(location, "invalid cross-lane arithmetic for opcode '" + operation.opcode +
                             "': lhs lane enum " + lhs_type->laneEnum +
                             " is incompatible with rhs lane enum " + rhs_type->laneEnum);
      return;
    }

    if (lhs_type->width != rhs_type->width ||
        lhs_type->tileWidth != rhs_type->tileWidth ||
        lhs_type->dtype != rhs_type->dtype) {
      add_diag(location, "binary operand type mismatch for opcode '" + operation.opcode +
                             "': " + registry_type_to_text(*lhs_type) + " vs " +
                             registry_type_to_text(*rhs_type));
      return;
    }

    const std::optional<RegistryType> dst_type =
        resolve_register_type(operation.dst, operation.dstType, location);
    if (!dst_type.has_value()) {
      return;
    }

    if (dst_type->laneEnum != lhs_type->laneEnum ||
        dst_type->width != lhs_type->width ||
        dst_type->tileWidth != lhs_type->tileWidth ||
        dst_type->dtype != lhs_type->dtype) {
      add_diag(location, "binary result type mismatch for opcode '" + operation.opcode +
                             "': result " + registry_type_to_text(*dst_type) +
                             " is incompatible with operand type " +
                             registry_type_to_text(*lhs_type));
      return;
    }

    register_types[operation.dst] = *lhs_type;
  }

  void validate_load_store_type_compatibility(const LdStOperation& operation,
                                              const SourceLocation& location) {
    const std::optional<RegistryType> rid_type =
        resolve_register_type(operation.rid, operation.registryType, location);
    if (!rid_type.has_value()) {
      return;
    }

    auto memory_it = memory_types.find(operation.target);
    if (memory_it == memory_types.end()) {
      memory_types[operation.target] = *rid_type;
      return;
    }

    const RegistryType& expected = memory_it->second;
    if (expected.laneEnum != rid_type->laneEnum ||
        expected.width != rid_type->width ||
        expected.tileWidth != rid_type->tileWidth ||
        expected.dtype != rid_type->dtype) {
      const std::string op_name = operation.isLoad ? "load" : "store";
      add_diag(location, op_name + " type mismatch on '" + operation.target + "': expected " +
                             registry_type_to_text(expected) + " but got " +
                             registry_type_to_text(*rid_type));
    }
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

  std::unique_ptr<Operation> parse_map_surface(MapSurfaceOperation::SurfaceKind kind) {
    auto operation = std::make_unique<MapSurfaceOperation>(previous().location);
    operation->surfaceKind = kind;

    consume_keyword("rid", "expected 'rid' in MAP operation");
    consume(TokenKind::Equal, "expected '=' after rid");
    operation->rid = expect_identifier("expected RID in MAP operation");

    consume_keyword("lane", "expected 'lane' in MAP operation");
    consume(TokenKind::Equal, "expected '=' after lane");
    const Token lane_token = advance();
    operation->lane = parse_lane_token(lane_token, /*for_directive=*/true);

    if (kind == MapSurfaceOperation::SurfaceKind::Store) {
      consume_keyword("to", "expected 'to' in MAP store operation");
    } else {
      consume_keyword("from", "expected 'from' in MAP operation");
    }
    operation->target = expect_identifier("expected MAP target");
    consume(TokenKind::Semicolon, "expected ';' after MAP operation");
    return operation;
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
    diagnostics.push_back(make_diagnostic(location, std::move(message)));
  }
};

class RopVerifier {
 public:
  explicit RopVerifier(const ModuleNode& module) : module_(module) {}

  VerifyResult run() {
    add_pass("phase_monotonicity_and_annotations",
             [this]() { verify_phase_monotonicity_and_annotations(); });
    add_pass("lane_consistency_per_rid",
             [this]() { verify_lane_consistency_per_rid(); });
    add_pass("offset_alignment_legality_by_lane",
             [this]() { verify_offset_alignment_legality_by_lane(); });
    add_pass("barrier_placement_and_hazard_contract",
             [this]() { verify_barrier_placement_and_hazard_contract(); });
    add_pass("concurrent_registry_access_by_lane_and_atomic_constraints",
             [this]() {
               verify_concurrent_registry_access_by_lane_and_atomic_constraints();
             });
    result_.ok = result_.diagnostics.empty();
    return result_;
  }

 private:
  struct RidState {
    int lane = -1;
    SourceLocation firstUse;
  };

  const ModuleNode& module_;
  VerifyResult result_;

  template <typename Fn>
  void add_pass(const std::string& name, Fn&& fn) {
    const std::size_t before = result_.diagnostics.size();
    fn();
    const std::size_t after = result_.diagnostics.size();
    result_.passes.push_back(
        VerifierPassResult{name, after == before, after - before});
  }

  void add_diag(const SourceLocation& location, std::string pass, std::string code,
                std::string message) {
    result_.diagnostics.push_back(
        make_diagnostic(location, std::move(message), std::move(pass),
                        std::move(code)));
  }

  void verify_phase_monotonicity_and_annotations() {
    constexpr const char* kPass = "phase_monotonicity_and_annotations";
    for (const FunctionNode& function : module_.functions) {
      int current_phase = -1;
      for (const auto& operation : function.body.operations) {
        if (operation->kind != Operation::Kind::Phase) {
          continue;
        }
        const auto* phase_op = static_cast<const PhaseOperation*>(operation.get());
        if (phase_op->phase < 0 || phase_op->phase > 3) {
          add_diag(phase_op->location, kPass, "phase.annotation.invalid",
                   "phase annotation must be 0..3");
          continue;
        }
        if (current_phase > phase_op->phase) {
          add_diag(phase_op->location, kPass, "phase.monotonicity.violation",
                   "phase directives must be monotonic within a function");
        }
        current_phase = phase_op->phase;
      }
    }
  }

  void verify_lane_consistency_per_rid() {
    constexpr const char* kPass = "lane_consistency_per_rid";
    for (const FunctionNode& function : module_.functions) {
      std::map<std::string, RidState> rid_lanes;
      for (const auto& operation : function.body.operations) {
        if (operation->kind == Operation::Kind::LdSt) {
          const auto* ld_st = static_cast<const LdStOperation*>(operation.get());
          verify_rid_lane(kPass, ld_st->rid, ld_st->lane, ld_st->location, rid_lanes);
        }
      }
    }
  }

  void verify_offset_alignment_legality_by_lane() {
    constexpr const char* kPass = "offset_alignment_legality_by_lane";
    for (const FunctionNode& function : module_.functions) {
      for (const auto& operation : function.body.operations) {
        if (operation->kind != Operation::Kind::LdSt) {
          continue;
        }
        const auto* ld_st = static_cast<const LdStOperation*>(operation.get());
        if (ld_st->lane < 0 || ld_st->lane > 63) {
          add_diag(ld_st->location, kPass, "lane.range.invalid",
                   "lane for memory access must be in range 0..63");
          continue;
        }
        const std::optional<int> offset = parse_numeric_suffix(ld_st->target);
        if (!offset.has_value()) {
          continue;
        }
        if ((*offset % 2) != (ld_st->lane % 2)) {
          add_diag(ld_st->location, kPass, "alignment.illegal_for_lane",
                   "memory offset alignment is illegal for lane parity");
        }
      }
    }
  }

  void verify_barrier_placement_and_hazard_contract() {
    constexpr const char* kPass = "barrier_placement_and_hazard_contract";
    for (const FunctionNode& function : module_.functions) {
      bool has_pending_memory_hazard = false;
      bool seen_phase = false;
      for (const auto& operation : function.body.operations) {
        if (operation->kind == Operation::Kind::LdSt) {
          has_pending_memory_hazard = true;
          continue;
        }
        if (operation->kind == Operation::Kind::Barrier) {
          has_pending_memory_hazard = false;
          continue;
        }
        if (operation->kind == Operation::Kind::Phase) {
                if (seen_phase && has_pending_memory_hazard) {
            add_diag(operation->location, kPass, "barrier.missing_before_phase",
                     "phase transition requires barrier after memory hazard");
          }
          seen_phase = true;
        }
      }
      if (has_pending_memory_hazard) {
        add_diag(function.body.operations.empty() ? SourceLocation{} :
                     function.body.operations.back()->location,
                 kPass, "barrier.missing_at_function_end",
                 "function ends with unresolved memory hazard; insert barrier");
      }
    }
  }

  void verify_concurrent_registry_access_by_lane_and_atomic_constraints() {
    constexpr const char* kPass =
        "concurrent_registry_access_by_lane_and_atomic_constraints";

    struct AccessState {
      int lane = -1;
      bool seen_atomic = false;
      bool seen_non_atomic = false;
      SourceLocation first_use;
    };

    for (const FunctionNode& function : module_.functions) {
      std::map<std::string, AccessState> epoch_registry_access;

      const auto flush_epoch = [&epoch_registry_access]() {
        epoch_registry_access.clear();
      };

      for (const auto& operation : function.body.operations) {
        if (operation->kind == Operation::Kind::Barrier ||
            operation->kind == Operation::Kind::Phase) {
          flush_epoch();
          continue;
        }

        if (operation->kind != Operation::Kind::MapSurface) {
          continue;
        }

        const auto* map_op = static_cast<const MapSurfaceOperation*>(operation.get());
        if (map_op->lane < 0 || map_op->lane > 63) {
          add_diag(map_op->location, kPass, "registry.lane.range.invalid",
                   "registry lane must be in range 0..63 for concurrent access");
          continue;
        }

        const bool is_atomic =
            map_op->surfaceKind == MapSurfaceOperation::SurfaceKind::AtomicAdd ||
            map_op->surfaceKind == MapSurfaceOperation::SurfaceKind::AtomicSub ||
            map_op->surfaceKind == MapSurfaceOperation::SurfaceKind::AtomicXor;

        auto [it, inserted] = epoch_registry_access.emplace(
            map_op->rid, AccessState{map_op->lane, false, false, map_op->location});
        AccessState& state = it->second;

        if (!inserted && state.lane != map_op->lane &&
            (!state.seen_atomic || !is_atomic)) {
          add_diag(map_op->location, kPass, "registry.concurrent_lane_conflict",
                   "RID '" + map_op->rid +
                       "' accessed by multiple lanes in one phase/epoch without atomic-only semantics");
        }

        if (is_atomic) {
          state.seen_atomic = true;
          if (state.seen_non_atomic) {
            add_diag(map_op->location, kPass, "atomic.constraints.mixed_access",
                     "atomic and non-atomic MAP access to RID '" + map_op->rid +
                         "' must be separated by phase/barrier");
          }
        } else {
          state.seen_non_atomic = true;
          if (state.seen_atomic) {
            add_diag(map_op->location, kPass, "atomic.constraints.mixed_access",
                     "non-atomic MAP access to RID '" + map_op->rid +
                         "' cannot follow atomic access in same phase/epoch");
          }
        }
      }
    }
  }


  void verify_rid_lane(const std::string& pass_name, const std::string& rid, int lane,
                       const SourceLocation& location,
                       std::map<std::string, RidState>& rid_lanes) {
    if (lane < 0) {
      return;
    }
    const auto it = rid_lanes.find(rid);
    if (it == rid_lanes.end()) {
      rid_lanes.emplace(rid, RidState{lane, location});
      return;
    }
    if (it->second.lane != lane) {
      add_diag(location, pass_name, "lane.inconsistent_for_rid",
               "RID '" + rid + "' used with multiple lanes");
    }
  }

  std::optional<int> parse_numeric_suffix(const std::string& text) const {
    std::size_t pos = text.size();
    while (pos > 0 &&
           std::isdigit(static_cast<unsigned char>(text[pos - 1]))) {
      --pos;
    }
    if (pos == text.size()) {
      return std::nullopt;
    }
    return std::stoi(text.substr(pos));
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
      diagnostics->push_back(make_diagnostic(location, message));
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
      case '<':
        emit(TokenKind::Less, "<", loc, &tokens);
        break;
      case '>':
        emit(TokenKind::Greater, ">", loc, &tokens);
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

VerifyResult verify_rop(const ModuleNode& module) {
  RopVerifier verifier(module);
  return verifier.run();
}

}  // namespace bcir
