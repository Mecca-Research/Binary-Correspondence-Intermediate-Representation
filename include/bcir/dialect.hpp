#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace bcir {

std::string dialect_component_banner();

struct SourceLocation {
  std::size_t line = 1;
  std::size_t column = 1;
  std::size_t offset = 0;
};

struct Diagnostic {
  SourceLocation location;
  std::string message;
};

enum class TokenKind {
  Identifier,
  Number,
  LBrace,
  RBrace,
  LParen,
  RParen,
  Comma,
  Semicolon,
  Equal,
  Less,
  Greater,
  EndOfFile,
};

struct Token {
  TokenKind kind = TokenKind::EndOfFile;
  std::string lexeme;
  SourceLocation location;
};

std::vector<Token> tokenize_dialect(std::string_view source,
                                    std::vector<Diagnostic>* diagnostics);

class Node {
 public:
  virtual ~Node() = default;
};

class Operation : public Node {
 public:
  enum class Kind { LdSt, Binary, Lane, Phase, Barrier, MacroExpansion };

  explicit Operation(Kind operation_kind) : kind(operation_kind) {}
  virtual ~Operation() = default;

  Kind kind;
};

class BlockNode : public Node {
 public:
  std::vector<std::unique_ptr<Operation>> operations;
};

class FunctionNode : public Node {
 public:
  std::string name;
  std::vector<std::string> params;
  BlockNode body;
};

class MacroDefinitionNode : public Node {
 public:
  std::string name;
  std::vector<std::string> params;
  BlockNode body;
};

class ModuleNode : public Node {
 public:
  std::string name;
  std::vector<MacroDefinitionNode> macros;
  std::vector<FunctionNode> functions;
};

class LdStOperation final : public Operation {
 public:
  explicit LdStOperation(bool is_load)
      : Operation(Kind::LdSt), isLoad(is_load) {}

  bool isLoad;
  std::string rid;
  std::optional<std::string> registryType;
  int lane = 0;
  std::string target;
};

class BinaryOpOperation final : public Operation {
 public:
  BinaryOpOperation() : Operation(Kind::Binary) {}

  std::string opcode;
  std::string dst;
  std::optional<std::string> dstType;
  std::string lhs;
  std::optional<std::string> lhsType;
  std::string rhs;
  std::optional<std::string> rhsType;
};

class LaneOpOperation final : public Operation {
 public:
  LaneOpOperation() : Operation(Kind::Lane) {}

  std::string opcode;
  std::string rid;
  int lane = 0;
};

class PhaseOperation final : public Operation {
 public:
  PhaseOperation() : Operation(Kind::Phase) {}

  int phase = 0;
};

class BarrierOperation final : public Operation {
 public:
  BarrierOperation() : Operation(Kind::Barrier) {}

  std::string scope;
};

class MacroExpansionOperation final : public Operation {
 public:
  MacroExpansionOperation() : Operation(Kind::MacroExpansion) {}

  std::string macroName;
  std::vector<std::string> args;
};

struct ParseResult {
  ModuleNode module;
  std::vector<Diagnostic> diagnostics;
};

ParseResult parse_dialect(std::string_view source);

}  // namespace bcir
