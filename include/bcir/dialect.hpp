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
  std::string pass;
  std::string code;
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
  enum class Kind { LdSt, Binary, Lane, Phase, Barrier, MacroExpansion, MapSurface };

  explicit Operation(Kind operation_kind, SourceLocation loc = {})
      : kind(operation_kind), location(loc) {}
  virtual ~Operation() = default;

  Kind kind;
  SourceLocation location;
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
  SourceLocation location;
};

class ModuleNode : public Node {
 public:
  std::string name;
  std::vector<MacroDefinitionNode> macros;
  std::vector<FunctionNode> functions;
};

class LdStOperation final : public Operation {
 public:
  explicit LdStOperation(bool is_load, SourceLocation loc = {})
      : Operation(Kind::LdSt, loc), isLoad(is_load) {}

  bool isLoad;
  std::string rid;
  std::optional<std::string> registryType;
  int lane = 0;
  std::string target;
};

class BinaryOpOperation final : public Operation {
 public:
  explicit BinaryOpOperation(SourceLocation loc = {}) : Operation(Kind::Binary, loc) {}

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
  explicit LaneOpOperation(SourceLocation loc = {}) : Operation(Kind::Lane, loc) {}

  std::string opcode;
  std::string rid;
  int lane = 0;
};

class PhaseOperation final : public Operation {
 public:
  explicit PhaseOperation(SourceLocation loc = {}) : Operation(Kind::Phase, loc) {}

  int phase = 0;
};

class BarrierOperation final : public Operation {
 public:
  explicit BarrierOperation(SourceLocation loc = {}) : Operation(Kind::Barrier, loc) {}

  std::string scope;
};

class MacroExpansionOperation final : public Operation {
 public:
  explicit MacroExpansionOperation(SourceLocation loc = {})
      : Operation(Kind::MacroExpansion, loc) {}

  std::string macroName;
  std::vector<std::string> args;
};

class MapSurfaceOperation final : public Operation {
 public:
  enum class SurfaceKind { Load, Store, AtomicAdd, AtomicSub, AtomicXor };

  explicit MapSurfaceOperation(SourceLocation loc = {})
      : Operation(Kind::MapSurface, loc) {}

  SurfaceKind surfaceKind = SurfaceKind::Load;
  std::string rid;
  int lane = 0;
  std::string target;
};

struct ParseResult {
  ModuleNode module;
  std::vector<Diagnostic> diagnostics;
};

ParseResult parse_dialect(std::string_view source);

struct VerifierPassResult {
  std::string name;
  bool passed = true;
  std::size_t diagnosticCount = 0;
};

struct VerifyResult {
  bool ok = true;
  std::vector<VerifierPassResult> passes;
  std::vector<Diagnostic> diagnostics;
};

VerifyResult verify_rop(const ModuleNode& module);

}  // namespace bcir
