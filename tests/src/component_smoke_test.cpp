#include <cstdlib>
#include <iostream>
#include <string>
#include <algorithm>
#include <atomic>

#include "bcir/dialect.hpp"
#include "bcir/lowering.hpp"
#include "bcir/runtime.hpp"

namespace {

bool has_diagnostic_substring(const bcir::ParseResult& result,
                              const std::string& needle) {
  for (const auto& diagnostic : result.diagnostics) {
    if (diagnostic.message.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool has_diagnostic_substring(const bcir::VerifyResult& result,
                              const std::string& needle) {
  for (const auto& diagnostic : result.diagnostics) {
    if (diagnostic.message.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool metadata_equivalent(const bcir::BlockNode& lhs, const bcir::BlockNode& rhs) {
  if (lhs.operations.size() != rhs.operations.size()) {
    return false;
  }

  for (std::size_t i = 0; i < lhs.operations.size(); ++i) {
    const auto* left = lhs.operations[i].get();
    const auto* right = rhs.operations[i].get();
    if (left->kind != right->kind) {
      return false;
    }

    if (left->kind == bcir::Operation::Kind::LdSt) {
      const auto* l = static_cast<const bcir::LdStOperation*>(left);
      const auto* r = static_cast<const bcir::LdStOperation*>(right);
      if (l->isLoad != r->isLoad || l->rid != r->rid ||
          l->registryType != r->registryType || l->lane != r->lane ||
          l->target != r->target) {
        return false;
      }
    } else if (left->kind == bcir::Operation::Kind::Binary) {
      const auto* l = static_cast<const bcir::BinaryOpOperation*>(left);
      const auto* r = static_cast<const bcir::BinaryOpOperation*>(right);
      if (l->opcode != r->opcode || l->dst != r->dst || l->dstType != r->dstType ||
          l->lhs != r->lhs || l->rhs != r->rhs) {
        return false;
      }
    } else if (left->kind == bcir::Operation::Kind::Lane) {
      const auto* l = static_cast<const bcir::LaneOpOperation*>(left);
      const auto* r = static_cast<const bcir::LaneOpOperation*>(right);
      if (l->opcode != r->opcode || l->rid != r->rid || l->lane != r->lane) {
        return false;
      }
    } else if (left->kind == bcir::Operation::Kind::Phase) {
      const auto* l = static_cast<const bcir::PhaseOperation*>(left);
      const auto* r = static_cast<const bcir::PhaseOperation*>(right);
      if (l->phase != r->phase) {
        return false;
      }
    } else if (left->kind == bcir::Operation::Kind::Barrier) {
      const auto* l = static_cast<const bcir::BarrierOperation*>(left);
      const auto* r = static_cast<const bcir::BarrierOperation*>(right);
      if (l->scope != r->scope) {
        return false;
      }
    } else if (left->kind == bcir::Operation::Kind::MapSurface) {
      const auto* l = static_cast<const bcir::MapSurfaceOperation*>(left);
      const auto* r = static_cast<const bcir::MapSurfaceOperation*>(right);
      if (l->surfaceKind != r->surfaceKind || l->rid != r->rid ||
          l->lane != r->lane || l->target != r->target) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  const std::string dialect = bcir::dialect_component_banner();
  const std::string lowering = bcir::lowering_component_banner();
  const std::string runtime = bcir::runtime_component_banner();

  const bool banners_ok = !dialect.empty() && !lowering.empty() && !runtime.empty();
  if (!banners_ok) {
    std::cerr << "BCIR component banners must not be empty" << std::endl;
    return EXIT_FAILURE;
  }

  bcir::GemStatus create_status = bcir::GemStatus::RuntimeError;
  std::string create_message;
  bcir::GemCreateOptions gem_options;
  gem_options.workerThreads = 2;
  auto gem = bcir::gem_create(gem_options, &create_status, &create_message);
  if (gem == nullptr || create_status != bcir::GemStatus::Ok) {
    std::cerr << "Expected gem runtime creation to succeed: " << create_message
              << std::endl;
    return EXIT_FAILURE;
  }
  gem->registry().allocate("rid.r0", 32);
  if (gem->registry().bytes_for("rid.r0") != 32 ||
      gem->registry().lookup("rid.r0") == nullptr) {
    std::cerr << "Expected registry allocation/lookup to succeed" << std::endl;
    return EXIT_FAILURE;
  }

  std::atomic<int> execution_trace{0};
  bcir::GemGraph graph;
  graph.nodes.push_back({0, 0, {}, [&execution_trace]() { execution_trace += 1; }});
  graph.nodes.push_back({1, 0, {}, [&execution_trace]() { execution_trace += 10; }});
  graph.nodes.push_back({2, 1, {0, 1}, [&execution_trace]() { execution_trace += 100; }});

  const bcir::GemExecuteResult execution = bcir::gem_execute(gem.get(), graph);
  if (execution.status != bcir::GemStatus::Ok || execution.executedNodes != 3 ||
      execution_trace.load() != 111) {
    std::cerr << "Expected GEM execute to process phase-scheduled graph"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::string destroy_message;
  if (bcir::gem_destroy(std::move(gem), &destroy_message) != bcir::GemStatus::Ok) {
    std::cerr << "Expected GEM destroy to be thread-safe and successful" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string valid_program = R"(
module test {
  macro pair(x, y) {
    bin add r3, x, y;
    map_load rid=r3 lane=lane2 from mem8;
  }

  fn main() {
    ld rid=r0<U x 8 x i32> lane=lane2 from mem0;
    ld rid=r1<U x 8 x i32> lane=lane5 from mem1;
    st rid=r1<U x 8 x i32> lane=lane5 to mem1;
    bin add r2<U x 8 x i32>, r0<U x 8 x i32>, r1<U x 8 x i32>;
    lane shuffle r1 lane7;
    phase 2;
    barrier workgroup;
    expand pair(r0, r1);
    map_store rid=r2 lane=lane2 to mem8;
    barrier workgroup;
    map_atomic_add rid=r2 lane=lane2 from mem8;
  }
}
)";

  bcir::ParseResult parsed_valid = bcir::parse_dialect(valid_program);
  if (!parsed_valid.diagnostics.empty()) {
    std::cerr << "Expected valid program to parse without diagnostics" << std::endl;
    return EXIT_FAILURE;
  }
  std::vector<bcir::Diagnostic> expand_diags =
      bcir::expand_macros(&parsed_valid.module);
  if (!expand_diags.empty()) {
    std::cerr << "Expected macro expansion to succeed for valid program" << std::endl;
    return EXIT_FAILURE;
  }
  bcir::lower_map_surface_ops(&parsed_valid.module);
  const auto& lowered_ops = parsed_valid.module.functions.front().body.operations;
  if (std::any_of(lowered_ops.begin(), lowered_ops.end(), [](const auto& op) {
        return op->kind == bcir::Operation::Kind::MapSurface ||
               op->kind == bcir::Operation::Kind::MacroExpansion;
      })) {
    std::cerr << "Expected MAP and macro surface operations to be lowered away"
              << std::endl;
    return EXIT_FAILURE;
  }

  const bcir::VerifyResult verified_valid = bcir::verify_rop(parsed_valid.module);
  if (!verified_valid.ok || !verified_valid.diagnostics.empty() ||
      verified_valid.passes.size() != 5) {
    std::cerr << "Expected valid program to verify without diagnostics"
              << std::endl;
    return EXIT_FAILURE;
  }

  const std::string roundtrip_program = R"(
module test {
  fn roundtrip() {
    ld rid=r0<U x 8 x i32> lane=lane2 from mem8;
    bin add r1<U x 8 x i32>, r0<U x 8 x i32>, r0<U x 8 x i32>;
    lane shuffle r1 lane3;
    phase 1;
    barrier workgroup;
    map_atomic_xor rid=r1 lane=lane3 from mem12;
  }
}
)";
  const bcir::ParseResult parsed_roundtrip = bcir::parse_dialect(roundtrip_program);
  if (!parsed_roundtrip.diagnostics.empty()) {
    std::cerr << "Expected round-trip program to parse successfully" << std::endl;
    return EXIT_FAILURE;
  }
  const bcir::BlockNode& roundtrip_block =
      parsed_roundtrip.module.functions.front().body;
  const std::vector<bcir::RopInstruction> rop_stream =
      bcir::bcir_graph_to_rop_stream(roundtrip_block);
  const bcir::BlockNode reconstructed =
      bcir::rop_stream_to_bcir_graph(rop_stream);
  if (!metadata_equivalent(roundtrip_block, reconstructed)) {
    std::cerr << "BCIR <-> ROP translation failed to preserve graph metadata"
              << std::endl;
    return EXIT_FAILURE;
  }
  if (rop_stream.size() < 2 || rop_stream[0].opcode != "ld" ||
      !rop_stream[0].offset.has_value() || *rop_stream[0].offset != 8 ||
      rop_stream[2].dependencies.empty() ||
      rop_stream.back().dependencies.empty()) {
    std::cerr << "ROP stream metadata did not preserve opcode/offset/dependency"
              << std::endl;
    return EXIT_FAILURE;
  }
  const auto lowering_table = bcir::rop_opcode_to_llvm_lowerings();
  if (lowering_table.empty()) {
    std::cerr << "Expected non-empty ROP->LLVM lowering table" << std::endl;
    return EXIT_FAILURE;
  }
  const auto* ld_lowering = bcir::find_rop_opcode_lowering("ld");
  const auto* barrier_lowering = bcir::find_rop_opcode_lowering("barrier");
  const auto* atomic_lowering = bcir::find_rop_opcode_lowering("map_atomic_add");
  if (ld_lowering == nullptr || ld_lowering->category != "memory" ||
      barrier_lowering == nullptr || !barrier_lowering->isBarrier ||
      atomic_lowering == nullptr || !atomic_lowering->isAtomic) {
    std::cerr << "Expected opcode categories for memory/barrier/atomic lowering"
              << std::endl;
    return EXIT_FAILURE;
  }
  bcir::RopInstruction atomic_instruction;
  atomic_instruction.opcode = "map_atomic_add";
  atomic_instruction.rid = "r2";
  const std::string neutral_dispatch = bcir::lower_rop_instruction_to_llvm_dispatch(
      atomic_instruction, bcir::LlvmTargetBackend::Neutral);
  const std::string gpu_dispatch = bcir::lower_rop_instruction_to_llvm_dispatch(
      atomic_instruction, bcir::LlvmTargetBackend::GPU);
  if (neutral_dispatch.find("emit_map_atomic_add") == std::string::npos ||
      gpu_dispatch.find("gpu.emit_surface_atomic_add") == std::string::npos) {
    std::cerr << "Expected target-neutral and backend extension lowering"
              << std::endl;
    return EXIT_FAILURE;
  }

  const std::string invalid_program = R"(
module test {
  macro broken(a, a) {
    phase 9;
  }

  fn main() {
    ld rid=rx<Z x 8 x i32> lane=laneZZ from mem0;
    ld rid=r0<U x 0 x i32> lane=lane1 from mem2;
    ld rid=r1<U x 8 x nope> lane=lane2 from mem3;
    ld rid=r2<UX x 4 x i16> lane=lane3 from mem4;
    ld rid=r3<T x 4 x i16> lane=lane4 from mem5;
    bin add r4<UX x 4 x i16>, r2<UX x 4 x i16>, r3<T x 4 x i16>;
    st rid=r3<T x 4 x i16> lane=lane4 to mem4;
    lane rotate x lane77;
  }
}
)";

  const bcir::ParseResult parsed_invalid = bcir::parse_dialect(invalid_program);
  if (!has_diagnostic_substring(parsed_invalid, "malformed RID token") ||
      !has_diagnostic_substring(parsed_invalid, "malformed lane token") ||
      !has_diagnostic_substring(parsed_invalid, "malformed macro params") ||
      !has_diagnostic_substring(parsed_invalid, "invalid phase directive") ||
      !has_diagnostic_substring(parsed_invalid, "invalid lane directive") ||
      !has_diagnostic_substring(parsed_invalid, "invalid lane enum") ||
      !has_diagnostic_substring(parsed_invalid, "invalid width constraint") ||
      !has_diagnostic_substring(parsed_invalid, "invalid data type") ||
      !has_diagnostic_substring(parsed_invalid, "invalid cross-lane arithmetic") ||
      !has_diagnostic_substring(parsed_invalid, "type mismatch")) {
    std::cerr << "Expected parser diagnostics were not emitted" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string invalid_verify_program = R"(
module test {
  fn main() {
    ld rid=r0<U x 8 x i32> lane=lane3 from mem0;
    st rid=r0<U x 8 x i32> lane=lane4 to mem0;
    lane shuffle r0 lane5;
    phase 1;
    phase 0;
  }
}
)";
  const bcir::ParseResult parsed_invalid_verify =
      bcir::parse_dialect(invalid_verify_program);
  if (!parsed_invalid_verify.diagnostics.empty()) {
    std::cerr << "Expected verifier-negative program to parse successfully"
              << std::endl;
    return EXIT_FAILURE;
  }
  const bcir::VerifyResult verified_invalid =
      bcir::verify_rop(parsed_invalid_verify.module);
  if (verified_invalid.ok ||
      !has_diagnostic_substring(verified_invalid, "multiple lanes") ||
      !has_diagnostic_substring(verified_invalid, "monotonic") ||
      !has_diagnostic_substring(verified_invalid, "illegal for lane parity") ||
      !has_diagnostic_substring(verified_invalid, "requires barrier")) {
    std::cerr << "Expected verifier diagnostics were not emitted" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string bad_macro_program = R"(
module test {
  macro pair(x, y) {
    bin add r1, x, y;
  }
  fn main() {
    expand pair(r0);
  }
}
)";
  bcir::ParseResult parsed_bad_macro = bcir::parse_dialect(bad_macro_program);
  std::vector<bcir::Diagnostic> bad_macro_diags =
      bcir::expand_macros(&parsed_bad_macro.module);
  if (!has_diagnostic_substring(parsed_bad_macro, "expected macro argument") &&
      !has_diagnostic_substring(bcir::VerifyResult{true, {}, bad_macro_diags},
                                "expects 2 argument")) {
    std::cerr << "Expected macro expansion diagnostics were not emitted"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "BCIR smoke test passed" << std::endl;
  return EXIT_SUCCESS;
}
