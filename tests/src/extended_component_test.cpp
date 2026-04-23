#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "bcir/dialect.hpp"
#include "bcir/lowering.hpp"
#include "bcir/runtime.hpp"

namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    return {};
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

std::string trim(std::string value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' ')) {
    value.pop_back();
  }
  std::size_t i = 0;
  while (i < value.size() && (value[i] == ' ' || value[i] == '\n' || value[i] == '\r' || value[i] == '\t')) {
    ++i;
  }
  return value.substr(i);
}

bool has_diag(const std::vector<bcir::Diagnostic>& diags, const std::string& needle) {
  for (const auto& diag : diags) {
    if (diag.message.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

std::string dump_block(const bcir::BlockNode& block) {
  std::ostringstream out;
  for (const auto& op_ptr : block.operations) {
    const auto* op = op_ptr.get();
    switch (op->kind) {
      case bcir::Operation::Kind::LdSt: {
        const auto* o = static_cast<const bcir::LdStOperation*>(op);
        out << (o->isLoad ? "ld" : "st") << " rid=" << o->rid << " lane=" << o->lane
            << " target=" << o->target << "\n";
        break;
      }
      case bcir::Operation::Kind::Binary: {
        const auto* o = static_cast<const bcir::BinaryOpOperation*>(op);
        out << "bin " << o->opcode << " dst=" << o->dst << " lhs=" << o->lhs << " rhs=" << o->rhs << "\n";
        break;
      }
      case bcir::Operation::Kind::Lane: {
        const auto* o = static_cast<const bcir::LaneOpOperation*>(op);
        out << "lane " << o->opcode << " rid=" << o->rid << " lane=" << o->lane << "\n";
        break;
      }
      case bcir::Operation::Kind::Phase: {
        const auto* o = static_cast<const bcir::PhaseOperation*>(op);
        out << "phase " << o->phase << "\n";
        break;
      }
      case bcir::Operation::Kind::Barrier: {
        const auto* o = static_cast<const bcir::BarrierOperation*>(op);
        out << "barrier " << o->scope << "\n";
        break;
      }
      case bcir::Operation::Kind::MacroExpansion: {
        const auto* o = static_cast<const bcir::MacroExpansionOperation*>(op);
        out << "expand " << o->macroName << " argc=" << o->args.size() << "\n";
        break;
      }
      case bcir::Operation::Kind::MapSurface: {
        const auto* o = static_cast<const bcir::MapSurfaceOperation*>(op);
        out << "map rid=" << o->rid << " lane=" << o->lane << " target=" << o->target << " kind="
            << static_cast<int>(o->surfaceKind) << "\n";
        break;
      }
    }
  }
  return out.str();
}

bool test_parser_verifier_unit() {
  const std::string good = R"(
module t {
  fn main() {
    ld rid=r0<U x 8 x i32> lane=lane0 from mem0;
    map_atomic_add rid=r0 lane=lane0 from mem0;
    barrier workgroup;
    phase 1;
  }
}
)";
  const auto parsed_good = bcir::parse_dialect(good);
  if (!parsed_good.diagnostics.empty()) {
    std::cerr << "parser/verifier unit: expected clean parse" << std::endl;
    return false;
  }
  const auto verified_good = bcir::verify_rop(parsed_good.module);
  if (!verified_good.ok) {
    std::cerr << "parser/verifier unit: expected verifier success" << std::endl;
    return false;
  }

  const std::string bad = R"(
module t {
  fn main() {
    map_load rid=r0 lane=lane1 from mem0;
    map_atomic_xor rid=r0 lane=lane2 from mem0;
    phase 2;
    phase 1;
  }
}
)";
  const auto parsed_bad = bcir::parse_dialect(bad);
  const auto verified_bad = bcir::verify_rop(parsed_bad.module);
  if (verified_bad.ok ||
      !has_diag(verified_bad.diagnostics, "multiple lanes") ||
      !has_diag(verified_bad.diagnostics, "monotonic") ||
      !has_diag(verified_bad.diagnostics, "atomic and non-atomic")) {
    std::cerr << "parser/verifier unit: expected verifier diagnostics" << std::endl;
    return false;
  }
  return true;
}

bool run_macro_golden_case(const std::filesystem::path& input_path,
                           const std::filesystem::path& expected_path) {
  const auto source = read_file(input_path);
  const auto expected = read_file(expected_path);
  if (source.empty() || expected.empty()) {
    std::cerr << "macro golden: missing fixture " << input_path << " or " << expected_path << std::endl;
    return false;
  }

  auto parsed = bcir::parse_dialect(source);
  if (!parsed.diagnostics.empty()) {
    std::cerr << "macro golden: parse failed for " << input_path << std::endl;
    return false;
  }
  const auto diags = bcir::expand_macros(&parsed.module);
  if (!diags.empty()) {
    std::cerr << "macro golden: expansion failed for " << input_path << std::endl;
    return false;
  }

  const std::string actual = dump_block(parsed.module.functions.front().body);
  if (trim(actual) != trim(expected)) {
    std::cerr << "macro golden mismatch for " << input_path << "\nACTUAL:\n" << actual << "\nEXPECTED:\n" << expected << std::endl;
    return false;
  }
  return true;
}

bool test_macro_golden() {
  const std::filesystem::path base = std::filesystem::path(BCIR_SOURCE_DIR) / "tests/data/golden";
  return run_macro_golden_case(base / "macro_vector_add.in.bcir", base / "macro_vector_add.expected.txt") &&
         run_macro_golden_case(base / "macro_hygiene.in.bcir", base / "macro_hygiene.expected.txt");
}

bool test_roundtrip() {
  const std::string program = R"(
module t {
  fn main() {
    ld rid=r0<U x 4 x i32> lane=lane1 from mem4;
    bin add r1<U x 4 x i32>, r0<U x 4 x i32>, r0<U x 4 x i32>;
    phase 1;
    barrier workgroup;
    map_store rid=r1 lane=lane1 to mem8;
    map_atomic_sub rid=r1 lane=lane1 from mem8;
  }
}
)";
  const auto parsed = bcir::parse_dialect(program);
  if (!parsed.diagnostics.empty()) {
    std::cerr << "roundtrip: parse failed" << std::endl;
    return false;
  }

  const auto original_dump = dump_block(parsed.module.functions.front().body);
  const auto rop = bcir::bcir_graph_to_rop_stream(parsed.module.functions.front().body);
  const auto restored = bcir::rop_stream_to_bcir_graph(rop);
  const auto restored_dump = dump_block(restored);

  if (trim(original_dump) != trim(restored_dump)) {
    std::cerr << "roundtrip: metadata mismatch" << std::endl;
    return false;
  }
  if (rop.size() < 6 || rop.front().opcode != "ld" || rop.back().opcode != "map_atomic_sub") {
    std::cerr << "roundtrip: unexpected rop stream shape" << std::endl;
    return false;
  }
  return true;
}

struct ExampleResult {
  std::vector<int32_t> output;
  std::size_t executed = 0;
  std::size_t phase_count = 0;
};

ExampleResult execute_vector_add(std::size_t worker_threads) {
  bcir::GemStatus status = bcir::GemStatus::RuntimeError;
  auto runtime = bcir::gem_create(bcir::GemCreateOptions{worker_threads, true, 0}, &status, nullptr);
  if (!runtime || status != bcir::GemStatus::Ok) {
    return {};
  }

  auto* a = reinterpret_cast<int32_t*>(runtime->registry().allocate("vector_add.a", 4 * sizeof(int32_t)));
  auto* b = reinterpret_cast<int32_t*>(runtime->registry().allocate("vector_add.b", 4 * sizeof(int32_t)));
  auto* c = reinterpret_cast<int32_t*>(runtime->registry().allocate("vector_add.c", 4 * sizeof(int32_t)));

  bcir::GemGraph graph;
  graph.nodes.push_back({0, 0, {}, [a]() { int32_t v[] = {1, 2, 3, 4}; std::copy(v, v + 4, a); }});
  graph.nodes.push_back({1, 0, {}, [b]() { int32_t v[] = {10, 20, 30, 40}; std::copy(v, v + 4, b); }});
  graph.nodes.push_back({2, 1, {0, 1}, [a, b, c]() {
                           for (int i = 0; i < 4; ++i) c[i] = a[i] + b[i];
                         }});

  auto exec = bcir::gem_execute(runtime.get(), graph);
  ExampleResult out;
  out.executed = exec.executedNodes;
  out.phase_count = exec.telemetry.phaseStats.size();
  out.output.assign(c, c + 4);
  bcir::gem_destroy(std::move(runtime), nullptr);
  return out;
}

ExampleResult execute_matmul(std::size_t worker_threads) {
  bcir::GemStatus status = bcir::GemStatus::RuntimeError;
  auto runtime = bcir::gem_create(bcir::GemCreateOptions{worker_threads, false, 0}, &status, nullptr);
  if (!runtime || status != bcir::GemStatus::Ok) {
    return {};
  }

  auto* a = reinterpret_cast<int32_t*>(runtime->registry().allocate("matmul.a", 4 * sizeof(int32_t)));
  auto* b = reinterpret_cast<int32_t*>(runtime->registry().allocate("matmul.b", 4 * sizeof(int32_t)));
  auto* c = reinterpret_cast<int32_t*>(runtime->registry().allocate("matmul.c", 4 * sizeof(int32_t)));

  bcir::GemGraph graph;
  graph.nodes.push_back({0, 0, {}, [a]() { int32_t v[] = {1, 2, 3, 4}; std::copy(v, v + 4, a); }});
  graph.nodes.push_back({1, 0, {}, [b]() { int32_t v[] = {5, 6, 7, 8}; std::copy(v, v + 4, b); }});
  graph.nodes.push_back({2, 1, {0, 1}, [a, b, c]() {
                           c[0] = a[0] * b[0] + a[1] * b[2];
                           c[1] = a[0] * b[1] + a[1] * b[3];
                           c[2] = a[2] * b[0] + a[3] * b[2];
                           c[3] = a[2] * b[1] + a[3] * b[3];
                         }});

  auto exec = bcir::gem_execute(runtime.get(), graph);
  ExampleResult out;
  out.executed = exec.executedNodes;
  out.phase_count = exec.telemetry.phaseStats.size();
  out.output.assign(c, c + 4);
  bcir::gem_destroy(std::move(runtime), nullptr);
  return out;
}

bool parse_expected_example(const std::filesystem::path& path,
                            std::vector<int32_t>* expected_out,
                            std::size_t* expected_nodes,
                            std::size_t* expected_phases) {
  const std::string text = read_file(path);
  if (text.empty()) {
    return false;
  }
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("output=", 0) == 0) {
      std::istringstream values(line.substr(7));
      std::string token;
      while (std::getline(values, token, ',')) {
        expected_out->push_back(std::stoi(token));
      }
    } else if (line.rfind("executed_nodes=", 0) == 0) {
      *expected_nodes = static_cast<std::size_t>(std::stoul(line.substr(15)));
    } else if (line.rfind("phase_count=", 0) == 0) {
      *expected_phases = static_cast<std::size_t>(std::stoul(line.substr(12)));
    }
  }
  return !expected_out->empty();
}

bool test_gem_execution_single_and_multi_thread() {
  const auto single = execute_vector_add(1);
  const auto multi = execute_vector_add(4);
  if (single.output != std::vector<int32_t>({11, 22, 33, 44}) ||
      multi.output != std::vector<int32_t>({11, 22, 33, 44})) {
    std::cerr << "gem execution: vector_add output mismatch" << std::endl;
    return false;
  }
  if (single.executed != 3 || multi.executed != 3 || single.phase_count != 2 || multi.phase_count != 2) {
    std::cerr << "gem execution: vector_add stats mismatch" << std::endl;
    return false;
  }

  const auto mat_single = execute_matmul(1);
  const auto mat_multi = execute_matmul(4);
  if (mat_single.output != std::vector<int32_t>({19, 22, 43, 50}) ||
      mat_multi.output != std::vector<int32_t>({19, 22, 43, 50})) {
    std::cerr << "gem execution: matmul output mismatch" << std::endl;
    return false;
  }
  return true;
}

bool test_reference_examples() {
  std::vector<int32_t> out;
  std::size_t nodes = 0;
  std::size_t phases = 0;

  if (!parse_expected_example(std::filesystem::path(BCIR_SOURCE_DIR) / "tests/data/examples/vector_add.expected", &out, &nodes, &phases)) {
    std::cerr << "reference examples: missing vector_add expected" << std::endl;
    return false;
  }
  const auto actual_v = execute_vector_add(2);
  if (actual_v.output != out || actual_v.executed != nodes || actual_v.phase_count != phases) {
    std::cerr << "reference examples: vector_add expected mismatch" << std::endl;
    return false;
  }

  out.clear();
  nodes = 0;
  phases = 0;
  if (!parse_expected_example(std::filesystem::path(BCIR_SOURCE_DIR) / "tests/data/examples/matmul.expected", &out, &nodes, &phases)) {
    std::cerr << "reference examples: missing matmul expected" << std::endl;
    return false;
  }
  const auto actual_m = execute_matmul(2);
  if (actual_m.output != out || actual_m.executed != nodes || actual_m.phase_count != phases) {
    std::cerr << "reference examples: matmul expected mismatch" << std::endl;
    return false;
  }
  return true;
}

bool test_runtime_regression_checks() {
  const auto baseline = execute_vector_add(1);
  const auto candidate = execute_vector_add(8);
  if (baseline.output != candidate.output) {
    std::cerr << "runtime regression: output changed across worker counts" << std::endl;
    return false;
  }
  if (baseline.executed != candidate.executed || baseline.phase_count != candidate.phase_count) {
    std::cerr << "runtime regression: telemetry shape changed" << std::endl;
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "all";

  const std::unordered_map<std::string, std::function<bool()>> tests = {
      {"parser_verifier", test_parser_verifier_unit},
      {"macro_golden", test_macro_golden},
      {"roundtrip", test_roundtrip},
      {"gem_exec", test_gem_execution_single_and_multi_thread},
      {"reference_examples", test_reference_examples},
      {"runtime_regression", test_runtime_regression_checks},
  };

  if (mode == "all") {
    for (const auto& [name, fn] : tests) {
      if (!fn()) {
        std::cerr << "failed test suite: " << name << std::endl;
        return EXIT_FAILURE;
      }
    }
    std::cout << "extended BCIR tests passed" << std::endl;
    return EXIT_SUCCESS;
  }

  const auto it = tests.find(mode);
  if (it == tests.end()) {
    std::cerr << "unknown mode: " << mode << std::endl;
    return EXIT_FAILURE;
  }
  if (!it->second()) {
    std::cerr << "failed test suite: " << mode << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "passed test suite: " << mode << std::endl;
  return EXIT_SUCCESS;
}
