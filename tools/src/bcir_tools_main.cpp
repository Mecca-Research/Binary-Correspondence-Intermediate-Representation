#include <iostream>
#include <stdexcept>
#include <string>

#include "bcir/dialect.hpp"
#include "bcir/lowering.hpp"
#include "bcir/runtime.hpp"

namespace {

std::string error_class_string(bcir::GemErrorClass c) {
  switch (c) {
    case bcir::GemErrorClass::None:
      return "none";
    case bcir::GemErrorClass::Recoverable:
      return "recoverable";
    case bcir::GemErrorClass::InvalidInput:
      return "invalid_input";
    case bcir::GemErrorClass::Timeout:
      return "timeout";
    case bcir::GemErrorClass::ResourceExhausted:
      return "resource_exhausted";
    case bcir::GemErrorClass::Internal:
      return "internal";
  }
  return "unknown";
}

void print_runtime_diagnostics() {
  bcir::GemStatus status = bcir::GemStatus::RuntimeError;
  auto runtime = bcir::gem_create({}, &status, nullptr);
  if (runtime == nullptr || status != bcir::GemStatus::Ok) {
    std::cerr << "failed to create runtime diagnostics probe\n";
    return;
  }

  runtime->set_error_classifier([](const std::exception_ptr& ex) {
    try {
      if (ex != nullptr) {
        std::rethrow_exception(ex);
      }
    } catch (const std::runtime_error&) {
      return bcir::GemErrorClass::Recoverable;
    } catch (...) {
      return bcir::GemErrorClass::Internal;
    }
    return bcir::GemErrorClass::Internal;
  });
  runtime->set_correction_hook([](const bcir::GemExecutionEvent&) { return true; });

  int attempts = 0;
  runtime->registry().allocate("diag.r0", 64);

  bcir::GemGraph graph;
  graph.nodes.push_back(
      {0,
       0,
       {},
       [&attempts]() {
         attempts += 1;
         if (attempts == 1) {
           throw std::runtime_error("transient");
         }
       },
       []() {},
       1,
       "diag.r0"});
  graph.nodes.push_back({1, 1, {0}, []() {}, nullptr, 0, "diag.r0"});

  const bcir::GemExecuteResult result = bcir::gem_execute(runtime.get(), graph);
  std::cout << "runtime.status=" << static_cast<int>(result.status)
            << " executed=" << result.executedNodes << " message=" << result.message
            << '\n';

  for (const auto& phase : result.telemetry.phaseStats) {
    std::cout << "phase=" << phase.phase << " scheduled=" << phase.scheduledNodes
              << " attempted=" << phase.attemptedNodes << " succeeded="
              << phase.succeededNodes << " failed=" << phase.failedNodes
              << " retries=" << phase.retryCount << " elapsed_ms=" << phase.elapsedMs
              << '\n';
  }

  for (const auto& event : result.telemetry.events) {
    std::cout << "event.node=" << event.nodeId << " phase=" << event.phase
              << " attempt=" << event.attempt
              << " class=" << error_class_string(event.classification)
              << " rollback=" << (event.rollbackAttempted ? "yes" : "no")
              << " corrected=" << (event.correctionApplied ? "yes" : "no")
              << " recovered=" << (event.recovered ? "yes" : "no") << " msg="
              << event.message << '\n';
  }

  for (const auto& reg : result.telemetry.registryStats) {
    std::cout << "registry=" << reg.name << " bytes=" << reg.bytes
              << " allocs=" << reg.allocations << " deallocs=" << reg.deallocations
              << " lookups=" << reg.lookups << '\n';
  }

  bcir::gem_destroy(std::move(runtime), nullptr);
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << bcir::dialect_component_banner() << '\n';
  std::cout << bcir::lowering_component_banner() << '\n';
  std::cout << bcir::runtime_component_banner() << '\n';
  if (argc > 1 && std::string(argv[1]) == "--runtime-diag") {
    print_runtime_diagnostics();
  }
  return 0;
}
