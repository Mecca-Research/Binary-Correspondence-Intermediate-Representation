#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace bcir {

// GEM graph execution runtime entry point.
std::string runtime_component_banner();

enum class GemStatus {
  Ok = 0,
  InvalidArgument,
  InvalidGraph,
  RuntimeShutdown,
  RuntimeError,
};

enum class GemErrorClass {
  None = 0,
  Recoverable,
  InvalidInput,
  Timeout,
  ResourceExhausted,
  Internal,
};

struct GemNode {
  std::size_t id = 0;
  std::size_t phase = 0;
  std::vector<std::size_t> dependencies;
  std::function<void()> work;
  std::function<void()> rollback;
  std::size_t maxReexecuteAttempts = 0;
  std::string registry;
};

struct GemGraph {
  std::vector<GemNode> nodes;
};

struct GemCreateOptions {
  std::size_t workerThreads = 0;
  bool deterministicOrdering = false;
  std::size_t phaseWaitTimeoutMs = 0;
};

struct GemExecutionEvent {
  std::size_t nodeId = 0;
  std::size_t phase = 0;
  std::size_t attempt = 0;
  GemErrorClass classification = GemErrorClass::None;
  bool rollbackAttempted = false;
  bool correctionApplied = false;
  bool recovered = false;
  std::string message;
};

struct GemPhaseStats {
  std::size_t phase = 0;
  std::size_t scheduledNodes = 0;
  std::size_t attemptedNodes = 0;
  std::size_t succeededNodes = 0;
  std::size_t failedNodes = 0;
  std::size_t retryCount = 0;
  std::size_t elapsedMs = 0;
};

struct GemRegistryStats {
  std::string name;
  std::size_t bytes = 0;
  std::size_t allocations = 0;
  std::size_t deallocations = 0;
  std::size_t lookups = 0;
};

struct GemExecutionTelemetry {
  std::vector<GemPhaseStats> phaseStats;
  std::vector<GemRegistryStats> registryStats;
  std::vector<GemExecutionEvent> events;
  std::size_t totalReexecuteCount = 0;
};

struct GemExecuteResult {
  GemStatus status = GemStatus::Ok;
  std::size_t executedNodes = 0;
  std::string message;
  GemExecutionTelemetry telemetry;
};

using GemErrorClassifier = std::function<GemErrorClass(const std::exception_ptr&)>;
using GemCorrectionHook = std::function<bool(const GemExecutionEvent&)>;

// Registry manager + allocator facade used by node executors.
class GemRegistryManager {
 public:
  GemRegistryManager() = default;
  ~GemRegistryManager() = default;

  GemRegistryManager(const GemRegistryManager&) = delete;
  GemRegistryManager& operator=(const GemRegistryManager&) = delete;

  std::uint8_t* allocate(std::string name, std::size_t bytes);
  bool deallocate(const std::string& name);
  std::uint8_t* lookup(const std::string& name);
  std::size_t bytes_for(const std::string& name) const;
  std::vector<GemRegistryStats> stats() const;

 private:
  struct Allocation {
    std::vector<std::uint8_t> bytes;
  };

  mutable std::mutex mutex_;
  std::unordered_map<std::string, Allocation> allocations_;
  std::unordered_map<std::string, GemRegistryStats> stats_;
};

class GemRuntime;

// Lifecycle APIs.
std::unique_ptr<GemRuntime> gem_create(const GemCreateOptions& options,
                                       GemStatus* status = nullptr,
                                       std::string* message = nullptr);
GemExecuteResult gem_execute(GemRuntime* runtime, const GemGraph& graph);
GemStatus gem_destroy(std::unique_ptr<GemRuntime> runtime,
                      std::string* message = nullptr);

// Runtime implementation handle.
class GemRuntime {
 public:
  explicit GemRuntime(std::size_t worker_threads);
  explicit GemRuntime(const GemCreateOptions& options);
  ~GemRuntime();

  GemRuntime(const GemRuntime&) = delete;
  GemRuntime& operator=(const GemRuntime&) = delete;

  GemExecuteResult execute(const GemGraph& graph);
  GemStatus shutdown(std::string* message = nullptr);
  bool is_shutdown() const;
  GemRegistryManager& registry();

  void set_error_classifier(GemErrorClassifier classifier);
  void set_correction_hook(GemCorrectionHook hook);
  GemExecutionTelemetry last_telemetry() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bcir
