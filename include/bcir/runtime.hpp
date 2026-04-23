#pragma once

#include <cstddef>
#include <cstdint>
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

struct GemNode {
  std::size_t id = 0;
  std::size_t phase = 0;
  std::vector<std::size_t> dependencies;
  std::function<void()> work;
};

struct GemGraph {
  std::vector<GemNode> nodes;
};

struct GemCreateOptions {
  std::size_t workerThreads = 0;
  bool deterministicOrdering = false;
  std::size_t phaseWaitTimeoutMs = 0;
};

struct GemExecuteResult {
  GemStatus status = GemStatus::Ok;
  std::size_t executedNodes = 0;
  std::string message;
};

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

 private:
  struct Allocation {
    std::vector<std::uint8_t> bytes;
  };

  mutable std::mutex mutex_;
  std::unordered_map<std::string, Allocation> allocations_;
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

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bcir
