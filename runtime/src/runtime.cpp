#include "bcir/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <thread>

namespace bcir {

namespace {

struct LoadedNode {
  std::size_t id = 0;
  std::size_t phase = 0;
  std::function<void()> work;
  std::vector<std::size_t> successors;
  std::size_t indegree = 0;
};

struct LoadedGraph {
  std::vector<LoadedNode> nodes;
  std::vector<std::size_t> phase_order;
};

bool validate_graph(const GemGraph& graph,
                    LoadedGraph* loaded,
                    std::string* message) {
  if (graph.nodes.empty()) {
    if (message != nullptr) {
      *message = "graph is empty";
    }
    return false;
  }

  std::vector<std::size_t> seen_ids;
  seen_ids.reserve(graph.nodes.size());
  std::unordered_map<std::size_t, std::size_t> id_to_index;
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    const GemNode& node = graph.nodes[i];
    if (!node.work) {
      if (message != nullptr) {
        *message = "node " + std::to_string(node.id) + " has no executor";
      }
      return false;
    }
    if (!id_to_index.emplace(node.id, i).second) {
      if (message != nullptr) {
        *message = "duplicate node id " + std::to_string(node.id);
      }
      return false;
    }

    LoadedNode loaded_node;
    loaded_node.id = node.id;
    loaded_node.phase = node.phase;
    loaded_node.work = node.work;
    loaded->nodes.push_back(std::move(loaded_node));
    seen_ids.push_back(node.phase);
  }

  std::sort(seen_ids.begin(), seen_ids.end());
  seen_ids.erase(std::unique(seen_ids.begin(), seen_ids.end()), seen_ids.end());
  loaded->phase_order = std::move(seen_ids);

  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    const GemNode& node = graph.nodes[i];
    for (std::size_t dep : node.dependencies) {
      auto it = id_to_index.find(dep);
      if (it == id_to_index.end()) {
        if (message != nullptr) {
          *message = "node " + std::to_string(node.id) +
                     " depends on unknown node " + std::to_string(dep);
        }
        return false;
      }
      if (loaded->nodes[it->second].phase > loaded->nodes[i].phase) {
        if (message != nullptr) {
          *message = "node " + std::to_string(node.id) +
                     " depends on a future phase node";
        }
        return false;
      }
      loaded->nodes[i].indegree += 1;
      loaded->nodes[it->second].successors.push_back(i);
    }
  }

  return true;
}

}  // namespace

std::string runtime_component_banner() {
  return "gem-runtime: graph execution engine";
}

std::uint8_t* GemRegistryManager::allocate(std::string name, std::size_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  Allocation allocation;
  allocation.bytes.assign(bytes, 0);
  std::uint8_t* ptr = allocation.bytes.data();
  allocations_[std::move(name)] = std::move(allocation);
  return ptr;
}

bool GemRegistryManager::deallocate(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocations_.erase(name) > 0;
}

std::uint8_t* GemRegistryManager::lookup(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(name);
  if (it == allocations_.end()) {
    return nullptr;
  }
  return it->second.bytes.data();
}

std::size_t GemRegistryManager::bytes_for(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(name);
  if (it == allocations_.end()) {
    return 0;
  }
  return it->second.bytes.size();
}

struct GemRuntime::Impl {
  explicit Impl(const GemCreateOptions& create_options)
      : worker_count(create_options.workerThreads == 0 ? 1 : create_options.workerThreads),
        deterministic_ordering(create_options.deterministicOrdering),
        phase_wait_timeout_ms(create_options.phaseWaitTimeoutMs) {}

  struct PhaseWorklist {
    std::deque<std::size_t> queue;
  };

  std::size_t worker_count = 1;
  bool deterministic_ordering = false;
  std::size_t phase_wait_timeout_ms = 0;
  GemRegistryManager registry;
  mutable std::mutex mutex;
  std::condition_variable cv;
  std::condition_variable phase_cv;
  bool shutdown_requested = false;
  bool execute_active = false;
  bool phase_done = false;
  std::size_t active_phase = 0;
  std::size_t outstanding_phase_nodes = 0;
  std::size_t active_workers = 0;
  std::size_t completed_phase_nodes = 0;
  std::size_t executed_in_run = 0;
  LoadedGraph graph;
  std::unordered_map<std::size_t, PhaseWorklist> phase_worklists;
  std::vector<std::thread> workers;

  void start_workers() {
    for (std::size_t i = 0; i < worker_count; ++i) {
      workers.emplace_back([this]() { worker_loop(); });
    }
  }

  void worker_loop() {
    while (true) {
      std::function<void()> work;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this]() {
          if (shutdown_requested) {
            return true;
          }
          if (!execute_active) {
            return false;
          }
          if (deterministic_ordering && active_workers > 0) {
            return false;
          }
          auto phase_it = phase_worklists.find(active_phase);
          return phase_it != phase_worklists.end() && !phase_it->second.queue.empty();
        });

        if (shutdown_requested) {
          return;
        }

        auto phase_it = phase_worklists.find(active_phase);
        if (phase_it == phase_worklists.end() || phase_it->second.queue.empty()) {
          continue;
        }

        const std::size_t node_index = phase_it->second.queue.front();
        phase_it->second.queue.pop_front();
        work = graph.nodes[node_index].work;
        active_workers += 1;
      }

      try {
        work();
      } catch (...) {
        // Runtime keeps progress accounting coherent even when node work throws.
      }

      {
        std::lock_guard<std::mutex> lock(mutex);
        active_workers -= 1;
        executed_in_run += 1;
        completed_phase_nodes += 1;
        if (outstanding_phase_nodes > 0) {
          outstanding_phase_nodes -= 1;
        }

        if (outstanding_phase_nodes == 0 && active_workers == 0) {
          phase_done = true;
          phase_cv.notify_one();
        }
        cv.notify_all();
      }
    }
  }

  GemStatus stop(std::string* message) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (shutdown_requested) {
        if (message != nullptr) {
          *message = "runtime already shut down";
        }
        return GemStatus::RuntimeShutdown;
      }
      shutdown_requested = true;
      execute_active = false;
      phase_worklists.clear();
    }
    cv.notify_all();
    phase_cv.notify_all();
    for (std::thread& worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    workers.clear();
    if (message != nullptr) {
      *message = "runtime shut down";
    }
    return GemStatus::Ok;
  }
};

GemRuntime::GemRuntime(std::size_t worker_threads)
    : GemRuntime(GemCreateOptions{worker_threads, false, 0}) {}

GemRuntime::GemRuntime(const GemCreateOptions& options)
    : impl_(std::make_unique<Impl>(options)) {
  impl_->start_workers();
}

GemRuntime::~GemRuntime() {
  if (impl_ != nullptr) {
    impl_->stop(nullptr);
  }
}

GemExecuteResult GemRuntime::execute(const GemGraph& graph) {
  GemExecuteResult result;

  LoadedGraph loaded;
  if (!validate_graph(graph, &loaded, &result.message)) {
    result.status = GemStatus::InvalidGraph;
    return result;
  }

  std::unique_lock<std::mutex> lock(impl_->mutex);
  if (impl_->shutdown_requested) {
    result.status = GemStatus::RuntimeShutdown;
    result.message = "runtime is shut down";
    return result;
  }
  if (impl_->execute_active) {
    result.status = GemStatus::RuntimeError;
    result.message = "runtime is already executing a graph";
    return result;
  }

  impl_->graph = std::move(loaded);
  impl_->phase_worklists.clear();
  impl_->execute_active = true;

  std::vector<std::size_t> ready_by_phase;
  for (std::size_t idx = 0; idx < impl_->graph.nodes.size(); ++idx) {
    if (impl_->graph.nodes[idx].indegree == 0) {
      ready_by_phase.push_back(idx);
    }
  }

  impl_->executed_in_run = 0;

  for (std::size_t phase : impl_->graph.phase_order) {
    auto& worklist = impl_->phase_worklists[phase];
    for (std::size_t idx : ready_by_phase) {
      if (impl_->graph.nodes[idx].phase == phase) {
        worklist.queue.push_back(idx);
      }
    }

    if (impl_->deterministic_ordering) {
      std::stable_sort(worklist.queue.begin(), worklist.queue.end(),
                       [this](std::size_t lhs, std::size_t rhs) {
                         return impl_->graph.nodes[lhs].id < impl_->graph.nodes[rhs].id;
                       });
    }

    impl_->active_phase = phase;
    impl_->outstanding_phase_nodes = worklist.queue.size();
    impl_->completed_phase_nodes = 0;
    impl_->phase_done = impl_->outstanding_phase_nodes == 0;

    if (!impl_->phase_done) {
      impl_->cv.notify_all();
      const auto wait_predicate = [this]() {
        return impl_->shutdown_requested || impl_->phase_done;
      };
      bool completed = false;
      if (impl_->phase_wait_timeout_ms == 0) {
        impl_->phase_cv.wait(lock, wait_predicate);
        completed = true;
      } else {
        completed = impl_->phase_cv.wait_for(
            lock, std::chrono::milliseconds(impl_->phase_wait_timeout_ms),
            wait_predicate);
      }

      if (!completed) {
        impl_->execute_active = false;
        result.status = GemStatus::RuntimeError;
        result.message = "phase execution timed out; possible deadlock/livelock";
        return result;
      }
      if (impl_->shutdown_requested) {
        result.status = GemStatus::RuntimeShutdown;
        result.message = "runtime shut down during execute";
        impl_->execute_active = false;
        return result;
      }
    }

    worklist.queue.clear();

    std::vector<std::size_t> newly_ready;
    for (std::size_t node_idx = 0; node_idx < impl_->graph.nodes.size(); ++node_idx) {
      auto& node = impl_->graph.nodes[node_idx];
      if (node.phase != phase) {
        continue;
      }
      for (std::size_t successor : node.successors) {
        auto& dep = impl_->graph.nodes[successor];
        if (dep.indegree > 0) {
          dep.indegree -= 1;
          if (dep.indegree == 0) {
            newly_ready.push_back(successor);
          }
        }
      }
    }
    ready_by_phase = std::move(newly_ready);
  }

  impl_->execute_active = false;

  result.status = GemStatus::Ok;
  result.executedNodes = impl_->executed_in_run;
  result.message = "execution completed";
  return result;
}

GemStatus GemRuntime::shutdown(std::string* message) { return impl_->stop(message); }

bool GemRuntime::is_shutdown() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->shutdown_requested;
}

GemRegistryManager& GemRuntime::registry() { return impl_->registry; }

std::unique_ptr<GemRuntime> gem_create(const GemCreateOptions& options,
                                       GemStatus* status,
                                       std::string* message) {
  if (options.workerThreads > 4096) {
    if (status != nullptr) {
      *status = GemStatus::InvalidArgument;
    }
    if (message != nullptr) {
      *message = "workerThreads exceeds supported limit";
    }
    return nullptr;
  }

  if (status != nullptr) {
    *status = GemStatus::Ok;
  }
  if (message != nullptr) {
    *message = "runtime created";
  }
  return std::make_unique<GemRuntime>(options);
}

GemExecuteResult gem_execute(GemRuntime* runtime, const GemGraph& graph) {
  if (runtime == nullptr) {
    return GemExecuteResult{GemStatus::InvalidArgument, 0, "runtime is null"};
  }
  return runtime->execute(graph);
}

GemStatus gem_destroy(std::unique_ptr<GemRuntime> runtime, std::string* message) {
  if (runtime == nullptr) {
    if (message != nullptr) {
      *message = "runtime is null";
    }
    return GemStatus::InvalidArgument;
  }

  const GemStatus status = runtime->shutdown(message);
  runtime.reset();
  return status;
}

}  // namespace bcir
