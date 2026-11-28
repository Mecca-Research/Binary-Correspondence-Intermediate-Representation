//===- bcir_solver.hpp - the hosted solver library's memory and errors (internal) ---===//
//
// What every solver of the hosted C++ library (runtime/cpp, the native port audit's N5) shares, so it has one
// spelling: the caller's `bcir_host_allocator` as a `std::pmr::memory_resource`, the arena a solve allocates from,
// the containers over it, and `bcir::expected` for failures as values. Internal to the library: no type here
// crosses a C ABI façade, and nothing here is global state.
//===----------------------------------------------------------------------===//
#ifndef BCIR_SOLVER_HPP
#define BCIR_SOLVER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <new>
#include <vector>

#include "bcir_expected.hpp"
#include "bcir_host_alloc.h"

namespace bcir::solver {

// The caller's allocator as a memory resource. `bcir_host_allocate` gives max_align_t alignment; a pool asks its
// upstream for more (its chunks), which this provides by over-allocating and keeping the allocation's own address
// in the word before the aligned block.
class host_resource final : public std::pmr::memory_resource {
 public:
  explicit host_resource(const bcir_host_allocator *a) : a_(bcir_host_allocator_or_default(a)) {}

 private:
  void *do_allocate(std::size_t bytes, std::size_t align) override {
    if (bytes == 0) bytes = 1;
    if (align <= alignof(std::max_align_t)) {
      void *p = bcir_host_allocate(&a_, bytes);
      if (!p) throw std::bad_alloc();
      return p;
    }
    std::size_t total = 0;
    if (!bcir_size_add(bytes, align + sizeof(void *), &total)) throw std::bad_alloc();
    void *raw = bcir_host_allocate(&a_, total);
    if (!raw) throw std::bad_alloc();
    std::uintptr_t at = reinterpret_cast<std::uintptr_t>(raw) + sizeof(void *);
    at = (at + align - 1) & ~(static_cast<std::uintptr_t>(align) - 1);
    reinterpret_cast<void **>(at)[-1] = raw;
    return reinterpret_cast<void *>(at);
  }
  void do_deallocate(void *p, std::size_t, std::size_t align) override {
    bcir_host_deallocate(&a_, align <= alignof(std::max_align_t) ? p : static_cast<void **>(p)[-1]);
  }
  bool do_is_equal(const std::pmr::memory_resource &o) const noexcept override { return this == &o; }
  bcir_host_allocator a_;
};

// One phase's memory: blocks drawn from the caller's allocator and carved into power-of-two size classes whose freed
// blocks are reused, all of it handed back when the phase ends -- the one release point. A request larger than the
// largest class, or aligned past max_align_t, goes to the caller's allocator directly. An allocation the caller
// refuses throws `std::bad_alloc` with the arena unchanged, so the containers' own guarantees hold and the façade
// reports NOMEM. Written here rather than taken from `std::pmr::unsynchronized_pool_resource`, which in libstdc++
// (13 through 16) does not survive its upstream refusing its first allocation: the next one reads a pool it never
// finished building (the allocation-failure campaign's first finding, reproduced in isolation).
class phase_arena final : public std::pmr::memory_resource {
 public:
  explicit phase_arena(std::pmr::memory_resource *upstream) : up_(upstream) {}
  phase_arena(const phase_arena &) = delete;
  phase_arena &operator=(const phase_arena &) = delete;
  ~phase_arena() override {
    while (chunks_) {
      Chunk *next = chunks_->next;
      up_->deallocate(chunks_, chunks_->size, alignof(std::max_align_t));
      chunks_ = next;
    }
  }

 private:
  static constexpr std::size_t kMinShift = 4;            // the smallest class: 16 bytes
  static constexpr std::size_t kClasses = 13;            // 16 B .. 64 KiB
  static constexpr std::size_t kLargest = std::size_t{1} << (kMinShift + kClasses - 1);
  static constexpr std::size_t kChunk = 2 * kLargest;    // what one refill asks the caller for, at least
  struct Free { Free *next; };
  struct Chunk { Chunk *next; std::size_t size; };
  static constexpr std::size_t kHeader =
      (sizeof(Chunk) + alignof(std::max_align_t) - 1) / alignof(std::max_align_t) * alignof(std::max_align_t);

  static std::size_t class_of(std::size_t bytes) {
    std::size_t c = 0;
    while ((std::size_t{1} << (kMinShift + c)) < bytes) c++;
    return c;
  }
  void *do_allocate(std::size_t bytes, std::size_t align) override {
    if (bytes == 0) bytes = 1;
    if (align > alignof(std::max_align_t) || bytes > kLargest) return up_->allocate(bytes, align);
    std::size_t c = class_of(bytes), size = std::size_t{1} << (kMinShift + c);
    if (Free *f = free_[c]) { free_[c] = f->next; return f; }
    if (left_ < size) {   // a fresh chunk; the old one's tail is left unused
      std::size_t want = std::max(kChunk, kHeader + size);
      Chunk *chunk = static_cast<Chunk *>(up_->allocate(want, alignof(std::max_align_t)));   // may throw: unchanged
      chunk->next = chunks_; chunk->size = want; chunks_ = chunk;
      cur_ = reinterpret_cast<unsigned char *>(chunk) + kHeader; left_ = want - kHeader;
    }
    void *p = cur_;
    cur_ += size; left_ -= size;
    return p;
  }
  void do_deallocate(void *p, std::size_t bytes, std::size_t align) override {
    if (bytes == 0) bytes = 1;
    if (align > alignof(std::max_align_t) || bytes > kLargest) { up_->deallocate(p, bytes, align); return; }
    std::size_t c = class_of(bytes);
    Free *f = static_cast<Free *>(p);
    f->next = free_[c]; free_[c] = f;
  }
  bool do_is_equal(const std::pmr::memory_resource &o) const noexcept override { return this == &o; }

  std::pmr::memory_resource *up_;
  Free *free_[kClasses] = {};
  Chunk *chunks_ = nullptr;
  unsigned char *cur_ = nullptr;
  std::size_t left_ = 0;
};

template <class T> using pvec = std::pmr::vector<T>;

}  // namespace bcir::solver

#endif  // BCIR_SOLVER_HPP
