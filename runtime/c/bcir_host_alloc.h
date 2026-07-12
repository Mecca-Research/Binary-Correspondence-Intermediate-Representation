/*===- bcir_host_alloc.h - hosted allocation discipline for BCIR ----------===*/
#ifndef BCIR_HOST_ALLOC_H
#define BCIR_HOST_ALLOC_H

/*
 * HOSTED-ONLY.  Freestanding runtime components must not include this file.
 *
 * Ownership vocabulary used by the hosted public APIs:
 *   borrowed  - valid only for the duration documented by the callee;
 *   owned     - must be released by the matching BCIR destroy/free function;
 *   consumed  - ownership transfers to the callee, including on success only
 *               when the API explicitly says so.
 *
 * The allocator is deliberately small enough to inject deterministic failures
 * without replacing process-global malloc.  All growth helpers are checked and
 * update the caller's pointer only after realloc succeeds.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#define BCIR_HOST_ALIGNOF(T) alignof(T)
#else
#define BCIR_HOST_ALIGNOF(T) _Alignof(T)
#endif

typedef void *(*bcir_host_allocate_fn)(void *context, size_t size);
typedef void *(*bcir_host_reallocate_fn)(void *context, void *allocation, size_t size);
typedef void (*bcir_host_deallocate_fn)(void *context, void *allocation);

typedef struct bcir_host_allocator {
  void *context;                         /* borrowed for the allocator's lifetime */
  bcir_host_allocate_fn allocate;
  bcir_host_reallocate_fn reallocate;
  bcir_host_deallocate_fn deallocate;
} bcir_host_allocator;

static inline void *bcir_host_libc_allocate(void *context, size_t size) {
  (void)context;
  return malloc(size ? size : 1u);
}

static inline void *bcir_host_libc_reallocate(void *context, void *allocation, size_t size) {
  (void)context;
  return realloc(allocation, size ? size : 1u);
}

static inline void bcir_host_libc_deallocate(void *context, void *allocation) {
  (void)context;
  free(allocation);
}

static inline bcir_host_allocator bcir_host_allocator_default(void) {
  bcir_host_allocator allocator;
  allocator.context = NULL;
  allocator.allocate = bcir_host_libc_allocate;
  allocator.reallocate = bcir_host_libc_reallocate;
  allocator.deallocate = bcir_host_libc_deallocate;
  return allocator;
}

static inline int bcir_host_allocator_valid(const bcir_host_allocator *allocator) {
  return allocator && allocator->allocate && allocator->reallocate && allocator->deallocate;
}

static inline bcir_host_allocator bcir_host_allocator_or_default(
    const bcir_host_allocator *allocator) {
  /* NULL deliberately selects libc. A non-NULL but incomplete allocator is a
   * configuration error and must fail closed at the first validity check. */
  return allocator ? *allocator : bcir_host_allocator_default();
}

static inline void *bcir_host_allocate(const bcir_host_allocator *allocator, size_t size) {
  if(!bcir_host_allocator_valid(allocator)) return NULL;
  if(allocator->allocate==bcir_host_libc_allocate)
    return malloc(size?size:1u);
  return allocator->allocate(allocator->context,size?size:1u);
}

static inline void bcir_host_deallocate(const bcir_host_allocator *allocator, void *allocation) {
  if (allocation && bcir_host_allocator_valid(allocator)) {
    if(allocator->deallocate==bcir_host_libc_deallocate) free(allocation);
    else allocator->deallocate(allocator->context, allocation);
  }
}

static inline int bcir_size_add(size_t left, size_t right, size_t *result) {
  if (!result || right > SIZE_MAX - left) return 0;
  *result = left + right;
  return 1;
}

static inline int bcir_size_mul(size_t left, size_t right, size_t *result) {
  if (!result || (right && left > SIZE_MAX / right)) return 0;
  *result = left * right;
  return 1;
}

static inline int bcir_size_align_up(size_t value, size_t alignment, size_t *result) {
  size_t remainder;
  if (!result || !alignment) return 0;
  remainder = value % alignment;
  if (!remainder) {
    *result = value;
    return 1;
  }
  return bcir_size_add(value, alignment - remainder, result);
}

/* Choose a geometric capacity >= minimum, checking both count and byte size. */
static inline int bcir_host_grow_capacity(size_t current, size_t minimum,
                                          size_t element_size, size_t *result) {
  size_t capacity = current ? current : 8u;
  size_t bytes;
  if (!result || !element_size) return 0;
  while (capacity < minimum) {
    if (capacity > SIZE_MAX / 2u) {
      capacity = minimum;
      break;
    }
    capacity *= 2u;
  }
  if (capacity < minimum || !bcir_size_mul(capacity, element_size, &bytes)) return 0;
  (void)bytes;
  *result = capacity;
  return 1;
}

/*
 * Two-phase realloc.  On failure, *allocation and the original bytes remain
 * untouched.  When zero_new is true, newly added elements are initialized.
 */
static inline int bcir_host_realloc_array(const bcir_host_allocator *allocator,
                                          void **allocation,
                                          size_t old_count,
                                          size_t new_count,
                                          size_t element_size,
                                          int zero_new) {
  size_t bytes;
  void *replacement;
  if (!allocation || new_count < old_count ||
      !bcir_size_mul(new_count, element_size, &bytes) ||
      !bcir_host_allocator_valid(allocator))
    return 0;
  replacement = allocator->reallocate==bcir_host_libc_reallocate
                    ?realloc(*allocation,bytes?bytes:1u)
                    :allocator->reallocate(allocator->context,*allocation,bytes?bytes:1u);
  if (!replacement) return 0;
  if (zero_new && new_count > old_count)
    memset((unsigned char *)replacement + old_count * element_size, 0,
           (new_count - old_count) * element_size);
  *allocation = replacement;
  return 1;
}

typedef struct bcir_host_arena_block {
  struct bcir_host_arena_block *next;
  size_t used;
  size_t capacity;
  max_align_t alignment_anchor;
  unsigned char data[];
} bcir_host_arena_block;

typedef struct bcir_host_arena {
  bcir_host_allocator allocator;
  bcir_host_arena_block *blocks;          /* owned */
  size_t preferred_block_size;
} bcir_host_arena;

static inline int bcir_host_arena_init(bcir_host_arena *arena,
                                       const bcir_host_allocator *allocator,
                                       size_t preferred_block_size) {
  if (!arena) return 0;
  memset(arena, 0, sizeof *arena);
  arena->allocator = bcir_host_allocator_or_default(allocator);
  if (!bcir_host_allocator_valid(&arena->allocator)) return 0;
  arena->preferred_block_size = preferred_block_size ? preferred_block_size : 4096u;
  return 1;
}

static inline void bcir_host_arena_reset(bcir_host_arena *arena) {
  bcir_host_arena_block *block;
  if (!arena || !bcir_host_allocator_valid(&arena->allocator)) return;
  block = arena->blocks;
  while (block) {
    bcir_host_arena_block *next = block->next;
    bcir_host_deallocate(&arena->allocator, block);
    block = next;
  }
  arena->blocks = NULL;
}

static inline void bcir_host_arena_destroy(bcir_host_arena *arena) {
  if (!arena) return;
  bcir_host_arena_reset(arena);
  memset(arena, 0, sizeof *arena);
}

static inline void *bcir_host_arena_allocate(bcir_host_arena *arena, size_t size,
                                             size_t alignment) {
  bcir_host_arena_block *block;
  size_t offset;
  if (!arena || !bcir_host_allocator_valid(&arena->allocator)) return NULL;
  if (!size) size = 1u;
  if (!alignment) alignment = BCIR_HOST_ALIGNOF(max_align_t);
  /* Arena blocks are guaranteed only to max_align_t.  Reject over-aligned
   * requests instead of returning a pointer with a stronger but false
   * alignment promise. */
  if (alignment > BCIR_HOST_ALIGNOF(max_align_t)) return NULL;
  block = arena->blocks;
  if (block && bcir_size_align_up(block->used, alignment, &offset) &&
      offset <= block->capacity && size <= block->capacity - offset) {
    void *result = block->data + offset;
    block->used = offset + size;
    return result;
  }
  {
    size_t capacity = arena->preferred_block_size;
    size_t needed;
    size_t total;
    if (!bcir_size_add(size, alignment - 1u, &needed)) return NULL;
    if (capacity < needed) capacity = needed;
    if (!bcir_size_add(offsetof(bcir_host_arena_block, data), capacity, &total)) return NULL;
    block = (bcir_host_arena_block *)bcir_host_allocate(&arena->allocator, total);
    if (!block) return NULL;
    block->next = arena->blocks;
    block->used = 0;
    block->capacity = capacity;
    arena->blocks = block;
  }
  if (!bcir_size_align_up(block->used, alignment, &offset) ||
      offset > block->capacity || size > block->capacity - offset)
    return NULL;
  block->used = offset + size;
  return block->data + offset;
}

#ifdef __cplusplus
}
#endif

#undef BCIR_HOST_ALIGNOF

#endif /* BCIR_HOST_ALLOC_H */
