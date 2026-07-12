/*===- bcir_runtime_channel.h - versioned in-process driver hook ABI -------===*/
#ifndef BCIR_RUNTIME_CHANNEL_H
#define BCIR_RUNTIME_CHANNEL_H

/*
 * Platform-neutral, allocation-free direct driver ABI. This is not an IPC
 * transport: a future Linux adapter may marshal these value types, but raw
 * device pointers never appear in the contract. Resources, mappings, and
 * submissions cross the boundary as generation-tagged handles plus offsets.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_RUNTIME_CHANNEL_ABI_V1 1u

typedef enum bcir_channel_status {
  BCIR_CHANNEL_OK = 0,
  BCIR_CHANNEL_INVALID = 1,
  BCIR_CHANNEL_VERSION = 2,
  BCIR_CHANNEL_STALE = 3,
  BCIR_CHANNEL_BACKPRESSURE = 4,
  BCIR_CHANNEL_WOULD_BLOCK = 5,
  BCIR_CHANNEL_CLOSED = 6,
  BCIR_CHANNEL_UNSUPPORTED = 7,
  BCIR_CHANNEL_IO = 8,
  BCIR_CHANNEL_CANCELED = 9
} bcir_channel_status;

typedef struct bcir_channel_handle {
  uint64_t value;
  uint32_t generation;
  uint32_t kind;
} bcir_channel_handle;

typedef enum bcir_channel_handle_kind {
  BCIR_CHANNEL_HANDLE_NONE = 0,
  BCIR_CHANNEL_HANDLE_SESSION = 1,
  BCIR_CHANNEL_HANDLE_RESOURCE = 2,
  BCIR_CHANNEL_HANDLE_MAPPING = 3
} bcir_channel_handle_kind;

typedef struct bcir_channel_mapping {
  bcir_channel_handle handle;
  bcir_channel_handle resource;
  uint64_t byte_offset;
  uint64_t byte_length;
} bcir_channel_mapping;

typedef struct bcir_channel_submission {
  bcir_channel_handle mapping;
  uint64_t byte_offset;
  uint64_t byte_length;
  uint64_t sequence;
  uint32_t flags;
  uint32_t reserved;
} bcir_channel_submission;

typedef enum bcir_channel_sync_flags {
  BCIR_CHANNEL_SYNC_WAIT = 1u << 0,
  BCIR_CHANNEL_SYNC_CANCEL = 1u << 1
} bcir_channel_sync_flags;

typedef enum bcir_channel_event_kind {
  BCIR_CHANNEL_EVENT_COMPLETE = 1,
  BCIR_CHANNEL_EVENT_CANCELED = 2,
  BCIR_CHANNEL_EVENT_PEER_CLOSED = 3
} bcir_channel_event_kind;

typedef struct bcir_channel_event {
  uint64_t sequence;
  uint32_t generation;
  uint16_t kind;
  uint16_t status;
} bcir_channel_event;

typedef enum bcir_channel_ring_policy {
  BCIR_CHANNEL_RING_BACKPRESSURE = 0,
  BCIR_CHANNEL_RING_OVERWRITE_OLDEST = 1
} bcir_channel_ring_policy;

typedef bcir_channel_status (*bcir_channel_open_fn)(
    void *implementation, uint64_t device_id, bcir_channel_handle *session);
typedef bcir_channel_status (*bcir_channel_claim_fn)(
    void *implementation, bcir_channel_handle session, uint64_t resource_id,
    uint32_t flags, bcir_channel_handle *resource);
typedef bcir_channel_status (*bcir_channel_map_fn)(
    void *implementation, bcir_channel_handle session,
    bcir_channel_handle resource, uint64_t byte_offset, uint64_t byte_length,
    bcir_channel_mapping *mapping);
typedef bcir_channel_status (*bcir_channel_submit_fn)(
    void *implementation, bcir_channel_handle session,
    const bcir_channel_submission *submission);
typedef bcir_channel_status (*bcir_channel_sync_fn)(
    void *implementation, bcir_channel_handle session, uint64_t sequence,
    uint32_t flags);
typedef bcir_channel_status (*bcir_channel_event_fn)(
    void *implementation, bcir_channel_handle session, bcir_channel_event *event);
typedef bcir_channel_status (*bcir_channel_close_fn)(
    void *implementation, bcir_channel_handle session);

/* Append-only hook table. Consumers must gate every tail field with struct_size. */
typedef struct bcir_runtime_channel_hooks_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t capabilities;
  void *implementation;                 /* opaque and process-local */
  bcir_channel_open_fn open;
  bcir_channel_claim_fn claim;
  bcir_channel_map_fn map;
  bcir_channel_submit_fn submit;
  bcir_channel_sync_fn sync;
  bcir_channel_event_fn event;
  bcir_channel_close_fn close;
} bcir_runtime_channel_hooks_v1;

typedef struct bcir_runtime_channel {
  const bcir_runtime_channel_hooks_v1 *hooks; /* borrowed for channel lifetime */
} bcir_runtime_channel;

bcir_channel_status bcir_runtime_channel_init(
    bcir_runtime_channel *channel,
    const bcir_runtime_channel_hooks_v1 *hooks);
void bcir_runtime_channel_reset(bcir_runtime_channel *channel);
bcir_channel_status bcir_channel_open(bcir_runtime_channel *channel,
                                      uint64_t device_id,
                                      bcir_channel_handle *session);
bcir_channel_status bcir_channel_claim(bcir_runtime_channel *channel,
                                       bcir_channel_handle session,
                                       uint64_t resource_id, uint32_t flags,
                                       bcir_channel_handle *resource);
bcir_channel_status bcir_channel_map(bcir_runtime_channel *channel,
                                     bcir_channel_handle session,
                                     bcir_channel_handle resource,
                                     uint64_t byte_offset,
                                     uint64_t byte_length,
                                     bcir_channel_mapping *mapping);
bcir_channel_status bcir_channel_submit(bcir_runtime_channel *channel,
                                        bcir_channel_handle session,
                                        const bcir_channel_submission *submission);
bcir_channel_status bcir_channel_sync(bcir_runtime_channel *channel,
                                      bcir_channel_handle session,
                                      uint64_t sequence, uint32_t flags);
bcir_channel_status bcir_channel_next_event(bcir_runtime_channel *channel,
                                            bcir_channel_handle session,
                                            bcir_channel_event *event);
bcir_channel_status bcir_channel_close(bcir_runtime_channel *channel,
                                       bcir_channel_handle session);

/*
 * Allocation-free resident reference driver. The caller owns event_storage and
 * the loopback object. It models one bounded byte resource and makes queue-full
 * behavior explicit, providing the direct-vtable baseline for future IPC parity.
 */
typedef struct bcir_loopback_channel {
  bcir_runtime_channel_hooks_v1 hooks;
  bcir_channel_event *events;
  size_t event_capacity;
  size_t event_head;
  size_t event_count;
  uint64_t resource_capacity;
  bcir_channel_mapping active_mapping;
  uint64_t next_mapping_id;
  uint64_t next_sequence;
  uint64_t completed_sequence;
  uint64_t dropped_events;
  uint32_t generation;
  uint32_t ring_policy;
  uint8_t open_state;
} bcir_loopback_channel;

bcir_channel_status bcir_loopback_channel_init(
    bcir_loopback_channel *loopback, bcir_channel_event *event_storage,
    size_t event_capacity, uint64_t resource_capacity,
    bcir_channel_ring_policy policy, bcir_runtime_channel *channel);
void bcir_loopback_channel_reset(bcir_loopback_channel *loopback);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_RUNTIME_CHANNEL_H */
