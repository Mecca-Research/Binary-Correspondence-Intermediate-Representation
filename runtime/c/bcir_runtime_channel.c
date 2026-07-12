/*===- bcir_runtime_channel.c - direct driver ABI and resident reference ----===*/
#include "bcir_runtime_channel.h"

#include <limits.h>
#include <string.h>

#define REQUIRED_HOOK_SIZE (offsetof(bcir_runtime_channel_hooks_v1, close) + sizeof(bcir_channel_close_fn))

static int handle_is(bcir_channel_handle handle, uint32_t kind,
                     uint32_t generation) {
  return handle.value != 0 && handle.kind == kind &&
         handle.generation == generation;
}

static bcir_channel_status ready(const bcir_runtime_channel *channel) {
  if(!channel||!channel->hooks) return BCIR_CHANNEL_INVALID;
  if(channel->hooks->abi_version!=BCIR_RUNTIME_CHANNEL_ABI_V1 ||
     channel->hooks->struct_size<REQUIRED_HOOK_SIZE) return BCIR_CHANNEL_VERSION;
  return BCIR_CHANNEL_OK;
}

bcir_channel_status bcir_runtime_channel_init(
    bcir_runtime_channel *channel,
    const bcir_runtime_channel_hooks_v1 *hooks) {
  if(!channel) return BCIR_CHANNEL_INVALID;
  channel->hooks=NULL;
  if(!hooks||hooks->abi_version!=BCIR_RUNTIME_CHANNEL_ABI_V1 ||
     hooks->struct_size<REQUIRED_HOOK_SIZE) return BCIR_CHANNEL_VERSION;
  if(!hooks->open||!hooks->claim||!hooks->map||!hooks->submit||
     !hooks->sync||!hooks->event||!hooks->close) return BCIR_CHANNEL_INVALID;
  channel->hooks=hooks;
  return BCIR_CHANNEL_OK;
}

void bcir_runtime_channel_reset(bcir_runtime_channel *channel) {
  if(channel) channel->hooks=NULL;
}

#define DISPATCH(name, member, args) do { \
  bcir_channel_status status=ready(channel); \
  if(status!=BCIR_CHANNEL_OK) return status; \
  return channel->hooks->member args; \
} while(0)

bcir_channel_status bcir_channel_open(bcir_runtime_channel *channel,
                                      uint64_t device_id,
                                      bcir_channel_handle *session) {
  if(session) memset(session,0,sizeof *session);
  if(!session) return BCIR_CHANNEL_INVALID;
  DISPATCH(open,open,(channel->hooks->implementation,device_id,session));
}

bcir_channel_status bcir_channel_claim(bcir_runtime_channel *channel,
                                       bcir_channel_handle session,
                                       uint64_t resource_id,uint32_t flags,
                                       bcir_channel_handle *resource) {
  if(resource) memset(resource,0,sizeof *resource);
  if(!resource) return BCIR_CHANNEL_INVALID;
  DISPATCH(claim,claim,(channel->hooks->implementation,session,resource_id,flags,resource));
}

bcir_channel_status bcir_channel_map(bcir_runtime_channel *channel,
                                     bcir_channel_handle session,
                                     bcir_channel_handle resource,
                                     uint64_t byte_offset,uint64_t byte_length,
                                     bcir_channel_mapping *mapping) {
  if(mapping) memset(mapping,0,sizeof *mapping);
  if(!mapping) return BCIR_CHANNEL_INVALID;
  DISPATCH(map,map,(channel->hooks->implementation,session,resource,byte_offset,byte_length,mapping));
}

bcir_channel_status bcir_channel_submit(bcir_runtime_channel *channel,
                                        bcir_channel_handle session,
                                        const bcir_channel_submission *submission) {
  if(!submission) return BCIR_CHANNEL_INVALID;
  DISPATCH(submit,submit,(channel->hooks->implementation,session,submission));
}

bcir_channel_status bcir_channel_sync(bcir_runtime_channel *channel,
                                      bcir_channel_handle session,
                                      uint64_t sequence,uint32_t flags) {
  DISPATCH(sync,sync,(channel->hooks->implementation,session,sequence,flags));
}

bcir_channel_status bcir_channel_next_event(bcir_runtime_channel *channel,
                                            bcir_channel_handle session,
                                            bcir_channel_event *event) {
  if(event) memset(event,0,sizeof *event);
  if(!event) return BCIR_CHANNEL_INVALID;
  DISPATCH(event,event,(channel->hooks->implementation,session,event));
}

bcir_channel_status bcir_channel_close(bcir_runtime_channel *channel,
                                       bcir_channel_handle session) {
  DISPATCH(close,close,(channel->hooks->implementation,session));
}

#undef DISPATCH

static bcir_channel_handle make_handle(uint64_t value,uint32_t generation,
                                       uint32_t kind) {
  bcir_channel_handle handle;
  handle.value=value;handle.generation=generation;handle.kind=kind;
  return handle;
}

static bcir_channel_status loop_session(bcir_loopback_channel *loopback,
                                        bcir_channel_handle session) {
  if(!loopback->open_state) return BCIR_CHANNEL_CLOSED;
  return handle_is(session,BCIR_CHANNEL_HANDLE_SESSION,loopback->generation)
             ?BCIR_CHANNEL_OK:BCIR_CHANNEL_STALE;
}

static bcir_channel_status loop_push(bcir_loopback_channel *loopback,
                                     bcir_channel_event event) {
  size_t tail;
  if(!loopback->event_capacity) return BCIR_CHANNEL_BACKPRESSURE;
  if(loopback->event_count==loopback->event_capacity){
    if(loopback->ring_policy==BCIR_CHANNEL_RING_BACKPRESSURE)
      return BCIR_CHANNEL_BACKPRESSURE;
    loopback->event_head=(loopback->event_head+1u)%loopback->event_capacity;
    loopback->event_count--;
    loopback->dropped_events++;
  }
  tail=(loopback->event_head+loopback->event_count)%loopback->event_capacity;
  loopback->events[tail]=event;
  loopback->event_count++;
  return BCIR_CHANNEL_OK;
}

static bcir_channel_status loop_open(void *implementation,uint64_t device_id,
                                     bcir_channel_handle *session) {
  bcir_loopback_channel *loopback=(bcir_loopback_channel *)implementation;
  (void)device_id;
  if(loopback->open_state) return BCIR_CHANNEL_INVALID;
  loopback->generation++;
  if(!loopback->generation) loopback->generation=1;
  loopback->open_state=1;
  loopback->event_head=0;loopback->event_count=0;
  loopback->next_mapping_id=1;loopback->next_sequence=1;loopback->completed_sequence=0;
  memset(&loopback->active_mapping,0,sizeof loopback->active_mapping);
  *session=make_handle(1u,loopback->generation,BCIR_CHANNEL_HANDLE_SESSION);
  return BCIR_CHANNEL_OK;
}

static bcir_channel_status loop_claim(void *implementation,
                                      bcir_channel_handle session,
                                      uint64_t resource_id,uint32_t flags,
                                      bcir_channel_handle *resource) {
  bcir_loopback_channel *loopback=(bcir_loopback_channel *)implementation;
  bcir_channel_status status=loop_session(loopback,session);
  (void)flags;
  if(status!=BCIR_CHANNEL_OK) return status;
  if(resource_id==UINT64_MAX) return BCIR_CHANNEL_INVALID;
  *resource=make_handle(resource_id+1u,loopback->generation,
                        BCIR_CHANNEL_HANDLE_RESOURCE);
  return BCIR_CHANNEL_OK;
}

static bcir_channel_status loop_map(void *implementation,
                                    bcir_channel_handle session,
                                    bcir_channel_handle resource,
                                    uint64_t byte_offset,uint64_t byte_length,
                                    bcir_channel_mapping *mapping) {
  bcir_loopback_channel *loopback=(bcir_loopback_channel *)implementation;
  bcir_channel_status status=loop_session(loopback,session);
  if(status!=BCIR_CHANNEL_OK) return status;
  if(!handle_is(resource,BCIR_CHANNEL_HANDLE_RESOURCE,loopback->generation))
    return BCIR_CHANNEL_STALE;
  if(byte_offset>loopback->resource_capacity ||
     byte_length>loopback->resource_capacity-byte_offset)
    return BCIR_CHANNEL_INVALID;
  if(loopback->next_mapping_id==UINT64_MAX) return BCIR_CHANNEL_INVALID;
  mapping->handle=make_handle(loopback->next_mapping_id++,loopback->generation,
                              BCIR_CHANNEL_HANDLE_MAPPING);
  mapping->resource=resource;
  mapping->byte_offset=byte_offset;
  mapping->byte_length=byte_length;
  loopback->active_mapping=*mapping;
  return BCIR_CHANNEL_OK;
}

static bcir_channel_status loop_submit(void *implementation,
                                       bcir_channel_handle session,
                                       const bcir_channel_submission *submission) {
  bcir_loopback_channel *loopback=(bcir_loopback_channel *)implementation;
  bcir_channel_event event;
  bcir_channel_status status=loop_session(loopback,session);
  uint64_t sequence;
  if(status!=BCIR_CHANNEL_OK) return status;
  if(!handle_is(submission->mapping,BCIR_CHANNEL_HANDLE_MAPPING,loopback->generation) ||
     submission->mapping.value!=loopback->active_mapping.handle.value)
    return BCIR_CHANNEL_STALE;
  if(submission->byte_offset>loopback->active_mapping.byte_length ||
     submission->byte_length>loopback->active_mapping.byte_length-submission->byte_offset)
    return BCIR_CHANNEL_INVALID;
  sequence=submission->sequence?submission->sequence:loopback->next_sequence;
  if(sequence<loopback->next_sequence) return BCIR_CHANNEL_STALE;
  if(sequence==UINT64_MAX) return BCIR_CHANNEL_INVALID;
  event.sequence=sequence;event.generation=loopback->generation;
  event.kind=BCIR_CHANNEL_EVENT_COMPLETE;event.status=BCIR_CHANNEL_OK;
  status=loop_push(loopback,event);
  if(status!=BCIR_CHANNEL_OK) return status;
  loopback->completed_sequence=sequence;
  loopback->next_sequence=sequence+1u;
  return BCIR_CHANNEL_OK;
}

static bcir_channel_status loop_sync(void *implementation,
                                     bcir_channel_handle session,
                                     uint64_t sequence,uint32_t flags) {
  bcir_loopback_channel *loopback=(bcir_loopback_channel *)implementation;
  bcir_channel_status status=loop_session(loopback,session);
  if(status!=BCIR_CHANNEL_OK) return status;
  if(flags&~(BCIR_CHANNEL_SYNC_WAIT|BCIR_CHANNEL_SYNC_CANCEL))
    return BCIR_CHANNEL_INVALID;
  /* This reference driver completes submit synchronously, so there is no
   * cancellable interval. Completed and never-submitted sequence numbers are
   * both stale; an asynchronous hardware driver may emit CANCELED instead. */
  if(flags&BCIR_CHANNEL_SYNC_CANCEL) return BCIR_CHANNEL_STALE;
  return sequence<=loopback->completed_sequence?BCIR_CHANNEL_OK:BCIR_CHANNEL_WOULD_BLOCK;
}

static bcir_channel_status loop_event(void *implementation,
                                      bcir_channel_handle session,
                                      bcir_channel_event *event) {
  bcir_loopback_channel *loopback=(bcir_loopback_channel *)implementation;
  bcir_channel_status status=loop_session(loopback,session);
  if(status!=BCIR_CHANNEL_OK) return status;
  if(!loopback->event_count) return BCIR_CHANNEL_WOULD_BLOCK;
  *event=loopback->events[loopback->event_head];
  loopback->event_head=(loopback->event_head+1u)%loopback->event_capacity;
  loopback->event_count--;
  return BCIR_CHANNEL_OK;
}

static bcir_channel_status loop_close(void *implementation,
                                      bcir_channel_handle session) {
  bcir_loopback_channel *loopback=(bcir_loopback_channel *)implementation;
  bcir_channel_status status=loop_session(loopback,session);
  if(status!=BCIR_CHANNEL_OK) return status;
  loopback->open_state=0;
  loopback->event_head=0;loopback->event_count=0;
  memset(&loopback->active_mapping,0,sizeof loopback->active_mapping);
  loopback->generation++;
  if(!loopback->generation) loopback->generation=1;
  return BCIR_CHANNEL_OK;
}

bcir_channel_status bcir_loopback_channel_init(
    bcir_loopback_channel *loopback,bcir_channel_event *event_storage,
    size_t event_capacity,uint64_t resource_capacity,
    bcir_channel_ring_policy policy,bcir_runtime_channel *channel) {
  if(!loopback||!channel||(!event_storage&&event_capacity) ||
     (policy!=BCIR_CHANNEL_RING_BACKPRESSURE&&
      policy!=BCIR_CHANNEL_RING_OVERWRITE_OLDEST)) return BCIR_CHANNEL_INVALID;
  memset(loopback,0,sizeof *loopback);
  loopback->events=event_storage;loopback->event_capacity=event_capacity;
  loopback->resource_capacity=resource_capacity;loopback->ring_policy=(uint32_t)policy;
  loopback->hooks.abi_version=BCIR_RUNTIME_CHANNEL_ABI_V1;
  loopback->hooks.struct_size=sizeof loopback->hooks;
  loopback->hooks.implementation=loopback;
  loopback->hooks.open=loop_open;loopback->hooks.claim=loop_claim;
  loopback->hooks.map=loop_map;loopback->hooks.submit=loop_submit;
  loopback->hooks.sync=loop_sync;loopback->hooks.event=loop_event;
  loopback->hooks.close=loop_close;
  return bcir_runtime_channel_init(channel,&loopback->hooks);
}

void bcir_loopback_channel_reset(bcir_loopback_channel *loopback) {
  if(!loopback) return;
  loopback->event_head=0;loopback->event_count=0;
  loopback->open_state=0;loopback->next_sequence=0;
  loopback->completed_sequence=0;loopback->generation++;
  memset(&loopback->active_mapping,0,sizeof loopback->active_mapping);
  if(!loopback->generation) loopback->generation=1;
}
