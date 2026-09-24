/*===- test_stage3.h - the Stage 3 exit flow, shared by the C and C++ harnesses ==========
 *
 * One artifact generation through every Stage 3 boundary on the loopback, then the next one:
 *
 *   control   ControlRecordV1 bytes carried by a live CONTROL ring (BACKPRESSURE) into the
 *             resident plane (G14 over G15);
 *   plan      the ExecutionPlanV1 admitted by the plane (bcir_ctl_admit_plan);
 *   data      the StreamPack written once into a pack-table slot, admitted against the live
 *             plane, its shard manifest gated there, and dispatched in place (G16);
 *   telemetry one TelemetryEnvelopeV0 per event, bound to the generation the data plane admitted,
 *             through a live OVERWRITE telemetry ring of two slots into the host intake (G15);
 *   evidence  one line that reconciles them: the intake's counts, the ring's loss against the
 *             intake's gap count, the plane's and the table's state digests.
 *
 * After the switch to the next generation the old one is offered at every boundary -- a record
 * witnessed against it, its plan, its pack (dispatch and admission), its manifest, a late sample
 * bound to it -- and every boundary must refuse it. bcir/tests/handoff_fixtures.py
 * (run_stage3_python) is the oracle and prints the same lines; the harnesses supply only the data
 * boundary (the C pack table directly, or the C++ seam's arena, owners and views) through
 * s3_ops, so the flow is written once for both native rails.
 *
 * The input is little-endian and u32-length framed: "BST3", the plane's key, scope (u8) and
 * subject (u64), then the records grant, gen-a, gen-b, stale, then plan, pack and manifest of
 * artifact a, then of artifact b. Test code: libc, no allocation.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_TEST_STAGE3_H
#define BCIR_TEST_STAGE3_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bcir_control_plane.h"
#include "bcir_handoff.h"
#include "bcir_ring.h"
#include "bcir_sha256.h"
#include "bcir_telemetry_envelope.h"

#define S3_CLAIMS_MAX 256u
#define S3_CONTROL_SLOTS 4u
#define S3_TELEMETRY_SLOT 128u
#define S3_TELEMETRY_SLOTS 2u
#define S3_SIGNAL 1u /* a signal the generated table defines */

/* The data boundary a harness supplies; `which` names the artifact (0 = a, 1 = b). */
typedef struct s3_ops {
  void *ctx;
  bcir_ho_outcome (*store)(void *ctx, int which, const uint8_t *data, size_t len);
  bcir_ho_outcome (*admit)(void *ctx, int which, bcir_ctl_state *plane);
  bcir_ho_outcome (*dispatch)(void *ctx, int which, bcir_ctl_state *plane, bcir_seg_fn fn,
                              void *fn_ctx);
  bcir_ho_outcome (*release)(void *ctx, int which);
  void (*digest)(void *ctx, uint8_t out[32]);
  const char *(*status)(bcir_status s);
} s3_ops;

typedef struct s3_in {
  const uint8_t *d;
  size_t len, pos;
  int err;
} s3_in;

static const uint8_t *s3_take(s3_in *r, size_t n) {
  if (r->err || n > r->len - r->pos) {
    r->err = 1;
    return NULL;
  }
  r->pos += n;
  return r->d + r->pos - n;
}

static uint64_t s3_le(s3_in *r, size_t n) {
  const uint8_t *p = s3_take(r, n);
  uint64_t v = 0;
  for (size_t i = 0; p && i < n; i++) v |= (uint64_t)p[i] << (8u * i);
  return v;
}

static const uint8_t *s3_blob(s3_in *r, size_t *n) {
  *n = (size_t)s3_le(r, 4);
  return s3_take(r, *n);
}

typedef struct s3_claims {
  uint64_t ids[S3_CLAIMS_MAX];
  uint32_t n;
  bcir_sha256 h;
} s3_claims;

static int s3_collect(const bcir_segment_view *seg, void *ctx) {
  s3_claims *c = (s3_claims *)ctx;
  uint8_t b[8];
  for (int i = 0; i < 8; i++) b[i] = (uint8_t)(seg->claim_id >> (8 * i));
  bcir_sha256_update(&c->h, b, sizeof b);
  if (c->n < S3_CLAIMS_MAX) c->ids[c->n] = seg->claim_id;
  c->n++;
  return 0;
}

static void s3_hex(const uint8_t *p, size_t n) {
  for (size_t i = 0; i < n; i++) printf("%02x", p[i]);
}

typedef struct s3_flow {
  const s3_ops *ops;
  bcir_ctl_state plane;
  bcir_tev_intake intake;
  uint64_t control_words[(BCIR_RING_HEADER_SIZE + S3_CONTROL_SLOTS * BCIR_RING_CONTROL_SLOT_MIN) / 8u];
  uint64_t telemetry_words[(BCIR_RING_HEADER_SIZE + S3_TELEMETRY_SLOTS * S3_TELEMETRY_SLOT) / 8u];
  bcir_ring_producer cp, tp;
  bcir_ring_consumer cc, tc;
  uint32_t seq;
  uint64_t control_delivered;
} s3_flow;

static int s3_rings(s3_flow *f) {
  bcir_ring_geometry g;
  uint8_t *cr = (uint8_t *)f->control_words, *tr = (uint8_t *)f->telemetry_words;
  memset(&g, 0, sizeof g);
  g.policy = BCIR_RING_BACKPRESSURE;
  g.payload = BCIR_RING_CONTROL;
  g.slot_size = BCIR_RING_CONTROL_SLOT_MIN;
  g.slot_count = S3_CONTROL_SLOTS;
  g.ring_id = 1u;
  if (bcir_ring_format(cr, sizeof f->control_words, &g) != BCIR_OK) return 0;
  g.policy = BCIR_RING_OVERWRITE;
  g.payload = BCIR_RING_TELEMETRY;
  g.slot_size = S3_TELEMETRY_SLOT;
  g.slot_count = S3_TELEMETRY_SLOTS;
  g.ring_id = 2u;
  if (bcir_ring_format(tr, sizeof f->telemetry_words, &g) != BCIR_OK) return 0;
  bcir_ring_producer_init(&f->cp, cr, sizeof f->control_words);
  bcir_ring_consumer_init(&f->cc, cr, sizeof f->control_words);
  bcir_ring_producer_init(&f->tp, tr, sizeof f->telemetry_words);
  bcir_ring_consumer_init(&f->tc, tr, sizeof f->telemetry_words);
  return bcir_ring_producer_attach(&f->cp, 0).verdict == BCIR_RING_OK &&
         bcir_ring_consumer_attach(&f->cc, 0).verdict == BCIR_RING_OK &&
         bcir_ring_producer_attach(&f->tp, 0).verdict == BCIR_RING_OK &&
         bcir_ring_consumer_attach(&f->tc, 0).verdict == BCIR_RING_OK;
}

/* A control record through the control ring into the plane; the intake follows the plane. */
static void s3_control(s3_flow *f, const char *label, const uint8_t *rec, size_t n) {
  uint8_t carried[BCIR_RING_CONTROL_SLOT_MIN];
  bcir_ring_outcome w = bcir_ring_publish(&f->cp, rec, n);
  bcir_ring_outcome r = w.verdict == BCIR_RING_OK
                            ? bcir_ring_consume(&f->cc, carried, sizeof carried)
                            : w;
  bcir_ctl_outcome o;
  if (w.verdict != BCIR_RING_OK || r.verdict != BCIR_RING_DELIVERED || r.position != w.position) {
    printf("control:%s ring %u %s\n", label, (unsigned)r.verdict, f->ops->status(r.status));
    return;
  }
  f->control_delivered++;
  o = bcir_ctl_submit(&f->plane, carried, r.length);
  bcir_tev_intake_set_live(&f->intake, f->plane.generation);
  printf("control:%s %u %u %s g=%u\n", label, (unsigned)o.verdict, (unsigned)o.refusal,
         f->ops->status(o.status), o.generation);
}

static void s3_plan(s3_flow *f, const char *label, const uint8_t *plan, size_t n) {
  bcir_ctl_outcome o = bcir_ctl_admit_plan(&f->plane, plan, n);
  printf("plan:%s %u %u %s g=%u\n", label, (unsigned)o.verdict, (unsigned)o.refusal,
         f->ops->status(o.status), o.generation);
}

static void s3_data(s3_flow *f, const char *label, bcir_ho_outcome o, const s3_claims *walked) {
  uint8_t digest[32];
  printf("data:%s %u %u %s g=%u ", label, (unsigned)o.verdict, (unsigned)o.refusal,
         f->ops->status(o.status), o.generation);
  if (o.handle.epoch) printf("h=%u:%u c=", o.handle.index, o.handle.epoch);
  else printf("h=- c=");
  if (walked && o.verdict == BCIR_HO_APPLIED) {
    bcir_sha256 h = walked->h;
    bcir_sha256_final(&h, digest);
    s3_hex(digest, 16);
  } else {
    printf("-");
  }
  printf("\n");
}

/* One sample bound to `generation` into the telemetry ring (never waiting: OVERWRITE). */
static void s3_emit(s3_flow *f, uint32_t generation, int64_t value) {
  bcir_tev env;
  uint8_t out[128];
  size_t n = 0;
  memset(&env, 0, sizeof env);
  env.kind = BCIR_TEV_SAMPLE;
  env.source = 1u;
  env.session = 1u;
  env.seq = ++f->seq;
  env.generation = generation;
  env.signal = S3_SIGNAL;
  env.required = 1u;
  env.value = value;
  if (bcir_tev_encode(&env, out, sizeof out, &n) != BCIR_OK ||
      bcir_ring_publish(&f->tp, out, n).verdict != BCIR_RING_OK)
    printf("tel:emit refused\n");
}

/* Read the telemetry ring dry into the intake. */
static void s3_drain(s3_flow *f) {
  uint8_t in[S3_TELEMETRY_SLOT];
  for (;;) {
    bcir_ring_outcome r = bcir_ring_consume(&f->tc, in, sizeof in);
    if (r.verdict == BCIR_RING_EMPTY) return;
    if (r.verdict == BCIR_RING_LOST) {
      printf("tel:lost %llu\n", (unsigned long long)r.count);
      continue;
    }
    if (r.verdict != BCIR_RING_DELIVERED) {
      printf("tel:ring %u %s\n", (unsigned)r.verdict, f->ops->status(r.status));
      return;
    }
    {
      bcir_tev env;
      bcir_tev_outcome o;
      memset(&env, 0, sizeof env);
      o = bcir_tev_intake_admit(&f->intake, in, r.length, &env);
      printf("tel:%u %u %u %s g=%u k=%u\n", env.seq, (unsigned)o.verdict, (unsigned)o.reason,
             f->ops->status(o.status), env.generation, (unsigned)o.classification);
    }
  }
}

static void s3_evidence(s3_flow *f) {
  bcir_tev_report rep;
  bcir_ring_accounting acct;
  uint8_t pd[32], td[32];
  bcir_tev_intake_report(&f->intake, &rep);
  if (bcir_ring_accounting_of((const uint8_t *)f->telemetry_words, sizeof f->telemetry_words,
                              &acct) != BCIR_OK)
    memset(&acct, 0xFF, sizeof acct);
  bcir_ctl_state_digest(&f->plane, pd);
  f->ops->digest(f->ops->ctx, td);
  printf("evidence g=%u accepted=%llu refused=%llu stale=%llu missing=%llu lost=%llu "
         "delivered=%llu control=%llu p=",
         f->plane.generation, (unsigned long long)rep.accepted, (unsigned long long)rep.refused,
         (unsigned long long)rep.stale, (unsigned long long)rep.missing,
         (unsigned long long)acct.lost, (unsigned long long)acct.delivered,
         (unsigned long long)f->control_delivered);
  s3_hex(pd, 32);
  printf(" t=");
  s3_hex(td, 32);
  printf("\n");
}

/* Admit artifact `which`, and on admission emit its marker sample and drain. */
static bcir_ho_outcome s3_admit(s3_flow *f, const char *label, int which) {
  bcir_ho_outcome o = f->ops->admit(f->ops->ctx, which, &f->plane);
  s3_data(f, label, o, NULL);
  if (o.verdict == BCIR_HO_APPLIED) {
    s3_emit(f, o.generation, 0);
    s3_drain(f);
  }
  return o;
}

/* Dispatch artifact `which`; each claim run is one sample bound to the dispatched generation,
 * drained after every sample (`each`) or once after all of them. */
static void s3_dispatch(s3_flow *f, const char *label, int which, int each) {
  s3_claims walked;
  bcir_ho_outcome o;
  walked.n = 0;
  bcir_sha256_init(&walked.h);
  o = f->ops->dispatch(f->ops->ctx, which, &f->plane, s3_collect, &walked);
  s3_data(f, label, o, &walked);
  if (o.verdict != BCIR_HO_APPLIED) return;
  for (uint32_t i = 0; i < walked.n && i < S3_CLAIMS_MAX; i++) {
    s3_emit(f, o.generation, (int64_t)walked.ids[i]);
    if (each) s3_drain(f);
  }
  s3_drain(f);
}

static void s3_manifest(s3_flow *f, const char *label, const uint8_t *m, size_t n) {
  s3_data(f, label, bcir_ho_admit_manifest(&f->plane, m, n), NULL);
}

/* The flow. 0 when the input was well formed (the verdicts are data, graded by the caller);
 * 2 on an input or setup error. */
static int s3_run(const uint8_t *data, size_t len, const s3_ops *ops) {
  static s3_flow flow;
  s3_flow *f = &flow;
  s3_in r;
  size_t key_len, n_grant, n_ga, n_gb, n_stale, n_pa, n_ka, n_ma, n_pb, n_kb, n_mb;
  const uint8_t *key, *grant, *gen_a, *gen_b, *stale, *plan_a, *pack_a, *man_a, *plan_b,
      *pack_b, *man_b;
  uint8_t scope;
  uint64_t subject;
  uint32_t gen_admitted_a = 0;
  memset(f, 0, sizeof *f);
  f->ops = ops;
  r.d = data;
  r.len = len;
  r.pos = 0;
  r.err = 0;
  if (!s3_take(&r, 4) || memcmp(data, "BST3", 4) != 0) return 2;
  key = s3_blob(&r, &key_len);
  scope = (uint8_t)s3_le(&r, 1);
  subject = s3_le(&r, 8);
  grant = s3_blob(&r, &n_grant);
  gen_a = s3_blob(&r, &n_ga);
  gen_b = s3_blob(&r, &n_gb);
  stale = s3_blob(&r, &n_stale);
  plan_a = s3_blob(&r, &n_pa);
  pack_a = s3_blob(&r, &n_ka);
  man_a = s3_blob(&r, &n_ma);
  plan_b = s3_blob(&r, &n_pb);
  pack_b = s3_blob(&r, &n_kb);
  man_b = s3_blob(&r, &n_mb);
  if (r.err || r.pos != len) return 2;
  if (bcir_ctl_init(&f->plane, key, key_len, scope, subject) != BCIR_OK || !s3_rings(f)) return 2;
  bcir_tev_intake_init(&f->intake, f->plane.generation);

  /* generation a flows: control -> plan -> data -> telemetry */
  s3_control(f, "grant", grant, n_grant);
  s3_control(f, "gen-a", gen_a, n_ga);
  s3_plan(f, "a", plan_a, n_pa);
  s3_data(f, "store-a", ops->store(ops->ctx, 0, pack_a, n_ka), NULL);
  gen_admitted_a = s3_admit(f, "admit-a", 0).generation;
  s3_manifest(f, "manifest-a", man_a, n_ma);
  s3_dispatch(f, "dispatch-a", 0, 0); /* drained once: the two-slot ring loses the rest */

  /* the switch, then the old generation at every boundary */
  s3_control(f, "gen-b", gen_b, n_gb);
  s3_control(f, "stale", stale, n_stale);
  s3_plan(f, "a-stale", plan_a, n_pa);
  s3_dispatch(f, "dispatch-a-stale", 0, 1);
  (void)s3_admit(f, "admit-a-stale", 0);
  s3_manifest(f, "manifest-a-stale", man_a, n_ma);
  s3_emit(f, gen_admitted_a, -1); /* a late sample from a producer that missed the switch */
  s3_drain(f);

  /* generation b flows */
  s3_plan(f, "b", plan_b, n_pb);
  s3_data(f, "store-b", ops->store(ops->ctx, 1, pack_b, n_kb), NULL);
  (void)s3_admit(f, "admit-b", 1);
  s3_manifest(f, "manifest-b", man_b, n_mb);
  s3_dispatch(f, "dispatch-b", 1, 1);

  /* the old artifact retired: its handle is dead too */
  s3_data(f, "release-a", ops->release(ops->ctx, 0), NULL);
  s3_dispatch(f, "dispatch-a-released", 0, 1);
  s3_evidence(f);
  return 0;
}

#endif /* BCIR_TEST_STAGE3_H */
