/*===- bcir_kplan.c - the native K_BCIR planner (BKPI -> BKPR, version zero) -===
 *
 * See bcir_kplan.h. Each function names the oracle function it mirrors; the parity gate
 * (bcir/tests/planner_fixtures.py, `planner.parity`) holds the two byte for byte.
 *===----------------------------------------------------------------------===*/
#include "bcir_kplan.h"

/* ---- little-endian access (every scratch array is read and written through these, so the
 * caller's storage needs no alignment and no effective type) ------------------------------ */

static uint16_t kp_rd16(const uint8_t *p) { return (uint16_t)(p[0] | ((unsigned)p[1] << 8)); }
static uint32_t kp_rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t kp_rd64(const uint8_t *p) {
  return (uint64_t)kp_rd32(p) | ((uint64_t)kp_rd32(p + 4) << 32);
}
static void kp_wr32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}
static void kp_wr64(uint8_t *p, uint64_t v) {
  kp_wr32(p, (uint32_t)v);
  kp_wr32(p + 4, (uint32_t)(v >> 32));
}
static void kp_zero(uint8_t *p, size_t n) {
  for (size_t i = 0; i < n; i++) p[i] = 0;
}

/* ---- exact 128-bit arithmetic --------------------------------------------------------------
 * The declared domain keeps every multiplicand under 2**96 and every factor under 2**32, and
 * every path weight under 2**127 (docs/kernel/BCIR_PLANNER_ABI.md), so these never wrap. */

typedef struct kp_u128 {
  uint64_t lo, hi;
} kp_u128;

static kp_u128 kp_u(uint64_t x) {
  kp_u128 r;
  r.lo = x;
  r.hi = 0;
  return r;
}
static kp_u128 kp_add(kp_u128 a, kp_u128 b) {
  kp_u128 r;
  r.lo = a.lo + b.lo;
  r.hi = a.hi + b.hi + (r.lo < a.lo ? 1u : 0u);
  return r;
}
/* a * f, for a < 2**96 and f < 2**32 */
static kp_u128 kp_mul(kp_u128 a, uint32_t f) {
  uint64_t l0 = (a.lo & 0xFFFFFFFFu) * (uint64_t)f;
  uint64_t l1 = (a.lo >> 32) * (uint64_t)f;
  kp_u128 r;
  r.lo = l0 + (l1 << 32);
  r.hi = a.hi * (uint64_t)f + (l1 >> 32) + (r.lo < l0 ? 1u : 0u);
  return r;
}
static kp_u128 kp_shr8(kp_u128 a) {
  kp_u128 r;
  r.lo = (a.lo >> 8) | (a.hi << 56);
  r.hi = a.hi >> 8;
  return r;
}
static int kp_lt(kp_u128 a, kp_u128 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
static int kp_eq(kp_u128 a, kp_u128 b) { return a.hi == b.hi && a.lo == b.lo; }
static int kp_fits63(kp_u128 a) { return a.hi == 0 && a.lo <= (uint64_t)INT64_MAX; }

/* ---- the model's vocabulary (bcir/model: Opcode, Lane, StrideClass, Domain) --------------- */

enum {
  OP_NOP = 0, OP_ADD = 3, OP_SUB = 4, OP_MUL = 5, OP_ATOMIC_ADD = 6, OP_ATOMIC_SUB = 7,
  OP_ATOMIC_XOR = 8, OP_CMPXCHG = 9, OP_BARRIER = 10, OP_PHASE_ENTER = 11, OP_PHASE_LEAVE = 12,
  OP_T_MACC = 15, OP_GEM_DISPATCH = 16, OP_PROV_NOTE = 17, OP_MAX = 17
};
enum { LANE_U = 0, LANE_UX = 1, LANE_T = 2, LANE_GGG = 3, LANE_A = 4, LANE_H = 5, LANE_MAX = 5 };
enum { SC_SCALAR = 0, SC_UNIT = 1, SC_STRIDED = 2, SC_CACHELINE = 3, SC_TILE = 4, SC_RANDOM = 5 };
enum { DOM_MMIO = 3, DOM_MAX = 5 };
enum { HAZ_UNIQUE = 0, HAZ_BARRIERED = 2, HAZ_COUNT = 3 };
enum { VER_EXACT = 2, VER_HASH = 3, VER_COUNT = 4 };
enum { ACC_HAM = 1, ACC_COUNT = 2 };
enum {
  F_VOLATILE = 1, F_DYNAMIC = 2, F_CALLEE_SIG = 4, F_TIMING = 8, F_LIFETIME = 16, F_IMM = 32,
  F_TOLERANCE = 64, F_QUANTIZED = 128, F2_PRECISION = 1
};
/* cost.DIMS */
enum { D_COMPUTE = 0, D_MEMORY = 1, D_FABRIC = 2, D_SYNC = 3, D_COMPILE = 4, D_THERMAL = 5,
       D_POWER = 6, D_RELIABILITY = 7, D_CONTENTION = 10, D_VERIFICATION = 11, D_N = 12 };
/* cost._DOMAIN_TIER: Domain -> MemTier (RAM, VRAM, NVM, MMIO, CXL, HBM -> DRAM, HBM, SSD, DRAM,
 * CXL, HBM). */
static const uint8_t kp_domain_tier[DOM_MAX + 1] = {3, 4, 6, 3, 5, 4};

/* claim record fields */
#define CL_ID(p)       kp_rd32((p) + 0)
#define CL_OPCODE(p)   ((p)[4])
#define CL_LANE(p)     ((p)[5])
#define CL_SC(p)       ((p)[6])
#define CL_DOMAIN(p)   ((p)[7])
#define CL_HAZARD(p)   ((p)[8])
#define CL_VERIFY(p)   ((p)[9])
#define CL_FLAGS(p)    ((p)[10])
#define CL_FLAGS2(p)   ((p)[11])
#define CL_COUNT(p)    kp_rd32((p) + 12)
#define CL_OP(p)       kp_rd32((p) + 16)
#define CL_NRD(p)      kp_rd16((p) + 20)
#define CL_NWR(p)      kp_rd16((p) + 22)
#define CL_STRIDE_K(p) kp_rd64((p) + 24)
#define CL_OFFSET(p)   kp_rd64((p) + 32)
#define CL_PRIMARY(p)  kp_rd32((p) + 40)
#define CL_RESERVED(p) kp_rd32((p) + 44)

/* ---- BKPI: the decoder -------------------------------------------------------------------- */

static uint64_t kp_input_size(uint32_t n_widths, uint32_t n_phases, uint32_t n_deps,
                              uint32_t n_claims, uint32_t n_refs, uint32_t n_resources,
                              uint32_t n_ops, uint32_t op_bytes) {
  return (uint64_t)BCIR_KP_INPUT_HEADER_SIZE + BCIR_KP_SCOPE_SIZE + 4u * (uint64_t)n_widths +
         12u * (uint64_t)n_phases + 4u * (uint64_t)n_deps +
         (uint64_t)BCIR_KP_CLAIM_SIZE * n_claims + 4u * (uint64_t)n_refs +
         (uint64_t)BCIR_KP_RESOURCE_SIZE * n_resources + 2u * (uint64_t)n_ops + op_bytes +
         BCIR_KP_CRC_SIZE;
}

static int kp_bytes_lt(const uint8_t *a, size_t na, const uint8_t *b, size_t nb) {
  size_t n = na < nb ? na : nb;
  for (size_t i = 0; i < n; i++)
    if (a[i] != b[i]) return a[i] < b[i];
  return na < nb;
}

bcir_status bcir_kp_decode_input(const uint8_t *data, size_t len, bcir_kp_input *out) {
  bcir_kp_input v;
  kp_zero((uint8_t *)&v, sizeof v);
  if (out) kp_zero((uint8_t *)out, sizeof *out);
  if (!out) return BCIR_ERR_NOSPACE;
  /* 1. the framing laws */
  if (!data || len < (size_t)BCIR_KP_INPUT_HEADER_SIZE + BCIR_KP_SCOPE_SIZE + BCIR_KP_CRC_SIZE)
    return BCIR_ERR_TRUNCATED;
  if (data[0] != 'B' || data[1] != 'K' || data[2] != 'P' || data[3] != 'I') return BCIR_ERR_MAGIC;
  if (kp_rd16(data + 4) != BCIR_KP_VERSION) return BCIR_ERR_VERSION;
  if (kp_rd32(data + len - 4) != bcir_crc32(data, len - 4)) return BCIR_ERR_CRC;
  if (kp_rd16(data + 6) != 0) return BCIR_ERR_RESERVED;
  for (size_t i = 40; i < 64; i++)
    if (data[i]) return BCIR_ERR_RESERVED;
  /* 2. the counts, then the size */
  v.n_phases = kp_rd32(data + 8);
  v.n_deps = kp_rd32(data + 12);
  v.n_claims = kp_rd32(data + 16);
  v.n_refs = kp_rd32(data + 20);
  v.n_resources = kp_rd32(data + 24);
  v.n_ops = kp_rd32(data + 28);
  v.op_bytes = kp_rd32(data + 32);
  v.n_widths = kp_rd32(data + 36);
  if (v.n_widths < 1 || v.n_widths > BCIR_KP_WIDTHS_MAX) return BCIR_ERR_PLANNER;
  if (v.n_phases > BCIR_KP_CLAIMS_MAX || v.n_claims > BCIR_KP_CLAIMS_MAX ||
      v.n_ops > BCIR_KP_CLAIMS_MAX || v.n_deps > BCIR_KP_REFS_MAX || v.n_refs > BCIR_KP_REFS_MAX ||
      v.n_resources > BCIR_KP_REFS_MAX || v.op_bytes > BCIR_KP_OP_BYTES_MAX)
    return BCIR_ERR_PLANNER;
  uint64_t size = kp_input_size(v.n_widths, v.n_phases, v.n_deps, v.n_claims, v.n_refs,
                                v.n_resources, v.n_ops, v.op_bytes);
  if ((uint64_t)len < size) return BCIR_ERR_TRUNCATED;
  if ((uint64_t)len > size) return BCIR_ERR_TRAILING;
  v.data = data;
  v.len = len;
  v.off_widths = BCIR_KP_INPUT_HEADER_SIZE + BCIR_KP_SCOPE_SIZE;
  v.off_phases = v.off_widths + 4u * (size_t)v.n_widths;
  v.off_deps = v.off_phases + 12u * (size_t)v.n_phases;
  v.off_claims = v.off_deps + 4u * (size_t)v.n_deps;
  v.off_refs = v.off_claims + (size_t)BCIR_KP_CLAIM_SIZE * v.n_claims;
  v.off_resources = v.off_refs + 4u * (size_t)v.n_refs;
  v.off_op_lens = v.off_resources + (size_t)BCIR_KP_RESOURCE_SIZE * v.n_resources;
  v.off_ops = v.off_op_lens + 2u * (size_t)v.n_ops;
  /* 3. the scope: target (8), tiers (14), theta (8), policy (12) */
  const uint8_t *sc = data + BCIR_KP_INPUT_HEADER_SIZE;
  uint32_t cacheline = kp_rd32(sc), elem_bytes = kp_rd32(sc + 4), mem_unit = kp_rd32(sc + 12);
  if (cacheline == 0 || (cacheline & (cacheline - 1u)) != 0) return BCIR_ERR_PLANNER;
  if (elem_bytes == 0 || mem_unit == 0) return BCIR_ERR_PLANNER;
  for (unsigned i = 0; i < 8; i++)
    if (kp_rd32(sc + 4u * i) > BCIR_KP_PARAM_MAX) return BCIR_ERR_PLANNER;
  for (unsigned i = 8; i < 22; i++) {
    uint32_t f = kp_rd32(sc + 4u * i);
    if (f < 1 || f > BCIR_KP_PARAM_MAX) return BCIR_ERR_PLANNER;
  }
  for (unsigned i = 22; i < 30; i++)
    if (kp_rd32(sc + 4u * i) > BCIR_KP_THETA_MAX) return BCIR_ERR_PLANNER;
  for (unsigned i = 30; i < 42; i++)
    if (kp_rd32(sc + 4u * i) > BCIR_KP_PARAM_MAX) return BCIR_ERR_PLANNER;
  /* 4. the widths, the phases */
  for (uint32_t i = 0; i < v.n_widths; i++) {
    uint32_t w = kp_rd32(data + v.off_widths + 4u * (size_t)i);
    if (w < 1 || w > BCIR_KP_PARAM_MAX) return BCIR_ERR_PLANNER;
  }
  uint64_t sum_deps = 0, sum_claims = 0;
  for (uint32_t i = 0; i < v.n_phases; i++) {
    const uint8_t *p = data + v.off_phases + 12u * (size_t)i;
    sum_deps += kp_rd32(p + 4);
    sum_claims += kp_rd32(p + 8);
  }
  if (sum_deps != v.n_deps || sum_claims != v.n_claims) return BCIR_ERR_PLANNER;
  /* 5. the resources, the op table */
  for (uint32_t i = 0; i < v.n_resources; i++) {
    const uint8_t *r = data + v.off_resources + BCIR_KP_RESOURCE_SIZE * (size_t)i;
    if (r[3]) return BCIR_ERR_RESERVED;
    if (r[0] > 1 || r[1] > DOM_MAX || r[2] >= ACC_COUNT) return BCIR_ERR_PLANNER;
    if (!r[0] && (r[1] || r[2])) return BCIR_ERR_PLANNER;
  }
  uint64_t sum_ops = 0;
  for (uint32_t i = 0; i < v.n_ops; i++) sum_ops += kp_rd16(data + v.off_op_lens + 2u * (size_t)i);
  if (sum_ops != v.op_bytes) return BCIR_ERR_PLANNER;
  size_t pos = v.off_ops, prev = 0, prev_len = 0;
  for (uint32_t i = 0; i < v.n_ops; i++) {
    size_t n = kp_rd16(data + v.off_op_lens + 2u * (size_t)i);
    if (!bcir_utf8_valid(data + pos, n)) return BCIR_ERR_UTF8;
    if (i && !kp_bytes_lt(data + prev, prev_len, data + pos, n)) return BCIR_ERR_PLANNER;
    prev = pos;
    prev_len = n;
    pos += n;
  }
  /* 6. each claim, in order */
  uint64_t ref_pos = 0;
  uint32_t next_resource = 0;
  for (uint32_t c = 0; c < v.n_claims; c++) {
    const uint8_t *cl = data + v.off_claims + (size_t)BCIR_KP_CLAIM_SIZE * c;
    if (CL_RESERVED(cl)) return BCIR_ERR_RESERVED;
    if (CL_OPCODE(cl) > OP_MAX || CL_LANE(cl) > LANE_MAX || CL_SC(cl) > SC_RANDOM ||
        CL_DOMAIN(cl) > DOM_MAX || CL_HAZARD(cl) >= HAZ_COUNT || CL_VERIFY(cl) >= VER_COUNT ||
        (CL_FLAGS2(cl) & ~F2_PRECISION) != 0 || CL_OP(cl) >= v.n_ops)
      return BCIR_ERR_PLANNER;
    uint32_t operands = (uint32_t)CL_NRD(cl) + CL_NWR(cl);
    if (operands > BCIR_KP_OPERANDS_MAX) return BCIR_ERR_PLANNER;
    if (operands > (uint64_t)v.n_refs - ref_pos) return BCIR_ERR_PLANNER;
    for (uint32_t k = 0; k < operands; k++) {
      uint32_t x = kp_rd32(data + v.off_refs + 4u * (size_t)(ref_pos + k));
      if (x > next_resource || x >= v.n_resources) return BCIR_ERR_PLANNER;
      if (x == next_resource) next_resource++;
    }
    ref_pos += operands;
    uint32_t primary = CL_PRIMARY(cl);
    if (primary != BCIR_KP_NO_RESOURCE) {
      if (CL_NRD(cl) || primary > next_resource || primary >= v.n_resources)
        return BCIR_ERR_PLANNER;
      if (!data[v.off_resources + BCIR_KP_RESOURCE_SIZE * (size_t)primary]) return BCIR_ERR_PLANNER;
      if (primary == next_resource) next_resource++;
    }
  }
  /* 7. every operand and resource accounted for */
  if (ref_pos != v.n_refs) return BCIR_ERR_PLANNER;
  if (next_resource != v.n_resources) return BCIR_ERR_PLANNER;
  *out = v;
  return BCIR_OK;
}

/* ---- the planner's working set --------------------------------------------------------------
 * Byte offsets into the caller's scratch; every array is little-endian u32 or u8. */

typedef struct kp_layout {
  uint64_t ph_tab, ph_color, ph_stack, ph_dep, ph_claim, topo, pos_claim, pos_phase, cl_ref,
      rdver, res_ver, res_vstamp, res_bstamp, id_tab, id_stamp, op_seen, pos_mode, node_start,
      pred, chosen, total;
  uint32_t cap_p, cap_c;
  uint64_t n_nodes;
} kp_layout;

static uint32_t kp_pow2_at_least(uint64_t n) {
  uint32_t c = 2;
  while (c < n) c <<= 1;
  return c;
}

static int kp_is_reduce_gather(const bcir_kp_input *in, uint32_t op) {
  static const char name[] = "reduce.gather";
  size_t off = in->off_ops;
  for (uint32_t i = 0; i < op; i++) off += kp_rd16(in->data + in->off_op_lens + 2u * (size_t)i);
  size_t n = kp_rd16(in->data + in->off_op_lens + 2u * (size_t)op);
  if (n != sizeof name - 1u) return 0;
  for (size_t i = 0; i < n; i++)
    if (in->data[off + i] != (uint8_t)name[i]) return 0;
  return 1;
}

/* The op index spelled "reduce.gather", or n_ops when the table has none (it is sorted, but a
 * linear scan keeps this trivially the same predicate as `claim.op == "reduce.gather"`). */
static uint32_t kp_reduce_gather_op(const bcir_kp_input *in) {
  for (uint32_t i = 0; i < in->n_ops; i++)
    if (kp_is_reduce_gather(in, i)) return i;
  return in->n_ops;
}

/* realize._geometry: the ascending unique widths, the cacheline-bucket width, the tile width */
typedef struct kp_geom {
  uint32_t uw[BCIR_KP_WIDTHS_MAX];
  unsigned n_uw;
  uint32_t ux, tile;
} kp_geom;

static void kp_geometry(const bcir_kp_input *in, kp_geom *g) {
  g->n_uw = 0;
  for (uint32_t i = 0; i < in->n_widths; i++) {
    uint32_t w = kp_rd32(in->data + in->off_widths + 4u * (size_t)i);
    unsigned k = 0;
    while (k < g->n_uw && g->uw[k] < w) k++;
    if (k < g->n_uw && g->uw[k] == w) continue;
    for (unsigned m = g->n_uw; m > k; m--) g->uw[m] = g->uw[m - 1];
    g->uw[k] = w;
    g->n_uw++;
  }
  uint32_t widest = g->uw[g->n_uw - 1];
  uint32_t last = kp_rd32(in->data + in->off_widths + 4u * (size_t)(in->n_widths - 1u));
  g->ux = widest < 8u ? widest : 8u;
  g->tile = last < 16u ? last : 16u;
}

/* How many realizations realize._offer_rows enumerates for a claim. */
static unsigned kp_row_count(const uint8_t *cl, const kp_geom *g, uint32_t reduce_gather) {
  uint8_t op = CL_OPCODE(cl), sc = CL_SC(cl);
  if (op == OP_NOP || op == OP_PHASE_ENTER || op == OP_PHASE_LEAVE || op == OP_PROV_NOTE) return 1;
  if (op == OP_BARRIER) return 1;
  if (op >= OP_ATOMIC_ADD && op <= OP_CMPXCHG) return 1;
  if (CL_OP(cl) == reduce_gather) return 2;
  if (sc == SC_UNIT || sc == SC_SCALAR) return g->n_uw;
  if (sc == SC_STRIDED || sc == SC_CACHELINE) return 2;
  return 1; /* RANDOM, TILE */
}

static bcir_status kp_plan_layout(const bcir_kp_input *in, kp_layout *L) {
  /* The one invariant the geometry indexes by: a record bcir_kp_decode_input accepted holds it,
   * and a hand-built bcir_kp_input that does not is refused here, before kp_geometry reads
   * uw[n_uw - 1] or inserts past uw[BCIR_KP_WIDTHS_MAX - 1]. */
  if (in->n_widths == 0 || in->n_widths > BCIR_KP_WIDTHS_MAX) return BCIR_ERR_PLANNER;
  kp_geom g;
  kp_geometry(in, &g);
  uint32_t reduce_gather = kp_reduce_gather_op(in);
  uint64_t nodes = 0;
  for (uint32_t c = 0; c < in->n_claims; c++)
    nodes += kp_row_count(in->data + in->off_claims + (size_t)BCIR_KP_CLAIM_SIZE * c, &g,
                          reduce_gather);
  uint64_t np = in->n_phases, nc = in->n_claims, o = 0;
  L->cap_p = kp_pow2_at_least(2u * np);
  L->cap_c = kp_pow2_at_least(2u * nc);
  L->n_nodes = nodes;
  L->ph_tab = o;     o += 4u * (uint64_t)L->cap_p;
  L->ph_color = o;   o += np;
  L->ph_stack = o;   o += 8u * np;
  L->ph_dep = o;     o += 4u * np;
  L->ph_claim = o;   o += 4u * np;
  L->topo = o;       o += 4u * np;
  L->pos_claim = o;  o += 4u * nc;
  L->pos_phase = o;  o += 4u * nc;
  L->cl_ref = o;     o += 4u * nc;
  L->rdver = o;      o += 4u * (uint64_t)in->n_refs;
  L->res_ver = o;    o += 4u * (uint64_t)in->n_resources;
  L->res_vstamp = o; o += 4u * (uint64_t)in->n_resources;
  L->res_bstamp = o; o += 4u * (uint64_t)in->n_resources;
  L->id_tab = o;     o += 4u * (uint64_t)L->cap_c;
  L->id_stamp = o;   o += 4u * (uint64_t)L->cap_c;
  L->op_seen = o;    o += ((uint64_t)in->n_ops + 7u) / 8u;
  L->pos_mode = o;   o += nc;
  L->node_start = o; o += 4u * (nc + 1u);
  L->pred = o;       o += nodes;
  L->chosen = o;     o += nc;
  L->total = o;
  if (o > (uint64_t)SIZE_MAX) return BCIR_ERR_OVERFLOW;
  return BCIR_OK;
}

bcir_status bcir_kp_scratch_size(const bcir_kp_input *in, size_t *bytes) {
  if (bytes) *bytes = 0;
  if (!in || !in->data || !bytes) return BCIR_ERR_NOSPACE;
  kp_layout L;
  bcir_status st = kp_plan_layout(in, &L);
  if (st != BCIR_OK) return st;
  *bytes = (size_t)L.total;
  return BCIR_OK;
}

bcir_status bcir_kp_realization_size(const bcir_kp_input *in, size_t *bytes) {
  if (bytes) *bytes = 0;
  if (!in || !in->data || !bytes) return BCIR_ERR_NOSPACE;
  uint64_t n = (uint64_t)BCIR_KP_REALIZATION_HEADER_SIZE +
               (uint64_t)BCIR_KP_STEP_SIZE * in->n_claims + BCIR_KP_CRC_SIZE;
  if (n > (uint64_t)SIZE_MAX) return BCIR_ERR_OVERFLOW;
  *bytes = (size_t)n;
  return BCIR_OK;
}

/* ---- the planner -------------------------------------------------------------------------- */

typedef struct kp_ctx {
  const bcir_kp_input *in;
  uint8_t *s; /* scratch */
  kp_layout L;
  kp_geom g;
  uint32_t target[8]; /* cacheline, elem_bytes, gather_penalty, mem_unit, base_overhead,
                       * thermal_density, power_density, per_op_heat */
  uint32_t tier_bw[7], tier_lat[7];
  uint32_t w[D_N]; /* weights.weights(): the policy folded with Theta */
  int hot;         /* Theta.thermal >= 60 */
  uint32_t reduce_gather;
} kp_ctx;

enum { T_CACHELINE = 0, T_ELEM = 1, T_GATHER = 2, T_MEM_UNIT = 3, T_OVERHEAD = 4, T_THERMAL = 5,
       T_POWER = 6, T_HEAT = 7 };

#define S32(off, i)        kp_rd32(x->s + (off) + 4u * (uint64_t)(i))
#define S32_SET(off, i, v) kp_wr32(x->s + (off) + 4u * (uint64_t)(i), (v))

static const uint8_t *kp_claim(const kp_ctx *x, uint32_t c) {
  return x->in->data + x->in->off_claims + (size_t)BCIR_KP_CLAIM_SIZE * c;
}
static uint32_t kp_ref(const kp_ctx *x, uint64_t k) {
  return kp_rd32(x->in->data + x->in->off_refs + 4u * (size_t)k);
}
static const uint8_t *kp_resource(const kp_ctx *x, uint32_t r) {
  return x->in->data + x->in->off_resources + BCIR_KP_RESOURCE_SIZE * (size_t)r;
}

/* weights.weights(h, theta, phase, policy) */
static void kp_weights(kp_ctx *x) {
  const uint8_t *sc = x->in->data + BCIR_KP_INPUT_HEADER_SIZE;
  uint32_t thermal = kp_rd32(sc + 88), power = kp_rd32(sc + 92), mem = kp_rd32(sc + 96),
           contention = kp_rd32(sc + 100), wear = kp_rd32(sc + 108);
  for (unsigned i = 0; i < D_N; i++) x->w[i] = kp_rd32(sc + 120 + 4u * i);
  x->w[D_THERMAL] += thermal / 20u;
  x->w[D_POWER] += power / 20u;
  x->w[D_RELIABILITY] += (thermal + wear) / 50u;
  x->w[D_MEMORY] += mem / 40u;
  x->w[D_FABRIC] += mem / 60u;
  x->w[D_CONTENTION] += contention / 40u;
  x->hot = thermal >= 60u;
}

/* One realization: realize._offer_rows' spec (lane, width, name, stride penalty, extra
 * compile). */
typedef struct kp_spec {
  uint8_t lane, name;
  uint32_t width, sp, extra;
} kp_spec;

static uint32_t kp_bit_length(uint64_t v) {
  uint32_t n = 0;
  while (v) {
    n++;
    v >>= 1;
  }
  return n;
}

static unsigned kp_specs(const kp_ctx *x, const uint8_t *cl, int ham, kp_spec *out) {
  uint8_t op = CL_OPCODE(cl), sc = CL_SC(cl);
  unsigned n = 0;
#define KP_SPEC(L_, W_, N_, SP_, EX_) \
  (out[n].lane = (uint8_t)(L_), out[n].width = (W_), out[n].name = (uint8_t)(N_), \
   out[n].sp = (SP_), out[n].extra = (EX_), n++)
  if (op == OP_NOP || op == OP_PHASE_ENTER || op == OP_PHASE_LEAVE || op == OP_PROV_NOTE) {
    KP_SPEC(LANE_H, 1u, BCIR_KP_NAME_NOOP, 0u, 0u);
    return n;
  }
  if (op == OP_BARRIER) {
    KP_SPEC(LANE_H, 1u, BCIR_KP_NAME_BARRIER, 0u, 0u);
    return n;
  }
  if (op >= OP_ATOMIC_ADD && op <= OP_CMPXCHG) {
    KP_SPEC(LANE_A, 1u, BCIR_KP_NAME_ATOMIC, 1u, 0u);
    return n;
  }
  uint32_t gp = x->target[T_GATHER];
  if (ham) {
    uint32_t count = CL_COUNT(cl);
    uint64_t nn = count > 1u ? count : 1u;
    uint32_t b = kp_bit_length(nn - 1u);
    gp = b > 1u ? b : 1u;
  }
  if (CL_OP(cl) == x->reduce_gather) {
    KP_SPEC(LANE_U, 1u, BCIR_KP_NAME_BLOCKED, 1u, 0u);
    KP_SPEC(LANE_GGG, 1u, BCIR_KP_NAME_GATHER, gp, 0u);
  } else if (sc == SC_UNIT || sc == SC_SCALAR) {
    for (unsigned i = 0; i < x->g.n_uw; i++) {
      uint32_t w = x->g.uw[i];
      if (w == 1u)
        KP_SPEC(CL_LANE(cl), 1u, BCIR_KP_NAME_SCALAR, 1u, 0u);
      else
        KP_SPEC(LANE_U, w, BCIR_KP_NAME_VEC, 1u, 0u);
    }
  } else if (sc == SC_STRIDED) {
    /* realize._stride_penalty: min(max(stride_k, 1), cacheline // elem_bytes) */
    int64_t k = (int64_t)CL_STRIDE_K(cl);
    uint32_t cap = x->target[T_CACHELINE] / x->target[T_ELEM];
    uint64_t kk = k > 1 ? (uint64_t)k : 1u;
    KP_SPEC(LANE_U, 1u, BCIR_KP_NAME_STRIDED, kk < cap ? (uint32_t)kk : cap, 0u);
    KP_SPEC(LANE_GGG, 1u, BCIR_KP_NAME_GATHER, gp, 0u);
  } else if (sc == SC_CACHELINE) {
    KP_SPEC(LANE_UX, x->g.ux, BCIR_KP_NAME_UX_BUCKET, 2u, CL_COUNT(cl) / 4u);
    KP_SPEC(LANE_GGG, 1u, BCIR_KP_NAME_GATHER, gp, 0u);
  } else if (sc == SC_RANDOM) {
    KP_SPEC(LANE_GGG, 1u, BCIR_KP_NAME_GATHER, gp, 0u);
  } else { /* SC_TILE */
    KP_SPEC(LANE_T, x->g.tile, BCIR_KP_NAME_TILE, 1u, 0u);
  }
#undef KP_SPEC
  return n;
}

enum { MODE_PLAIN = 0, MODE_CSE = 1, MODE_DEFOREST = 2 };

/* realize._base_cost, then the intra-phase discount (the Q8 couple). `base` has 12 entries. */
static void kp_base(const kp_ctx *x, const uint8_t *cl, const kp_spec *sp, uint32_t bw_f,
                    uint32_t lat_f, unsigned mode, kp_u128 *base) {
  uint32_t count = CL_COUNT(cl);
  uint64_t n = count > 1u ? count : 1u;
  uint32_t streams = (uint32_t)CL_NRD(cl) + CL_NWR(cl);
  uint64_t ceil = (n + sp->width - 1u) / sp->width;
  uint8_t op = CL_OPCODE(cl);
  for (unsigned i = 0; i < D_N; i++) base[i] = kp_u(0);
  if (op == OP_NOP || op == OP_PHASE_ENTER || op == OP_PHASE_LEAVE || op == OP_PROV_NOTE) {
    /* the zero vector */
  } else if (op == OP_BARRIER) {
    base[D_SYNC] = kp_u(16u);
  } else {
    uint64_t compute = 0;
    if (op == OP_ADD || op == OP_SUB || op == OP_MUL) compute = ceil;
    else if (op == OP_T_MACC) compute = ceil * 2u;
    kp_u128 mem_shared = kp_shr8(kp_mul(
        kp_mul(kp_mul(kp_u(n), streams), x->target[T_MEM_UNIT]), bw_f));
    kp_u128 access = kp_mul(kp_mul(kp_mul(kp_mul(kp_u(ceil), streams), x->target[T_OVERHEAD]),
                                   sp->sp),
                            lat_f);
    uint64_t heat = ceil * x->target[T_HEAT];
    base[D_COMPUTE] = kp_u(compute);
    base[D_MEMORY] = kp_add(mem_shared, kp_shr8(access));
    base[D_COMPILE] = kp_u(sp->extra);
    base[D_THERMAL] = kp_u((uint64_t)sp->width * x->target[T_THERMAL] + heat);
    base[D_POWER] = kp_u((uint64_t)sp->width * x->target[T_POWER] + heat);
    uint8_t ver = CL_VERIFY(cl);
    base[D_VERIFICATION] = kp_u(ver == VER_EXACT || ver == VER_HASH ? n : 0u);
  }
  if (mode == MODE_CSE) {
    /* realize._cse_memory_q8: ((1 + |wr|) * 256) // max(1, |rd| + |wr|) */
    uint32_t full = streams ? streams : 1u;
    uint32_t q = ((1u + CL_NWR(cl)) * 256u) / full;
    base[D_COMPUTE] = kp_u(0);
    base[D_MEMORY] = kp_shr8(kp_mul(base[D_MEMORY], q));
  } else if (mode == MODE_DEFOREST) {
    base[D_MEMORY] = kp_shr8(kp_mul(base[D_MEMORY], 192u));
  }
}

/* realize._edge_cost_pair */
static void kp_edge(const kp_ctx *x, const kp_u128 *base, int hot, kp_u128 *plain,
                    kp_u128 *fused) {
  kp_u128 rest = kp_u(0);
  for (unsigned i = 0; i < D_N; i++) {
    if (i == D_MEMORY) continue;
    kp_u128 b = base[i];
    if (hot && (i == D_THERMAL || i == D_POWER)) b = kp_shr8(kp_mul(b, 320u));
    rest = kp_add(rest, kp_mul(b, x->w[i]));
  }
  *plain = kp_add(rest, kp_mul(base[D_MEMORY], x->w[D_MEMORY]));
  *fused = kp_add(rest, kp_mul(kp_shr8(kp_mul(base[D_MEMORY], 192u)), x->w[D_MEMORY]));
}

/* realize.cse_eligible */
static int kp_cse_eligible(const kp_ctx *x, const uint8_t *cl, uint32_t ref0) {
  uint8_t op = CL_OPCODE(cl), fl = CL_FLAGS(cl);
  if (CL_NRD(cl) == 0) return 0;
  if (CL_HAZARD(cl) != HAZ_UNIQUE || (fl & F_VOLATILE)) return 0;
  if ((op >= OP_ATOMIC_ADD && op <= OP_CMPXCHG) || op == OP_BARRIER || op == OP_PHASE_ENTER ||
      op == OP_PHASE_LEAVE || op == OP_GEM_DISPATCH || op == OP_PROV_NOTE)
    return 0;
  if (fl & (F_CALLEE_SIG | F_TIMING | F_LIFETIME)) return 0;
  if (CL_LANE(cl) == LANE_GGG || CL_SC(cl) == SC_RANDOM) return 0;
  if ((fl & (F_IMM | F_TOLERANCE | F_QUANTIZED)) || (CL_FLAGS2(cl) & F2_PRECISION)) return 0;
  if (CL_DOMAIN(cl) == DOM_MMIO) return 0;
  uint32_t operands = (uint32_t)CL_NRD(cl) + CL_NWR(cl);
  for (uint32_t k = 0; k < operands; k++) {
    const uint8_t *r = kp_resource(x, kp_ref(x, (uint64_t)ref0 + k));
    if (r[0] && r[1] == DOM_MMIO) return 0;
  }
  return 1;
}

/* FNV-1a over realize.cse_identity's fields (the reads at their versions). */
static uint64_t kp_mix(uint64_t h, uint64_t v) {
  for (unsigned i = 0; i < 8; i++) {
    h ^= (uint8_t)(v >> (8u * i));
    h *= 1099511628211ull;
  }
  return h;
}

static uint64_t kp_identity_hash(const kp_ctx *x, const uint8_t *cl, uint32_t ref0) {
  uint64_t h = 1469598103934665603ull;
  h = kp_mix(h, CL_OP(cl));
  h = kp_mix(h, CL_OPCODE(cl) | ((uint64_t)CL_LANE(cl) << 8) | ((uint64_t)CL_SC(cl) << 16) |
                    ((uint64_t)CL_DOMAIN(cl) << 24) |
                    ((uint64_t)((CL_FLAGS(cl) & F_DYNAMIC) != 0) << 32));
  h = kp_mix(h, CL_COUNT(cl));
  h = kp_mix(h, CL_STRIDE_K(cl));
  h = kp_mix(h, CL_OFFSET(cl));
  for (uint32_t k = 0; k < CL_NRD(cl); k++) {
    h = kp_mix(h, kp_ref(x, (uint64_t)ref0 + k));
    h = kp_mix(h, S32(x->L.rdver, (uint64_t)ref0 + k));
  }
  return h;
}

static int kp_same_identity(const kp_ctx *x, uint32_t a, uint32_t b) {
  const uint8_t *ca = kp_claim(x, a), *cb = kp_claim(x, b);
  if (CL_OP(ca) != CL_OP(cb) || CL_OPCODE(ca) != CL_OPCODE(cb) || CL_LANE(ca) != CL_LANE(cb) ||
      CL_SC(ca) != CL_SC(cb) || CL_COUNT(ca) != CL_COUNT(cb) ||
      CL_STRIDE_K(ca) != CL_STRIDE_K(cb) || CL_OFFSET(ca) != CL_OFFSET(cb) ||
      CL_DOMAIN(ca) != CL_DOMAIN(cb) ||
      ((CL_FLAGS(ca) & F_DYNAMIC) != 0) != ((CL_FLAGS(cb) & F_DYNAMIC) != 0) ||
      CL_NRD(ca) != CL_NRD(cb))
    return 0;
  uint32_t ra = S32(x->L.cl_ref, a), rb = S32(x->L.cl_ref, b);
  for (uint32_t k = 0; k < CL_NRD(ca); k++) {
    if (kp_ref(x, (uint64_t)ra + k) != kp_ref(x, (uint64_t)rb + k)) return 0;
    if (S32(x->L.rdver, (uint64_t)ra + k) != S32(x->L.rdver, (uint64_t)rb + k)) return 0;
  }
  return 1;
}

/* The phase-id table: an id -> its record index (+1; 0 empty). 0 on a duplicate id. */
static uint32_t kp_hash32(uint32_t v, uint32_t cap) {
  return (uint32_t)(((uint64_t)v * 0x9E3779B97F4A7C15ull) >> 32) & (cap - 1u);
}

static int kp_phase_insert(kp_ctx *x, uint32_t id, uint32_t index) {
  uint32_t cap = x->L.cap_p, i = kp_hash32(id, cap);
  for (;;) {
    uint32_t e = S32(x->L.ph_tab, i);
    if (!e) {
      S32_SET(x->L.ph_tab, i, index + 1u);
      return 1;
    }
    if (kp_rd32(x->in->data + x->in->off_phases + 12u * (size_t)(e - 1u)) == id) return 0;
    i = (i + 1u) & (cap - 1u);
  }
}

static uint32_t kp_phase_find(const kp_ctx *x, uint32_t id) {
  uint32_t cap = x->L.cap_p, i = kp_hash32(id, cap);
  for (;;) {
    uint32_t e = S32(x->L.ph_tab, i);
    if (!e) return 0xFFFFFFFFu;
    if (kp_rd32(x->in->data + x->in->off_phases + 12u * (size_t)(e - 1u)) == id) return e - 1u;
    i = (i + 1u) & (cap - 1u);
  }
}

/* The laws that need memory, in order: phase ids unique, claim ids unique, every op
 * referenced (bcir.abi.planner_abi.check_input). */
static bcir_status kp_check(kp_ctx *x) {
  const bcir_kp_input *in = x->in;
  for (uint32_t i = 0; i < x->L.cap_p; i++) S32_SET(x->L.ph_tab, i, 0u);
  for (uint32_t p = 0; p < in->n_phases; p++)
    if (!kp_phase_insert(x, kp_rd32(in->data + in->off_phases + 12u * (size_t)p), p))
      return BCIR_ERR_PLANNER;
  uint32_t cap = x->L.cap_c;
  for (uint32_t i = 0; i < cap; i++) S32_SET(x->L.id_tab, i, 0u);
  for (uint32_t c = 0; c < in->n_claims; c++) {
    uint32_t id = CL_ID(kp_claim(x, c)), i = kp_hash32(id, cap);
    for (;;) {
      uint32_t e = S32(x->L.id_tab, i);
      if (!e) {
        S32_SET(x->L.id_tab, i, c + 1u);
        break;
      }
      if (CL_ID(kp_claim(x, e - 1u)) == id) return BCIR_ERR_PLANNER;
      i = (i + 1u) & (cap - 1u);
    }
  }
  uint64_t bytes = ((uint64_t)in->n_ops + 7u) / 8u;
  kp_zero(x->s + x->L.op_seen, (size_t)bytes);
  uint32_t seen = 0;
  for (uint32_t c = 0; c < in->n_claims; c++) {
    uint32_t op = CL_OP(kp_claim(x, c));
    uint8_t *b = x->s + x->L.op_seen + op / 8u;
    if (!(*b & (1u << (op % 8u)))) {
      *b = (uint8_t)(*b | (1u << (op % 8u)));
      seen++;
    }
  }
  if (seen != in->n_ops) return BCIR_ERR_PLANNER;
  return BCIR_OK;
}

/* model.topological_phase_ids: dependency-first, roots in declaration order, a missing
 * dependency ignored, a back edge skipped. Fills `topo` with phase record indices. */
static void kp_topo(kp_ctx *x) {
  const bcir_kp_input *in = x->in;
  uint32_t np = in->n_phases, dep_at = 0, claim_at = 0, out = 0;
  for (uint32_t p = 0; p < np; p++) {
    const uint8_t *ph = in->data + in->off_phases + 12u * (size_t)p;
    S32_SET(x->L.ph_dep, p, dep_at);
    S32_SET(x->L.ph_claim, p, claim_at);
    dep_at += kp_rd32(ph + 4);
    claim_at += kp_rd32(ph + 8);
    x->s[x->L.ph_color + p] = 0;
  }
  for (uint32_t root = 0; root < np; root++) {
    if (x->s[x->L.ph_color + root]) continue;
    x->s[x->L.ph_color + root] = 1;
    uint32_t depth = 0;
    S32_SET(x->L.ph_stack, 0, root);
    S32_SET(x->L.ph_stack, 1, 0u);
    depth = 1;
    while (depth) {
      uint32_t pid = S32(x->L.ph_stack, 2u * (depth - 1u));
      uint32_t next = S32(x->L.ph_stack, 2u * (depth - 1u) + 1u);
      const uint8_t *ph = in->data + in->off_phases + 12u * (size_t)pid;
      uint32_t nd = kp_rd32(ph + 4), d0 = S32(x->L.ph_dep, pid);
      int pushed = 0;
      while (next < nd) {
        uint32_t dep_id = kp_rd32(in->data + in->off_deps + 4u * (size_t)(d0 + next));
        next++;
        S32_SET(x->L.ph_stack, 2u * (depth - 1u) + 1u, next);
        uint32_t dep = kp_phase_find(x, dep_id);
        if (dep != 0xFFFFFFFFu && x->s[x->L.ph_color + dep] == 0) {
          x->s[x->L.ph_color + dep] = 1;
          S32_SET(x->L.ph_stack, 2u * depth, dep);
          S32_SET(x->L.ph_stack, 2u * depth + 1u, 0u);
          depth++;
          pushed = 1;
          break;
        }
      }
      if (!pushed) {
        depth--;
        if (x->s[x->L.ph_color + pid] != 2) {
          x->s[x->L.ph_color + pid] = 2;
          S32_SET(x->L.topo, out, pid);
          out++;
        }
      }
    }
  }
}

/* The Q8 resource factors of a claim's primary resource (realize.fused_offer): the first read,
 * else the primary resource; a declared resource's tier and addressing model, else DRAM/flat. */
static void kp_factors(const kp_ctx *x, const uint8_t *cl, uint32_t ref0, uint32_t *bw,
                       uint32_t *lat, int *ham) {
  uint32_t r = CL_NRD(cl) ? kp_ref(x, ref0) : CL_PRIMARY(cl);
  *bw = 256u;
  *lat = 256u;
  *ham = 0;
  if (r == BCIR_KP_NO_RESOURCE) return;
  const uint8_t *res = kp_resource(x, r);
  if (!res[0]) return;
  uint8_t tier = kp_domain_tier[res[1]];
  *bw = x->tier_bw[tier];
  *lat = x->tier_lat[tier];
  *ham = res[2] == ACC_HAM;
}

static int kp_shares_reads(const kp_ctx *x, uint32_t a, uint32_t b) {
  const uint8_t *ca = kp_claim(x, a), *cb = kp_claim(x, b);
  uint32_t ra = S32(x->L.cl_ref, a), rb = S32(x->L.cl_ref, b);
  for (uint32_t i = 0; i < CL_NRD(ca); i++)
    for (uint32_t k = 0; k < CL_NRD(cb); k++)
      if (kp_ref(x, (uint64_t)ra + i) == kp_ref(x, (uint64_t)rb + k)) return 1;
  return 0;
}

bcir_status bcir_kp_plan(const bcir_kp_input *in, void *scratch, size_t scratch_len,
                         uint8_t *out, size_t cap, size_t *out_len) {
  if (out_len) *out_len = 0;
  size_t need = 0;
  if (!in || !in->data || !out_len) {
    if (out && cap) kp_zero(out, cap);
    return BCIR_ERR_NOSPACE;
  }
  bcir_status st = bcir_kp_realization_size(in, &need);
  if (st != BCIR_OK) {
    if (out && cap) kp_zero(out, cap);
    return st;
  }
  kp_ctx ctx;
  kp_ctx *x = &ctx;
  kp_zero((uint8_t *)x, sizeof *x);
  x->in = in;
  st = kp_plan_layout(in, &x->L);
  if (st == BCIR_OK && (!out || cap < need || !scratch || scratch_len < x->L.total))
    st = BCIR_ERR_NOSPACE;
  if (st != BCIR_OK) {
    if (out && cap) kp_zero(out, cap);
    return st;
  }
  x->s = (uint8_t *)scratch;
  st = kp_check(x);
  if (st != BCIR_OK) {
    kp_zero(out, cap);
    return st;
  }
  const uint8_t *sc = in->data + BCIR_KP_INPUT_HEADER_SIZE;
  for (unsigned i = 0; i < 8; i++) x->target[i] = kp_rd32(sc + 4u * i);
  for (unsigned i = 0; i < 7; i++) {
    x->tier_bw[i] = kp_rd32(sc + 32 + 8u * i);
    x->tier_lat[i] = kp_rd32(sc + 36 + 8u * i);
  }
  kp_geometry(in, &x->g);
  kp_weights(x);
  x->reduce_gather = kp_reduce_gather_op(in);
  kp_topo(x);

  /* The claims' first operand reference, in record order. */
  uint64_t ref = 0;
  for (uint32_t c = 0; c < in->n_claims; c++) {
    const uint8_t *cl = kp_claim(x, c);
    S32_SET(x->L.cl_ref, c, (uint32_t)ref);
    ref += (uint32_t)CL_NRD(cl) + CL_NWR(cl);
  }
  for (uint32_t r = 0; r < in->n_resources; r++) {
    S32_SET(x->L.res_vstamp, r, 0u);
    S32_SET(x->L.res_bstamp, r, 0u);
    S32_SET(x->L.res_ver, r, 0u);
  }
  for (uint32_t i = 0; i < x->L.cap_c; i++) {
    S32_SET(x->L.id_tab, i, 0u);
    S32_SET(x->L.id_stamp, i, 0u);
  }

  /* One pass: the offer of each column (realize.fused_offer) and its relaxation (optimize). */
  kp_u128 dist[2][BCIR_KP_WIDTHS_MAX];
  uint32_t width[2][BCIR_KP_WIDTHS_MAX];
  unsigned cur = 0, prev_rows = 0;
  int narrow = -1, wide = -1; /* the previous column's cheapest narrow / wide row */
  uint32_t pos = 0, node = 0, prev_claim = 0;
  for (uint32_t t = 0; t < in->n_phases; t++) {
    uint32_t p = S32(x->L.topo, t), stamp = t + 1u;
    const uint8_t *ph = in->data + in->off_phases + 12u * (size_t)p;
    uint32_t c0 = S32(x->L.ph_claim, p), nc = kp_rd32(ph + 8);
    for (uint32_t c = c0; c < c0 + nc; c++, pos++) {
      const uint8_t *cl = kp_claim(x, c);
      uint32_t ref0 = S32(x->L.cl_ref, c), n_rd = CL_NRD(cl), n_wr = CL_NWR(cl);
      S32_SET(x->L.pos_claim, pos, c);
      S32_SET(x->L.pos_phase, pos, p);
      /* the discount: CSE by value-numbered identity, else deforestation */
      int eligible = kp_cse_eligible(x, cl, ref0), found = 0;
      uint32_t slot = 0;
      if (eligible) {
        for (uint32_t k = 0; k < n_rd; k++) {
          uint32_t r = kp_ref(x, (uint64_t)ref0 + k);
          S32_SET(x->L.rdver, (uint64_t)ref0 + k,
                  S32(x->L.res_vstamp, r) == stamp ? S32(x->L.res_ver, r) : 0u);
        }
        uint32_t mask = x->L.cap_c - 1u;
        slot = (uint32_t)kp_identity_hash(x, cl, ref0) & mask;
        for (;;) {
          uint32_t e = S32(x->L.id_tab, slot);
          if (!e || S32(x->L.id_stamp, slot) != stamp) break;
          if (kp_same_identity(x, e - 1u, c)) {
            found = 1;
            break;
          }
          slot = (slot + 1u) & mask;
        }
      }
      unsigned mode = MODE_PLAIN;
      if (eligible && found) {
        mode = MODE_CSE;
      } else {
        int shared = 0, fenced = 0;
        for (uint32_t k = 0; k < n_rd; k++) {
          uint32_t r = kp_ref(x, (uint64_t)ref0 + k);
          if (S32(x->L.res_vstamp, r) == stamp) shared = 1;
          if (S32(x->L.res_bstamp, r) == stamp) fenced = 1;
        }
        if (shared && CL_HAZARD(cl) != HAZ_BARRIERED && !fenced) mode = MODE_DEFOREST;
      }
      if (eligible && !found) {
        S32_SET(x->L.id_tab, slot, c + 1u);
        S32_SET(x->L.id_stamp, slot, stamp);
      }
      for (uint32_t k = 0; k < n_wr; k++) {
        uint32_t r = kp_ref(x, (uint64_t)ref0 + n_rd + k);
        if (S32(x->L.res_vstamp, r) == stamp) {
          S32_SET(x->L.res_ver, r, S32(x->L.res_ver, r) + 1u);
        } else {
          S32_SET(x->L.res_vstamp, r, stamp);
          S32_SET(x->L.res_ver, r, 1u);
        }
        if (CL_HAZARD(cl) == HAZ_BARRIERED) S32_SET(x->L.res_bstamp, r, stamp);
      }
      x->s[x->L.pos_mode + pos] = (uint8_t)mode;

      /* the column's rows and their relaxation */
      uint32_t bw, lat;
      int ham;
      kp_factors(x, cl, ref0, &bw, &lat, &ham);
      kp_spec specs[BCIR_KP_WIDTHS_MAX];
      unsigned rows = kp_specs(x, cl, ham, specs);
      int shares = pos > 0 && kp_shares_reads(x, prev_claim, c);
      unsigned nxt = cur ^ 1u;
      S32_SET(x->L.node_start, pos, node);
      for (unsigned i = 0; i < rows; i++) {
        kp_u128 base[D_N], plain, fused;
        kp_base(x, cl, &specs[i], bw, lat, mode, base);
        kp_edge(x, base, x->hot && specs[i].width >= 16u, &plain, &fused);
        width[nxt][i] = specs[i].width;
        uint8_t pred = 0xFF;
        kp_u128 best = plain;
        if (pos > 0) {
          kp_u128 via_wide = specs[i].width > 1u && shares ? fused : plain;
          if (narrow >= 0) {
            best = kp_add(dist[cur][narrow], plain);
            pred = (uint8_t)narrow;
            if (wide >= 0) {
              kp_u128 d = kp_add(dist[cur][wide], via_wide);
              if (kp_lt(d, best) || (kp_eq(d, best) && wide < narrow)) {
                best = d;
                pred = (uint8_t)wide;
              }
            }
          } else {
            best = kp_add(dist[cur][wide], via_wide);
            pred = (uint8_t)wide;
          }
        }
        dist[nxt][i] = best;
        x->s[x->L.pred + node + i] = pred;
      }
      narrow = wide = -1;
      for (unsigned i = 0; i < rows; i++) {
        if (width[nxt][i] > 1u) {
          if (wide < 0 || kp_lt(dist[nxt][i], dist[nxt][wide])) wide = (int)i;
        } else if (narrow < 0 || kp_lt(dist[nxt][i], dist[nxt][narrow])) {
          narrow = (int)i;
        }
      }
      cur = nxt;
      prev_rows = rows;
      node += rows;
      prev_claim = c;
    }
  }
  S32_SET(x->L.node_start, pos, node);

  /* SINK: the first last-column row with the least distance; then the path, backwards. */
  kp_u128 score = kp_u(0);
  if (in->n_claims) {
    unsigned b = 0;
    for (unsigned i = 1; i < prev_rows; i++)
      if (kp_lt(dist[cur][i], dist[cur][b])) b = i;
    score = dist[cur][b];
    for (uint32_t j = in->n_claims; j-- > 0;) {
      x->s[x->L.chosen + j] = (uint8_t)b;
      b = x->s[x->L.pred + S32(x->L.node_start, j) + b];
    }
  }
  if (!kp_fits63(score)) {
    kp_zero(out, cap);
    return BCIR_ERR_OVERFLOW;
  }

  /* The realization: each chosen row re-derived, its cost the edge the path took. */
  kp_zero(out, need);
  out[0] = 'B';
  out[1] = 'K';
  out[2] = 'P';
  out[3] = 'R';
  kp_wr32(out + 8, in->n_claims);
  kp_wr64(out + 16, score.lo);
  uint32_t prev_width = 0;
  for (uint32_t j = 0; j < in->n_claims; j++) {
    uint32_t c = S32(x->L.pos_claim, j);
    const uint8_t *cl = kp_claim(x, c);
    uint32_t ref0 = S32(x->L.cl_ref, c), bw, lat;
    int ham;
    kp_factors(x, cl, ref0, &bw, &lat, &ham);
    kp_spec specs[BCIR_KP_WIDTHS_MAX];
    (void)kp_specs(x, cl, ham, specs);
    const kp_spec *sp = &specs[x->s[x->L.chosen + j]];
    kp_u128 base[D_N], plain, fused;
    kp_base(x, cl, sp, bw, lat, x->s[x->L.pos_mode + j], base);
    kp_edge(x, base, x->hot && sp->width >= 16u, &plain, &fused);
    int shares = j > 0 && kp_shares_reads(x, S32(x->L.pos_claim, j - 1u), c);
    kp_u128 cost = j > 0 && prev_width > 1u && sp->width > 1u && shares ? fused : plain;
    uint8_t *step = out + BCIR_KP_REALIZATION_HEADER_SIZE + (size_t)BCIR_KP_STEP_SIZE * j;
    if (!kp_fits63(cost)) {
      kp_zero(out, cap);
      return BCIR_ERR_OVERFLOW;
    }
    for (unsigned i = 0; i < D_N; i++) {
      if (!kp_fits63(base[i])) {
        kp_zero(out, cap);
        return BCIR_ERR_OVERFLOW;
      }
      kp_wr64(step + 24 + 8u * i, base[i].lo);
    }
    kp_wr32(step, CL_ID(cl));
    kp_wr32(step + 4, kp_rd32(in->data + in->off_phases + 12u * (size_t)S32(x->L.pos_phase, j)));
    kp_wr32(step + 8, sp->width);
    step[12] = sp->lane;
    step[13] = sp->name;
    kp_wr64(step + 16, cost.lo);
    prev_width = sp->width;
  }
  size_t body = need - BCIR_KP_CRC_SIZE;
  kp_wr32(out + body, bcir_crc32(out, body));
  *out_len = need;
  return BCIR_OK;
}

/* ---- BKPR: the decoder -------------------------------------------------------------------- */

bcir_status bcir_kp_decode_realization(const uint8_t *data, size_t len,
                                       bcir_kp_realization *out) {
  if (out) kp_zero((uint8_t *)out, sizeof *out);
  if (!out) return BCIR_ERR_NOSPACE;
  if (!data || len < (size_t)BCIR_KP_REALIZATION_HEADER_SIZE + BCIR_KP_CRC_SIZE)
    return BCIR_ERR_TRUNCATED;
  if (data[0] != 'B' || data[1] != 'K' || data[2] != 'P' || data[3] != 'R') return BCIR_ERR_MAGIC;
  if (kp_rd16(data + 4) != BCIR_KP_VERSION) return BCIR_ERR_VERSION;
  if (kp_rd32(data + len - 4) != bcir_crc32(data, len - 4)) return BCIR_ERR_CRC;
  if (kp_rd16(data + 6) != 0) return BCIR_ERR_RESERVED;
  uint32_t n = kp_rd32(data + 8);
  if (kp_rd32(data + 12) != 0 || kp_rd64(data + 24) != 0) return BCIR_ERR_RESERVED;
  uint64_t size =
      (uint64_t)BCIR_KP_REALIZATION_HEADER_SIZE + (uint64_t)BCIR_KP_STEP_SIZE * n + BCIR_KP_CRC_SIZE;
  if ((uint64_t)len < size) return BCIR_ERR_TRUNCATED;
  if ((uint64_t)len > size) return BCIR_ERR_TRAILING;
  uint64_t score = kp_rd64(data + 16);
  if (score > (uint64_t)INT64_MAX) return BCIR_ERR_OVERFLOW;
  kp_u128 total = kp_u(0);
  for (uint32_t i = 0; i < n; i++) {
    const uint8_t *s = data + BCIR_KP_REALIZATION_HEADER_SIZE + (size_t)BCIR_KP_STEP_SIZE * i;
    if (kp_rd16(s + 14) != 0) return BCIR_ERR_RESERVED;
    for (unsigned k = 0; k < 13; k++)
      if (kp_rd64(s + 16 + 8u * k) > (uint64_t)INT64_MAX) return BCIR_ERR_OVERFLOW;
    uint32_t width = kp_rd32(s + 8);
    uint8_t lane = s[12], name = s[13];
    if (lane > LANE_MAX || name >= BCIR_KP_NAME_COUNT || width == 0) return BCIR_ERR_PLANNER;
    if (name == BCIR_KP_NAME_VEC ? width < 2u
                                 : (name != BCIR_KP_NAME_UX_BUCKET && name != BCIR_KP_NAME_TILE &&
                                    width != 1u))
      return BCIR_ERR_PLANNER;
    total = kp_add(total, kp_u(kp_rd64(s + 16)));
  }
  if (!kp_eq(total, kp_u(score))) return BCIR_ERR_PLANNER;
  out->data = data;
  out->len = len;
  out->n_steps = n;
  out->score = score;
  return BCIR_OK;
}
