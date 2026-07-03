/*===- bcir_cir.h - BCIR claim-graph IR (the C twin of bcir/model) ---------===
 *
 * The in-memory BCIR intermediate representation the C frontend (bcir_cfront.c)
 * produces: resources, claims, and phases -- the exact model bcir/model/ defines
 * in the Python oracle (enum values match bcir/model/{opcodes,lanes,graph}.py and
 * bcir_streampack.h's bcir_lane). This is the production substrate for the
 * plug-in C compiler: C source -> this IR -> verify -> emit / lower to a
 * StreamPack the freestanding runtime (bcir_runtime.c / bcir_exec.c) executes.
 *
 * Freestanding-safe (only <stdint.h>/<stddef.h>): the IR links into a driver. The
 * *frontend* that builds it (bcir_cfront.c) is a host compiler tool and may use
 * libc; the IR it emits does not.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_CIR_H
#define BCIR_CIR_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_streampack.h"   /* bcir_lane (shared enum) */

#ifdef __cplusplus
extern "C" {
#endif

/* opcode / domain / stride-class enums -- values MUST match bcir/model. */
typedef enum bcir_opcode {
  BCIR_OP_NOP = 0, BCIR_OP_LOAD = 1, BCIR_OP_STORE = 2, BCIR_OP_ADD = 3, BCIR_OP_SUB = 4,
  BCIR_OP_MUL = 5, BCIR_OP_ATOMIC_ADD = 6, BCIR_OP_ATOMIC_SUB = 7, BCIR_OP_ATOMIC_XOR = 8,
  BCIR_OP_CMPXCHG = 9, BCIR_OP_BARRIER = 10, BCIR_OP_PHASE_ENTER = 11, BCIR_OP_PHASE_LEAVE = 12,
  BCIR_OP_GGG_LOAD = 13, BCIR_OP_GGG_STORE = 14, BCIR_OP_T_MACC = 15, BCIR_OP_GEM_DISPATCH = 16,
  BCIR_OP_PROV_NOTE = 17
} bcir_opcode;

typedef enum bcir_domain {
  BCIR_DOM_RAM = 0, BCIR_DOM_VRAM = 1, BCIR_DOM_NVM = 2, BCIR_DOM_MMIO = 3,
  BCIR_DOM_CXL = 4, BCIR_DOM_HBM = 5
} bcir_domain;

typedef enum bcir_stride {
  BCIR_STRIDE_SCALAR = 0, BCIR_STRIDE_UNIT = 1, BCIR_STRIDE_STRIDED = 2,
  BCIR_STRIDE_CACHELINE = 3, BCIR_STRIDE_TILE = 4, BCIR_STRIDE_RANDOM = 5
} bcir_stride;

/* hazard / bounds contracts (R5 / R7) -- string-free, matched to verify/__init__.py. */
typedef enum bcir_hazard { BCIR_HZ_UNIQUE = 0, BCIR_HZ_ATOMIC = 1, BCIR_HZ_BARRIERED = 2 } bcir_hazard;
typedef enum bcir_bounds { BCIR_BND_STRICT = 0, BCIR_BND_MASKED = 1, BCIR_BND_ASSUMED = 2 } bcir_bounds;

#define BCIR_CLAIM_MAX_RD 6
#define BCIR_CLAIM_MAX_WR 2
#define BCIR_CLAIM_MAX_IMM 4   /* off, size, flag, stride -- the array-of-structs `arr[i].field` store */
#define BCIR_CIR_NAME 32

/* shape kind of a resource (drives how the emitter takes its address). */
typedef enum bcir_rkind { BCIR_RK_SCALAR = 0, BCIR_RK_AGGREGATE = 1, BCIR_RK_POINTER = 2 } bcir_rkind;

typedef struct bcir_resource {
  uint32_t rid;
  bcir_domain domain;
  uint32_t elem_bytes;
  uint32_t count;            /* element count (product of shape) */
  uint8_t  is_volatile;      /* MMIO/volatile access */
  uint8_t  read_only;
  uint8_t  kind;             /* bcir_rkind */
  uint8_t  is_float;         /* a floating value (emit float/double, not uint32_t) -- complex sets this too */
  uint8_t  is_complex;       /* a C99 `_Complex` value (a pair of floats): emit `<elem> _Complex`, native
                              * operators; the element float width is elem_bytes/2 (so 16-byte = double) */
  uint8_t  is_signed;        /* a signed integer value (drives signed C emit + the usual arith conv) */
  uint8_t  is_bool;          /* a _Bool/bool object: emit `_Bool` so a store normalizes to 0/1 (§6.3.1.2) */
  uint8_t  is_plain_char;    /* a plain `char` (not signed/unsigned char): emit `char`, whose signedness
                              * is implementation-defined -- NOT int8_t (always signed -> wrong on ARM) */
  uint8_t  zinit;            /* an aggregate local declared with a `= {0}` zero baseline (§6.7.10) */
  uint8_t  ptr_depth;        /* pointer indirection depth (POINTER kind): 1 `T*`, 2 `T**`, ...; 0 == 1 */
  uint8_t  is_valist;        /* a `va_list` object (variadic): emit `va_list`, opaque to load/store */
  uint8_t  is_voidptr;       /* a `void *` pointer (a `&&L` label address / void-pointee local): emit `void *`,
                              * not a width-derived `uintN *` (the pointee has no width/sign to render) */
  uint8_t  is_funcptr;       /* a function-pointer value (a named function decayed, or a funcptr param):
                              * stored to a member by a DIRECT `memcpy(&fn, ...)`, not via a `uintN _v` temp
                              * (which would be an invalid pointer->integer conversion) */
  int      bit_width;        /* a C23 `_BitInt(N)` value's EXACT width N (0 == a normal type; >0 == `_BitInt(N)`).
                              * A distinct, NON-promoting integer type: the emit spells `_BitInt(N)` /
                              * `unsigned _BitInt(N)` (signed by is_signed) so Clang applies the N-bit semantics
                              * in both rails; it is kept OUT of the power-of-two width canonicalization. */
  uint8_t  is_vla;           /* a 1-D stack VLA `T a[n]`: NAMED but NOT declared up front (its runtime size
                              * isn't known until execution reaches the decl) -- the `c.vladecl` claim emits
                              * `<elem> a[<ext_var>];` IN-BODY; `a[i]` masks against ext_var via ptr_extent */
  uint32_t ext_var;          /* the snapshot extent rid (`__bcir_extK`) sizing a VLA (is_vla); else 0 */
  uint8_t  vla_ndims;        /* a MULTI-dim stack VLA `T a[d0][d1]...`: the dim count (2..3); 0 == not a
                              * multi-dim VLA. The array is a FLAT `T a[__ext_total];` (is_vla also set), but
                              * `a[i][j]` Horner-flattens with RUNTIME strides (no c.const) -- see vla_strides */
  uint32_t vla_strides[3];   /* the per-dim snapshot rids (`__bcir_extK`), the runtime Horner multipliers:
                              * `a[i][j]` -> i*vla_strides[1] + j (dim d's snapshot is the step-d multiplier) */
  char     name[BCIR_CIR_NAME];
  char     agg[BCIR_CIR_NAME]; /* struct tag (aggregate resources, for emission); else "" */
} bcir_resource;

/* a C type descriptor (for signatures + faithful emission). */
typedef struct bcir_ctype {
  uint8_t  kind;             /* 0 scalar, 1 struct-by-value, 2 pointer, 3 function-pointer */
  int      size;             /* scalar size, pointee size for a pointer, or 8 for a funcptr */
  int      signd;
  uint8_t  is_volatile;      /* volatile-qualified (MMIO) */
  uint8_t  is_atomic;        /* _Atomic-qualified (C11/C23 atomics) */
  uint8_t  ptr_to_struct;    /* a pointer whose pointee is a struct */
  uint8_t  ptr_depth;        /* pointer indirection depth: 1 `T*`, 2 `T**`, ... (kind 2); 0 == 1 */
  uint8_t  is_union;         /* the aggregate is a union (emit `union` not `struct`) */
  uint8_t  is_float;         /* a floating type (float/double) -- a _Complex type sets this too */
  uint8_t  is_complex;       /* a C99 `_Complex` type (size is 2x the element float; element-aligned) */
  uint8_t  is_bool;          /* a _Bool/bool type: emit `_Bool` so conversions normalize to 0/1 (§6.3.1.2) */
  uint8_t  is_plain_char;    /* a plain `char` (vs signed/unsigned char): emit `char` (impl-defined sign) */
  uint8_t  is_valist;        /* the `va_list` type (variadic argument cursor) -- emit `va_list` */
  int      bit_width;        /* a C23 `_BitInt(N)` type's exact width N (0 == a normal type; >0 == `_BitInt(N)`),
                              * carried on signatures so a param/return spells `_BitInt(N)` faithfully */
  char     tag[BCIR_CIR_NAME]; /* struct/union tag (kind 1/ptr_to_struct), or funcptr alias (kind 3) */
  int      fp_ret_size;      /* kind-3 funcptr: the captured RETURN type (sign/width/float), used to type a
                              * c.call.indirect / c.call.imember result temp; ZERO if the return wasn't captured */
  uint8_t  fp_ret_signd;
  uint8_t  fp_ret_float;
  int      adims[3];         /* decayed multi-dim array-param shape (outer-first), for m[i][j] */
  int      nadims;           /* number of array dims (0 == not an array parameter) */
} bcir_ctype;

typedef struct bcir_param { char name[BCIR_CIR_NAME]; uint32_t rid; bcir_ctype type; } bcir_param;

typedef struct bcir_claim {
  uint32_t id;
  bcir_opcode opcode;
  uint8_t  lane;             /* bcir_lane */
  bcir_stride stride;
  uint32_t count;
  uint32_t rd[BCIR_CLAIM_MAX_RD]; uint8_t n_rd;
  uint32_t wr[BCIR_CLAIM_MAX_WR]; uint8_t n_wr;
  int64_t  imm[BCIR_CLAIM_MAX_IMM]; uint8_t n_imm;
  bcir_domain domain;
  bcir_hazard hazard;
  bcir_bounds bounds;
  uint8_t  lifetime;          /* R21 pointer-lifetime event (§5.12): 0 none/use, 1 alloc, 2 free.
                               * OPTIONAL annotation -- excluded from the R13 provenance digest (it does
                               * not change the claim-graph identity); default 0 via new_claim's memset. */
  char     op[BCIR_CIR_NAME]; /* semantic label, e.g. "c.bin.add" / "c.load" / "c.bf.get" */
} bcir_claim;

/* A static-local variable (static storage duration: a once-only constant init). */
typedef struct bcir_static { char name[BCIR_CIR_NAME]; long long init; } bcir_static;

/* A function lowered to a single phase of claims over its resources. Every variable-length member
 * grows on demand (geometric realloc) -- the IR has no fixed BCIR_MAX_* ceilings, so a real
 * translation unit (any number of functions / params / calls / resources / claims) lowers. */
typedef struct bcir_func {
  char name[BCIR_CIR_NAME];
  bcir_resource *res; size_t n_res, cap_res;
  bcir_claim    *claims; size_t n_claims, cap_claims;
  bcir_param    *params; int n_params, cap_params;
  bcir_ctype  ret;
  uint8_t  variadic;          /* the function is variadic: a trailing `...` after its named params */
  uint8_t  reproducible;      /* a C23 `[[reproducible]]`/`[[unsequenced]]` hint is on the definition --
                               * a fusion-legality signal. The hint is value-neutral, so the emit drops it
                               * (it is never re-spelled); recorded only for the dual-rail repro= summary. */
  uint8_t  static_fn;         /* source `static` on the definition (internal linkage). The default emit
                               * is static regardless; --linkable keeps `static` only when this is set
                               * (source-static honoring, the oracle's LoweredFunc.static_fn twin). */
  uint32_t return_rid; uint8_t has_return;
  char (*calls)[BCIR_CIR_NAME]; int n_calls, cap_calls;   /* callee names (R18 call graph) */
  bcir_static *statics; int n_statics, cap_statics;       /* static locals */
} bcir_func;

/* A translation unit: a growable list of functions sharing struct definitions + a call graph. */
typedef struct bcir_unit {
  bcir_func *funcs; int n_funcs, cap_funcs;
} bcir_unit;

/* opcode name (for emission / diagnostics). */
const char *bcir_opcode_name(bcir_opcode op);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_CIR_H */
