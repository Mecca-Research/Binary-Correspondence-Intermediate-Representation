/*===- bcir_devman.h - device manifests bound at instantiation -------------------------------------------===
 *
 * N11 of the native port audit (docs/research/BCIR_NATIVE_PORT_AUDIT_2026-11-09.md): the C twin of
 * bcir/kbcir/device_manifest.py's `load_device_manifest`, `check_device_manifest` and `probe_agree`, so an
 * instance binds its device's static hardware schema (D-R1) when it starts, from the persisted envelope the
 * planner was given, rather than from a header generated per device.
 *
 * ONE JSON READER. The envelope is read by the bounded JER reader (bcir_jer.h) on this rail and by its Python
 * twin (`jer_bounded.scan`, then `json.loads`) on the oracle's, under one limits profile (bcir_devman_limits here,
 * `MANIFEST_LIMITS` there): the scan bounds depth, nodes, members, elements, string and number lengths -- an
 * integer of at most 18 digits, so every one the scan admits is an int64 on both rails -- and work, then UTF-8 (an
 * escaped lone surrogate refused), then ECMA-404's grammar, with no member name repeated in one object and no
 * NaN. A refusal there is BCIR_DEVMAN_ERR_READ, the reader's own diagnostic in *diag.
 *
 * THE LAWS, in the oracle's order: 16 MiB at most; the reader; the root an object; its `kind`
 * "bcir.device_manifest"; its `schema` an integer, not newer than 1, and 1; exactly the eight fields; the target
 * the caller expects (when it names one); the calibration generation it expects (when it names one, against an
 * integer `cal_gen`); the fields' types; each bank five fields, typed, its domain a `Domain`; each distance row
 * integers; the digest 64 lowercase hex digits, and the SHA-256 the oracle's `DeviceManifest.digest` takes --
 * over `json.dumps(..., sort_keys=True)` of the manifest, which this file writes octet for octet, ASCII-escaped
 * as Python escapes it -- re-derived; then `check_device_manifest`'s consistency laws, refusals as bits, one per
 * kind of message the oracle writes. A JSON number with a fraction or an exponent is the oracle's float, never an
 * integer.
 *
 * Then the binding: the banks as the table bcir_dma_admit and bcir_kv_init take (a tile or an alignment past 32
 * bits refused, which no bcir_dma_bank can hold), and `probe_agree`'s veto -- what a runtime probe observed
 * either equals the manifest or the instance refuses to run; discovery never steers.
 *
 * Freestanding: <stddef.h> and <stdint.h> (and the reader and SHA-256), no heap, no libc, no recursion. Every
 * buffer is the caller's: one workspace holds the reader's stack and scratch, the document's nodes and the loaded
 * manifest, which points into it and into `data`. bcir_devman_workspace_size(len) is enough for any envelope of
 * `len` octets; the load asks only for what the document the scan admitted needs, after the scan, so a short
 * workspace never hides the reader's own refusal.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_DEVMAN_H
#define BCIR_DEVMAN_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_dma.h"
#include "bcir_jer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_DEVMAN_MAX_BYTES (16u << 20) /* device_manifest._MAX_MANIFEST_BYTES */
#define BCIR_DEVMAN_SCHEMA 1u
#define BCIR_DEVMAN_DOMAINS 6u            /* bcir.model.Domain: RAM VRAM NVM MMIO CXL HBM */

enum {
  BCIR_DEVMAN_OK = 0,
  BCIR_DEVMAN_ERR_ARG = 1,          /* a NULL (C only) */
  BCIR_DEVMAN_ERR_SIZE = 2,         /* more than 16 MiB */
  BCIR_DEVMAN_ERR_READ = 3,         /* the bounded reader refused the document (*diag) */
  BCIR_DEVMAN_ERR_ROOT = 4,         /* the root is not an object */
  BCIR_DEVMAN_ERR_KIND = 5,         /* not a bcir.device_manifest document */
  BCIR_DEVMAN_ERR_SCHEMA_TYPE = 6,  /* `schema` not an integer */
  BCIR_DEVMAN_ERR_SCHEMA_NEWER = 7, /* a schema newer than this build's */
  BCIR_DEVMAN_ERR_SCHEMA = 8,       /* an unsupported older schema */
  BCIR_DEVMAN_ERR_FIELDS = 9,       /* missing or unknown fields */
  BCIR_DEVMAN_ERR_TARGET = 10,      /* authored for another target: retrain */
  BCIR_DEVMAN_ERR_STALE = 11,       /* authored under another calibration generation: retrain */
  BCIR_DEVMAN_ERR_TYPES = 12,       /* a field of the wrong type */
  BCIR_DEVMAN_ERR_BANK_SHAPE = 13,  /* a bank that is not five fields (*where: the bank) */
  BCIR_DEVMAN_ERR_BANK_TYPES = 14,  /* a bank field of the wrong type (*where: the bank) */
  BCIR_DEVMAN_ERR_DOMAIN = 15,      /* an unknown domain (*where: the bank) */
  BCIR_DEVMAN_ERR_DISTANCE = 16,    /* a distance row that is not integers (*where: the row) */
  BCIR_DEVMAN_ERR_DIGEST_FORM = 17, /* the digest is not 64 lowercase hex digits */
  BCIR_DEVMAN_ERR_TAMPERED = 18,    /* the digest does not re-derive */
  BCIR_DEVMAN_ERR_INVALID = 19,     /* check_device_manifest refused it (*bits) */
  BCIR_DEVMAN_ERR_SPACE = 20,       /* the workspace or the bank table is short (C only) */
  BCIR_DEVMAN_ERR_WIDTH = 21        /* a tile or alignment past 32 bits, which a bank table cannot hold (C only) */
};

/* check_device_manifest's refusals: "no banks" and the matrix's shape end the walk, the rest accumulate */
#define BCIR_DEVMAN_BAD_DEVICE 0x001u     /* the device name empty */
#define BCIR_DEVMAN_BAD_NO_BANKS 0x002u   /* no bank */
#define BCIR_DEVMAN_BAD_NAME 0x004u       /* a bank name empty */
#define BCIR_DEVMAN_BAD_DUPLICATE 0x008u  /* two banks of one name */
#define BCIR_DEVMAN_BAD_CAPACITY 0x010u   /* a capacity below 1 */
#define BCIR_DEVMAN_BAD_TILE 0x020u       /* a native tile below 1 */
#define BCIR_DEVMAN_BAD_ALIGN 0x040u      /* an alignment below 1 */
#define BCIR_DEVMAN_BAD_ALIGN_POW2 0x080u /* an alignment not a power of two */
#define BCIR_DEVMAN_BAD_SHAPE 0x100u      /* the distance matrix not n x n */
#define BCIR_DEVMAN_BAD_DIAGONAL 0x200u   /* a nonzero diagonal */
#define BCIR_DEVMAN_BAD_DISTANCE 0x400u   /* an off-diagonal distance below 1 */
#define BCIR_DEVMAN_BAD_ASYMMETRIC 0x800u /* distance[i][j] != distance[j][i] */

/* probe_agree's vetoes */
#define BCIR_DEVMAN_PROBE_UNKNOWN 0x1u  /* a bank the manifest does not name */
#define BCIR_DEVMAN_PROBE_CAPACITY 0x2u /* a capacity other than the manifest's */
#define BCIR_DEVMAN_PROBE_TILE 0x4u     /* a native tile other than the manifest's */

typedef struct bcir_devman_bank {
  const uint8_t *name; /* UTF-8, in the workspace */
  size_t name_len;
  uint32_t domain;
  int64_t capacity, native_tile, align;
} bcir_devman_bank;

typedef struct bcir_devman {
  const uint8_t *device, *target; /* UTF-8, in the workspace */
  size_t device_len, target_len;
  int64_t cal_gen;
  uint32_t n_banks;
  const bcir_devman_bank *banks;
  uint32_t n_rows;          /* the distance matrix's rows, each its own length until checked square */
  const uint32_t *row_off;  /* n_rows + 1 offsets into `distance` */
  const int64_t *distance;
  uint8_t digest[32];       /* the re-derived SHA-256 */
} bcir_devman;

typedef struct bcir_devman_probe {
  const uint8_t *name;
  size_t name_len;
  int has_capacity, has_tile;
  uint64_t capacity, native_tile;
} bcir_devman_probe;

/* The reader's limits for an envelope (`MANIFEST_LIMITS` on the oracle's rail). */
void bcir_devman_limits(bcir_jer_limits *out);

/* The workspace bcir_devman_load needs for an envelope of `len` octets (0 past what a size_t holds). */
size_t bcir_devman_workspace_size(size_t len);

/* Load and check the envelope data[0 .. len). `expect_target` (NULL or empty: any) and `expect_cal_gen` (when
 * `has_expect_cal_gen`) are the caller's live target and calibration generation. On BCIR_DEVMAN_OK *out is the
 * manifest, valid while `work` and `data` are; on BCIR_DEVMAN_ERR_TAMPERED only out->digest is written, the digest
 * the envelope's contents re-derive; on any other refusal *out is untouched. *where (may be NULL) is the bank or row a per-bank or per-row
 * refusal names; *bits (may be NULL) the consistency refusals; *diag (may be NULL) the reader's diagnostic. */
int bcir_devman_load(const uint8_t *data, size_t len, const uint8_t *expect_target, size_t expect_target_len,
                     int has_expect_cal_gen, uint64_t expect_cal_gen, void *work, size_t work_len, bcir_devman *out,
                     uint32_t *where, unsigned *bits, bcir_jer_diag *diag);

/* The bank a name names, or -1. */
long bcir_devman_bank_index(const bcir_devman *man, const uint8_t *name, size_t name_len);

/* Bind the banks as the table bcir_dma_admit takes: out[0 .. n_banks), in the manifest's order. */
int bcir_devman_dma_banks(const bcir_devman *man, bcir_dma_bank *out, size_t cap);

/* probe_agree: the vetoes as bits, and how many messages the oracle writes in *messages (may be NULL). The
 * observations name distinct banks, as the oracle's dict keys do. */
unsigned bcir_devman_probe_agree(const bcir_devman *man, const bcir_devman_probe *observed, size_t n,
                                 size_t *messages);

#ifdef __cplusplus
}
#endif

#endif
