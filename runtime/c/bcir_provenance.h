/*===- bcir_provenance.h - BCIR provenance-digest twin (attestation) -------===
 *
 * The freestanding C twin of the DIGEST half of bcir/kbcir/provenance.py: the FNV-1a
 * item chain (`_fnv`) and the manifest digest (`_digest`) a plan's commit hash is made
 * of, byte-identical to the oracle -- so a driver / bare-metal host can VERIFY a recorded
 * ProvenanceManifest with NO Python on the attestation path (the R13 story off-host).
 *
 * Scope (deliberate): the twin CHECKS a digest from its component hashes + artifact
 * generations -- it does not recompute m_module from a claim graph (on the law rail the
 * MLIR -bcir-verify pass already recomputes hashModuleFromIR byte-identically; the C rail
 * receives component hashes as attested inputs and pins the CHAINING). The learned-organ
 * FIT halves stay Python per the two-truth policy; this is pure integer hashing.
 *
 * The oracle chain, mirrored exactly (provenance.py::_fnv):
 *   h = FNV_OFFSET
 *   per item: for each byte of str(item) utf-8: h = (h ^ byte) * FNV_PRIME (mod 2^64)
 *             then h = (h ^ 0xFF) * FNV_PRIME (the field separator)
 *   result   = h & ((1<<63)-1)   (i63 mask, MLIR signed-integer parity)
 * str(int) is the decimal rendering (with '-' for negatives), str(str) the text itself.
 * A digest = fnv(m_module, m_target, m_theta, m_policy, name1, gen1, name2, gen2, ...)
 * with the artifact (name, generation) pairs in the oracle's normalized (sorted) order.
 *
 * Depends only on <stddef.h> + <stdint.h> (freestanding-clean, C11 + C23). Differential
 * test: bcir/tests/test_provenance_twin.py (toolchain-gated, byte-identical vs oracle).
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_PROVENANCE_H
#define BCIR_PROVENANCE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_PROV_FNV_OFFSET 14695981039346656037ULL
#define BCIR_PROV_FNV_PRIME  1099511628211ULL
#define BCIR_PROV_I63_MASK   0x7FFFFFFFFFFFFFFFULL

/* One in-force artifact tag: a (name, generation/fingerprint) pair, in the manifest's
 * normalized order (sorted by name, then value -- the caller supplies them sorted, as
 * provenance._norm_artifacts records them). */
typedef struct bcir_prov_artifact {
    const char *name;
    int64_t     value;
} bcir_prov_artifact;

/* The running FNV chain (an unmasked 64-bit accumulator; mask only at the end). */
uint64_t bcir_prov_begin(void);
uint64_t bcir_prov_item_str(uint64_t h, const char *s);   /* one string item + separator */
uint64_t bcir_prov_item_i64(uint64_t h, int64_t v);       /* one integer item + separator */
uint64_t bcir_prov_end(uint64_t h);                       /* i63 mask (the oracle result) */

/* The manifest digest: chain the four component hashes then the artifact pairs
 * (name, value interleaved), exactly provenance._digest. */
uint64_t bcir_prov_digest(int64_t m_module, int64_t m_target, int64_t m_theta,
                          int64_t m_policy, const bcir_prov_artifact *arts, size_t n);

/* The attestation check: 1 iff the recomputed digest equals the recorded one. */
int bcir_prov_verify(uint64_t recorded, int64_t m_module, int64_t m_target,
                     int64_t m_theta, int64_t m_policy,
                     const bcir_prov_artifact *arts, size_t n);

#ifdef __cplusplus
}
#endif
#endif /* BCIR_PROVENANCE_H */
