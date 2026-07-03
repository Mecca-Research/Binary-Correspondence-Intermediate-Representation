/*===- bcir_decode.h - the rung-5 LLM decode stage kernels (C twins) ----------------------===
 *
 * Open-weight ladder rung 5, second half (ML/AI roadmap SS7.4): the rung-3 reference
 * decoder's LLM-specific stages as C twins, kernel-for-kernel with the oracle references
 * (kbcir/transformer_grads.py rmsnorm_reference / rope_reference; unsupervised.py
 * embedding_lookup) -- the same accumulation order, so on a shared libm the two rails agree
 * to the last bit (the differential gate allows only float round-off for cross-libm CI).
 * Host-tool posture (libm for sqrt/cos/sin), like bcir_train.c; caller-owned memory.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_DECODE_H
#define BCIR_DECODE_H

#ifdef __cplusplus
extern "C" {
#endif

/* RMSNorm (the Gemma/Llama-family per-row normalizer): out = x / sqrt(mean(x^2) + eps) * gamma,
 * per row of `dim`; gamma has dim entries; eps mirrors the oracle default (1e-6). */
void bcir_rmsnorm(const double *x, int rows, int dim, const double *gamma, double eps,
                  double *out);

/* Rotary position embedding: row r (position pos_offset + r) has each channel pair (2k, 2k+1)
 * rotated by theta_k = pos * base^(-2k/dim). `dim` must be EVEN (the caller's law). */
void bcir_rope(const double *x, int rows, int dim, double base, int pos_offset, double *out);

/* Embedding lookup: out[t*dim + d] = table[ids[t]*dim + d] -- a pure row gather, EXACT.
 * Returns 0, or -1 on an out-of-range id (the oracle raises; the twin refuses too). */
int bcir_embedding(const double *table, int vocab, int dim, const int *ids, int n_ids,
                   double *out);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_DECODE_H */
