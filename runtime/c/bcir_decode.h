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
 * Returns 0, or -1 on invalid geometry/pointers or an out-of-range id. Every id is
 * preflighted before row zero is written, so a late bad id leaves no partial output. */
int bcir_embedding(const double *table, int vocab, int dim, const int *ids, int n_ids,
                   double *out);

/* Causal GROUPED-QUERY attention over PRE-ROPED projections (rung 5's GQA primitive; the
 * oracle twin is frontends/models/decode.py::gqa_attention_reference). q is row-major
 * seq x (n_heads*d_k); k/v are seq x (n_kv_heads*d_k); query head h reads kv head
 * h / (n_heads/n_kv_heads). Per head: (Q_h @ K_g^T)/sqrt(d_k) + causal mask -> stable
 * softmax -> @ V_g, ascending-index accumulation everywhere (the oracle order).
 * n_kv_heads == n_heads is exactly multi-head attention. `scores` is caller-owned scratch
 * (seq doubles -- one score row at a time). out: seq x (n_heads*d_k). Returns 0, or -1 when
 * dimensions/pointers are invalid, n_kv_heads is out of [1, n_heads], or it does not divide
 * n_heads (the oracle raises; the twin refuses). */
int bcir_gqa_attention(const double *q, const double *k, const double *v, int seq,
                       int n_heads, int n_kv_heads, int d_k, double *scores, double *out);

/* One KV-CACHED decode step: the attention output row of the LATEST position, given the
 * t_len cached token rows (k_rows/v_rows: t_len x (n_kv_heads*d_k), token-major -- exactly
 * the rows a cache appends, K stored POST-RoPE; the caller appends the new row BEFORE the
 * call, so row t_len-1 is the current token). q_row is the roped n_heads*d_k query row.
 * Bit-for-bit the i == t_len-1 row of bcir_gqa_attention on the same buffers -- the
 * naive-vs-cached invariant at kernel level (the rung-3 twin gate, pinned in
 * test_decode_c_kernels.py). Same refusal contract; t_len must be positive. `scores`:
 * t_len scratch doubles. */
int bcir_gqa_attention_row(const double *q_row, const double *k_rows, const double *v_rows,
                           int t_len, int n_heads, int n_kv_heads, int d_k,
                           double *scores, double *out);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_DECODE_H */
