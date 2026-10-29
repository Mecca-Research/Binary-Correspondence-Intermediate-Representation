"""NEXT-12: write the register-blocked prototype into a copy of runtime/c/bcir_ai_kernels.c and print the unified diff.

usage: make_block_patch.py REPO_KERNELS_C OUT_C  -> writes OUT_C (the patched unit); the caller diffs the two.

The prototype runs only where it is the committed kernel's twin: a whole call (columns [0, output_count)) on the
AVX-512 instance with group size 32 where Q8-BOUND proves every x * s exact and Q8-PAIR's conditions hold (output a
multiple of eight). Each block of 32 columns keeps four 8-column accumulators in registers down every input row and
adds each output's terms code * (x * s) in ascending-row order, as Q8-PAIR adds them: every output the same."""

import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
s = src.read_text(encoding="utf-8")

KERNEL = r"""/* NEXT-12's prototype (measured, not taken): the register-blocked AVX-512 matvec for a whole call of group size 32
 * where Q8-BOUND proves it -- each block of 32 columns' four 8-column accumulators held in registers down every input
 * row, each output's terms code * (x * s) added in ascending-row order as Q8-PAIR adds them, so every output is the
 * same. A vector of eight columns never straddles a group: k0 and output_count are multiples of eight, the group 32. */
__attribute__((target("avx512f"))) static void q8_matvec_block512_xs(const double *input, size_t input_count,
    const uint8_t *codes, const uint8_t *exponents_le, size_t output_count, double *output) {
  size_t k0, i;
  for (k0 = 0u; k0 < output_count; k0 += 32u) {
    size_t nv = (output_count - k0) / 8u < 4u ? (output_count - k0) / 8u : 4u;
    __m512d a0 = _mm512_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
    for (i = 0u; i < input_count; ++i) {
      size_t e = i * output_count + k0, g = e >> 5, b = ((g + 1u) << 5) - e;
      double xi = input[i];
      __m512d s0 = _mm512_set1_pd(xi * q8_scale(exponents_le, g));
      __m512d s1 = b < 8u * nv ? _mm512_set1_pd(xi * q8_scale(exponents_le, g + 1u)) : s0;
      const uint8_t *c = codes + e;
      a0 = _mm512_add_pd(a0, q8_terms512_xs(c, s0));
      if (nv > 1u) a1 = _mm512_add_pd(a1, q8_terms512_xs(c + 8, b > 8u ? s0 : s1));
      if (nv > 2u) a2 = _mm512_add_pd(a2, q8_terms512_xs(c + 16, b > 16u ? s0 : s1));
      if (nv > 3u) a3 = _mm512_add_pd(a3, q8_terms512_xs(c + 24, b > 24u ? s0 : s1));
    }
    _mm512_storeu_pd(output + k0, a0);
    if (nv > 1u) _mm512_storeu_pd(output + k0 + 8, a1);
    if (nv > 2u) _mm512_storeu_pd(output + k0 + 16, a2);
    if (nv > 3u) _mm512_storeu_pd(output + k0 + 24, a3);
  }
}
"""
anchor = "#undef Q8_PAIR_ROW\n"
assert s.count(anchor) == 1, "anchor: #undef Q8_PAIR_ROW"
s = s.replace(anchor, anchor + KERNEL)

old = (
    "    else if (instance == 4)\n"
    "      Q8_LANES_RUN(q8_matvec_pair512_xs, input, input_count, codes, exponents_le, output_count, group_size);\n"
)
new = (
    "    else if (instance == 4 && whole && group_size == 32u) {\n"
    "      Q8_COLUMNS_SAW(1);\n"
    "      q8_matvec_block512_xs(input, input_count, codes, exponents_le, output_count, output);\n"
    "    } else if (instance == 4)\n"
    "      Q8_LANES_RUN(q8_matvec_pair512_xs, input, input_count, codes, exponents_le, output_count, group_size);\n"
)
assert s.count(old) == 1, "anchor: the AVX-512 Q8-BOUND dispatch"
s = s.replace(old, new)
out.write_text(s, encoding="utf-8")
print("patched", out)
