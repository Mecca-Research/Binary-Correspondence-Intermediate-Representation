/*===- bcir_provenance.c - BCIR provenance-digest twin (attestation) -------===
 * The freestanding FNV-1a manifest-digest chain, byte-identical to
 * bcir/kbcir/provenance.py::_fnv/_digest. See bcir_provenance.h for the contract.
 *===----------------------------------------------------------------------===*/
#include "bcir_provenance.h"

uint64_t bcir_prov_begin(void) { return BCIR_PROV_FNV_OFFSET; }

static uint64_t prov_bytes(uint64_t h, const char *s) {
    while (*s) {
        h = (h ^ (uint64_t)(unsigned char)*s++) * BCIR_PROV_FNV_PRIME;
    }
    return h;
}

static uint64_t prov_sep(uint64_t h) {
    return (h ^ 0xFFULL) * BCIR_PROV_FNV_PRIME;   /* the oracle's per-item field separator */
}

uint64_t bcir_prov_item_str(uint64_t h, const char *s) {
    if (!s) return 0;   /* invalid-input sentinel; never hash NULL as empty */
    return prov_sep(prov_bytes(h, s));
}

uint64_t bcir_prov_item_i64(uint64_t h, int64_t v) {
    /* str(int): the decimal rendering, '-' for negatives. 21 chars covers INT64_MIN. */
    char buf[21];
    int  i = 20;
    buf[20] = '\0';
    uint64_t u = (v < 0) ? (uint64_t)(-(v + 1)) + 1ULL : (uint64_t)v;   /* no UB on INT64_MIN */
    if (u == 0) {
        buf[--i] = '0';
    }
    while (u) {
        buf[--i] = (char)('0' + (u % 10ULL));
        u /= 10ULL;
    }
    if (v < 0) {
        buf[--i] = '-';
    }
    return prov_sep(prov_bytes(h, &buf[i]));
}

uint64_t bcir_prov_end(uint64_t h) { return h & BCIR_PROV_I63_MASK; }

uint64_t bcir_prov_digest(int64_t m_module, int64_t m_target, int64_t m_theta,
                          int64_t m_policy, const bcir_prov_artifact *arts, size_t n) {
    if (n && !arts) return 0;
    uint64_t h = bcir_prov_begin();
    h = bcir_prov_item_i64(h, m_module);
    h = bcir_prov_item_i64(h, m_target);
    h = bcir_prov_item_i64(h, m_theta);
    h = bcir_prov_item_i64(h, m_policy);
    for (size_t i = 0; i < n; ++i) {          /* pairs flatten in order: name, value, ... */
        if (!arts[i].name) return 0;
        h = bcir_prov_item_str(h, arts[i].name);
        h = bcir_prov_item_i64(h, arts[i].value);
    }
    return bcir_prov_end(h);
}

int bcir_prov_verify(uint64_t recorded, int64_t m_module, int64_t m_target,
                     int64_t m_theta, int64_t m_policy,
                     const bcir_prov_artifact *arts, size_t n) {
    if ((n && !arts)) return 0;
    for (size_t i = 0; i < n; ++i) if (!arts[i].name) return 0;
    return bcir_prov_digest(m_module, m_target, m_theta, m_policy, arts, n) == recorded;
}
