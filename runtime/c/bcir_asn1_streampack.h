/*===- bcir_asn1_streampack.h - DER -> native StreamPack fast path --------===
 *
 * Reconstruct a native StreamPack artifact directly from its X.690 DER projection,
 * with no Python and no allocation.
 *
 * WHY THIS EXISTS. The ASN.1 projection is additive (law A3, docs/BCIR_ASN1_X690_ABI.md):
 * the native octets are the frozen truth and the DER is a second transfer syntax for the
 * same abstract value. Until now the DER -> native direction existed only on the Python
 * rail (`bcir.asn1.streampack.value_to_pack` + `bcir.abi.encode`), so a driver that
 * received a DER projection from a peer could not turn it back into an executable pack
 * without a Python interpreter. This is that path in freestanding C.
 *
 * THE LAW IT UPHOLDS. For every pack P:
 *
 *     bcir_asn1_to_streampack(encode_pack(P))  ==  encode(P)      byte for byte
 *
 * That is A3 proven on the C rail. Byte identity, not equivalence: the reconstruction
 * has to pick the same StreamPack VERSION the native encoder would (v1/v2/v3 is derived
 * from content, not carried in the projection -- the module's own `version` field is the
 * PROJECTION version and is deliberately independent), emit `stride_k` as the reserved
 * zero the projection does not carry, and re-derive the CRC.
 *
 * TRUST BOUNDARY. `der` is untrusted input. Every read is bounds-checked through
 * bcir_asn1.c's non-recursive walker, every write is bounded by `out_cap`, and a
 * malformed projection returns a status rather than producing a partial artifact. The
 * output is additionally required to pass `bcir_sp_verify_semantic` before it is
 * blessed, so a well-formed-but-nonsense projection cannot mint an unexecutable pack.
 *
 * Rec. ITU-T X.690 (02/2021) over the BCIR-StreamPack module in
 * bcir/asn1/BCIR-StreamPack.asn1.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_ASN1_STREAMPACK_H
#define BCIR_ASN1_STREAMPACK_H

#include "bcir_asn1.h"
#include "bcir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Rebuild native StreamPack octets from a DER projection.
 *
 * `out_len` receives the artifact length on success and 0 on every failure. Returns
 * BCIR_OK, or:
 *   BCIR_ERR_TRUNCATED  the projection is malformed, or a component is missing
 *   BCIR_ERR_NOSPACE    `out_cap` is too small, or a count exceeds the native format
 *   BCIR_ERR_LANE / _WIDTH / _DISPATCH / _PROVENANCE / _TRAILING / _UTF8
 *                       the reconstruction is well-formed but not a valid pack
 *                       (the semantic gate's verdict, unchanged)
 *
 * A safe upper bound for `out_cap` is `bcir_asn1_streampack_bound(der_len)`. */
BCIR_NODISCARD bcir_status bcir_asn1_to_streampack(
    const uint8_t *BCIR_RESTRICT der, size_t der_len,
    uint8_t *BCIR_RESTRICT out, size_t out_cap, size_t *BCIR_RESTRICT out_len);

/* An output capacity that is always sufficient for a projection of `der_len` octets.
 *
 * The native encoding is not uniformly smaller than the DER: DER omits every component
 * equal to its DEFAULT (X.690 11.5) while the native format writes every field
 * unconditionally, so an omitted `channel` costs 0 DER octets and 2 + 4 native ones.
 * The bound accounts for that expansion rather than assuming the projection is larger. */
size_t bcir_asn1_streampack_bound(size_t der_len);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_ASN1_STREAMPACK_H */
