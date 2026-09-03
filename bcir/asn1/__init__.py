"""ASN.1 / X.690 binary format compatibility for the BCIR ABI.

Implements Rec. ITU-T X.690 (02/2021) | ISO/IEC 8825-1:2021 — the Basic and
Distinguished Encoding Rules — over the tag assignments of Rec. ITU-T X.680 (02/2021).
Erratum 1 (09/2021) redraws Figure 4 only and changes no normative rule.

**The contract is DER out, BER in.** BCIR digests, replays, and byte-compares its
artifacts, and BER's sender's options (indefinite lengths, constructed strings,
non-minimal lengths, `TRUE` as any non-zero octet) make a value's octets non-unique —
so nothing here ever emits them. Decoding accepts the full BER surface, and `der`
reports precisely which clauses an incoming encoding broke, so an artifact can be
accepted from a BER-speaking peer and re-emitted canonically. CER is deliberately not
implemented: §9.1 makes the indefinite form mandatory for constructed encodings, which
is irreconcilable with a frozen, digested artifact.

This package is a **cold organ** — it is registered in `tools/perf/import_graph.py`
and must never be eagerly imported from a hot entry point. Import it directly
(`from bcir.asn1 import encode_der`) at the point of use.

Layers, in the order X.690 defines them:

* `tags`   — §8.1.2 identifier octets, X.680 Table 1 universal tag assignments;
* `length` — §8.1.3 the definite short/long and indefinite length forms;
* `tlv`    — §8.1.1 the structure of an encoding, and the total decoder over it;
* `values` — §8.2–§8.26 contents octets for every universal type;
* `der`    — clause 10 + 11, as a checker over a decoded tree and a BER→DER rewrite;
* `schema` — the ASN.1 type model, which owns the rules needing a type definition
  (§8.9 OPTIONAL/DEFAULT presence and §11.5 DEFAULT-value omission).
"""

from __future__ import annotations

from .codec import Strictness, decode_der, decode_value, encode_der, reencode_as_der
from .der import Violation, der_violations, is_der, require_der, to_der
from .tags import Asn1Error, Tag, TagClass, Universal
from .tlv import Tlv, decode_one, decode_tlv, encode_tlv, iter_tlv
from .values import BitString

#: The distinguished encoding rules, as identified by X.690 §12.4:
#: {joint-iso-itu-t asn1(1) ber-derived(2) distinguished-encoding(1)}.
DER_OID: tuple[int, ...] = (2, 1, 2, 1)
DER_OID_IRI = "/ASN.1/BER-Derived/Distinguished-Encoding"

#: The basic encoding rules, as identified by X.690 §12.2:
#: {joint-iso-itu-t asn1(1) basic-encoding(1)}.
BER_OID: tuple[int, ...] = (2, 1, 1)
BER_OID_IRI = "/ASN.1/Basic-Encoding"

#: The X.680 modules shipped beside this package (see `pyproject.toml`'s
#: package-data note): they are SOURCE the front end compiles, not
#: documentation about it.
STREAMPACK_MODULE = "BCIR-StreamPack.asn1"
ARTIFACT_BUNDLE_MODULE = "BCIR-ArtifactBundle.asn1"


def module_source(name: str) -> str:
    """The text of a module shipped inside this package.

    Read through `importlib.resources`, never by path arithmetic and never
    relative to the working directory. `__file__` may sit inside a zip in an
    installed wheel, and a checkout-relative `open("bcir/asn1/...")` resolves
    against the CWD — so it works when a test happens to run from the repo
    root and fails everywhere else, including in the wheel that ships the
    very file it is opening. `ecn_syntax.frame_header_source` has read its
    module this way since it was written; this is the same predicate, now
    shared by all three (L14).
    """
    from importlib import resources

    return (resources.files(__package__) / name).read_text(encoding="utf-8")

__all__ = [
    "ARTIFACT_BUNDLE_MODULE", "Asn1Error", "BER_OID", "BER_OID_IRI", "BitString",
    "DER_OID", "DER_OID_IRI", "STREAMPACK_MODULE", "Strictness", "Tag", "TagClass",
    "Tlv", "Universal", "Violation", "decode_der", "decode_one", "decode_tlv",
    "decode_value", "der_violations", "encode_der", "encode_tlv", "is_der",
    "iter_tlv", "module_source", "reencode_as_der", "require_der", "to_der",
]
