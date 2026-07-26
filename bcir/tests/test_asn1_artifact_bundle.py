"""ASN.1 DER/OER interoperability for the native BCAB artifact bundle."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile

from bcir.abi.artifact_bundle import (
    ArtifactFormat,
    ArtifactKind,
    BundleError,
    encode_bundle,
)
from bcir.abi.artifact_tool import main as bundle_tool
from bcir.asn1 import Strictness
from bcir.asn1.artifact_bundle import (
    ARTIFACT_BUNDLE_MODULE_OID,
    MODULE,
    PROJECTION_VERSION,
    bundle_to_value,
    decode_bundle_der,
    decode_bundle_oer,
    der_to_native,
    encode_bundle_der,
    encode_bundle_oer,
    native_to_der,
    native_to_oer,
    oer_to_native,
    value_to_bundle,
)
from bcir.asn1.tags import Asn1Error, encode_tag
from bcir.asn1.tlv import decode_one, encode_tlv
from bcir.frontends.asn1 import compile_module
from bcir.lower.artifact_bundle import asn1_contract_to_mlir
from bcir.tests.test_artifact_bundle import _three_bundle


_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _ROOT / "bcir" / "asn1" / "BCIR-ArtifactBundle.asn1"


def _indefinite_outer(der: bytes) -> bytes:
    tree = decode_one(der)
    assert tree.constructed
    body = b"".join(encode_tlv(child) for child in tree.children)
    return encode_tag(tree.tag) + b"\x80" + body + b"\x00\x00"


def test_x680_module_compiles_to_the_same_der_and_oer_type_contract():
    bundle = _three_bundle()
    value = bundle_to_value(bundle)
    lowered = compile_module(_MODULE_PATH.read_text(encoding="utf-8"), str(_MODULE_PATH))
    assert lowered.module.oid == ARTIFACT_BUNDLE_MODULE_OID
    assert ARTIFACT_BUNDLE_MODULE_OID == (1, 3, 6, 1, 4, 1, 62596, 2)
    assert lowered.module.encode("ArtifactBundle", value) == encode_bundle_der(bundle)

    from bcir.asn1.oer import OerRules, encode_oer

    assert encode_oer(
        lowered.module.types["ArtifactBundle"], value, rules=OerRules.CANONICAL,
    ) == encode_bundle_oer(bundle)


def test_native_der_and_oer_round_trips_are_byte_identical():
    bundle = _three_bundle()
    native = encode_bundle(bundle)
    der = native_to_der(native)
    oer = native_to_oer(native)
    assert decode_bundle_der(der) == bundle
    assert decode_bundle_oer(oer, canonical=True) == bundle
    assert der_to_native(der) == native
    assert oer_to_native(oer, canonical=True) == native
    assert encode_bundle_der(decode_bundle_der(der)) == der
    assert encode_bundle_oer(decode_bundle_oer(oer, canonical=True)) == oer


def test_canonical_oer_rejects_a_basic_oer_nonminimal_length_spelling():
    bundle = _three_bundle()
    canonical = encode_bundle_oer(bundle)
    # The top-level presence map and version occupy two octets; rootVariant's
    # seven-octet UTF-8 value then starts with a short-form length of 7. BASIC-OER
    # admits the equivalent long form, while CANONICAL-OER requires the short form.
    assert canonical[:3] == b"\xc0\x01\x07"
    basic = canonical[:2] + b"\x81\x07" + canonical[3:]
    assert decode_bundle_oer(basic) == bundle
    try:
        decode_bundle_oer(basic, canonical=True)
        assert False, "CANONICAL-OER path accepted a non-minimal length determinant"
    except Asn1Error:
        pass


def test_ber_is_admitted_only_when_requested_and_recanonicalizes_native_bytes():
    native = encode_bundle(_three_bundle())
    der = native_to_der(native)
    ber = _indefinite_outer(der)
    assert ber != der
    try:
        der_to_native(ber)
        assert False, "strict DER path accepted an indefinite-length BER spelling"
    except Asn1Error:
        pass
    assert der_to_native(ber, strictness=Strictness.BER) == native


def test_projection_rejects_unknown_or_noncanonical_directory_metadata():
    value = bundle_to_value(_three_bundle())
    cases = []

    changed = {**value, "version": PROJECTION_VERSION + 1}
    cases.append(changed)

    changed = {**value, "variants": [dict(item) for item in value["variants"]]}
    changed["variants"][0]["flags"] = 0x80
    cases.append(changed)

    changed = {**value, "variants": [dict(item) for item in value["variants"]]}
    changed["variants"][0]["targetManifest"] = b"short"
    cases.append(changed)

    changed = {**value, "variants": [dict(item) for item in value["variants"]]}
    changed["variants"][0]["kind"] = 999
    cases.append(changed)

    changed = {**value, "variants": [dict(item) for item in value["variants"]]}
    changed["variants"][2]["requiredFeatures"] = ["z", "a"]
    cases.append(changed)

    changed = {**value, "variants": [dict(item) for item in value["variants"]]}
    changed["variants"][1]["payload"] = b"bad\x00text"
    cases.append(changed)

    changed = {**value, "unknown": 1}
    cases.append(changed)

    changed = {**value, "variants": [dict(item) for item in value["variants"]]}
    changed["variants"][0]["unknown"] = 1
    cases.append(changed)

    for malformed in cases:
        try:
            value_to_bundle(malformed)
            assert False, malformed
        except BundleError:
            pass


def test_projection_schema_requires_every_nonoptional_directory_field():
    value = bundle_to_value(_three_bundle())
    del value["variants"][0]["payload"]
    try:
        MODULE.encode("ArtifactBundle", value)
        assert False, "schema accepted a variant without payload octets"
    except Asn1Error as exc:
        assert "payload" in str(exc)


def test_bundle_cli_transcodes_der_and_oer_atomically():
    native = encode_bundle(_three_bundle())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "bundle.bcab"
        der = root / "bundle.der"
        oer = root / "bundle.oer"
        from_der = root / "from-der.bcab"
        from_oer = root / "from-oer.bcab"
        source.write_bytes(native)
        for argv in (
            ["to-der", str(source), str(der)],
            ["from-der", str(der), str(from_der)],
            ["to-oer", str(source), str(oer)],
            ["from-oer", "--canonical", str(oer), str(from_oer)],
        ):
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                assert bundle_tool(argv) == 0, (argv, stderr.getvalue())
            assert stdout.getvalue().startswith(argv[0] + " ")
        assert from_der.read_bytes() == native
        assert from_oer.read_bytes() == native
        assert not list(root.glob("*.tmp"))


def test_mlir_contract_names_the_additive_projection_and_r24_identity():
    text = asn1_contract_to_mlir()
    assert "bcir.asn1.module @BCIR_ArtifactBundle" in text
    assert "array<i64: 1, 3, 6, 1, 4, 1, 62596, 2>" in text
    assert 'native = "artifact_bundle"' in text
    assert "bcir.asn1.projection @artifact_bundle_projection" in text
    assert "additive" in text
    assert "rules = #bcir.asn1_rules<der>" in text


def test_artifact_kind_and_format_enumerations_remain_native_bcab_v1_values():
    """The ASN.1 rail projects the native enum; it does not allocate competing IDs."""
    bundle = _three_bundle()
    value = bundle_to_value(bundle)
    for variant, projected in zip(bundle.variants, value["variants"]):
        assert projected["kind"] == int(variant.kind)
        assert projected["format"] == int(variant.format)
        assert ArtifactKind(projected["kind"]) is variant.kind
        assert ArtifactFormat(projected["format"]) is variant.format
    assert value_to_bundle(value) == bundle
