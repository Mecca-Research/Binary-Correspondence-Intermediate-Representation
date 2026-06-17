"""Fuzz the trust boundaries -- seeded by the differential generator.

The differential campaign proves the *optimizer* rails agree; this fuzzes the
*parsers/decoders* on the trust boundary (the BCIR_Roadmap Phase B.3 lever). Two
invariants, per target:

  * **round-trip identity** on valid inputs -- encode->decode / to_json->from_json /
    emit->re-read must reproduce the object (a correctness gate);
  * **graceful rejection** of malformed inputs -- a mutated/garbage input must never
    crash the interpreter with an *unexpected* exception class (a robustness gate);
    a clean domain error (AbiError / ParseError / ValueError / struct.error / ...)
    is the contract, an AttributeError/RecursionError is a bug.

Valid seeds come from `gen_module` (the same generator that drives the parity
campaign), so the fuzz surface tracks the real IR shapes. Pure Python, no deps;
`run_fuzz` returns the findings (empty == clean).
"""

from __future__ import annotations

import json
import random
import struct
from dataclasses import dataclass

from .cost import TARGETS, Theta
from .differential import gen_module
from .realize import optimize

# Exception classes that count as *graceful* rejection of malformed input. Anything
# outside this set (e.g. AttributeError, TypeError, AssertionError, RecursionError,
# SystemError) is a finding: the boundary should reject hostile input with a domain
# error, not a logic crash (a TypeError/AssertionError signals the decoder reached an
# internal contract on attacker input instead of validating up front).
_GRACEFUL = (ValueError, KeyError, IndexError, struct.error, EOFError,
             UnicodeDecodeError, OverflowError, json.JSONDecodeError,
             ZeroDivisionError)


@dataclass(frozen=True)
class FuzzFinding:
    target: str
    kind: str        # roundtrip | ungraceful
    detail: str


def _mutate_bytes(data: bytes, rng: random.Random) -> bytes:
    """A random byte-level corruption: flip, truncate, or pad."""
    if not data:
        return b"\x00" * rng.randint(0, 8)
    b = bytearray(data)
    mode = rng.randint(0, 2)
    if mode == 0:                                   # flip some bytes
        for _ in range(rng.randint(1, max(1, len(b) // 4))):
            i = rng.randrange(len(b))
            b[i] ^= 1 << rng.randint(0, 7)
    elif mode == 1:                                 # truncate
        return bytes(b[:rng.randint(0, len(b))])
    else:                                           # pad with garbage
        b.extend(rng.randbytes(rng.randint(1, 16)))
    return bytes(b)


def _mutate_text(text: str, rng: random.Random) -> str:
    if not text:
        return rng.choice(["", "{", "\x00", "garbage", "[]"])
    chars = list(text)
    for _ in range(rng.randint(1, max(1, len(chars) // 8))):
        i = rng.randrange(len(chars))
        chars[i] = rng.choice("{}[],:\"\\ \n0xZ;")
    out = "".join(chars)
    return out[:rng.randint(0, len(out))] if rng.random() < 0.3 else out


# --- StreamPack ABI codec --------------------------------------------------------

def _fuzz_streampack(rng: random.Random, findings: list[FuzzFinding]) -> None:
    from ..abi import streampack_abi as abi
    from ..gem import hydrate

    m = gen_module(rng)
    pack = hydrate(m, optimize(m, TARGETS["x86_avx512"], Theta.cool()))
    blob = abi.encode(pack)
    # round-trip identity on the valid encoding (full segment equality, not just ids).
    back = abi.decode(blob)
    if back.segments != pack.segments:
        findings.append(FuzzFinding("streampack", "roundtrip",
                                    f"segments differ after decode ({m.name})"))
    # graceful rejection of the corrupted encoding.
    try:
        abi.decode(_mutate_bytes(blob, rng))
    except abi.AbiError:
        pass
    except _GRACEFUL:
        pass
    except Exception as e:  # noqa: BLE001 - anything else is a robustness bug
        findings.append(FuzzFinding("streampack", "ungraceful",
                                    f"{type(e).__name__}: {e}"))


# --- frozen-calibrator / certificate JSON ----------------------------------------

def _fuzz_json(rng: random.Random, findings: list[FuzzFinding]) -> None:
    from .calibrate import FrozenCalibrator

    fc = FrozenCalibrator(w_q8=tuple(rng.randint(-512, 512) for _ in range(4)),
                          gen=rng.randint(1, 9), samples=rng.randint(0, 99))
    if FrozenCalibrator.from_json(fc.to_json()) != fc:
        findings.append(FuzzFinding("calibrator_json", "roundtrip",
                                    "FrozenCalibrator round-trip changed the object"))
    for payload in (_mutate_text(fc.to_json(), rng), rng.choice(["", "{}", "null", "[1,2]"])):
        try:
            FrozenCalibrator.from_json(payload)
        except _GRACEFUL:
            pass
        except Exception as e:  # noqa: BLE001
            findings.append(FuzzFinding("calibrator_json", "ungraceful",
                                        f"{type(e).__name__}: {e!r} on {payload!r:.40}"))


# --- text front-ends (ROP / MAP / ETL) -------------------------------------------

def _fuzz_frontends(rng: random.Random, findings: list[FuzzFinding]) -> None:
    from ..frontends.rop import parse_rop_program
    from ..frontends.map import parse_map
    from ..etl.parse import parse_rop as etl_parse_rop

    junk = "".join(rng.choice("abc 012 .,;:[]{}()@%$#\n\t=->") for _ in range(rng.randint(0, 60)))
    for name, fn in (("rop", parse_rop_program), ("map", parse_map), ("etl", etl_parse_rop)):
        try:
            fn(junk)
        except _GRACEFUL:
            pass
        except Exception as e:  # noqa: BLE001 - frontends define their own *Error subclasses of these
            # a frontend's declared error (MapError/ParseError/LexError) is graceful too.
            if any(b.__name__ in ("MapError", "ParseError", "LexError", "RopError")
                   for b in type(e).__mro__):
                continue
            findings.append(FuzzFinding(f"frontend_{name}", "ungraceful",
                                        f"{type(e).__name__}: {e}"))


# --- ETL binary-record decoder (the binary twin of the text front-ends) ----------

def _fuzz_etl_binary(rng: random.Random, findings: list[FuzzFinding]) -> None:
    """Fuzz `etl.binary.decode` -- the binary trust boundary whose C twin is
    `runtime/c/bcir_binrec.c` (libFuzzer'd separately). A random (often illegal:
    unaligned, oversized, out-of-bounds) record over a random/short buffer must reject
    with a domain error, never an ungraceful crash -- the Python-side analog of the C
    decoder's bounds-check contract (and a round-trip on a clean byte-aligned record)."""
    from ..etl.binary import BinaryField, BinaryRecord, decode

    # A legal byte-aligned record round-trips to the values its own bytes encode.
    width = rng.choice((8, 16, 32))
    nbytes = width // 8
    fld = BinaryField("v", offset_bits=0, width_bits=width, kind="u")
    payload = rng.randbytes(nbytes)
    rec = BinaryRecord("clean", (fld,))
    if decode(rec, payload)["v"] != int.from_bytes(payload, "little"):
        findings.append(FuzzFinding("etl_binary", "roundtrip",
                                    "byte-aligned decode != int.from_bytes"))

    # Adversarial: random fields (often unaligned / oversized) over a short buffer.
    fields = tuple(
        BinaryField(f"f{i}", offset_bits=rng.randint(0, 96), width_bits=rng.randint(0, 96),
                    kind=rng.choice(("u", "s", "f", "bytes")))
        for i in range(rng.randint(1, 4)))
    record = BinaryRecord("fuzz", fields)
    data = _mutate_bytes(rng.randbytes(rng.randint(0, 8)), rng)
    try:
        decode(record, data)
    except _GRACEFUL:
        pass
    except Exception as e:  # noqa: BLE001 - anything else is a robustness bug
        findings.append(FuzzFinding("etl_binary", "ungraceful", f"{type(e).__name__}: {e}"))


# --- the MLIR emitter (structure-aware, seeded by gen_module) ---------------------

def _fuzz_mlir(rng: random.Random, findings: list[FuzzFinding]) -> None:
    from ..lower.mlir import plan_view, to_mlir
    from ..kbcir.realize import optimize
    from .differential import law_select

    m = gen_module(rng)
    h = TARGETS[rng.choice(list(TARGETS))]
    th = Theta.cool()
    # Plan once and thread it through both emissions + the plan_view; the optimizer is
    # deterministic (covered by the differential campaign), so re-running it per call
    # here just burned CPU. The determinism check now validates emit-side determinism
    # (text building) from a fixed plan.
    result = optimize(m, h, th)
    a = to_mlir(m, h, th, result=result)
    if to_mlir(m, h, th, result=result) != a:        # deterministic emission
        findings.append(FuzzFinding("to_mlir", "roundtrip", f"non-deterministic emit ({m.name})"))
    if a.count("{") != a.count("}"):
        findings.append(FuzzFinding("to_mlir", "roundtrip", f"unbalanced braces ({m.name})"))
    # emit -> read-back: the law's argmin over the emitted plan_view reproduces the
    # oracle's per-claim selection (the bridge is lossless).
    pv = plan_view(m, h, th, result=result)
    law = law_select(pv)
    for cv in pv.claims:
        if law[cv.claim_id][0] != cv.selected:
            findings.append(FuzzFinding("to_mlir", "roundtrip",
                                        f"plan_view->law_select lost claim {cv.claim_id}"))


def run_fuzz(n: int = 400, seed: int = 0) -> list[FuzzFinding]:
    """Run `n` fuzz iterations over every trust boundary; return the findings
    (empty == clean: valid inputs round-trip and malformed inputs are rejected
    gracefully)."""
    rng = random.Random(seed)
    findings: list[FuzzFinding] = []
    for _ in range(n):
        _fuzz_streampack(rng, findings)
        _fuzz_json(rng, findings)
        _fuzz_frontends(rng, findings)
        _fuzz_etl_binary(rng, findings)
        _fuzz_mlir(rng, findings)
    return findings


def main(argv=None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="bcir.kbcir.fuzz",
                                description="Fuzz the BCIR trust boundaries (seeded by gen_module)")
    p.add_argument("-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    findings = run_fuzz(n=args.n, seed=args.seed)
    print(f"fuzz: {args.n} iters, {len(findings)} finding(s)")
    for f in findings[:20]:
        print(f"  {f.target} [{f.kind}] {f.detail}")
    return 1 if findings else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
