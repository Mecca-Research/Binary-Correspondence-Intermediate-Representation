#!/usr/bin/env python3
"""QUAL-3: qualify BCIR on a concrete target -- one model, end to end, on this machine.

Takes a BCIR-native model (the output directory of `tools/models/build_lab_model.py`: its
BCIRQ8 weights, its tokenizer and its model card) and a prompt, and runs the whole vertical
stack on it:

1. **Program** -- the greedy generation as a BCIR module, one claim per decoder operation per
   token (`decoder_program`), verified under every R-law (`verify_all`: zero diagnostics).
2. **Plan** -- K_BCIR plans it for this host's target profile; GEM hydrates the plan into a
   pipelined StreamPack.
3. **Projections** -- JSON <-> ASN.1 <-> GEM: the module as JER, the plan as its binary ABI,
   DER, OER and JER, the pack on the native wire and as DER, OER and JER. Every projection is
   decoded back and must equal its source and re-encode to the same octets.
4. **Execution** -- the C rail (`bcir-qualify`) runs the program through GEM's executor with
   the decoder kernels and the same request through the monolithic C runner, from the native
   pack and from the DER converted in C; the oracle runs the decoded DER pack through its own
   interpreter (`run_decoder_pack`) and the monolithic oracle decode (`decode_with_kv_cache`),
   and, when the float checkpoint is present, the float reference the BCIRQ8 weights quantize.
5. **Correctness** -- the four Q8 rails must generate the same tokens; the C GEM run and the
   monolithic C run must agree on every logit to the bit; the oracle's GEM and monolithic runs
   likewise; the C and oracle logits within the cross-rail bound of the BCIRQ8 parity gate
   (1e-9). The float reference's agreement is reported, not gated: it measures quantization.
6. **Latency** -- interleaved medians and minima of the GEM and monolithic C runs, the time to
   first token and per-token latency of an instrumented GEM run, per-operation kernel time,
   load and DER-conversion time, and the oracle-side plan, hydrate and encode times. Wall
   times are INDICATIVE (docs/PERFORMANCE_AUDIT.md); the GEM/monolithic ratio is gated with a
   wide band (`--max-gem-ratio`), which a dispatch overhead that grew with the program would
   breach.
7. **Memory traffic** -- the octets the program PLANS (decoder_program.program_traffic), the
   octets the C kernels COUNT as they run (they must be equal, op for op), and the memory
   references and misses a cache model SIMULATES for one run of each rail (callgrind's cache
   simulation over bcir-qualify's callgrind build, which brackets each rail's first run with
   client requests). No hardware counter is read: a simulated row is labelled simulated, and
   an unavailable valgrind is a recorded skip unless `--require-callgrind`.

One JSON report (`bcir.qualification_report.v1`) carries every number with its provenance:
the model card's digests, every artifact's SHA-256, the host and the toolchain. The exit status
is 0 only when every gated check passed; each check is listed with its verdict.

    python3 tools/models/run_qualification.py --model-dir build/lab-model \\
        --output build/qualification/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcir.abi import encode  # noqa: E402
from bcir.abi import execution_plan_abi as plan_abi  # noqa: E402
from bcir.abi.streampack_abi import decode  # noqa: E402
from bcir.asn1 import execution_plan as plan_asn1  # noqa: E402
from bcir.asn1 import program as program_asn1  # noqa: E402
from bcir.asn1 import streampack as pack_asn1  # noqa: E402
from bcir.frontends.models.decode import KVCache, decode_with_kv_cache, head_logits  # noqa: E402
from bcir.frontends.models.decoder_gem import run_decoder_pack  # noqa: E402
from bcir.frontends.models.decoder_program import (  # noqa: E402
    OPS,
    decoder_program,
    program_traffic,
)
from bcir.frontends.models.weights_io import read_q8_decoder  # noqa: E402
from bcir.gem.execution_plan import plan_from_realization  # noqa: E402
from bcir.gem.streampack import hydrate_pipelined  # noqa: E402
from bcir.kbcir import optimize  # noqa: E402
from bcir.kbcir.cost import TargetProfile, Theta  # noqa: E402
from bcir.verify import verify_all  # noqa: E402

REPORT_SCHEMA = "bcir.qualification_report.v1"
RUNTIME = ROOT / "runtime" / "c"
#: bcir-qualify's sources: the `bcir-qualify` tool of runtime/manifest.json, the units it links.
QUALIFY_SOURCES = (
    "bcir_qualify_cli.c",
    "bcir_decoder_gem.c",
    "bcir_llama.c",
    "bcir_q8_model.c",
    "bcir_q4_kernel.c",
    "bcir_ai_kernels.c",
    "bcir_decode.c",
    "bcir_exec.c",
    "bcir_runtime.c",
    "bcir_asn1_streampack.c",
    "bcir_asn1.c",
    "bcir_sha256.c",
)
#: The cross-rail logit bound of the BCIRQ8 parity gate (test_model_weights_io.py).
CROSS_RAIL_BOUND = 1e-9
#: The rails the callgrind build of bcir-qualify (`-DBCIR_QUALIFY_CALLGRIND`) brackets: the first
#: run of each is one profile part, dumped by a callgrind client request under the rail's name.
CALLGRIND_RAILS = ("gem", "monolithic")
#: The simulated cache geometry, fixed rather than read from the host, so a report's simulated
#: rows compare across hosts (size, associativity, line): 32 KiB L1s, an 8 MiB last level.
CACHE_MODEL = {"I1": (32768, 8, 64), "D1": (32768, 8, 64), "LL": (8388608, 16, 64)}


class QualificationError(Exception):
    """The qualification could not run what it was asked to (not a failed check)."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ms(seconds: float) -> float:
    return round(seconds * 1e3, 3)


class Checks:
    """The gated verdicts, in the order they were reached."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, ok: bool, detail: str) -> bool:
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def passed(self) -> bool:
        return bool(self.rows) and all(row["ok"] for row in self.rows)


# --- the C rail --------------------------------------------------------------------------------


def _valgrind_include() -> list[str]:
    """The include directory of the valgrind on PATH, appended after every other directory
    (`-idirafter`), when its client-request header is there: a compiler with a sysroot of its
    own (a conda clang) does not search the system's, and appended last the directory supplies
    only what the compiler's own headers do not."""
    valgrind = shutil.which("valgrind")
    if valgrind is None:
        return []
    include = Path(os.path.realpath(valgrind)).parent.parent / "include"
    return ["-idirafter", str(include)] if (include / "valgrind" / "callgrind.h").is_file() else []


def build_qualify(directory: Path, *, callgrind: bool = False) -> Path:
    """bcir-qualify from the checkout's runtime sources, under the host C compiler; with
    `callgrind`, its callgrind build, which brackets the first run of each rail with callgrind
    client requests from the valgrind on PATH (<valgrind/callgrind.h>)."""
    from bcir.toolchain import host_c_compiler, host_link_args

    cc = host_c_compiler()
    if cc is None:
        raise QualificationError("no C compiler: the C rail cannot be built")
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / ("bcir-qualify.exe" if os.name == "nt" else "bcir-qualify")
    command = [cc, "-std=c11", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-g"]
    command += ["-DBCIR_QUALIFY_CALLGRIND", *_valgrind_include()] if callgrind else []
    command += ["-I", str(RUNTIME), *(str(RUNTIME / s) for s in QUALIFY_SOURCES)]
    command += ["-o", str(exe), *host_link_args(["-lm"])]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise QualificationError(f"bcir-qualify build failed:\n{result.stderr[-4000:]}")
    return exe


def _compiler_version() -> str:
    from bcir.toolchain import host_c_compiler

    cc = host_c_compiler()
    if cc is None:
        return "none"
    try:
        out = subprocess.run([cc, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return cc
    return out.stdout.splitlines()[0] if out.stdout else cc


def run_qualify(exe: Path, model: Path, pack: Path, *, der: bool, prompt_ids, reps: int, logits):
    argv = [str(exe), str(model), str(pack), "--der" if der else "--native"]
    argv += ["--prompt", ",".join(map(str, prompt_ids)), "--reps", str(reps)]
    argv += ["--logits-out", str(logits)]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    if r.returncode not in (0, 3) or not r.stdout.strip():
        raise QualificationError(f"bcir-qualify exited {r.returncode}: {r.stderr.strip()[-2000:]}")
    return r.returncode, json.loads(r.stdout)


def _read_logits(path: Path, vocab: int) -> list[float]:
    return list(struct.unpack(f"<{vocab}d", path.read_bytes()))


#: The events callgrind's cache simulation counts, each of which a rail's part must carry.
_CALLGRIND_EVENTS = ("Ir", "Dr", "Dw", "I1mr", "D1mr", "D1mw", "ILmr", "DLmr", "DLmw")
_CALLGRIND_EVENT = re.compile(r"[A-Za-z][A-Za-z0-9]*", re.ASCII)
_CALLGRIND_COUNT = re.compile(r"[0-9]+", re.ASCII)
_CALLGRIND_REQUEST = "Client Request: "


def callgrind_part(text: str) -> tuple[str, dict[str, int]]:
    """One part of a callgrind profile, read by the Callgrind format's own grammar: the dump's
    trigger (`desc: Trigger:`) and its `totals:` by event (`events:`). The format lets a cost line
    leave out trailing zero counts, so a short `totals:` is padded with zeros. Anything else is a
    ValueError, never a guessed number: one of the three lines missing or repeated, an event
    named twice or outside the grammar, more counts than events, or a count that is not ASCII
    decimal digits (no host-language parser reads one: `int` would take `1_000`, a sign or a
    non-ASCII digit)."""
    found: dict[str, list[str]] = {"events:": [], "totals:": [], "desc: Trigger:": []}
    for line in text.split("\n"):
        for key, values in found.items():
            if line.startswith(key + " "):
                values.append(line[len(key) + 1 :])
    for key, values in found.items():
        if len(values) != 1:
            raise ValueError(f"a callgrind part needs one '{key}' line, not {len(values)}")
    events_text, totals_text, trigger = (values[-1] for values in found.values())
    events, counts = events_text.split(" "), totals_text.split(" ")
    if len(set(events)) != len(events) or not all(map(_CALLGRIND_EVENT.fullmatch, events)):
        raise ValueError(f"callgrind events outside the grammar: {events_text[:200]!r}")
    if len(counts) > len(events) or not all(map(_CALLGRIND_COUNT.fullmatch, counts)):
        raise ValueError(f"callgrind totals outside the grammar: {totals_text[:200]!r}")
    totals = dict.fromkeys(events, 0)
    totals.update(zip(events, map(int, counts)))
    return trigger, totals


def callgrind_rails(texts) -> dict[str, dict[str, int]]:
    """Each rail's one run, from the parts of one callgrind profile of bcir-qualify's callgrind
    build: exactly one part per rail -- the one its client request dumped under the rail's name
    -- counting every event of the cache simulation and at least one instruction. Every part is
    read by the grammar (callgrind_part); the dump at program termination is the process around
    the rails and is not used. A rail with no part (a binary built without the brackets) or with
    two, a part from a request no rail makes, or a rail's part that counted nothing is a
    ValueError: a measurement that did not happen is not a zero."""
    rails: dict[str, dict[str, int]] = {}
    for text in texts:
        trigger, totals = callgrind_part(text)
        if not trigger.startswith(_CALLGRIND_REQUEST):
            continue
        rail = trigger[len(_CALLGRIND_REQUEST) :]
        if rail not in CALLGRIND_RAILS:
            raise ValueError(f"a callgrind part from a client request no rail makes: {rail[:80]!r}")
        if rail in rails:
            raise ValueError(f"two callgrind parts for the {rail} rail")
        missing = [event for event in _CALLGRIND_EVENTS if event not in totals]
        if missing:
            raise ValueError(f"the {rail} part does not count {', '.join(missing)}")
        if not totals["Ir"]:
            raise ValueError(f"the {rail} part counted no instructions")
        rails[rail] = totals
    absent = [rail for rail in CALLGRIND_RAILS if rail not in rails]
    if absent:
        raise ValueError(
            f"no callgrind part for the {' and '.join(absent)} rail: the profiled binary does "
            "not bracket it (bcir-qualify's callgrind build, -DBCIR_QUALIFY_CALLGRIND)"
        )
    return rails


def traffic_row(totals: dict[str, int]) -> dict[str, int]:
    """One rail's simulated row: its counts, and the data references, last-level data misses and
    octets those misses move (one last-level line each, CACHE_MODEL) that they sum to."""
    row = {event: totals[event] for event in _CALLGRIND_EVENTS}
    row["data_refs"] = totals["Dr"] + totals["Dw"]
    row["ll_data_misses"] = totals["DLmr"] + totals["DLmw"]
    row["ll_miss_octets"] = row["ll_data_misses"] * CACHE_MODEL["LL"][2]
    return row


def profile_rails(exe: Path, model: Path, pack: Path, prompt_ids, out_dir: Path, *, reps: int = 1):
    """Run `exe` (a bcir-qualify) on the native pack under callgrind's cache simulation with the
    fixed cache model, writing its parts into `out_dir`, which must hold no earlier profile, and
    read each rail's part (callgrind_rails). QualificationError when callgrind cannot run;
    ValueError when its profile is not one part per rail."""
    valgrind = shutil.which("valgrind")
    if not valgrind:
        raise QualificationError("valgrind not found")
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("callgrind.out*")):
        raise QualificationError(f"{out_dir} already holds a callgrind profile")
    argv = [valgrind, "--tool=callgrind", "--cache-sim=yes"]
    argv += [f"--callgrind-out-file={out_dir / 'callgrind.out'}"]
    argv += [
        f"--{level}={size},{assoc},{line}" for level, (size, assoc, line) in CACHE_MODEL.items()
    ]
    argv += [str(exe), str(model), str(pack), "--native"]
    argv += ["--prompt", ",".join(map(str, prompt_ids)), "--reps", str(reps)]
    try:
        r = subprocess.run(
            argv, capture_output=True, text=True, errors="replace", timeout=7200, check=False
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise QualificationError(f"callgrind could not run: {error}") from error
    if r.returncode not in (0, 3):
        raise QualificationError(
            f"callgrind run failed ({r.returncode}): {r.stderr.strip()[-500:]}"
        )
    parts = sorted(out_dir.glob("callgrind.out*"))
    return callgrind_rails(p.read_bytes().decode("utf-8", "replace") for p in parts)


def callgrind_traffic(model: Path, pack: Path, prompt_ids, workdir: Path):
    """Per-run simulated memory references and misses of each rail, or None and the reason the
    measurement could not be made. bcir-qualify's callgrind build brackets the first run of each
    rail with callgrind client requests (runtime/c/bcir_qualify_cli.c), so a rail's part is one
    run of that rail on any architecture valgrind runs on. The parts are read in callgrind's own
    machine format: QUAL-3 first scraped callgrind_annotate's human-readable function table for
    each rail's entry point, and on AArch64 the line it took was not that function's row."""
    if not shutil.which("valgrind"):
        return None, "valgrind not found"
    try:
        exe = build_qualify(workdir / "callgrind-build", callgrind=True)
        rails = profile_rails(exe, model, pack, prompt_ids, workdir / "callgrind")
    except (QualificationError, ValueError, OSError) as error:
        return None, str(error)
    rows = {rail: traffic_row(totals) for rail, totals in rails.items()}
    caches = {
        level: {"bytes": size, "ways": assoc, "line": line}
        for level, (size, assoc, line) in CACHE_MODEL.items()
    }
    return {
        "simulated": True,
        "tool": "callgrind --cache-sim=yes",
        "measured": "one run of each rail, bracketed by callgrind client requests",
        "caches": caches,
        **rows,
    }, ""


# --- the oracle rail ---------------------------------------------------------------------------


def oracle_monolithic(spec, weights, prompt_ids, max_new):
    """decode_with_kv_cache's ids and the logits that chose the last of them."""
    ids = decode_with_kv_cache(prompt_ids, spec, weights, max_new)
    cache, row = KVCache(spec), None
    for token in list(prompt_ids) + ids[:-1]:
        row = cache._step_row(weights.embedding.row(token), spec, weights)
    return ids, head_logits(row, weights)


def float_reference(model_dir: Path, workdir: Path):
    """The float decoder the BCIRQ8 weights quantize (model.bcirdt), or None without it."""
    checkpoint = model_dir / "model.bcirdt"
    if not checkpoint.is_file():
        return None
    from bcir.hosted.models.native import NativeDecoder

    library = NativeDecoder.build(workdir / "native")
    return NativeDecoder.resume(library, checkpoint).decoder_weights()


# --- the qualification -------------------------------------------------------------------------


#: The JER reader's §4.3 maxima (bcir/asn1/jer_bounded.py): a caller may tighten them, never raise
#: them, so a document past one is refused by design. Only these refusals are a bound, not a fault.
_JER_BOUNDS = frozenset(
    {"input-too-large", "nodes-exceeded", "elements-exceeded", "members-exceeded", "work-exceeded"}
)


def _round_trip(name: str, value, enc, dec, checks: Checks, *, bounded: bool = False) -> dict:
    t0 = time.perf_counter()
    octets = enc(value)
    encode_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    try:
        back = dec(octets)
    except Exception as exc:  # a bounded reader's refusal is a verdict; anything else is not
        diag = getattr(exc, "diagnostic", None)
        code = getattr(getattr(diag, "code", None), "value", None)
        if not bounded or code not in _JER_BOUNDS:
            raise
        checks.add(
            f"round trip: {name}",
            True,
            f"{len(octets)} octets: refused by the bounded JER reader's maxima ({code}); "
            "the JSON chain is qualified on a request within them",
        )
        return {
            "bytes": len(octets),
            "sha256": sha256(octets),
            "round_trip": None,
            "refused_by_bound": {"code": code, "needed": diag.needed, "offset": diag.offset},
            "encode_ms": _ms(encode_s),
        }
    decode_s = time.perf_counter() - t0
    ok = back == value and enc(back) == octets
    checks.add(f"round trip: {name}", ok, f"{len(octets)} octets")
    return {
        "bytes": len(octets),
        "sha256": sha256(octets),
        "round_trip": ok,
        "encode_ms": _ms(encode_s),
        "decode_ms": _ms(decode_s),
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _host() -> dict:
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "cpu model", "hardware")):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": cpu or platform.processor() or "unknown",
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "compiler": _compiler_version(),
        "counters": "none read: wall clocks only, cache traffic simulated",
    }


def qualify(
    model_dir,
    *,
    prompt: str | None = None,
    max_new: int = 48,
    json_max_new: int = 12,
    reps: int = 7,
    max_gem_ratio: float = 1.5,
    callgrind: str = "auto",
    float_check: bool = True,
    log=print,
) -> tuple[dict, bool]:
    """Run the qualification; (report, every gated check passed)."""
    model_dir = Path(model_dir)
    checks = Checks()
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    q8_path = model_dir / "model.bcirq8"
    q8_sha = _file_sha256(q8_path)
    checks.add(
        "model artifact is the card's",
        q8_sha == card["artifacts"]["bcirq8_sha256"],
        f"BCIRQ8 sha256 {q8_sha[:16]}",
    )
    from bcir.hosted.training.bpe import BytePairTokenizer

    spec, weights, meta = read_q8_decoder(q8_path)
    tok_path = model_dir / "tokenizer.json"
    tok_sha = _file_sha256(tok_path)
    checks.add(
        "the tokenizer is the one the card and the BCIRQ8 artifact name",
        tok_sha == card["tokenizer"]["sha256"] == meta.tokenizer_sha256,
        f"tokenizer sha256 {tok_sha[:16]}",
    )
    tokenizer = BytePairTokenizer.from_json(tok_path.read_text("utf-8"))
    prompt = card["sample"]["prompt"] if prompt is None else prompt
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    log(f"model {card['name']}: {card['parameters']} parameters, prompt {len(prompt_ids)} ids")

    # 1-2: the program, its plan, its pack
    t0 = time.perf_counter()
    program = decoder_program(spec, len(prompt_ids), max_new)
    program_s = time.perf_counter() - t0
    h, theta = TargetProfile.for_host(), Theta.cool()
    t0 = time.perf_counter()
    result = optimize(program.module, h, theta)
    plan_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pack = hydrate_pipelined(program.module, result, depth=2)
    hydrate_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    diagnostics = verify_all(program.module, result=result, pack=pack, h=h, theta=theta)
    verify_s = time.perf_counter() - t0
    checks.add(
        "the program verifies (R1-R25)",
        not diagnostics,
        f"{len(diagnostics)} diagnostics over {program.claims} claims",
    )
    ops = {op: 0 for op in OPS}
    for phase in program.module.phases:
        for claim in phase.claims:
            ops[claim.op] += 1
    log(f"program: {program.claims} claims, {len(program.module.phases)} phases; plan + hydrate ok")

    # 3: the projections
    execution_plan = plan_from_realization(program.module, result, h, "eft")
    artifacts = {
        "module": {
            "jer": _round_trip(
                "module JER",
                program.module,
                program_asn1.module_to_jer,
                program_asn1.jer_to_module,
                checks,
            )
        },
        "plan": {
            name: _round_trip(f"plan {name}", execution_plan, enc, dec, checks)
            for name, enc, dec in (
                ("binary", plan_abi.encode_plan, plan_abi.decode_plan),
                ("der", plan_asn1.encode_plan_der, plan_asn1.decode_plan_der),
                ("oer", plan_asn1.encode_plan_oer, plan_asn1.decode_plan_oer),
                ("jer", plan_asn1.encode_plan_jer, plan_asn1.decode_plan_jer),
            )
        },
        "pack": {
            name: _round_trip(f"pack {name}", pack, enc, dec, checks)
            for name, enc, dec in (
                ("native", encode, decode),
                ("der", pack_asn1.encode_pack, pack_asn1.decode_pack),
                ("oer", pack_asn1.encode_pack_oer, pack_asn1.decode_pack_oer),
            )
        },
    }
    artifacts["pack"]["jer"] = _round_trip(
        "pack jer", pack, pack_asn1.encode_pack_jer, pack_asn1.decode_pack_jer, checks, bounded=True
    )
    # The JSON chain: a shorter request of the same model, its pack carried as JSON (JER), decoded,
    # re-encoded as ASN.1 (DER), converted to the native wire in C and executed by GEM. Greedy
    # decoding is prefix-consistent, so it must generate the headline run's first tokens.
    json_new = min(json_max_new, max_new)
    json_program = decoder_program(spec, len(prompt_ids), json_new)
    json_result = optimize(json_program.module, h, theta)
    json_pack = hydrate_pipelined(json_program.module, json_result, depth=2)
    json_doc = pack_asn1.encode_pack_jer(json_pack)
    from_json = pack_asn1.decode_pack_jer(json_doc)
    json_der = pack_asn1.encode_pack(from_json)
    native = encode(pack)
    der = pack_asn1.encode_pack(pack)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "pack.bin").write_bytes(native)
        (work / "pack.der").write_bytes(der)
        (work / "json.der").write_bytes(json_der)
        exe = build_qualify(work)

        # 4: the C rail, from the native pack and from the DER converted in C
        c_runs = {}
        for form, path, is_der in (
            ("native", work / "pack.bin", False),
            ("der", work / "pack.der", True),
        ):
            logits_path = work / f"logits-{form}.f64"
            rc, report = run_qualify(
                exe, q8_path, path, der=is_der, prompt_ids=prompt_ids, reps=reps, logits=logits_path
            )
            report["exit"] = rc
            report["logits_values"] = _read_logits(logits_path, spec.vocab_size)
            c_runs[form] = report
            log(f"C rail ({form}): {report['status']}, tokens {report['tokens']['gem'][:8]}...")
        _, json_run = run_qualify(
            exe,
            q8_path,
            work / "json.der",
            der=True,
            prompt_ids=prompt_ids,
            reps=1,
            logits=work / "logits-json.f64",
        )
        log(f"JSON chain ({json_program.claims} claims): {json_run['status']}")
        traffic_sim, sim_reason = (None, "not requested")
        if callgrind != "off":
            log("simulating each rail's memory traffic under callgrind (slow) ...")
            traffic_sim, sim_reason = callgrind_traffic(
                q8_path, work / "pack.bin", prompt_ids, work
            )
            if traffic_sim is None and callgrind == "require":
                checks.add("simulated memory traffic", False, sim_reason)

        # 4: the oracle rail
        t0 = time.perf_counter()
        twin = run_decoder_pack(pack_asn1.decode_pack(der), spec, weights, prompt_ids)
        twin_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        oracle_ids, oracle_logits = oracle_monolithic(spec, weights, prompt_ids, max_new)
        oracle_s = time.perf_counter() - t0
        reference = float_reference(model_dir, work) if float_check else None
        if reference is not None:
            float_ids, float_logits = oracle_monolithic(spec, reference, prompt_ids, max_new)

    # 5: correctness
    c = c_runs["native"]
    tokens = {
        "c_gem": c["tokens"]["gem"],
        "c_monolithic": c["tokens"]["monolithic"],
        "c_gem_from_der": c_runs["der"]["tokens"]["gem"],
        "python_gem": twin.tokens,
        "python_monolithic": oracle_ids,
    }
    checks.add(
        "every Q8 rail generates the same tokens",
        len({tuple(v) for v in tokens.values()}) == 1,
        f"{len(oracle_ids)} tokens on {len(tokens)} rails",
    )
    for form, run in c_runs.items():
        checks.add(
            f"C GEM == monolithic C, bit for bit ({form} pack)",
            run["exit"] == 0 and run["logits"]["bitwise_identical"],
            f"logits sha256 {run['logits']['sha256_gem'][:16]}",
        )
        checks.add(
            f"every C GEM run identical ({form} pack)",
            run["logits"]["gem_runs_identical"],
            f"{reps} repetitions + 1 instrumented run",
        )
    checks.add(
        "the C rail executes the oracle's native octets (DER converted in C)",
        c_runs["der"]["program"]["native_sha256"] == sha256(native)
        and c_runs["native"]["program"]["native_sha256"] == sha256(native),
        f"native sha256 {sha256(native)[:16]}",
    )
    checks.add(
        "oracle GEM == oracle monolithic, bit for bit",
        twin.logits == oracle_logits,
        "run_decoder_pack against decode_with_kv_cache",
    )
    c_vs_py = max(abs(a - b) for a, b in zip(c["logits_values"], oracle_logits))
    checks.add(
        f"C logits within {CROSS_RAIL_BOUND} of the oracle's",
        c_vs_py <= CROSS_RAIL_BOUND,
        f"max |C - oracle| = {c_vs_py:.3g}",
    )
    checks.add(
        "JSON -> ASN.1 -> GEM: the JER-carried program runs bit for bit, the headline's prefix",
        from_json == json_pack
        and json_run["status"] == "PASS"
        and json_run["tokens"]["gem"] == oracle_ids[:json_new]
        and json_run["program"]["native_sha256"] == sha256(encode(json_pack)),
        f"{json_program.claims} claims, {len(json_doc)} JER octets -> {len(json_der)} DER octets",
    )
    correctness = {
        "json_chain": {
            "claims": json_program.claims,
            "max_new_tokens": json_new,
            "jer_bytes": len(json_doc),
            "jer_sha256": sha256(json_doc),
            "der_bytes": len(json_der),
            "der_sha256": sha256(json_der),
            "native_sha256": json_run["program"]["native_sha256"],
            "tokens": json_run["tokens"]["gem"],
            "bitwise_identical": json_run["logits"]["bitwise_identical"],
        },
        "tokens": tokens,
        "logits": {
            "c_gem_sha256": c["logits"]["sha256_gem"],
            "c_monolithic_sha256": c["logits"]["sha256_monolithic"],
            "c_bitwise_identical": c["logits"]["bitwise_identical"],
            "python_bitwise_identical": twin.logits == oracle_logits,
            "max_abs_c_vs_python": c_vs_py,
        },
        "model_card_sample_matches": card["sample"]["generated_ids"][:max_new]
        == oracle_ids[: len(card["sample"]["generated_ids"])]
        if prompt == card["sample"]["prompt"]
        else None,
    }
    if reference is not None:
        agree = sum(1 for a, b in zip(float_ids, oracle_ids) if a == b)
        prefix = next(
            (i for i, (a, b) in enumerate(zip(float_ids, oracle_ids)) if a != b), len(oracle_ids)
        )
        correctness["float_reference"] = {
            "tokens": float_ids,
            "tokens_agreeing": agree,
            "agreeing_prefix": prefix,
            "max_abs_logit_quantization_error": max(
                abs(a - b) for a, b in zip(float_logits, oracle_logits)
            ),
            "note": "reported, not gated: BCIRQ8 quantizes the float decoder",
        }

    # 6: latency
    t = c["timing"]
    ratio = t["gem_median_ns"] / t["monolithic_median_ns"]
    checks.add(
        f"GEM within {max_gem_ratio}x of the monolithic runner (ratio row)",
        ratio <= max_gem_ratio,
        f"median GEM/monolithic = {ratio:.3f}",
    )
    per_token = [b - a for a, b in zip([0] + t["token_ns"][:-1], t["token_ns"])]
    latency = {
        "class": "wall: INDICATIVE, never a gate; the ratio row is gated with a wide band",
        "c": {
            "reps": reps,
            "gem_median_ms": t["gem_median_ns"] / 1e6,
            "gem_min_ms": t["gem_min_ns"] / 1e6,
            "monolithic_median_ms": t["monolithic_median_ns"] / 1e6,
            "monolithic_min_ms": t["monolithic_min_ns"] / 1e6,
            "gem_over_monolithic_median": ratio,
            # what the GEM run spends beyond the kernels: the executor's trust boundary (the
            # pack's CRC and R10 walk, re-verified on every run), then dispatch and bookkeeping
            "gem_overhead_ms": (t["gem_median_ns"] - t["monolithic_median_ns"]) / 1e6,
            "trust_boundary_ms": t["trust_boundary_median_ns"] / 1e6,
            "time_to_first_token_ms": t["time_to_first_token_ns"] / 1e6,
            "per_token_ms": {
                "median": statistics.median(per_token) / 1e6,
                "min": min(per_token) / 1e6,
                "max": max(per_token) / 1e6,
            },
            "tokens_per_second": len(per_token) / (t["instrumented_run_ns"] / 1e9),
            "load_ms": t["load_ns"] / 1e6,
            "der_to_native_ms": c_runs["der"]["timing"]["der_to_native_ns"] / 1e6,
            "per_operation_ms": {op: row["ns"] / 1e6 for op, row in c["operations"].items()},
        },
        "oracle": {
            "program_ms": _ms(program_s),
            "plan_ms": _ms(plan_s),
            "hydrate_ms": _ms(hydrate_s),
            "verify_ms": _ms(verify_s),
            "python_gem_ms": _ms(twin_s),
            "python_monolithic_ms": _ms(oracle_s),
        },
    }

    # 7: memory traffic
    planned = program_traffic(program)
    counted = {
        op: (row["bytes_read"], row["bytes_written"])
        for op, row in c["operations"].items()
        if row["claims"]
    }
    counted_total = (c["traffic"]["bytes_read"], c["traffic"]["bytes_written"])
    checks.add(
        "counted traffic == planned traffic, op for op",
        all(counted.get(op) == tuple(planned[op]) for op in planned if op != "total")
        and counted_total == tuple(planned["total"]),
        f"{planned['total'][0]} octets read, {planned['total'][1]} written",
    )
    traffic = {
        "planned": {op: {"read": r, "written": w} for op, (r, w) in planned.items()},
        "counted": {op: {"read": r, "written": w} for op, (r, w) in counted.items()},
        "simulated": traffic_sim if traffic_sim is not None else {"skipped": sim_reason},
    }

    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if checks.passed else "FAIL",
        "checks": checks.rows,
        "commit": _git("rev-parse", "HEAD"),
        # A report of a checkout with local changes is not a report of its commit: say so.
        "tree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "host": _host(),
        "model": {
            "name": card["name"],
            "parameters": card["parameters"],
            "decoder": card["decoder"],
            "bcirq8_sha256": q8_sha,
            "bcirq8_bytes": q8_path.stat().st_size,
            "recipe_sha256": card["recipe_sha256"],
            "corpus_sha256": card["corpus"]["corpus_sha256"],
            "tokenizer_sha256": card["tokenizer"]["sha256"],
            "validation_perplexity": card["training"]["validation_perplexity_final"],
            "training_steps": card["training"]["steps"],
            "checkpoint_sha256": meta.source_model_sha256,
        },
        "request": {"prompt": prompt, "prompt_ids": prompt_ids, "max_new_tokens": max_new},
        "program": {
            "claims": program.claims,
            "phases": len(program.module.phases),
            "operations": ops,
            "positions": program.positions,
            "verify_diagnostics": len(diagnostics),
            "target_profile": getattr(h, "name", str(h)),
        },
        "artifacts": artifacts,
        "correctness": correctness,
        "latency": latency,
        "traffic": traffic,
        "text": tokenizer.decode(prompt_ids + oracle_ids),
    }
    return report, checks.passed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", help="default: the model card's evaluation prompt")
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument(
        "--json-max-new", type=int, default=12, help="tokens of the JSON-chain request"
    )
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--max-gem-ratio", type=float, default=1.5)
    parser.add_argument("--no-float", action="store_true", help="skip the float reference")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--require-callgrind", action="store_true")
    group.add_argument("--no-callgrind", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(args.max_new, args.json_max_new, args.reps) < 1
        or not math.isfinite(args.max_gem_ratio)
        or args.max_gem_ratio <= 0
    ):
        print(
            "run_qualification: --max-new, --json-max-new and --reps must be >= 1, "
            "--max-gem-ratio a positive number",
            file=sys.stderr,
        )
        return 2
    callgrind = "require" if args.require_callgrind else "off" if args.no_callgrind else "auto"
    try:
        report, ok = qualify(
            args.model_dir,
            prompt=args.prompt,
            max_new=args.max_new,
            json_max_new=args.json_max_new,
            reps=args.reps,
            max_gem_ratio=args.max_gem_ratio,
            callgrind=callgrind,
            float_check=not args.no_float,
        )
    except (OSError, ValueError, KeyError, QualificationError, subprocess.SubprocessError) as exc:
        print(f"run_qualification: FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for row in report["checks"]:
        print(f"  {'PASS' if row['ok'] else 'FAIL'} {row['check']}: {row['detail']}")
    print(f"run_qualification: {report['status']} ({out})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
