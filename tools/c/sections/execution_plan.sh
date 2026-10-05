#!/usr/bin/env bash
# execution_plan: ExecutionPlanV1 (G11): freestanding decode + Python->C->Python byte-identical round trip + plan/pack binding
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: BCIR Make
# builds them from runtime/manifest.json and runs this script as a task, whose verdict the gate
# shows; the CMake project builds them from the same manifest and runs this script as the
# `c-section-execution_plan` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical.
#
#   usage: execution_plan.sh <test_execution_plan>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_execution_plan>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# bcir_execution_plan.h is the C twin of bcir/abi/execution_plan_abi.py -- the plan the pack was
# derived from, as bytes (its decoder/verifier lives in bcir_runtime.c, freestanding-checked above).
# Python encodes the audit fixture's plan and pack; the C harness decodes, verifies, binds the pack
# to the plan (bcir_ep_check_pack) and checks the vector against the live registry; Python rebuilds
# the plan from the C decode and must re-encode it byte for byte. A registry that moved after the
# plan was minted must be refused as STALE on the C rail.
"${PYTHON}" - "${tmp}" <<'PY' || { echo "  FAIL: python plan encode"; exit 1; }
import sys
from dataclasses import replace
from bcir.abi import encode, encode_plan
from bcir.gem.execution_plan import plan_from_realization
from bcir.gem.streampack import generation_vector, hydrate
from bcir.kbcir.realize import optimize
from bcir.tests.plan_fixtures import audit_fixture
tmp = sys.argv[1]
module, target, theta, result = audit_fixture()
plan = plan_from_realization(module, result, target, "tokens", plan="plan0")
open(f"{tmp}/plan.bin", "wb").write(encode_plan(plan))
open(f"{tmp}/plan_pack.bin", "wb").write(encode(hydrate(module, result, "plan0")))
# v2 (G5): the same plan carrying schedule-liveness lifetimes from the static memory planner
from bcir.gem.schedule import schedule_plan
from bcir.kbcir.static_memory import plan_static_memory
from bcir.performance_audit import _AuditHardware, static_memory_module
sm = static_memory_module(1)
sm_result = optimize(sm, target, theta)
placement = schedule_plan(sm, sm_result, target, "tokens")
static = plan_static_memory(sm, {rid: "ram" for rid in sm.resources}, _AuditHardware(), schedule=placement)
v2 = plan_from_realization(sm, sm_result, target, "tokens", static_plan=static)
assert encode_plan(v2)[4] == 2
open(f"{tmp}/plan_v2.bin", "wb").write(encode_plan(v2))
open(f"{tmp}/live.txt", "w").write(" ".join(f"{g.rid}:{g.map_gen}:{g.data_gen}" for g in generation_vector(module)))
rid = min(module.resources)
module.resources[rid] = replace(module.resources[rid], map_gen=module.resources[rid].map_gen + 1)
module.touch()
open(f"{tmp}/moved.txt", "w").write(" ".join(f"{g.rid}:{g.map_gen}:{g.data_gen}" for g in generation_vector(module)))
PY
# shellcheck disable=SC2046
plan_out="$("${HARNESS}" "${tmp}/plan.bin" --dump --pack "${tmp}/plan_pack.bin" --live $(cat "${tmp}/live.txt"))" \
  || { echo "  FAIL: C plan decode/verify/pack/vector"; echo "${plan_out}" | tail -5; exit 1; }
printf '%s\n' "${plan_out}" > "${tmp}/plan_dump.txt"
"${PYTHON}" - "${tmp}" <<'PY' || { echo "  FAIL: Python re-encode of the C decode is not byte-identical"; exit 1; }
import sys
from bcir.abi import encode_plan
from bcir.tests.plan_fixtures import parse_c_dump
tmp = sys.argv[1]
original = open(f"{tmp}/plan.bin", "rb").read()
again = encode_plan(parse_c_dump(open(f"{tmp}/plan_dump.txt").read()))
sys.exit(0 if again == original else 1)
PY
echo "  PASS ExecutionPlanV1 parity (Python encode -> C decode -> Python re-encode, byte-identical; plan/pack bound; vector live)"
v2_out="$("${HARNESS}" "${tmp}/plan_v2.bin" --dump)" \
  || { echo "  FAIL: C v2 plan decode/verify"; echo "${v2_out}" | tail -5; exit 1; }
printf '%s\n' "${v2_out}" > "${tmp}/plan_v2_dump.txt"
"${PYTHON}" - "${tmp}" <<'PY' || { echo "  FAIL: Python re-encode of the C v2 decode is not byte-identical"; exit 1; }
import sys
from bcir.abi import encode_plan
from bcir.tests.plan_fixtures import parse_c_dump
tmp = sys.argv[1]
original = open(f"{tmp}/plan_v2.bin", "rb").read()
dump = open(f"{tmp}/plan_v2_dump.txt").read()
assert "header version=2 mode=1 liveness=1" in dump, dump[:200]
sys.exit(0 if encode_plan(parse_c_dump(dump)) == original else 1)
PY
echo "  PASS ExecutionPlanV1 v2 parity (schedule-liveness lifetimes: Python encode -> C decode -> Python re-encode, byte-identical)"
# shellcheck disable=SC2046
if "${HARNESS}" "${tmp}/plan.bin" --live $(cat "${tmp}/moved.txt") > "${tmp}/plan_stale.txt" 2>&1; then
  echo "  FAIL: a plan minted under an older generation vector was accepted on the C rail"; exit 1
fi
grep -q "^vector=BCIR_ERR_STALE$" "${tmp}/plan_stale.txt" \
  && echo "  PASS ExecutionPlanV1 stale vector refused on the C rail (BCIR_ERR_STALE)" \
  || { echo "  FAIL: unexpected stale verdict"; cat "${tmp}/plan_stale.txt"; exit 1; }
# v3 (G8, S5-C): every plan the movement planner mints (bcir/kbcir/movement.py) -- the move tail
# and the source/spec binding trailer -- decodes on the C rail, binds its pack and the live
# registry of the module it realizes (the vector sits before the binding trailer) and re-encodes
# byte for byte; every malformed move and binding (plan_fixtures.v3_variants, the list the tests
# read) is refused on both rails.
"${PYTHON}" - "${tmp}" "${HARNESS}" <<'PY' || { echo "  FAIL: ExecutionPlan v3 (G8) parity"; exit 1; }
import sys
from bcir.abi import encode, encode_plan
from bcir.abi.execution_plan_abi import plan_version
from bcir.gem.streampack import generation_vector, hydrate
from bcir.kbcir.movement import execution_plan_of
from bcir.tests import movement_fixtures as mf
from bcir.tests.plan_fixtures import (
    c_refuses, c_roundtrip, parse_c_dump, run_harness, v3_variants, wire_refuses,
)
tmp, exe = sys.argv[1], sys.argv[2]
h, _theta = mf.target_and_theta()
moved = set()
for name, (_module, _spec, mp) in mf.planned().items():
    plan = execution_plan_of(mp.best, h)
    blob = encode_plan(plan)
    if encode_plan(parse_c_dump(c_roundtrip(exe, tmp, blob))) != blob:
        print(f"  {name}: the C decode does not re-encode byte for byte")
        sys.exit(1)
    mod = mp.best.transform.module
    pack = encode(hydrate(mod, mp.best.result, "plan0"))
    code, out = run_harness(exe, tmp, blob, pack_bytes=pack, live=generation_vector(mod))
    if code != 0 or "pack=BCIR_OK" not in out or "vector=BCIR_OK" not in out:
        print(f"  {name}: the plan/pack/vector binding was refused on the C rail")
        print(out)
        sys.exit(1)
    if plan_version(plan) == 3:
        moved.add(name)
variants = v3_variants()
accepted = [name for name, blob, _plan in variants if not (wire_refuses(blob) and c_refuses(exe, tmp, blob))]
if moved != set(mf.planned()) - mf.IDENTITY or accepted:
    print(f"  v3 plans {sorted(moved)}; malformed variants accepted by a rail: {accepted}")
    sys.exit(1)
print(
    f"  PASS ExecutionPlan v3 (G8): {len(moved)} movement plans Python -> C -> Python byte-identical, "
    f"plan/pack/vector bound; {len(variants)} malformed moves/bindings refused on both rails"
)
PY
