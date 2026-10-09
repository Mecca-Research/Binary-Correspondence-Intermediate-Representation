#!/usr/bin/env bash
# plan_sign: Ed25519 (RFC 8032) + PlanStatementV1 / TrustStoreV1 (G21) on the C rail, against the oracle
#
# One section of tools/c/check_runtime.sh, run against a harness built elsewhere: BCIR Make builds it
# from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake
# project builds it from the same manifest and runs this script as the `c-section-plan_sign` CTest
# entry. tools/build/section_parity.py holds the two builds' outputs byte-identical.
#
#   usage: plan_sign.sh <test_plan_sign>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_plan_sign>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# bcir_ed25519.c (SHA-512 + Ed25519, freestanding) holds RFC 8032 section 7.1's vectors and
# FIPS 180-4's "abc", and refuses an S moved off its one spelling.
out="$("${HARNESS}" vectors)" && [ "${out}" = "vectors=ok" ] \
  || { echo "  FAIL: Ed25519 / SHA-512 vectors on the C rail"; echo "${out}"; exit 1; }
echo "  PASS Ed25519 (RFC 8032 section 7.1) + SHA-512 (FIPS 180-4) vectors on the C rail"

# The twin signs as the oracle signs, and every case of bcir/tests/plan_sign_fixtures.py -- the
# two accepted statements and every forgery of trust, binding, the signed fields, the statement's
# wire and the store's (a small-order key among them) -- draws the oracle's verdict.
"${PYTHON}" - "${tmp}" "${HARNESS}" <<'PY' || { echo "  FAIL: PlanStatementV1 parity (G21)"; exit 1; }
import hashlib
import os
import subprocess
import sys
from bcir.abi import ed25519
from bcir.abi.plan_sign_abi import PlanSignError, decode_store, verify_statement
from bcir.tests import plan_sign_fixtures as fx
tmp, exe = sys.argv[1], sys.argv[2]
def put(name, data):
    path = os.path.join(tmp, name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path
for i in range(16):
    sk = hashlib.sha256(b"parity %d" % i).digest()
    msg = (hashlib.sha512(b"m %d" % i).digest() * 3)[: i * 11]
    got = subprocess.run([exe, "sign", sk.hex(), put("m", msg)], capture_output=True, text=True).stdout.split()
    if got != [ed25519.public_key(sk).hex(), ed25519.sign(sk, msg).hex()]:
        print(f"  signing differs from the oracle on seed {i}")
        sys.exit(1)
cases = fx.cases()
for name, statement, store, plan, pack, now, want in cases:
    try:
        verify_statement(statement, decode_store(store), plan, now, pack_bytes=pack)
        oracle = "ok"
    except PlanSignError as exc:
        oracle = exc.code
    args = [exe, "verify", put("st", statement), put("store", store), put("plan", plan),
            put("pack", pack) if pack is not None else "-", str(now)]
    twin = subprocess.run(args, capture_output=True, text=True).stdout.strip().removeprefix("verdict=")
    if not oracle == twin == want:
        print(f"  {name}: oracle {oracle}, twin {twin}, expected {want}")
        sys.exit(1)
print(f"  PASS PlanStatementV1 (G21): 16 signatures byte-identical to the oracle's; {len(cases)} cases, the same verdict on both rails")
PY
