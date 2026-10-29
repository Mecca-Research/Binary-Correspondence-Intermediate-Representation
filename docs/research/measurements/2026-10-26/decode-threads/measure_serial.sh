#!/usr/bin/env bash
# DECODE-THREADS' serial records, run after measure_threads.sh from <REPO> on the committed tree -- HEAD
# DECODE-THREADS with each vector kernel's whole call an instance of its own, ef6717a DECODE-THREADS as first
# committed (one kernel for the whole call and a range, its column range a variable) and 4bac89e the tree before (the
# paste builder's bounds, on HEAD-PACK's records) -- one step at a time with nothing else running; <SCRATCH> a scratch
# directory and <LAB> the qualification's lab model (BCIR-Corpus-860K). What the column ranges cost a decode lent no
# runner, every operation one call as before: the three trees' kernel units in one binary (kernels_ab.c) and their
# whole-call kernels' instructions, with 0c24748's (kernel_code.py); Q8-LANES' decode A/B
# (../../2026-10-25/q8-lanes/decode_ab.py, as committed) between the three trees' bcir-llama; and the qualification
# run from each tree six times, each tree twice in each place of the order. The A/Bs build each tree twice, as the
# decoder's builds do and with every function on a 64-byte boundary (-falign-functions=64): on the session's second
# host the same instructions read up to x1.23 apart at the compilers' own placements, so the aligned builds are where
# two trees differ in their code alone. Writes kernels.json, serial.json, qualification-repeats.json and
# hashes-serial.json beside itself.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/decode-threads-serial"; lab="<LAB>"
before=4bac89e66d708b6963cac110e1fc999372bbb44d; first=ef6717a8ef4dcf77209fc159cde396ea52af2dd0
inlined=0c247486c6e1077f4d3859acfa6c13fc45ca02fc  # its whole-call kernels one body inlined into both instances
here="${repo}/docs/research/measurements/2026-10-26/decode-threads"
lanes="${repo}/docs/research/measurements/2026-10-25/q8-lanes"
rm -rf "${s:?}"; mkdir -p "${s}"
git -C "${repo}" worktree add --detach "${s}/before" "${before}" >/dev/null
git -C "${repo}" worktree add --detach "${s}/first" "${first}" >/dev/null
git -C "${repo}" worktree add --detach "${s}/after" HEAD >/dev/null
git -C "${repo}" worktree add --detach "${s}/inlined" "${inlined}" >/dev/null
flags=(-std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror)

# kernels.json: each tree's bcir_ai_kernels.c compiled as the decoder's builds compile it, as they place its
# functions and with every function on a 64-byte boundary, its exported symbols renamed b_, f_ and a_ (objcopy) and
# linked into kernels_ab.c -- each compiler and placement, on the widest instance and on AVX2, 101 rounds -- and
# kernel_code.py over the three objects as the decoder's builds compile them
for cc in clang gcc; do
  for place in default aligned; do
    extra=(); a=""
    if [ "${place}" = aligned ]; then extra=(-falign-functions=64); a=-a64; fi
    for t in before first after; do
      o="${s}/kernels-${t}-${cc}${a}"
      "${cc}" "${flags[@]}" "${extra[@]}" -I "${s}/${t}/runtime/c" -c "${s}/${t}/runtime/c/bcir_ai_kernels.c" \
        -o "${o}.o"
      nm -g --defined-only "${o}.o" | awk -v p="${t:0:1}_" '{print $3, p $3}' > "${o}.map"
      objcopy --redefine-syms="${o}.map" "${o}.o" "${o}-renamed.o"
    done
    "${cc}" "${flags[@]}" "${here}/kernels_ab.c" "${s}/kernels-before-${cc}${a}-renamed.o" \
      "${s}/kernels-first-${cc}${a}-renamed.o" "${s}/kernels-after-${cc}${a}-renamed.o" -o "${s}/kernels_ab-${cc}${a}" -lm
    for instance in 4 2; do
      "${s}/kernels_ab-${cc}${a}" "${instance}" 101 | sed "s/^{/{\"cc\":\"${cc}\",\"place\":\"${place}\",/"
    done
  done
done > "${s}/kernels-timing.jsonl"
for cc in clang gcc; do
  python3 "${here}/kernel_code.py" objdump "${s}/kernels-before-${cc}.o" "${s}/kernels-first-${cc}.o" \
    "${s}/kernels-after-${cc}.o" | sed "s/^{/{\"cc\":\"${cc}\",/"
done > "${s}/kernels-code.jsonl"
# and 0c24748's kernels in the after's place: its whole-call kernels against the tree before's
for cc in clang gcc; do
  "${cc}" "${flags[@]}" -I "${s}/inlined/runtime/c" -c "${s}/inlined/runtime/c/bcir_ai_kernels.c" \
    -o "${s}/kernels-inlined-${cc}.o"
  python3 "${here}/kernel_code.py" objdump "${s}/kernels-before-${cc}.o" "${s}/kernels-first-${cc}.o" \
    "${s}/kernels-inlined-${cc}.o" | sed "s/^{/{\"cc\":\"${cc}\",/"
done > "${s}/kernels-code-inlined.jsonl"
python3 -c 'import json, sys; json.dump({"timing": [json.loads(l) for l in open(sys.argv[1])], "code": [json.loads(l) for l in open(sys.argv[2])], "code_0c24748": [json.loads(l) for l in open(sys.argv[3])]}, open(sys.argv[4], "w"), indent=1)' \
  "${s}/kernels-timing.jsonl" "${s}/kernels-code.jsonl" "${s}/kernels-code-inlined.jsonl" "${here}/kernels.json"

# serial.json: decode_ab.py -- bcir-llama from each tree (the same build of each: the trees after's CLI links the
# trainer's pool, which the tree before's links unused), each compiler, 24 rounds, on the widest instance and with
# every tree capped at AVX2, as the decoder's builds place their functions and with every function aligned; none
# lent a runner
llama() {  # tree cc name [define]
  local c="$1/runtime/c"
  "$2" "${flags[@]}" "${@:4}" -I "${c}" "${c}/bcir_ai_kernels.c" "${c}/bcir_q8_model.c" "${c}/bcir_decode.c" \
    "${c}/bcir_llama.c" "${c}/bcir_llama_cli.c" "${c}/bcir_tensor.c" -o "${s}/$3" -lm -pthread
}
for cc in clang gcc; do
  for t in before first after; do
    llama "${s}/${t}" "${cc}" "llama-${t}-${cc}"
    llama "${s}/${t}" "${cc}" "llama-${t}-avx2-${cc}" -DBCIR_AI_MAX_INSTANCE=2
    llama "${s}/${t}" "${cc}" "llama-${t}-a64-${cc}" -falign-functions=64
    llama "${s}/${t}" "${cc}" "llama-${t}-a64-avx2-${cc}" -DBCIR_AI_MAX_INSTANCE=2 -falign-functions=64
  done
  for v in "" "avx2-" "a64-" "a64-avx2-"; do
    python3 "${lanes}/decode_ab.py" 24 "${lab}/model.bcirq8" "${s}/decode-${v}${cc}.json" \
      "before${v:+-${v%-}}=${s}/llama-before-${v}${cc}" "first${v:+-${v%-}}=${s}/llama-first-${v}${cc}" \
      "after${v:+-${v%-}}=${s}/llama-after-${v}${cc}" > "${s}/decode-${v}${cc}.log"
  done
done
python3 -c 'import json, sys; json.dump({f"{cc} {w}": json.load(open(f"{sys.argv[1]}/decode-{x}{cc}.json")) for cc in ("clang", "gcc") for w, x in (("widest", ""), ("avx2", "avx2-"), ("widest aligned", "a64-"), ("avx2 aligned", "a64-avx2-"))}, open(sys.argv[2], "w"), indent=1)' \
  "${s}" "${here}/serial.json"

# qualification-repeats.json: six runs of each tree without callgrind, the order rotated a run, each run's C latency
# rows
for run in 1 2 3 4 5 6; do
  case "$(( (run - 1) % 3 ))" in
    0) order="before first after" ;;
    1) order="first after before" ;;
    *) order="after before first" ;;
  esac
  for t in ${order}; do
    (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" \
      --output "${s}/q-${t}-${run}.json" --no-callgrind > /dev/null)
    python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); c = d["latency"]["c"]; o = c["per_operation_ms"]; print(json.dumps({"tree": sys.argv[2], "run": int(sys.argv[3]), "commit": d["commit"], "status": d["status"], "per_token_ms_median": round(c["per_token_ms"]["median"], 4), "tokens_per_second": round(c["tokens_per_second"], 1), "time_to_first_token_ms": round(c["time_to_first_token_ms"], 3), "monolithic_median_ms": round(c["monolithic_median_ms"], 3), "gem_median_ms": round(c["gem_median_ms"], 3), "gem_over_monolithic_median": round(c["gem_over_monolithic_median"], 4), "head_ms": round(o["dec.head"], 3), "matvec_ms": round(o["dec.matvec"], 3), "matvec_add_ms": round(o["dec.matvec_add"], 3)}))' \
      "${s}/q-${t}-${run}.json" "${t}" "${run}"
  done
done | python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' \
  "${here}/qualification-repeats.json"

# hashes-serial.json: this script's sha256 and its tools', decode_ab.py's, and the sources each tree's builds read
python3 - "${s}" "${here}" "${lanes}" <<'PY'
import hashlib, json, sys
from pathlib import Path
s, here, lanes = (Path(a) for a in sys.argv[1:4])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
json.dump({"tools": {t: sha(here / t) for t in ("measure_serial.sh", "kernels_ab.c", "kernel_code.py")},
           "tools run from ../../2026-10-25/q8-lanes": {"decode_ab.py": sha(lanes / "decode_ab.py")},
           "sources": {f"{t}/runtime/c/{u}": sha(s / t / "runtime/c" / u)
                       for t in ("before", "first", "after", "inlined")
                       for u in ("bcir_ai_kernels.c", "bcir_ai_kernels.h", "bcir_q8_model.c", "bcir_decode.c",
                                 "bcir_llama.c", "bcir_llama_cli.c", "bcir_tensor.c")}},
          open(here / "hashes-serial.json", "w"), indent=1)
PY
for t in before first after inlined; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
git -C "${repo}" worktree prune
echo "serial records written"
