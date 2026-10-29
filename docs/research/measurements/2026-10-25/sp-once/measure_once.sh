#!/usr/bin/env bash
# SP-ONCE's records, run from <REPO> on the committed tree -- HEAD SP-ONCE, HEAD~1 the tree before -- one step at a time
# with nothing else running; <SCRATCH> a scratch directory and <LAB> the qualification's lab model (BCIR-Corpus-860K).
# Writes once.json, qualification.json, qualification-repeats.json and hashes.json beside itself. The pack is the
# qualification's own (CRC-SLICE's dump_pack.py, ../crc-slice/, as committed), its rows CRC-SLICE's qual_crc.py.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/sp-once"; lab="<LAB>"
here="${repo}/docs/research/measurements/2026-10-25/sp-once"
crc="${repo}/docs/research/measurements/2026-10-25/crc-slice"
rm -rf "${s}"; mkdir -p "${s}"
for t in before after; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo HEAD~1 || echo HEAD)" >/dev/null
done
B="${s}/before/runtime/c"; A="${s}/after/runtime/c"
flags=(-std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror)
units=(bcir_runtime bcir_exec bcir_decoder_gem bcir_llama bcir_q8_model bcir_q4_kernel bcir_ai_kernels bcir_decode)
lines() { python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' "$1"; }

# the qualification's native pack and its prompt (the same bytes on both trees: the program, plan and hydration)
python3 "${crc}/dump_pack.py" "${s}/after" "${lab}" "${s}/pack.bin"

# once.json: once_ab.c -- the tree before's GEM units, every global renamed before_*, against the tree after's
for cc in clang gcc; do
  objs=()
  for u in "${units[@]}"; do
    "${cc}" "${flags[@]}" -I "${B}" -c "${B}/${u}.c" -o "${s}/b-${u}-${cc}.o"
    objs+=("${s}/b-${u}-${cc}.o")
  done
  nm --defined-only -g "${objs[@]}" | awk 'NF == 3 {print $3 " before_" $3}' | sort -u > "${s}/renames-${cc}.txt"
  renamed=()
  for o in "${objs[@]}"; do
    objcopy --redefine-syms="${s}/renames-${cc}.txt" "${o}" "${o%.o}-r.o"
    renamed+=("${o%.o}-r.o")
  done
  srcs=()
  for u in "${units[@]}"; do srcs+=("${A}/${u}.c"); done
  "${cc}" "${flags[@]}" -I "${A}" "${here}/once_ab.c" "${srcs[@]}" "${renamed[@]}" -o "${s}/once_ab-${cc}" -lm
done
for cc in clang gcc; do
  "${s}/once_ab-${cc}" 15 "${lab}/model.bcirq8" "${s}/pack.bin" "$(cat "${s}/pack.bin.ids")" | sed "s/^{/{\"cc\":\"${cc}\",/"
done | lines "${here}/once.json"

# qualification.json: run_qualification.py from each tree, its traffic under callgrind
for t in before after; do
  (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" --output "${s}/qual-${t}.json" \
    --require-callgrind > "${s}/qual-${t}.log")
done
python3 "${crc}/qual_crc.py" "${s}/qual-before.json" "${s}/qual-after.json" > "${here}/qualification.json"

# qualification-repeats.json: three runs of each tree in alternation without callgrind, each run's C latency rows
for run in 1 2 3; do
  for t in $([ $((run % 2)) = 1 ] && echo "before after" || echo "after before"); do
    (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" --output "${s}/q-${t}-${run}.json" \
      --no-callgrind > /dev/null)
    python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); c = d["latency"]["c"]; print(json.dumps({"tree": sys.argv[2], "run": int(sys.argv[3]), "commit": d["commit"], "status": d["status"], "load_ms": round(c["load_ms"], 3), "trust_boundary_ms": round(c["trust_boundary_ms"], 3), "monolithic_median_ms": round(c["monolithic_median_ms"], 3), "gem_median_ms": round(c["gem_median_ms"], 3), "gem_overhead_ms": round(c["gem_overhead_ms"], 3), "gem_over_monolithic_median": round(c["gem_over_monolithic_median"], 4), "time_to_first_token_ms": round(c["time_to_first_token_ms"], 3)}))' \
      "${s}/q-${t}-${run}.json" "${t}" "${run}"
  done
done | lines "${here}/qualification-repeats.json"

# hashes.json: the tools' sha256 (this directory's and those run from CRC-SLICE's) and the sources'
python3 - "${s}" "${here}" "${crc}" <<'PY'
import hashlib, json, sys
from pathlib import Path
s, here, crc = (Path(a) for a in sys.argv[1:4])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {
    "tools": {t: sha(here / t) for t in ("measure_once.sh", "once_ab.c")},
    "tools run from ../crc-slice": {t: sha(crc / t) for t in ("dump_pack.py", "qual_crc.py")},
    "sources": {f"{t}/{u}": sha(s / t / "runtime" / "c" / u)
                for t in ("before", "after")
                for u in ("bcir_runtime.c", "bcir_runtime.h", "bcir_exec.c", "bcir_decoder_gem.c", "bcir_qualify_cli.c")},
    "pack.bin": sha(s / "pack.bin"),
}
(here / "hashes.json").write_text(json.dumps(record, indent=1) + "\n")
PY
for t in before after; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
