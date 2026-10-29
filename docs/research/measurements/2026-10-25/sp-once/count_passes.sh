#!/usr/bin/env bash
# SP-ONCE's pass counts, run from <REPO> on the committed tree -- HEAD SP-ONCE, HEAD~1 the tree before -- with nothing
# else running; <SCRATCH> a scratch directory and <LAB> the qualification's lab model (BCIR-Corpus-860K). Each tree's
# units are built at -O0 -fno-inline (every call a call) and each entry point is run under callgrind with a profile
# dumped before and after each of its calls, so a part holds one call alone: bcir_sp_execute, bcir_dgem_load and
# bcir_dgem_run on the qualification's pack (count_passes.c), and bcir_decoder_lower_pack on the pack fuzz harness's
# first seed (replay_pack.c over fuzz_decoder_pack.c, four lowerings). (A whole-run profile with collection toggled on
# inside the call is not used: callgrind counts an edge's calls made before the toggle too, where costs it does not.)
# Writes passes.json beside itself: each call's CRC-32 calls, by caller.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/sp-once-passes"; lab="<LAB>"
here="${repo}/docs/research/measurements/2026-10-25/sp-once"
crc="${repo}/docs/research/measurements/2026-10-25/crc-slice"
rm -rf "${s}"; mkdir -p "${s}"
for t in before after; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo HEAD~1 || echo HEAD)" >/dev/null
done
python3 "${crc}/dump_pack.py" "${s}/after" "${lab}" "${s}/pack.bin"
(cd "${s}/after" && python3 - "${s}/seed.bin" <<'PY'
import sys
from bcir.tests.native_decoder_fixtures import fuzz_pack_seeds
name, data, admitted = fuzz_pack_seeds()[0]
assert name == "c0_1x1" and admitted, name
open(sys.argv[1], "wb").write(data)
PY
)
gem=(bcir_runtime bcir_exec bcir_decoder_gem bcir_llama bcir_q8_model bcir_q4_kernel bcir_ai_kernels bcir_decode)
low=(bcir_decoder_pack bcir_decoder_train bcir_tensor bcir_runtime bcir_exec)
for t in before after; do
  C="${s}/${t}/runtime/c"
  srcs=(); for u in "${gem[@]}"; do srcs+=("${C}/${u}.c"); done
  clang -std=c11 -O0 -fno-inline -ffp-contract=off -Wall -Wextra -Werror -I "${C}" "${here}/count_passes.c" \
    "${srcs[@]}" -o "${s}/count-${t}" -lm
  srcs=(); for u in "${low[@]}"; do srcs+=("${C}/${u}.c"); done
  clang -std=c23 -O0 -fno-inline -Wall -Wextra -I "${C}" "${here}/replay_pack.c" "${C}/fuzz_decoder_pack.c" \
    "${srcs[@]}" -o "${s}/replay-${t}" -lm -lpthread
done
cg() { valgrind --tool=callgrind --dump-before="$1" --dump-after="$1" --callgrind-out-file="$2" "${@:3}" \
  > "$2.log" 2>&1; }
for t in before after; do
  for mode in execute load run; do
    entry="$(case "${mode}" in execute) echo bcir_sp_execute;; load) echo bcir_dgem_load;; run) echo bcir_dgem_run;; esac)"
    cg "${entry}" "${s}/cg-${t}-${mode}.out" "${s}/count-${t}" "${mode}" "${lab}/model.bcirq8" "${s}/pack.bin" \
      "$(cat "${s}/pack.bin.ids")"
  done
  cg bcir_decoder_lower_pack "${s}/cg-${t}-lower.out" "${s}/replay-${t}" "${s}/seed.bin"
done
python3 - "${s}" "${here}" <<'PY'
import collections, hashlib, json, re, sys
from pathlib import Path
s, here = Path(sys.argv[1]), Path(sys.argv[2])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def edges(path):
    """one profile part's call counts, (caller, callee) -> calls, from callgrind's own output (fn= / cfn= / calls=)."""
    names, calls, fn, cfn = {}, collections.Counter(), None, None
    def name(tok):
        m = re.match(r"\((\d+)\)(?:\s+(.*))?$", tok.strip())
        if not m:
            return tok.strip()
        if m.group(2):
            names[m.group(1)] = m.group(2)
        return names[m.group(1)]
    for line in path.read_text().splitlines():
        if line.startswith("fn="):
            fn = name(line[3:])
        elif line.startswith("cfn="):
            cfn = name(line[4:])
        elif line.startswith("calls="):
            calls[(fn, cfn)] += int(line[6:].split()[0])
    return calls

rows = []
for t in ("before", "after"):
    for case, entry in (("execute", "bcir_sp_execute"), ("load", "bcir_dgem_load"), ("run", "bcir_dgem_run"),
                        ("lower", "bcir_decoder_lower_pack")):
        parts = [p for p in sorted(s.glob(f"cg-{t}-{case}.out*"))
                 if not p.name.endswith(".log") and f"desc: Trigger: --dump-after={entry}\n" in p.read_text()]
        each = [{caller: n for (caller, callee), n in sorted(edges(p).items()) if callee == "bcir_crc32"}
                for p in parts]
        assert parts and all(c == each[0] for c in each), (t, case, each)
        rows.append({"tree": t, "entry": entry, "calls": len(parts), "crc32_by_caller_each_call": each[0],
                     "pack_passes_a_call": each[0].get("bcir_sp_validate", 0),
                     "plan_passes_a_call": each[0].get("bcir_ep_validate", 0)})
record = {
    "tools": {f: sha(here / f) for f in ("count_passes.sh", "count_passes.c", "replay_pack.c")},
    "inputs": {"pack.bin": sha(s / "pack.bin"), "seed.bin (fuzz_pack_seeds' c0_1x1)": sha(s / "seed.bin")},
    "sources": {f"{t}/{u}": sha(s / t / "runtime" / "c" / u) for t in ("before", "after")
                for u in ("bcir_runtime.c", "bcir_exec.c", "bcir_decoder_gem.c", "bcir_decoder_pack.c",
                          "fuzz_decoder_pack.c")},
    "rows": rows,
}
(here / "passes.json").write_text(json.dumps(record, indent=1) + "\n")
for r in rows:
    print(r["tree"], r["entry"], r["calls"], r["crc32_by_caller_each_call"])
PY
for t in before after; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
