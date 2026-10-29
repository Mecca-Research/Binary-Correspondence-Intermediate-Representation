#!/usr/bin/env bash
# CRC-SLICE's qualification, repeated: run_qualification.py from each tree (HEAD~1 the tree before, HEAD CRC-SLICE)
# three times in alternation without callgrind, each run's C latency rows -- whether the monolithic runner's and the
# generated token's figures move between the trees or between runs. Run from <REPO> after measure_crc.sh;
# <SCRATCH> and <LAB> as there. One JSON line a run.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/crc-slice-repeat"; lab="<LAB>"
rm -rf "${s}"; mkdir -p "${s}"
for t in before after; do
  git -C "${repo}" worktree add --detach "${s}/${t}" "$([ "${t}" = before ] && echo HEAD~1 || echo HEAD)" >/dev/null
done
for run in 1 2 3; do
  for t in $([ $((run % 2)) = 1 ] && echo "before after" || echo "after before"); do
    (cd "${s}/${t}" && python3 tools/models/run_qualification.py --model-dir "${lab}" --output "${s}/q-${t}-${run}.json" \
      --no-callgrind > /dev/null)
    python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); c = d["latency"]["c"]; print(json.dumps({"tree": sys.argv[2], "run": int(sys.argv[3]), "commit": d["commit"], "status": d["status"], "load_ms": round(c["load_ms"], 3), "monolithic_median_ms": round(c["monolithic_median_ms"], 3), "gem_median_ms": round(c["gem_median_ms"], 3), "gem_over_monolithic_median": round(c["gem_over_monolithic_median"], 4), "per_token_ms_median": round(c["per_token_ms"]["median"], 4), "trust_boundary_ms": round(c["trust_boundary_ms"], 3)}))' \
      "${s}/q-${t}-${run}.json" "${t}" "${run}"
  done
done
for t in before after; do git -C "${repo}" worktree remove --force "${s}/${t}"; done
