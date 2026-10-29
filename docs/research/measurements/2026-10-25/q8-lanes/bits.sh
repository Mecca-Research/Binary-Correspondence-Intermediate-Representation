#!/usr/bin/env bash
# Q8-LANES' bit check: the standalone C decoder (bcir-llama) built from BEFORE (the tree before Q8-LANES) with clang,
# and from AFTER with clang and gcc at each instance cap (BCIR_AI_MAX_INSTANCE 4: AVX-512, 2: AVX2, 1: the portable
# loops) and with clang the dispatch compiled out (BCIR_AI_NO_DISPATCH), each -std=c11 -O2 -ffp-contract=off -Wall
# -Wextra -Werror; every build's generated tokens and final logits for each model x prompt x new-token count against
# BEFORE's bytes. One JSON line a case.   usage: bits.sh BEFORE AFTER OUT_DIR MODEL...
set -euo pipefail
before="$1"; after="$2"; out="$3"; shift 3
mkdir -p "${out}"
build() {  # tree cc name [define]
  local c="$1/runtime/c"
  "$2" -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror "${@:4}" -I "${c}" \
    "${c}/bcir_ai_kernels.c" "${c}/bcir_q8_model.c" "${c}/bcir_decode.c" "${c}/bcir_llama.c" \
    "${c}/bcir_llama_cli.c" -o "${out}/$3" -lm
}
build "${before}" clang before
builds=()
for cc in clang gcc; do
  for cap in 4 2 1; do
    build "${after}" "${cc}" "after-${cc}-${cap}" "-DBCIR_AI_MAX_INSTANCE=${cap}"
    builds+=("after-${cc}-${cap}")
  done
done
build "${after}" clang after-clang-nodispatch -DBCIR_AI_NO_DISPATCH
builds+=(after-clang-nodispatch)
for model in "$@"; do
  for prompt in 1,100,200,300,400 1 5,9,13,700,1000,2,3; do
    for new in 1 7 64 200; do
      "${out}/before" --model "${model}" --prompt-ids "${prompt}" --max-new "${new}" --logits-out "${out}/ref.bin" \
        > "${out}/ref.txt"
      ref=$(cat "${out}/ref.bin" "${out}/ref.txt" | sha256sum | cut -d' ' -f1)
      agree=true
      for b in "${builds[@]}"; do
        "${out}/${b}" --model "${model}" --prompt-ids "${prompt}" --max-new "${new}" --logits-out "${out}/t.bin" \
          > "${out}/t.txt"
        got=$(cat "${out}/t.bin" "${out}/t.txt" | sha256sum | cut -d' ' -f1)
        [ "${got}" = "${ref}" ] || agree=false
      done
      printf '{"model_sha256":"%s","prompt":"%s","new_tokens":%s,"tokens_and_logits_sha256":"%s","builds":%d,"every_build_agrees":%s}\n' \
        "$(sha256sum "${model}" | cut -c1-16)" "${prompt}" "${new}" "${ref}" "${#builds[@]}" "${agree}"
    done
  done
done
