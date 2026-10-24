#!/bin/bash
# POOL-AFFINITY's steady-state reading, declared (2026-10-24, 14:55Z) after measure_aff.sh's one-process rows and before
# any of these ran: the one-process A/B runs one step an arm a round, so each arm's step starts after the two other
# arms' steps have flushed the caches, where training runs its steps back to back on one pool. Here each library runs
# alone in its own process, STEPS steps back to back after WARM untimed ones (step_one.c; fwd_one.c for the inference
# forward), the libraries alternating run by run, six runs each: the tree before (A), the tree after (B) and B's two
# design variants (nohead, nosteal; proto_aff_variants.py, built by measure_aff.sh). Each run prints its weights'
# digest (the step) or its last forward's logits' digest (the forward), which must agree across the libraries.
# Last, a control of the GEMM rig (ab_gemm.c), declared after section 7 and before this script ran.
set -o pipefail
S=<SCRATCH>
R=<REPO>
C=$S/next0/aff; O=$C/final; T=$S/next0/tools
export OMP_WAIT_POLICY=PASSIVE
cd $R || exit 2
echo "== $(date -u +%H:%M:%S) before=$(git rev-parse HEAD~1) after=$(git rev-parse HEAD) dirty=$(git status --short | wc -l)"
for v in A B nohead nosteal; do
  [ -f $O/bin/libdec_clang_$v.so ] || { echo "missing libdec_clang_$v.so (measure_aff.sh builds it)"; exit 1; }
done
clang -std=c11 -O2 -I $C/srcB $T/step_one.c -o $O/bin/step_one -ldl -lm || exit 1
clang -std=c11 -O2 -I $C/srcB $T/fwd_one.c -o $O/bin/fwd_one -ldl -lm || exit 1
echo "== $(date -u +%H:%M:%S) the training step, one library a process"
: > $O/one-step.jsonl
for tt in 128 256 512 1024; do
  steps=40; [ $tt = 1024 ] && steps=20
  for k in 1 2 3 4 5 6; do for v in A B nohead nosteal; do
    $O/bin/step_one $O/bin/libdec_clang_$v.so 260 128 4 4 3 256 4 $tt 14 4 $steps 4 \
      | sed "s/^{/{\"build\":\"$v\",\"run\":$k,/" >> $O/one-step.jsonl || echo "step_one $v T$tt FAILED"
  done; done
done
echo "== $(date -u +%H:%M:%S) the inference forward, one library a process"
: > $O/one-forward.jsonl
for spec in "4 256 60" "4 1024 30" "1 128 200" "1 256 120"; do
  set -- $spec
  for k in 1 2 3 4 5 6; do for v in A B; do
    $O/bin/fwd_one $O/bin/libdec_clang_$v.so 260 128 4 4 3 256 $1 $2 14 4 $3 4 \
      | sed "s/^{/{\"build\":\"$v\",\"run\":$k,/" >> $O/one-forward.jsonl || echo "fwd_one $v B$1 T$2 FAILED"
  done; done
done
echo "== $(date -u +%H:%M:%S) the GEMM rig's control: ab_gemm.c with the tree after against a byte copy of itself"
# Declared (15:27Z, after measure_aff.sh's section 7 read the nohead variant x1.361 on a single GEMM whose code path
# it does not change): two libraries, two pools, one code, so any ratio away from 1 is the rig's own (the second
# library's arm always follows an arm on the other library's pool).
cp $O/bin/libtensor_clang_B.so $O/bin/libtensor_clang_Bcopy.so || exit 1
: > $O/rig-control.jsonl
for M in 1024 2048 8192; do
  $O/bin/ab_gemm $O/bin/libtensor_clang_B.so $O/bin/libtensor_clang_Bcopy.so 32 4 5 \
    g:0,1,$M,128,128 g:1,0,128,128,$M p:0,0,$M,128,128:1,0,128,128,$M c3:0,0,$M,128,128:3x1,0,128,128,$M \
    >> $O/rig-control.jsonl || echo "ab_gemm control $M FAILED"
done
echo "== $(date -u +%H:%M:%S) done dirty=$(git status --short | wc -l)"
