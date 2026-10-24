#!/bin/bash
# NEXT-0's survey (before POOL-AFFINITY, the tree at HEAD~1) and the prototypes measured and not taken, serially, as
# records: every tool here is in this directory, every build -std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread.
#  s1 parts      -- ../../2026-10-23/bwd-keep/step_parts.py at T256, T512 and T1024 on four threads and T512 on one
#  s2 forward    -- the banded stored forward a phase at a time in place (instrument_fwd.py, fwd_bench.c) and its group
#                   kernels on cache-resident operands (fwd_micro.c) and on a cold band (fwd_cold.c: in place, or computed
#                   in a resident scratch and copied out with non-temporal or ordinary stores)
#  s3 pool       -- the pool's own cost a job and the step's elementwise jobs (pool_cost.c); RMSNorm, its VJP and the
#                   residual add warm, cold and after a pool job wrote their operands (ew_cold.c); a GEMM's calls back to
#                   back against calls after a 1 ms pause (gemm_b2b.c)
#  s4 prototypes -- EXP8 (proto_exp8.py: the softmax eight exponentials at once, fwd_micro.c); FWD-BANDW-0 (proto_fwd.py:
#                   the next band prefetched for writing, or the group streamed out; ../p-bands/ab_bands.c); BWD-SWEEP-0
#                   (proto_bwd.py: dV and dK inside the groups' sweep) and BWD-PF-0 (proto_pf.py: P's lines prefetched
#                   in the banded walks), each A/B'd and timed a phase at a time (instrument_sweep.py, bwd_phase.c);
#                   WADJ-TILE-0 (proto_wadj.py: the weight adjoints' rows in ranges of 64; ../../2026-10-23/mm-rows/ab_gemm.c);
#                   POOL-YIELDS-0 (the pool's workers yielding 200000 rounds before they sleep; step_one.c, alternating
#                   processes)
set -o pipefail
S=<SCRATCH>
R=<REPO>
C=$S/next0/survey; O=$C/out; T=$S/next0/tools
D3=$R/docs/research/measurements/2026-10-23; D4=$R/docs/research/measurements/2026-10-24
export OMP_WAIT_POLICY=PASSIVE
cd $R || exit 2
A=$(git rev-parse HEAD~1)
echo "== $(date -u +%H:%M:%S) before=$A"
rm -rf $C; mkdir -p $C/src $C/srcd $O $C/bin
git show $A:runtime/c/bcir_tensor.c > $C/src/bcir_tensor_tree.c; git show $A:runtime/c/bcir_tensor.h > $C/src/bcir_tensor.h
for f in bcir_tensor.c bcir_tensor.h bcir_decoder_train.c bcir_decoder_train.h; do git show $A:runtime/c/$f > $C/srcd/$f; done
CF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread"
echo "== $(date -u +%H:%M:%S) s1 parts (the tree before, in a detached worktree)"
rm -rf $C/wt; git worktree add --detach $C/wt $A > /dev/null 2>&1 || exit 1
for spec in "256 4" "512 4" "1024 4" "512 1"; do set -- $spec
  python3 -I $C/wt/docs/research/measurements/2026-10-23/bwd-keep/step_parts.py $1 kblock+exp+reduce 12 $O/parts-T$1-t$2.json $2 > /dev/null 2>&1; echo "parts T$1 threads $2 exit $?"
done
git worktree remove --force $C/wt; git worktree prune
echo "== $(date -u +%H:%M:%S) s2 forward"
python3 -I $T/instrument_fwd.py $C/src/bcir_tensor_tree.c $C/src/bcir_tensor_fw.c > /dev/null || exit 1
clang $CF -I $C/src $T/fwd_bench.c $C/src/bcir_tensor_fw.c -o $C/bin/fwd_bench -lm || exit 1
clang $CF -D_GNU_SOURCE -I $C/src $T/fwd_micro.c -o $C/bin/fwd_micro -lm || exit 1
clang $CF -D_GNU_SOURCE -I $C/src $T/fwd_cold.c -o $C/bin/fwd_cold -lm || exit 1
: > $O/forward-phases.jsonl
for rep in 1 2 3; do for spec in "4 2048 4 4 32 4 16 49" "4 2048 4 4 32 1 4 49" "16 2048 1 1 32 4 12 49" "4 1024 4 4 32 4 32 49"; do
  $C/bin/fwd_bench $spec >> $O/forward-phases.jsonl
done; done
: > $O/forward-resident.jsonl
for R0 in 256 1024 2032; do for th in 1 4; do $C/bin/fwd_micro $R0 32 $th 400 >> $O/forward-resident.jsonl; done; done
: > $O/forward-cold.jsonl
for rep in 1 2; do for mode in 0 1 2 3; do $C/bin/fwd_cold 1024 32 4 1500 $mode 64 >> $O/forward-cold.jsonl; done; done
echo "== $(date -u +%H:%M:%S) s3 pool"
clang $CF -I $C/src $T/pool_cost.c $C/src/bcir_tensor_tree.c -o $C/bin/pool_cost -lm || exit 1
clang $CF -I $C/src $T/ew_cold.c $C/src/bcir_tensor_tree.c -o $C/bin/ew_cold -lm || exit 1
clang $CF -I $C/src $T/gemm_b2b.c $C/src/bcir_tensor_tree.c -o $C/bin/gemm_b2b -lm || exit 1
: > $O/pool-cost.txt
for th in 1 4; do $C/bin/pool_cost $th 16 200 9 2000 2048 128 >> $O/pool-cost.txt; done
$C/bin/pool_cost 4 128 200 9 500 2048 128 >> $O/pool-cost.txt; $C/bin/pool_cost 4 4 200 9 8000 2048 128 >> $O/pool-cost.txt
: > $O/elementwise-cold.txt
for th in 1 4; do for rows in 2048 8192; do $C/bin/ew_cold $th $rows 128 15 >> $O/elementwise-cold.txt; done; done
: > $O/gemm-gap.txt
for th in 1 4; do for shape in "0 1 2048 128 128" "1 0 128 128 2048" "0 0 2048 128 128" "0 1 1024 128 128" "0 1 8192 128 128"; do
  $C/bin/gemm_b2b $th $shape 50 9 1000 >> $O/gemm-gap.txt
done; done
echo "== $(date -u +%H:%M:%S) s4 prototypes"
python3 -I $T/proto_exp8.py $C/src/bcir_tensor_tree.c $C/src/bcir_tensor_exp8.c > /dev/null || exit 1
sed 's/bcir_tensor_tree.c/bcir_tensor_exp8.c/' $T/fwd_micro.c > $C/src/fwd_micro8.c
for w in 4 8; do clang $CF -D_GNU_SOURCE -DEXPW=$w -I $C/src $C/src/fwd_micro8.c -o $C/bin/fwd_micro_w$w -lm || exit 1; done
: > $O/exp8.jsonl
for rep in 1 2 3; do for w in 4 8; do for R0 in 1024 2032; do
  $C/bin/fwd_micro_w$w $R0 32 4 400 | sed "s/^{/{\"width\":$w,/" >> $O/exp8.jsonl
done; done; done
python3 -I $T/proto_fwd.py $C/src/bcir_tensor_tree.c $C/src/bcir_tensor_pfw.c $C/src/bcir_tensor_nt.c > /dev/null || exit 1
python3 -I $T/proto_bwd.py $C/src/bcir_tensor_tree.c $C/src/bcir_tensor_sweep.c > /dev/null || exit 1
for m in walk group both; do python3 -I $T/proto_pf.py $C/src/bcir_tensor_tree.c $C/src/bcir_tensor_pf_$m.c $m > /dev/null || exit 1; done
python3 -I $T/proto_wadj.py $C/src/bcir_tensor_tree.c $C/src/bcir_tensor_wadj.c > /dev/null || exit 1
for v in tree pfw nt sweep pf_walk pf_group pf_both; do clang $CF -fPIC -shared -I $C/src $C/src/bcir_tensor_$v.c -o $C/bin/lib_$v.so -lm || exit 1; done
clang $CF -DWADJ_RR=64 -fPIC -shared -I $C/src $C/src/bcir_tensor_wadj.c -o $C/bin/lib_wadj64.so -lm || exit 1
clang -std=c11 -O2 $D4/p-bands/ab_bands.c -o $C/bin/ab_bands -ldl -lm || exit 1
clang -std=c11 -O2 $D3/mm-rows/ab_gemm.c -o $C/bin/ab_gemm -ldl -lm || exit 1
: > $O/prototypes-attention.jsonl
for v in pfw nt sweep pf_walk; do
  $C/bin/ab_bands $C/bin/lib_tree.so $C/bin/lib_$v.so 32 4 2048 4 4 32 4 49 | sed "s/^{/{\"proto\":\"$v\",/" >> $O/prototypes-attention.jsonl || echo "ab_bands $v FAILED"
done
for v in tree sweep pf_walk pf_group pf_both; do
  python3 -I $T/instrument_sweep.py $([ $v = sweep ] && echo sweep || echo tree) $C/src/bcir_tensor_$v.c $C/src/inst_$v.c > /dev/null || exit 1
  clang $CF -I $C/src $T/bwd_phase.c $C/src/inst_$v.c -o $C/bin/bph_$v -lm || exit 1
done
: > $O/prototypes-phases.jsonl
for rep in 1 2 3; do for v in tree sweep pf_walk pf_group pf_both; do
  $C/bin/bph_$v 4 2048 4 4 32 4 8 49 | sed "s/^{/{\"proto\":\"$v\",/" >> $O/prototypes-phases.jsonl
done; done
: > $O/prototypes-wadj.jsonl
for M in 1024 2048 8192; do
  $C/bin/ab_gemm $C/bin/lib_tree.so $C/bin/lib_wadj64.so 16 4 5 p:0,0,$M,128,128:1,0,128,128,$M c3:0,0,$M,128,128:3x1,0,128,128,$M g:1,0,128,128,$M g:1,0,260,128,$M >> $O/prototypes-wadj.jsonl
done
clang $CF -fPIC -shared -I $C/srcd $C/srcd/bcir_tensor.c $C/srcd/bcir_decoder_train.c -o $C/bin/libdec_tree.so -lm || exit 1
clang $CF -DBCIR_POOL_YIELDS=200000 -fPIC -shared -I $C/srcd $C/srcd/bcir_tensor.c $C/srcd/bcir_decoder_train.c -o $C/bin/libdec_yield.so -lm || exit 1
clang -std=c11 -O2 -I $C/srcd $T/step_one.c -o $C/bin/step_one -ldl -lm || exit 1
: > $O/prototypes-yields.jsonl
for tt in 256 512; do for k in 1 2 3 4 5 6; do for v in tree yield; do
  $C/bin/step_one $C/bin/libdec_$v.so 260 128 4 4 3 256 4 $tt 14 4 40 4 | sed "s/^{/{\"build\":\"$v\",/" >> $O/prototypes-yields.jsonl
done; done; done
echo "== $(date -u +%H:%M:%S) done"
