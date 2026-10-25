#!/bin/bash
# KSPLIT-ROOM's records on the committed tree (HEAD) against the tree before it (HEAD~1), serially, two workers at
# most:
#  0 sizes  -- room_sizes.c on each tree's library: the need for any pool, at one thread and four, the need without
#              the split room (HEAD's alone), and the room stated apart from the unit, at every shape below
#  1 values -- the C harness on each tree and compiler: HEAD's output the tree before's line for line but the new
#              KSPLIT-ROOM line; every timed arm below holds its weights equal byte for byte
#  2 splits -- ../ksplit/need_ab.py's copy of step_ab (two rounds) over each tree's unit with the split rule's log
#              (../ksplit/split_probe.py on the tree before's, split_probe2.py on HEAD's): every weight adjoint the
#              rule weighs in the step lent that tree's own need at four threads, split or kept
#  3 step   -- the same copy of step_ab between the two trees, each state lent its own tree's need at four threads
#              (what the Python trainer lends): the corpus recipe's decoder B8 T256, the GEO dimensions B4 T512 and
#              the 9.84M decoder B8 T512 and T1024, where HEAD lends more; the GEO dimensions T1024 and T2048, whose
#              attention copies already hold the room (the need the same on both trees), as controls; clang and gcc
#  4 one    -- the same on one thread: the corpus decoder T256, the GEO dimensions T1024 (one thread's copies short
#              of the room) and the 9.84M decoder T512
set -o pipefail
S=<SCRATCH>
R=<REPO>
C=$S/next5/meas; O=$C/final; T=$R/docs/research/measurements/2026-10-25/ksplit-room; D=$R/docs/research/measurements
CF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread"
cd $R || exit 2
H=$(git rev-parse HEAD); P=$(git rev-parse HEAD~1)
echo "== $(date -u +%H:%M:%S) tree=$H before=$P dirty=$(git status --short | wc -l)"
rm -rf $C; mkdir -p $C/srcA $C/srcB $C/srcPA $C/srcPB $O/bin
for f in bcir_tensor.c bcir_tensor.h bcir_decoder_train.c bcir_decoder_train.h test_decoder_train.c; do
  git show $P:runtime/c/$f > $C/srcA/$f; git show $H:runtime/c/$f > $C/srcB/$f
done
for cc in clang gcc; do for side in A B; do
  src=$C/src$side
  $cc $CF -fPIC -shared -I $src $src/bcir_tensor.c $src/bcir_decoder_train.c -o $O/bin/libdec_${cc}_$side.so -lm || exit 1
done; done
echo "== $(date -u +%H:%M:%S) 0 sizes: each tree's need and HEAD's need without the room"
clang $CF -I $C/srcB $T/room_sizes.c -o $O/bin/room_sizes -ldl || exit 1
for side in A B; do
  $O/bin/room_sizes $O/bin/libdec_clang_$side.so | sed "s/^{/{\"tree\":\"$side\",/" | tee -a $O/sizes.jsonl || exit 1
done
echo "== $(date -u +%H:%M:%S) 1 values: the C harness on each tree and compiler"
for cc in clang gcc; do for side in A B; do
  src=$C/src$side
  $cc -std=c2x -O3 -ffp-contract=off -fno-math-errno -pthread -I $src $src/test_decoder_train.c $src/bcir_tensor.c \
    $src/bcir_decoder_train.c -o $O/bin/harness_${cc}_$side -lm || exit 1
  $O/bin/harness_${cc}_$side > $O/harness-$cc-$side.txt 2>&1; echo "harness $cc $side exit $?"
done
  if diff <(grep -v "(KSPLIT-ROOM)" $O/harness-$cc-B.txt) $O/harness-$cc-A.txt > /dev/null; then
    echo "harness $cc: the same output but the new line"
  else
    echo "harness $cc: OUTPUT DIFFERS"; diff <(grep -v "(KSPLIT-ROOM)" $O/harness-$cc-B.txt) $O/harness-$cc-A.txt | cut -c1-300
  fi
  grep "(KSPLIT-ROOM)" $O/harness-$cc-B.txt
done
echo "== $(date -u +%H:%M:%S) 2 splits: the calls the split rule weighs, each tree lent its own need at four threads"
python3 -I $D/2026-10-25/ksplit/need_ab.py $D/2026-10-22/attn-rate/step_ab.c $C/step_need_ab.c || exit 1
clang -std=c11 -O2 -I $C/srcB $C/step_need_ab.c -o $O/bin/step_need_ab -ldl -lm || exit 1
cp $C/srcA/* $C/srcPA/; cp $C/srcB/* $C/srcPB/
python3 -I $D/2026-10-25/ksplit/split_probe.py $C/srcA/bcir_tensor.c $C/srcPA/bcir_tensor.c || exit 1
python3 -I $T/split_probe2.py $C/srcB/bcir_tensor.c $C/srcPB/bcir_tensor.c || exit 1
for side in A B; do
  clang $CF -fPIC -shared -I $C/srcP$side $C/srcP$side/bcir_tensor.c $C/srcP$side/bcir_decoder_train.c \
    -o $O/bin/libdec_split_$side.so -lm || exit 1
done
: > $O/splits.txt
for args in "1024 128 4 2 4 344 8 256 14 4 2" "260 128 4 4 3 256 4 512 14 4 2" "260 128 4 4 3 256 4 1024 14 4 2" \
            "260 128 4 4 3 256 4 2048 14 4 2" "260 384 8 4 6 1024 8 512 46 4 2" "260 384 8 4 6 1024 8 1024 46 4 2"; do
  for side in A B; do
    echo "== $side step_need_ab $args" >> $O/splits.txt
    $O/bin/step_need_ab $O/bin/libdec_split_$side.so $O/bin/libdec_split_$side.so $args 2>&1 | grep "^KSPLIT" | \
      sort | uniq -c >> $O/splits.txt
  done
done
cat $O/splits.txt
echo "== $(date -u +%H:%M:%S) 3 the training step, each tree lent its own need at four threads"
: > $O/ab-step.jsonl
for cc in clang gcc; do
  for spec in "corpus B8 T256 kblock+exp+reduce|1024 128 4 2 4 344 8 256 14 4 48" \
              "geo-dims B4 T512 kblock+exp+reduce|260 128 4 4 3 256 4 512 14 4 48" \
              "decoder-9.84M B8 T512 kblock+exp+reduce+attn|260 384 8 4 6 1024 8 512 46 4 16" \
              "decoder-9.84M B8 T1024 kblock+exp+reduce+attn|260 384 8 4 6 1024 8 1024 46 4 16" \
              "geo-dims B4 T1024 kblock+exp+reduce (control: the need the same)|260 128 4 4 3 256 4 1024 14 4 24" \
              "geo-dims B4 T2048 kblock+exp+reduce (control: the need the same)|260 128 4 4 3 256 4 2048 14 4 24"; do
    IFS='|' read -r name args <<< "$spec"
    out=$($O/bin/step_need_ab $O/bin/libdec_${cc}_A.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
    echo "{\"compiler\":\"$cc\",\"threads\":4,\"case\":\"$name\",\"run\":$out}" >> $O/ab-step.jsonl
    echo "$cc $name: $(echo "$out" | cut -c1-260)"
  done
done
echo "== $(date -u +%H:%M:%S) 4 the training step on one thread, each tree lent its own need there"
: > $O/ab-step-one.jsonl
for cc in clang gcc; do
  for spec in "corpus B8 T256 kblock+exp+reduce, one thread|1024 128 4 2 4 344 8 256 14 1 24" \
              "geo-dims B4 T1024 kblock+exp+reduce, one thread|260 128 4 4 3 256 4 1024 14 1 12" \
              "decoder-9.84M B8 T512 kblock+exp+reduce+attn, one thread|260 384 8 4 6 1024 8 512 46 1 8"; do
    IFS='|' read -r name args <<< "$spec"
    out=$($O/bin/step_need_ab $O/bin/libdec_${cc}_A.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
    echo "{\"compiler\":\"$cc\",\"threads\":1,\"case\":\"$name\",\"run\":$out}" >> $O/ab-step-one.jsonl
    echo "$cc $name: $(echo "$out" | cut -c1-260)"
  done
done
echo "== $(date -u +%H:%M:%S) done"
