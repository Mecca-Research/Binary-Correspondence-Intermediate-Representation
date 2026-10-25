#!/bin/bash
# KSPLIT-ROOM's records' second run, after measure_room.sh (its libraries and sources), serially, two workers at most:
#  5 room -- room_ab.py's copy of step_ab over the committed tree's library alone: state A lent the need without the
#            split room (bcir_decoder_workspace_unsplit: the tree before's need), B and B again the need -- the room
#            alone, every instruction the same: the corpus decoder B8 T256, GEO B4 T512 and the 9.84M decoder B8 T512
#            and T1024 on four threads; GEO T1024 and T2048, whose two figures are equal (a pure check of the rig); and
#            on one thread the corpus decoder, GEO T1024 and the 9.84M decoder T512; clang and gcc. measure_room.sh's
#            tree-against-tree controls read x0.905-1.003 at GEO T1024 and T2048, where both trees lend the same
#            workspace and split the same calls
#  6 swap -- need_ab's copy of step_ab between the trees with their places swapped (the tree after's library as A),
#            GEO T1024 and T2048, clang and gcc: whether that reading follows the library or the place
set -o pipefail
S=<SCRATCH>
R=<REPO>
C=$S/next5/meas; O=$C/final; T=$R/docs/research/measurements/2026-10-25/ksplit-room; D=$R/docs/research/measurements
cd $R || exit 2
echo "== $(date -u +%H:%M:%S) tree=$(git rev-parse HEAD) dirty=$(git status --short | wc -l)"
python3 -I $T/room_ab.py $D/2026-10-22/attn-rate/step_ab.c $C/room_ab.c || exit 1
clang -std=c11 -O2 -I $C/srcB $C/room_ab.c -o $O/bin/room_ab -ldl -lm || exit 1
echo "== $(date -u +%H:%M:%S) 5 the room alone: one library, A lent the need without the room, B the need"
: > $O/ab-room.jsonl
for cc in clang gcc; do
  for spec in "4|corpus B8 T256 kblock+exp+reduce|1024 128 4 2 4 344 8 256 14 4 48" \
              "4|geo-dims B4 T512 kblock+exp+reduce|260 128 4 4 3 256 4 512 14 4 48" \
              "4|decoder-9.84M B8 T512 kblock+exp+reduce+attn|260 384 8 4 6 1024 8 512 46 4 16" \
              "4|decoder-9.84M B8 T1024 kblock+exp+reduce+attn|260 384 8 4 6 1024 8 1024 46 4 16" \
              "4|geo-dims B4 T1024 kblock+exp+reduce (control: the two figures equal)|260 128 4 4 3 256 4 1024 14 4 24" \
              "4|geo-dims B4 T2048 kblock+exp+reduce (control: the two figures equal)|260 128 4 4 3 256 4 2048 14 4 24" \
              "1|corpus B8 T256 kblock+exp+reduce, one thread|1024 128 4 2 4 344 8 256 14 1 24" \
              "1|geo-dims B4 T1024 kblock+exp+reduce, one thread|260 128 4 4 3 256 4 1024 14 1 12" \
              "1|decoder-9.84M B8 T512 kblock+exp+reduce+attn, one thread|260 384 8 4 6 1024 8 512 46 1 8"; do
    IFS='|' read -r th name args <<< "$spec"
    out=$($O/bin/room_ab $O/bin/libdec_${cc}_B.so $O/bin/libdec_${cc}_B.so $args 2>&1 | tail -1)
    echo "{\"compiler\":\"$cc\",\"threads\":$th,\"case\":\"$name\",\"run\":$out}" >> $O/ab-room.jsonl
    echo "$cc $name: $(echo "$out" | cut -c1-260)"
  done
done
echo "== $(date -u +%H:%M:%S) 6 the trees' places swapped: the tree after's library as A"
: > $O/ab-swap.jsonl
for cc in clang gcc; do
  for spec in "geo-dims B4 T1024 kblock+exp+reduce, places swapped|260 128 4 4 3 256 4 1024 14 4 24" \
              "geo-dims B4 T2048 kblock+exp+reduce, places swapped|260 128 4 4 3 256 4 2048 14 4 24"; do
    IFS='|' read -r name args <<< "$spec"
    out=$($O/bin/step_need_ab $O/bin/libdec_${cc}_B.so $O/bin/libdec_${cc}_A.so $args 2>&1 | tail -1)
    echo "{\"compiler\":\"$cc\",\"threads\":4,\"case\":\"$name\",\"run\":$out}" >> $O/ab-swap.jsonl
    echo "$cc $name: $(echo "$out" | cut -c1-260)"
  done
done
echo "== $(date -u +%H:%M:%S) done"
