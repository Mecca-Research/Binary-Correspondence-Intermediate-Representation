#!/bin/bash
# ATT-CONTIG's check over shapes, threads, contracts and both compilers. usage: ws_checks.sh BIN LIBPREFIX OUT
B=$1; L=$2; O=$3; : > $O; n=0; bad=0; cp=0
for cc in clang gcc; do
for shape in "1 1 1 1 16" "1 7 2 1 8" "2 9 2 2 16" "1 31 2 1 32" "2 33 4 2 32" "1 100 4 4 32" "2 127 4 2 48" \
             "1 257 2 1 64" "1 300 4 2 40" "3 300 8 4 48" "1 1030 2 1 32" "2 1100 4 2 40" "1 1024 2 1 128" \
             "1 300 2 2 128" "1 100 1 1 256" "1 70 1 1 264" "1 64 4 1 32" "2 2048 4 4 32" "1 2050 2 2 40"; do
for th in 1 3 8; do for ct in 0 1 2 3 8 9; do
  set -- $shape
  [ "$2" -ge 1024 ] && [ "$ct" != 1 ] && [ "$ct" != 9 ] && continue
  out=$($B/ws_check $B/${L}_$cc.so $shape $th $ct 2>&1); n=$((n+1))
  case "$out" in equal*) echo "ok $cc [$shape] t$th c$ct $out" >> $O; [[ "$out" == *copies=1* ]] && cp=$((cp+1));;
    *) bad=$((bad+1)); echo "DIFF $cc [$shape] t$th c$ct: $out" | tee -a $O;; esac
done; done; done; done
echo "cases=$n differ=$bad with_copies=$cp" | tee -a $O
