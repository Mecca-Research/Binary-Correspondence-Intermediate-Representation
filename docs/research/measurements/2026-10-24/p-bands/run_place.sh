#!/bin/bash
# BWD-NEXT's placement, on a given unit: the host's read and copy bandwidths (membw.c), the row-major walk's band and
# prefetch variants (variant_tr.py, timed by tr_place.c) and the walks over P in 16-row bands (band_place.c: its own
# btr and bctx, and the unit's att512_btransposed -- 'bp'), every result's bits checked against the unit's row walk.
# Eight interleaved passes, one process a configuration. usage: run_place.sh UNIT_DIR TOOLS_DIR OUT_DIR
SRC=$1; T=$2; O=$3; mkdir -p $O/bin $O/units
CF="-std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread"
clang -std=c11 -O2 -pthread $T/membw.c -o $O/bin/membw || exit 1
cp $SRC/bcir_tensor.c $O/units/base.c && cp $SRC/bcir_tensor.h $O/units/ || exit 1
VARS="b16p0T0:16:0:T0 b32p32T1:32:32:T1 b32p128T0:32:128:T0 b32p128T1:32:128:T1 b32p128T2:32:128:T2 b32p256T1:32:256:T1"
NAMES="base b16p0T0 b32p32T1 b32p128T0 b32p128T1 b32p128T2 b32p256T1"
for v in $VARS; do IFS=: read -r n b p h <<< "$v"; python3 -I $T/variant_tr.py $SRC/bcir_tensor.c $O/units/$n.c $b $p $h > /dev/null || exit 1; done
for u in $NAMES; do clang $CF -I $O/units -DUNIT="\"$u.c\"" $T/tr_place.c -o $O/bin/trp_$u -lm || exit 1; done
clang $CF -I $O/units -DUNIT='"base.c"' -DPROTO_BTR $T/band_place.c -o $O/bin/bandp -lm || exit 1
: > $O/membw.txt
for th in 1 4; do for mib in 1 2 4 8 16 32 48 64 128 256 1024; do
  $O/bin/membw $mib $th 5 | sed "s/^/membw /" >> $O/membw.txt
done; done
: > $O/place.txt
for pass in 1 2 3 4 5 6 7 8; do
  for nb in 1 16; do
    calls=12; [ $nb = 16 ] && calls=32
    for u in $NAMES; do echo "pass $pass trp $u $($O/bin/trp_$u $nb $calls tr)" >> $O/place.txt; done
    echo "pass $pass trp base $($O/bin/trp_base $nb $calls ctx)" >> $O/place.txt
    for w in "tr 32" "btr 32" "bp 32" "ctx 32" "bctx 32" "btr 16" "btr 64"; do
      set -- $w; echo "pass $pass bandp $($O/bin/bandp $nb $calls $1 $2)" >> $O/place.txt
    done
    echo "pass $pass bandp-rewrite1 $($O/bin/bandp $nb $calls btr 32 1)" >> $O/place.txt
    echo "pass $pass bandp-rewrite2 $($O/bin/bandp $nb $calls btr 32 2)" >> $O/place.txt
  done
done
grep -c "bits differ" $O/place.txt; wc -l < $O/place.txt; wc -l < $O/membw.txt
