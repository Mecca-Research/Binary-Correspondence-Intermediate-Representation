; Straight-line scalar operations for the SLP Vectorizer.
;
; Try:
;   opt -S -passes=slp-vectorizer slp-scalars.ll -o -
;   opt -S -passes='default<O3>' slp-scalars.ll -o -

source_filename = "slp-scalars.ll"
target triple = "x86_64-unknown-linux-gnu"

define void @slp_scalars(ptr noalias nocapture readonly %a,
                         ptr noalias nocapture readonly %b,
                         ptr noalias nocapture writeonly %c) {
entry:
  %a0p = getelementptr inbounds i32, ptr %a, i64 0
  %a1p = getelementptr inbounds i32, ptr %a, i64 1
  %a2p = getelementptr inbounds i32, ptr %a, i64 2
  %a3p = getelementptr inbounds i32, ptr %a, i64 3
  %b0p = getelementptr inbounds i32, ptr %b, i64 0
  %b1p = getelementptr inbounds i32, ptr %b, i64 1
  %b2p = getelementptr inbounds i32, ptr %b, i64 2
  %b3p = getelementptr inbounds i32, ptr %b, i64 3
  %c0p = getelementptr inbounds i32, ptr %c, i64 0
  %c1p = getelementptr inbounds i32, ptr %c, i64 1
  %c2p = getelementptr inbounds i32, ptr %c, i64 2
  %c3p = getelementptr inbounds i32, ptr %c, i64 3

  %a0 = load i32, ptr %a0p, align 4
  %a1 = load i32, ptr %a1p, align 4
  %a2 = load i32, ptr %a2p, align 4
  %a3 = load i32, ptr %a3p, align 4
  %b0 = load i32, ptr %b0p, align 4
  %b1 = load i32, ptr %b1p, align 4
  %b2 = load i32, ptr %b2p, align 4
  %b3 = load i32, ptr %b3p, align 4

  %s0 = add i32 %a0, %b0
  %s1 = add i32 %a1, %b1
  %s2 = add i32 %a2, %b2
  %s3 = add i32 %a3, %b3

  store i32 %s0, ptr %c0p, align 4
  store i32 %s1, ptr %c1p, align 4
  store i32 %s2, ptr %c2p, align 4
  store i32 %s3, ptr %c3p, align 4
  ret void
}
