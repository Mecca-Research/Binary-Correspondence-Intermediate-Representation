source_filename = "bcir_kbcost.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.costvec.q16 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

define i64 @bcir.kbdi.score.q16(ptr %cost, ptr %weight) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%next, %body]
  %acc = phi i64 [0, %entry], [%acc2, %body]
  %done = icmp uge i32 %i, 11
  br i1 %done, label %exit, label %body

body:
  %cp = getelementptr inbounds i32, ptr %cost, i32 %i
  %wp = getelementptr inbounds i32, ptr %weight, i32 %i
  %c32 = load i32, ptr %cp, align 4
  %w32 = load i32, ptr %wp, align 4
  %c = zext i32 %c32 to i64
  %w = zext i32 %w32 to i64
  %mul = mul i64 %c, %w
  %acc2 = add i64 %acc, %mul
  %next = add i32 %i, 1
  br label %loop

exit:
  ret i64 %acc
}

!bcir.kbdi = !{!200}
!200 = !{!"K_BDI(G|H,Theta)", !"score", !"sum_i weight_i(theta,phase,policy) * normalized_cost_i", !"costs", !"compute,memory,fabric,sync,compile,thermal,power,reliability,security,accuracy,contention"}


define i64 @bcir.kbdi.score.batch.q16(ptr %batch_cost, ptr %weight) {
entry:
  %score = call i64 @bcir.kbdi.score.q16(ptr %batch_cost, ptr %weight)
  ret i64 %score
}

!bcir.kbdi.phase3 = !{!310}
!310 = !{
  !"batch_score",
  !"compute + memory + fabric + sync + thermal + contention",
  !"selection", !"min cost legal batch plan"
}
