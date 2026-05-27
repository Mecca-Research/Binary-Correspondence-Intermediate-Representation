source_filename = "bcir_phase_epoch.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare void @llvm.trap() cold noreturn

define void @bcir.verify.epoch_phase(i32 %prev_epoch, i32 %prev_phase, i32 %next_epoch, i32 %next_phase) {
entry:
  %same_epoch = icmp eq i32 %prev_epoch, %next_epoch
  br i1 %same_epoch, label %same, label %different

same:
  %phase_ok = icmp uge i32 %next_phase, %prev_phase
  br i1 %phase_ok, label %ok, label %bad

different:
  %epoch_ok = icmp ugt i32 %next_epoch, %prev_epoch
  br i1 %epoch_ok, label %epoch_advanced, label %bad

epoch_advanced:
  %phase_reset = icmp eq i32 %next_phase, 0
  br i1 %phase_reset, label %ok, label %bad

bad:
  call void @llvm.trap()
  unreachable

ok:
  ret void
}
