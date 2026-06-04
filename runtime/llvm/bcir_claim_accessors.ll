source_filename = "bcir_claim_accessors.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }

define i64 @bcir.claim.control(ptr %c) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.claim, ptr %c, i32 0, i32 0
  %v = load i64, ptr %p, align 8
  ret i64 %v
}

define i64 @bcir.claim.opstride(ptr %c) alwaysinline {
entry:
  %v = call i64 @bcir.claim.control(ptr %c)
  ret i64 %v
}

define i8 @bcir.claim.opcode(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.control(ptr %c)
  %x = trunc i64 %h to i8
  ret i8 %x
}

define i8 @bcir.claim.lane(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.control(ptr %c)
  %s = lshr i64 %h, 8
  %x = trunc i64 %s to i8
  ret i8 %x
}

define i32 @bcir.claim.phase(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.control(ptr %c)
  %s = lshr i64 %h, 16
  %m = and i64 %s, 4095
  %x = trunc i64 %m to i32
  ret i32 %x
}

define i32 @bcir.claim.epoch(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.control(ptr %c)
  %s = lshr i64 %h, 28
  %m = and i64 %s, 4095
  %x = trunc i64 %m to i32
  ret i32 %x
}

define i32 @bcir.claim.flags(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.control(ptr %c)
  %s = lshr i64 %h, 40
  %m = and i64 %s, 255
  %x = trunc i64 %m to i32
  ret i32 %x
}

define i32 @bcir.claim.stride_code(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.control(ptr %c)
  %s = lshr i64 %h, 48
  %x = trunc i64 %s to i32
  ret i32 %x
}

define i32 @bcir.claim.rd(ptr %c, i32 %idx) alwaysinline {
entry:
  %arr = getelementptr inbounds %bcir.claim, ptr %c, i32 0, i32 1
  %p = getelementptr inbounds [4 x i32], ptr %arr, i32 0, i32 %idx
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i32 @bcir.claim.wr(ptr %c, i32 %idx) alwaysinline {
entry:
  %arr = getelementptr inbounds %bcir.claim, ptr %c, i32 0, i32 2
  %p = getelementptr inbounds [4 x i32], ptr %arr, i32 0, i32 %idx
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i64 @bcir.claim.hazard_domain(ptr %c) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.claim, ptr %c, i32 0, i32 3
  %v = load i64, ptr %p, align 8
  ret i64 %v
}

define i32 @bcir.claim.hazard_mode(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.hazard_domain(ptr %c)
  %m = and i64 %h, 15
  %x = trunc i64 %m to i32
  ret i32 %x
}

define i32 @bcir.claim.domain(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.hazard_domain(ptr %c)
  %s = lshr i64 %h, 4
  %m = and i64 %s, 15
  %x = trunc i64 %m to i32
  ret i32 %x
}

define i32 @bcir.claim.ordering(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.hazard_domain(ptr %c)
  %s = lshr i64 %h, 16
  %m = and i64 %s, 255
  %x = trunc i64 %m to i32
  ret i32 %x
}

define i32 @bcir.claim.alias_group(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.hazard_domain(ptr %c)
  %s = lshr i64 %h, 32
  %m = and i64 %s, 65535
  %x = trunc i64 %m to i32
  ret i32 %x
}

define i64 @bcir.claim.imm(ptr %c, i32 %idx) alwaysinline {
entry:
  %arr = getelementptr inbounds %bcir.claim, ptr %c, i32 0, i32 4
  %p = getelementptr inbounds [2 x i64], ptr %arr, i32 0, i32 %idx
  %v = load i64, ptr %p, align 8
  ret i64 %v
}
