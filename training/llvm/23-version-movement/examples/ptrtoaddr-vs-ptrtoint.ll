; REQUIRES: llvm >= 23
;
; `ptrtoaddr` arrived in LLVM 23. On LLVM 18 and 15 this file is not "wrong" -- it is
; unparseable: `llvm-as-18` stops with `expected instruction opcode`. That is why the
; directive above exists and why the corpus's LLVM 23 CI job is what verifies this file.
;
; The two casts differ in what they are allowed to produce. This address space carries a
; 128-bit pointer whose ADDRESS is only 64 bits -- the shape of a capability or fat
; pointer, where a pointer is an address plus bounds and permissions.
target datalayout = "e-p:64:64-p1:128:128:128:64"

; ptrtoaddr yields the ADDRESS. The verifier requires the result to be address width:
; asking for i128 here fails with "PtrToAddr result must be address width".
define i64 @address_of(ptr addrspace(1) %p) {
  %a = ptrtoaddr ptr addrspace(1) %p to i64
  ret i64 %a
}

; ptrtoint yields the whole POINTER, bounds and permissions included, so it is i128 wide
; on the same address space. The two casts are not spellings of one operation.
define i128 @pointer_bits_of(ptr addrspace(1) %p) {
  %i = ptrtoint ptr addrspace(1) %p to i128
  ret i128 %i
}

; On a flat address space both are 64 bits and look interchangeable. They are not, and
; `opt -O2` keeps them distinct rather than canonicalising one into the other: ptrtoaddr
; is the non-capturing cast, so it does not escape the pointer the way ptrtoint does.
define i64 @flat_address(ptr %p) {
  %a = ptrtoaddr ptr %p to i64
  ret i64 %a
}

define i64 @flat_integer(ptr %p) {
  %i = ptrtoint ptr %p to i64
  ret i64 %i
}
