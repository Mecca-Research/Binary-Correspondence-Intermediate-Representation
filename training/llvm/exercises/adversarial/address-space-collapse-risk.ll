; adversarial-class: assemble-valid-semantically-risky
; Risk: verifier-valid addrspacecast syntax does not establish that address
; space 1 is round-trippable or dereferenceable through address space 0.

define i32 @address_space_collapse_risk(ptr addrspace(1) %device_ptr) {
entry:
  %generic_ptr = addrspacecast ptr addrspace(1) %device_ptr to ptr
  %value = load i32, ptr %generic_ptr, align 4
  ret i32 %value
}
