// RUN: bcir-opt -split-input-file -verify-diagnostics %s

// expected-error@+1 {{'bcir.interrupt_trampoline' op interrupt vector must be in [0, 255] (got 256)}}
bcir.interrupt_trampoline @bad_vector vector 256 handler @dispatch

// -----

// Quoted MLIR symbols may contain assembler punctuation; the module-assembly edge must reject them
// rather than emitting attacker-controlled directives into the object stream.
// expected-error@+1 {{'bcir.entry' op entry symbol must be an unquoted x86 assembler identifier (got 'bad;label')}}
bcir.entry @"bad;label" stack @stack_top target @kernel_main

// -----

// expected-error@+1 {{'bcir.interrupt_trampoline' op handler must be an unquoted x86 assembler identifier (got 'bad handler')}}
bcir.interrupt_trampoline @irq32 vector 32 handler @"bad handler"

// -----

// A leading '$' is an AT&T immediate marker in generated call/jump operands.
// expected-error@+1 {{'bcir.interrupt_trampoline' op handler must be an unquoted x86 assembler identifier (got '$dispatch')}}
bcir.interrupt_trampoline @irq33 vector 33 handler @"$dispatch"

// -----

// NMI-like entries need paranoid GS/IST/nesting handling even when GS swapping is disabled.
// expected-error@+1 {{'bcir.interrupt_trampoline' op normal interrupt entry is unsafe for paranoid/IST vector 2; use a dedicated paranoid/IST trampoline}}
bcir.interrupt_trampoline @nmi vector 2 handler @dispatch

// -----

// #DB can arrive in transient entry state and requires paranoid GS handling.
// expected-error@+1 {{'bcir.interrupt_trampoline' op normal interrupt entry is unsafe for paranoid/IST vector 1; use a dedicated paranoid/IST trampoline}}
bcir.interrupt_trampoline @debug vector 1 handler @dispatch

// -----

// #DF has a hardware error code but still requires the paranoid/IST operation.
// expected-error@+1 {{'bcir.interrupt_trampoline' op normal interrupt entry is unsafe for paranoid/IST vector 8; use a dedicated paranoid/IST trampoline}}
bcir.interrupt_trampoline @double_fault vector 8 handler @dispatch

// -----

// #MC may interrupt transient machine state and uses a dedicated IST entry.
// expected-error@+1 {{'bcir.interrupt_trampoline' op normal interrupt entry is unsafe for paranoid/IST vector 18; use a dedicated paranoid/IST trampoline}}
bcir.interrupt_trampoline @machine_check vector 18 handler @dispatch

// -----

// AMD #VC uses an IST stack and must support nested #VC handling.
// expected-error@+1 {{'bcir.interrupt_trampoline' op normal interrupt entry is unsafe for paranoid/IST vector 29; use a dedicated paranoid/IST trampoline}}
bcir.interrupt_trampoline @vmm_communication vector 29 handler @dispatch
