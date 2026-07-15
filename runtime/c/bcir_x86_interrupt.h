/*===- bcir_x86_interrupt.h - x86-64 BCIR interrupt-frame ABI -----------===*/
#ifndef BCIR_X86_INTERRUPT_H
#define BCIR_X86_INTERRUPT_H

/*
 * Fixed frame passed by `bcir.interrupt_trampoline` to its C handler.
 *
 * Ownership: the trampoline owns the frame. The handler receives one borrowed,
 * mutable view that remains valid only for the call. Mutating a saved GPR, RIP,
 * CS, RFLAGS, or return stack state changes the architectural IRETQ frame, but CS
 * must retain its original RPL (ring 0 or ring 3). Changing privilege class would
 * violate the trampoline's return and SWAPGS policy, so the trampoline traps instead.
 * With CET shadow stacks enabled, a changed RIP/CS also requires the platform's
 * corresponding shadow-stack policy and may otherwise raise #CP.
 * The entry/exit swapgs pairing is based on private state. The handler must not
 * retain the pointer. It is compiled with:
 *
 *   -mno-red-zone -mgeneral-regs-only
 *
 * Extended SIMD/FPU state is not part of this ABI. A kernel that enables it must
 * establish and test a separate eager/lazy state policy before such instructions
 * are allowed in interrupt context. SMAP AC clearing, CET/IBT configuration, CR3/PTI,
 * and entry-side speculation mitigations are likewise platform policy outside this
 * normal trampoline.
 */

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
#  define BCIR_X86_STATIC_ASSERT(c, m) static_assert(c, m)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define BCIR_X86_STATIC_ASSERT(c, m) _Static_assert(c, m)
#else
#  define BCIR_X86_STATIC_ASSERT(c, m)
#endif

typedef struct bcir_x86_interrupt_stack_state {
  uint64_t rsp;
  uint64_t ss;
} bcir_x86_interrupt_stack_state;

typedef struct bcir_x86_interrupt_frame {
  uint64_t r15;
  uint64_t r14;
  uint64_t r13;
  uint64_t r12;
  uint64_t r11;
  uint64_t r10;
  uint64_t r9;
  uint64_t r8;
  uint64_t rdi;
  uint64_t rsi;
  uint64_t rbp;
  uint64_t rdx;
  uint64_t rcx;
  uint64_t rbx;
  uint64_t rax;
  uint64_t vector;
  uint64_t error_code; /* hardware supplied, or zero synthesized by the trampoline */
  uint64_t rip;
  uint64_t cs;
  uint64_t rflags;
  /* AMD64 long mode always pushes SS:RSP for an interrupt and IRETQ always
   * consumes them, including same-CPL returns. This state is therefore part of
   * the fixed frame, not an optional user-transition tail. */
  bcir_x86_interrupt_stack_state stack;
} bcir_x86_interrupt_frame;

typedef void (*bcir_x86_interrupt_handler)(bcir_x86_interrupt_frame *frame);

static inline int
bcir_x86_interrupt_from_user(const bcir_x86_interrupt_frame *frame) {
  return frame != NULL && (frame->cs & UINT64_C(3)) == UINT64_C(3);
}

static inline bcir_x86_interrupt_stack_state *
bcir_x86_interrupt_saved_stack(bcir_x86_interrupt_frame *frame) {
  return frame != NULL ? &frame->stack : NULL;
}

BCIR_X86_STATIC_ASSERT(offsetof(bcir_x86_interrupt_frame, r15) == 0,
                       "interrupt frame must begin with r15");
BCIR_X86_STATIC_ASSERT(offsetof(bcir_x86_interrupt_frame, rax) == 112,
                       "interrupt frame rax offset drifted");
BCIR_X86_STATIC_ASSERT(offsetof(bcir_x86_interrupt_frame, vector) == 120,
                       "interrupt frame vector offset drifted");
BCIR_X86_STATIC_ASSERT(offsetof(bcir_x86_interrupt_frame, error_code) == 128,
                       "interrupt frame error-code offset drifted");
BCIR_X86_STATIC_ASSERT(offsetof(bcir_x86_interrupt_frame, rip) == 136,
                       "interrupt frame rip offset drifted");
BCIR_X86_STATIC_ASSERT(offsetof(bcir_x86_interrupt_frame, cs) == 144,
                       "interrupt frame cs offset drifted");
BCIR_X86_STATIC_ASSERT(offsetof(bcir_x86_interrupt_frame, rflags) == 152,
                       "interrupt frame rflags offset drifted");
BCIR_X86_STATIC_ASSERT(offsetof(bcir_x86_interrupt_frame, stack) == 160,
                       "interrupt frame return-stack offset drifted");
BCIR_X86_STATIC_ASSERT(sizeof(bcir_x86_interrupt_frame) == 176,
                       "interrupt frame must be exactly 176 bytes");
BCIR_X86_STATIC_ASSERT(sizeof(bcir_x86_interrupt_stack_state) == 16,
                       "interrupt stack state must be 16 bytes");

#endif /* BCIR_X86_INTERRUPT_H */
