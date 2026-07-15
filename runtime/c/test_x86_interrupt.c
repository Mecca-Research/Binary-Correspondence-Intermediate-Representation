#include "bcir_x86_interrupt.h"

int main(void) {
  bcir_x86_interrupt_frame saved = {0};

  if (bcir_x86_interrupt_from_user(&saved))
    return 1;
  if (bcir_x86_interrupt_saved_stack(&saved) != &saved.stack)
    return 2;

  saved.cs = UINT64_C(3);
  saved.stack.rsp = UINT64_C(0x12345678);
  saved.stack.ss = UINT64_C(0x23);
  if (!bcir_x86_interrupt_from_user(&saved))
    return 3;
  bcir_x86_interrupt_stack_state *stack =
      bcir_x86_interrupt_saved_stack(&saved);
  if (stack == NULL || stack->rsp != saved.stack.rsp || stack->ss != saved.stack.ss)
    return 4;
  return 0;
}
