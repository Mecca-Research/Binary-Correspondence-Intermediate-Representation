.text
.globl sum_pair
.type sum_pair,@function
sum_pair:
  movl %edi, %eax
  addl %esi, %eax
  retq
.size sum_pair, .-sum_pair

.section .note.GNU-stack,"",@progbits
