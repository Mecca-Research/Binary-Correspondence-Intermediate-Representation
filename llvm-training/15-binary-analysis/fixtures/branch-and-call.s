.text
.globl increment
.type increment,@function
increment:
.Lincrement:
  leal 1(%rdi), %eax
  retq
.size increment, .-increment

.globl classify_and_increment
.type classify_and_increment,@function
classify_and_increment:
  testl %edi, %edi
  je .Lzero
  callq .Lincrement
  retq
.Lzero:
  xorl %eax, %eax
  retq
.size classify_and_increment, .-classify_and_increment

.section .note.GNU-stack,"",@progbits

.data
.globl fixture_bias
.type fixture_bias,@object
fixture_bias:
  .long 7
.size fixture_bias, .-fixture_bias

.bss
.globl fixture_scratch
.type fixture_scratch,@object
fixture_scratch:
  .zero 4
.size fixture_scratch, .-fixture_scratch
