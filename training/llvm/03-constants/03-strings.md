# String Constants

## TL;DR

LLVM IR has no string *type*. A "string" is a constant array of `i8`,
typically written with the `c"..."` syntax. Null termination is your
responsibility (write `\00`).

```llvm
@hello = constant [6 x i8]  c"hello\00"          ; null-terminated, length 6
@msg   = constant [12 x i8] c"Hi, world!\0A\00"  ; with newline + null, length 12
```

## Syntax

```
c"<bytes>"
```

Inside the quotes, `\NN` is a two-digit hex byte escape. There are no
C-style escape characters (no `\n`, `\t`); use `\0A`, `\09`, etc.

## Counting bytes

The bracketed length must match the byte count exactly. Common
gotcha:

```llvm
; "hello\00" = 6 bytes: 'h','e','l','l','o','\0'
@hello = constant [6 x i8] c"hello\00"

; If you write [5 x i8], the assembler errors out:
; "constant expression type mismatch: got '[6 x i8]' but expected '[5 x i8]'"
```

A handy rule: count the *displayed* characters, count the explicit
`\NN` escapes as 1 byte each (not 3 displayed characters), and add
them up.

## Common patterns

```llvm
; A printf-style format string
@.fmt = private unnamed_addr constant [4 x i8] c"%d\0A\00"

declare i32 @printf(ptr, ...)

define void @print_int(i32 %x) {
  call i32 (ptr, ...) @printf(ptr @.fmt, i32 %x)
  ret void
}
```

```llvm
; An array of bytes that isn't a "string" — same syntax
@magic = constant [4 x i8] c"\7FELF"   ; ELF file magic
```

```llvm
; Wide strings (UTF-16, UTF-32) aren't directly supported.
; Encode the wide chars as i16/i32 array constants:
@utf16 = constant [4 x i16] [i16 72, i16 105, i16 33, i16 0]   ; "Hi!"
```

## Using a string constant

A string constant is an array. To pass it to a function expecting
`ptr`, take its address (which is the GEP-0-0 pattern, or just use
the global symbol directly with opaque pointers):

```llvm
@.msg = private unnamed_addr constant [6 x i8] c"hello\00"

declare i32 @puts(ptr)

define i32 @greet() {
  ; With opaque pointers, you can pass the global directly:
  %r = call i32 @puts(ptr @.msg)
  ret i32 %r
}
```

The old typed-pointer idiom was:
```llvm
%p = getelementptr [6 x i8], ptr @.msg, i32 0, i32 0
%r = call i32 @puts(ptr %p)
```
Both work; the first is shorter and idiomatic for opaque pointers.

## Conventional attributes

For read-only string literals, these attributes are conventional:

```llvm
@.msg = private unnamed_addr constant [6 x i8] c"hello\00"
```

- **`private`** — symbol not exported; visible only within the module
  (and stripped from the symbol table).
- **`unnamed_addr`** — the address doesn't matter; duplicates can be
  merged.
- **`constant`** — placed in read-only memory; mutation is UB.

For exported strings (rare):

```llvm
@MESSAGE = constant [12 x i8] c"interface!\00"
```

## Pitfalls

- **Off-by-one length.** Most common mistake. "hello" + null = 6
  bytes; declare `[6 x i8]`. The assembler is precise here.

- **Forgetting the null terminator.** If the consumer (`printf`,
  `puts`, `strlen`) walks past your string looking for `\0`, it walks
  into undefined territory.

- **Trying to use `\n`, `\t`, etc.** The C-style escapes don't work.
  Use `\0A`, `\09`.

- **Mutating a `constant` string.** It's in read-only memory. Will
  segfault at runtime (or be optimized into a constant fold).

- **Counting `\NN` as more than 1 byte.** Each `\NN` is *one* byte in
  the resulting array.

## See also

- [`../02-types/02-composite-types.md`](../02-types/02-composite-types.md) — arrays
- [`../04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) — `global` vs `constant`,
  `private`, `unnamed_addr`
- [`04-global-vs-local.md`](04-global-vs-local.md) — constants at different scopes
