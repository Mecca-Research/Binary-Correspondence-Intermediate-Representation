# The AST and Semantic Analysis

## TL;DR

The AST is the frontend's semantic truth, and Sema *adds to it*. Most IR that
"came from nowhere" corresponds to an AST node the source never wrote:
conversions, materialized temporaries, destructor calls, and implicit
`this`/vtable machinery are all explicit nodes before they are instructions.

```bash
clang -Xclang -ast-dump -fsyntax-only file.c
clang++ -Xclang -ast-dump -fsyntax-only file.cpp
clang -Xclang -ast-dump=json -fsyntax-only file.c    # machine-readable
```

## Node families worth recognising

| Family | Examples | What it means |
| --- | --- | --- |
| `*Decl` | `FunctionDecl`, `ParmVarDecl`, `VarDecl`, `RecordDecl`, `FieldDecl`, `CXXMethodDecl` | a named entity |
| `*Stmt` | `CompoundStmt`, `IfStmt`, `ForStmt`, `SwitchStmt`, `ReturnStmt` | a statement |
| `*Expr` | `BinaryOperator`, `CallExpr`, `DeclRefExpr`, `MemberExpr`, `IntegerLiteral` | a value-producing expression |
| `*Type` | `BuiltinType`, `PointerType`, `RecordType`, `TypedefType` | a type, canonical or sugared |
| `*Attr` | `AlignedAttr`, `PackedAttr`, `NoInlineAttr` | an attribute attached to a decl |

Three properties that make the dump readable:

- **Every expression carries its type**, printed in quotes. That is how you see
  where an integer became a double.
- **`lvalue`/`xvalue`** annotations tell you the value category, which decides
  whether a load is coming.
- **Sugar is preserved.** A `TypedefType` prints as the typedef *and* its
  underlying type; `-ast-dump` shows what the user wrote, and Sema reasons on
  the canonical type underneath.

## Sema's implicit nodes

This is the part worth internalising. Given plain C:

```c
double f(int a, double b) { return a + b; }
```

Sema produces:

```
`-BinaryOperator 'double' '+'
  |-ImplicitCastExpr 'double' <IntegralToFloating>
  | `-ImplicitCastExpr 'int' <LValueToRValue>
  |   `-DeclRefExpr 'int' lvalue ParmVar 'a' 'int'
  `-ImplicitCastExpr 'double' <LValueToRValue>
    `-DeclRefExpr 'double' lvalue ParmVar 'b' 'double'
```

Two implicit casts per operand, and the cast *kinds* are named. `LValueToRValue`
becomes a `load`; `IntegralToFloating` becomes `sitofp`. The one-line C
statement already contains the whole lowering plan.

The C++ case is where this pays for itself. For `return Guard().n;`:

```
`-ExprWithCleanups 'int'
  `-ImplicitCastExpr 'int' <LValueToRValue>
    `-MemberExpr 'int' xvalue .n
      `-MaterializeTemporaryExpr 'Guard' xvalue
        `-CXXBindTemporaryExpr 'Guard' (CXXTemporary)
          `-CXXTemporaryObjectExpr 'Guard' 'void ()'
```

Read outward-in: a temporary is constructed, bound (meaning it has a non-trivial
destructor), materialized so a member can be named, loaded, and then
`ExprWithCleanups` marks the point where the destructor runs — the end of the
full expression. The destructor call in the IR is not the optimizer's doing and
not a mystery; it is the `ExprWithCleanups` node.

## What Sema decides, permanently

| Decision | Consequence in IR |
| --- | --- |
| Overload resolution | which mangled symbol is called |
| Implicit conversion sequence | which cast instructions appear, in which order |
| Template instantiation | which specializations exist, with `linkonce_odr` linkage |
| Access control, `const` correctness | nothing — these are checks, not lowering |
| Default arguments | argument expressions materialized at the call site |
| Name lookup and ADL | the callee, before any IR exists |

CodeGen does not revisit any of these. If IR calls the wrong overload, the AST
already called the wrong overload — dump it and look, instead of reading IR.

## Useful dumps beyond `-ast-dump`

```bash
clang -Xclang -ast-dump=json -fsyntax-only file.c        # structured, greppable
clang++ -Xclang -fdump-record-layouts -c file.cpp -o /dev/null   # field offsets
clang++ -Xclang -fdump-vtable-layouts -c file.cpp -o /dev/null   # vtable slots
clang++ -Xclang -ast-print -fsyntax-only file.cpp        # AST back to source
clang -Xclang -ast-dump-filter=name -Xclang -ast-dump -fsyntax-only file.c
```

`-ast-dump-filter` is the one that makes this usable on real code: a full dump
of a translation unit with system headers is tens of megabytes, and filtering by
declaration name cuts it to the part you asked about.

`-fdump-record-layouts` deserves special mention — it prints the exact offset,
size, and alignment the frontend chose for every field, which is the ground
truth that [`03-c-lowering-rules.md`](03-c-lowering-rules.md) reasons about.

## Diagnostics as a Sema artifact

Sema's diagnostics carry structure the text does not show:

```bash
clang -fdiagnostics-format=sarif -fsyntax-only file.c   # machine-readable
clang -fno-caret-diagnostics -fsyntax-only file.c       # compact for logs
clang -fdiagnostics-show-template-tree -fsyntax-only f.cpp
clang -Wall -Wextra -Werror ...                          # the useful default set
```

For tooling, prefer a structured format over parsing rendered text — the
rendering is tuned for humans and changes between releases.

## Pitfalls

- **Reading IR to answer a Sema question.** Wrong overload, unexpected
  conversion, surprising destructor timing: all visible in the AST, all
  obscured in IR.
- **Confusing sugar with the canonical type.** `-ast-dump` shows both; a
  diagnostic mentioning a typedef is talking about the same type as one
  mentioning its underlying type.
- **Assuming an implicit node is optional.** `ExprWithCleanups` is where a
  destructor runs. Removing "redundant" AST structure in a tool changes program
  meaning.
- **Dumping a whole TU.** Use `-ast-dump-filter`.
- **Expecting `-ast-dump` to be stable output.** It is a debugging aid; node
  spellings change. Use the JSON form for anything automated, and even then pin
  the Clang version.
- **Assuming templates exist in IR.** Only instantiations do, one symbol each,
  with `linkonce_odr` linkage and a comdat.

## BCIR notes

- BCIR's C-front twin has the same split: a checked semantic form comes first, a
  lowering decision comes second, and diagnostics belong to the first. Keeping
  the two apart is why a fallback-to-LLVM signal can be reported honestly rather
  than as a crash.
- When comparing the BCIR front-end against Clang on the same input, compare at
  the *semantic* level first (did both accept it, with the same types?) and only
  then at the IR level. Two front-ends can agree semantically and legitimately
  differ in emitted IR; disagreeing semantically is a defect in one of them.

## See also

- [`01-driver-and-frontend-pipeline.md`](01-driver-and-frontend-pipeline.md) — how to reach these flags
- [`03-c-lowering-rules.md`](03-c-lowering-rules.md) — what CodeGen does with the checked AST
- [`04-cxx-lowering-rules.md`](04-cxx-lowering-rules.md) — the C++ nodes in detail
