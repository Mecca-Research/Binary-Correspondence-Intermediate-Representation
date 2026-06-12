# Solution 041: Interpret static binary evidence

The linear fixture exports only `sum_pair`, has one basic block, and contains
one data-transfer, one integer-arithmetic instruction, and one return. The
branch-and-call fixture exports `increment` and `classify_and_increment`; it adds
a conditional branch, a direct call edge from `classify_and_increment` to
`increment`, a compare/test instruction, and three basic blocks in the caller.
Its `.text` section is consequently larger.

The `deterministic` classification means the declared assembly sources can be
rebuilt for the fixed x86-64 target and their normalized symbol, instruction
class, basic-block, call-edge, and section evidence can be checked for drift. It
does not establish semantic equivalence between the two functions, nor does it
make a wall-clock or hardware-counter prediction.

Even identical instruction-class counts and basic-block counts would lose
constants, predicates, operand relationships, and memory effects. Before an
equivalence claim I would compare exact control/data-flow semantics and test or
symbolically check representative and boundary inputs. I would also review ABI
and call targets; where side channels matter, I would separately collect a
controlled dynamic trace or host-sensitive counter experiment without treating
that experiment as a portable CI golden value.
