# Solution 037: TableGen-to-MCInst review

A TableGen record defines the instruction's opcode metadata, operands, register
classes, scheduling information, and whether it is a pseudo. If `BCIR_PREFETCHrr`
is a pseudo, it normally must be expanded before final emission, either during
instruction selection, a pseudo-expansion pass, or target lowering before the
`MCInst` stream is encoded.

The review should check that `(base, offset)` use legal register classes and
operand types, that the pseudo maps to a real target opcode with an encoding, and
that the asm printer or MC lowering does not receive an opcode marked as
non-encodable. If an `MCInst` still contains the pseudo opcode, the encoder lacks
a real binary encoding to emit. Fix the pseudo expansion path or define a real
instruction with complete encoder and printer support.
