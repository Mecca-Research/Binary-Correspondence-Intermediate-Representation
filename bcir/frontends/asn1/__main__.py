"""bcir-asn1c — the ASN.1 (X.680) schema compiler driver.

Compiles an ASN.1 module into the BCIR type model and reports what it found, or uses it
to encode/decode X.690 DER.

    python -m bcir.frontends.asn1 [options] module.asn1 ...

Options:
    --list                  print each module's types with their tags (the default)
    --print                 re-emit the module as ASN.1 notation (the round-trip form)
    --check                 parse, lower, and assert the round-trip law, printing nothing
                            on success -- the form to use in a build
    --encode TYPE           read a JSON value on stdin, write DER for TYPE to stdout
    --decode TYPE           read DER on stdin, write the decoded value as JSON to stdout
    --hex                   with --encode/--decode, use hex text instead of raw octets
    --ber                   with --decode, accept BER rather than requiring DER
    --jer                   with --encode/--decode, use X.697 JER instead of X.690 DER;
                            decoding runs the bounded J1 oracle (explicit limits, UTF-8,
                            and exact canonical-byte validation) and reports a stable
                            error code, byte offset and required capacity on refusal
    --basic                 with --jer, use BASIC JER rather than the BCIR canonical
                            profile -- accepts every encoder's option X.697 allows
    --framed                with --jer, wrap (or expect) the BCIR frame of roadmap 3.3:
                            version, sequence, generation, length and a CRC-32, all
                            verified before any payload is returned
    --transcode TYPE        read DER on stdin and write JER for TYPE to stdout, or the
                            reverse with --jer; the value never leaves the type model

Exit status is 0 on success and 1 on any lexical, syntactic, or semantic fault; every
diagnostic carries `file:line:column`, so the output drops straight into an editor.
"""

from __future__ import annotations

import json
import sys

from bcir.asn1.jer_bounded import JerBoundedError

from .lexer import Asn1SyntaxError
from .lower import Asn1SemanticError, compile_module
from .parser import parse_module
from .printer import print_module


def _describe(lowered) -> list[str]:
    from bcir.asn1.schema import Choice, Primitive, Sequence, SequenceOf, Set, SetOf

    lines = [
        f"module {lowered.module.name} "
        f"{{ {' '.join(map(str, lowered.module.oid))} }} "
        f"{lowered.tag_default} TAGS"
    ]
    for name, built in lowered.module.types.items():
        kind = type(built).__name__
        lines.append(f"  {name} : {kind}")
        members = ()
        if isinstance(built, (Sequence, Set)):
            members = built.components
        elif isinstance(built, Choice):
            members = built.alternatives
        elif isinstance(built, (SequenceOf, SetOf)):
            lines.append(f"      of {built.element.name}")
        elif isinstance(built, Primitive):
            lines[-1] += f" (UNIVERSAL {built.universal})"
        for comp in members:
            tag = (
                ""
                if comp.tag is None
                else f"[{comp.tag}] {'EXPLICIT' if comp.explicit else 'IMPLICIT'} "
            )
            suffix = (
                " OPTIONAL"
                if comp.optional
                else (f" DEFAULT {comp.default!r}" if comp.has_default else "")
            )
            lines.append(f"      {tag}{comp.name} : {comp.type.name}{suffix}")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or "-h" in args or "--help" in args:
        print(__doc__)
        return 0

    mode, encode_type, decode_type = "list", None, None
    use_hex = ber = jer = basic = framed = False
    transcode_type = None
    paths: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("--list", "--print", "--check"):
            mode = arg[2:]
        elif arg == "--hex":
            use_hex = True
        elif arg == "--ber":
            ber = True
        elif arg == "--jer":
            jer = True
        elif arg == "--basic":
            basic = True
        elif arg == "--framed":
            framed = True
        elif arg == "--transcode":
            index += 1
            if index >= len(args):
                sys.stderr.write(f"bcir-asn1c: {arg} needs a type name\n")
                return 1
            mode, transcode_type = "transcode", args[index]
        elif arg in ("--encode", "--decode"):
            index += 1
            if index >= len(args):
                sys.stderr.write(f"bcir-asn1c: {arg} needs a type name\n")
                return 1
            mode = arg[2:]
            if arg == "--encode":
                encode_type = args[index]
            else:
                decode_type = args[index]
        elif arg.startswith("-"):
            sys.stderr.write(f"bcir-asn1c: unknown option {arg}\n")
            return 1
        else:
            paths.append(arg)
        index += 1

    if not paths:
        sys.stderr.write("bcir-asn1c: no input files\n")
        return 1

    status = 0
    for path in paths:
        try:
            text = open(path, encoding="utf-8").read()
            if mode == "print":
                sys.stdout.write(print_module(parse_module(text, path)))
                continue
            lowered = compile_module(text, path)
            if mode == "check":
                node = parse_module(text, path)
                if parse_module(print_module(node), f"{path}<printed>") != node:
                    sys.stderr.write(
                        f"{path}: round-trip law failed: printing the "
                        f"parsed module yields a different module\n"
                    )
                    status = 1
                continue
            if mode == "list":
                print("\n".join(_describe(lowered)))
                continue
            status |= _transcode(
                lowered,
                mode,
                encode_type,
                decode_type,
                use_hex,
                ber,
                jer,
                basic,
                framed,
                transcode_type,
            )
        except JerBoundedError as exc:
            # The J1 diagnostic is structured; print it as such so a caller can branch on
            # the code rather than on the prose.
            diagnostic = exc.diagnostic
            sys.stderr.write(f"bcir-asn1c: {diagnostic.code.value}: {diagnostic}\n")
            status = 1
        except (Asn1SyntaxError, Asn1SemanticError) as exc:
            sys.stderr.write(f"{exc}\n")
            status = 1
        except OSError as exc:
            sys.stderr.write(f"bcir-asn1c: {exc}\n")
            status = 1
    return status


def _transcode(
    lowered,
    mode,
    encode_type,
    decode_type,
    use_hex,
    ber,
    jer=False,
    basic=False,
    framed=False,
    transcode_type=None,
) -> int:
    from bcir.asn1.codec import Strictness

    if mode == "transcode":
        # DER in, JER out -- or the reverse with --jer. The value passes through the type
        # model, so a transcode is two conformant codecs rather than a textual rewrite.
        kind = lowered.module.types[transcode_type]
        raw = sys.stdin.read().strip() if use_hex else sys.stdin.buffer.read()
        if jer:
            value = _jer_decode(bytes.fromhex(raw) if use_hex else raw, kind, basic, framed)
            octets = lowered.module.encode(transcode_type, value)
            print(octets.hex()) if use_hex else sys.stdout.buffer.write(octets)
            return 0
        octets = bytes.fromhex(raw) if use_hex else raw
        value = lowered.module.decode(
            transcode_type, octets, strictness=Strictness.BER if ber else Strictness.DER
        )
        sys.stdout.buffer.write(_jer_encode(kind, value, basic, framed))
        return 0

    if mode == "encode":
        value = json.loads(sys.stdin.read())
        if jer:
            sys.stdout.buffer.write(
                _jer_encode(lowered.module.types[encode_type], value, basic, framed)
            )
            return 0
        octets = lowered.module.encode(encode_type, value)
        if use_hex:
            print(octets.hex())
        else:
            sys.stdout.buffer.write(octets)
        return 0

    raw = sys.stdin.read().strip() if use_hex else sys.stdin.buffer.read()
    octets = bytes.fromhex(raw) if use_hex else raw
    if jer:
        decoded = _jer_decode(octets, lowered.module.types[decode_type], basic, framed)
    else:
        decoded = lowered.module.decode(
            decode_type, octets, strictness=Strictness.BER if ber else Strictness.DER
        )
    print(json.dumps(decoded, indent=2, default=str))
    return 0


def _jer_encode(kind, value, basic: bool, framed: bool) -> bytes:
    from bcir.asn1.jer import JerRules, encode_jer
    from bcir.asn1.jer_bounded import encode_framed

    rules = JerRules.BASIC if basic else JerRules.CANONICAL
    if framed:
        return encode_framed(kind, value, rules=rules)
    return encode_jer(kind, value, rules=rules)


def _jer_decode(octets: bytes, kind, basic: bool, framed: bool):
    """Always through the bounded oracle: J1's limits and canonical-byte check are not an
    opt-in, they are what makes the CLI safe to point at a file someone else wrote."""
    from bcir.asn1.jer import JerRules
    from bcir.asn1.jer_bounded import decode_bounded, decode_framed

    rules = JerRules.BASIC if basic else JerRules.CANONICAL
    if framed:
        return decode_framed(octets, kind, rules=rules)
    return decode_bounded(octets, kind, rules=rules)


if __name__ == "__main__":
    sys.exit(main())
