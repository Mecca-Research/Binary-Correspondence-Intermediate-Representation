"""Print a parsed module back as ASN.1 notation.

This exists for the round-trip law, not for pretty output: `parse(print(parse(t)))`
must equal `parse(t)`. That law is what makes the front-end's AST *complete* --
anything the parser silently dropped would vanish here too, and the two would agree
with each other while both disagreeing with the module. Comparing ASTs rather than
text is deliberate: layout and comments are not semantic, so requiring byte-identical
output would fail on formatting instead of on meaning.

The printer therefore emits everything the AST carries, including the tag mode when
the source stated one -- a module printed with its tags resolved would round-trip to a
DIFFERENT AST in an IMPLICIT-TAGS module, and the law would catch that.
"""

from __future__ import annotations

from . import ast

_INDENT = "    "


def _oid(value: ast.OidValue) -> str:
    parts = []
    for arc in value.arcs:
        if arc.name is not None and arc.number is not None:
            parts.append(f"{arc.name}({arc.number})")
        elif arc.name is not None:
            parts.append(arc.name)
        else:
            parts.append(str(arc.number))
    return "{ " + " ".join(parts) + " }"


def _value(node) -> str:
    if isinstance(node, ast.IntValue):
        return str(node.value)
    if isinstance(node, ast.StrValue):
        return '"' + node.value.replace('"', '""') + '"'
    if isinstance(node, ast.BoolValue):
        return "TRUE" if node.value else "FALSE"
    if isinstance(node, ast.NullValue):
        return "NULL"
    if isinstance(node, ast.BitsValue):
        # Print as hstring when the bit count is a whole number of hex digits, so a
        # value written that way survives the round trip in the same form.
        if node.bits % 4 == 0:
            return "'" + node.data.hex().upper()[:node.bits // 4] + "'H"
        bits = "".join(f"{byte:08b}" for byte in node.data)[:node.bits]
        return "'" + bits + "'B"
    if isinstance(node, ast.OidValue):
        return _oid(node)
    if isinstance(node, ast.BracedValue):
        # Print whichever reading the text supported; when both did, the value-list
        # spelling re-parses into the same pair of readings, so the law still holds.
        if node.items is not None:
            if not node.items:
                return "{}"
            return "{ " + ", ".join(_value(v) for v in node.items) + " }"
        return _oid(ast.OidValue(node.arcs or ()))
    if isinstance(node, ast.RefValue):
        return node.name
    raise TypeError(f"cannot print value {type(node).__name__}")


def _named_numbers(named: tuple[ast.NamedNumber, ...], extensible: bool = False) -> str:
    parts = [f"{n.name}({n.number})" for n in named]
    if extensible:
        parts.append("...")
    return " { " + ", ".join(parts) + " }"


def _type(node, depth: int) -> str:
    if isinstance(node, ast.TypeRef):
        return f"{node.module}.{node.name}" if node.module else node.name

    if isinstance(node, ast.Builtin):
        if node.named or node.extensible:
            return node.name + _named_numbers(node.named, node.extensible)
        return node.name

    if isinstance(node, ast.Tagged):
        head = f"[{node.tag_class} {node.number}]" if node.tag_class \
            else f"[{node.number}]"
        mode = f" {node.mode}" if node.mode else ""
        return f"{head}{mode} {_type(node.inner, depth)}"

    if isinstance(node, ast.SequenceOfType):
        name = f"{node.element_name} " if node.element_name else ""
        return f"SEQUENCE OF {name}{_type(node.element, depth)}"

    if isinstance(node, ast.SetOfType):
        name = f"{node.element_name} " if node.element_name else ""
        return f"SET OF {name}{_type(node.element, depth)}"

    if isinstance(node, ast.SequenceType):
        return "SEQUENCE " + _components(node.components, depth)
    if isinstance(node, ast.SetType):
        return "SET " + _components(node.components, depth)
    if isinstance(node, ast.ChoiceType):
        return "CHOICE " + _components(node.alternatives, depth)

    raise TypeError(f"cannot print type {type(node).__name__}")


def _components(items: tuple[object, ...], depth: int) -> str:
    if not items:
        return "{}"
    pad = _INDENT * (depth + 1)
    rendered = []
    for item in items:
        if isinstance(item, ast.ExtensionMarker):
            rendered.append(pad + "...")
            continue
        line = f"{pad}{item.name} {_type(item.type, depth + 1)}"
        if item.optional:
            line += " OPTIONAL"
        elif item.has_default:
            line += f" DEFAULT {_value(item.default)}"
        rendered.append(line)
    return "{\n" + ",\n".join(rendered) + "\n" + _INDENT * depth + "}"


def print_module(node: ast.ModuleNode) -> str:
    lines: list[str] = []
    header = node.name
    if node.oid is not None:
        header += " " + _oid(node.oid)
    lines.append(header)
    tags = f"{node.tag_default} TAGS "
    extend = "EXTENSIBILITY IMPLIED " if node.extensibility_implied else ""
    lines.append(f"DEFINITIONS {tags}{extend}::= BEGIN")
    lines.append("")

    if node.exports is not None:
        lines.append("EXPORTS " + ", ".join(node.exports) + ";")
    elif node.imports:
        lines.append("EXPORTS ALL;")
    for group in node.imports:
        target = group.module + (f" {_oid(group.oid)}" if group.oid else "")
        lines.append(f"IMPORTS {', '.join(group.symbols)} FROM {target};")
    if node.exports is not None or node.imports:
        lines.append("")

    for assignment in node.assignments:
        if isinstance(assignment, ast.TypeAssignment):
            lines.append(f"{assignment.name} ::= {_type(assignment.type, 0)}")
        elif isinstance(assignment, ast.ValueAssignment):
            lines.append(f"{assignment.name} {_type(assignment.type, 0)} ::= "
                         f"{_value(assignment.value)}")
        lines.append("")

    lines.append("END")
    return "\n".join(lines) + "\n"


__all__ = ["print_module"]
