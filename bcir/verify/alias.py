"""R12's reading of the alias facts an emitted elementwise kernel carries (S5-A).

The declared side is `lower.alias_facts.kernel_facts`, the one derivation every emitter uses
(docs/security/laws.md L14). The emitted side is read here from the kernel's text and nothing
else -- never from the emitter's own tables -- so a fact the emitter drops, adds or contradicts
is a finding in the law that owns it. Two kinds of finding, named apart in the messages:

  * a FALSE fact -- `noalias` / `restrict` on a pointer whose resource another operand names, an
    access declaring its own resource's scope disjoint from itself, a TBAA type the claim does
    not declare, a plain access in a volatile claim: LLVM reorders or elides on the strength of
    these, so each is undefined behaviour or a lost device access, not a missed optimization;
  * a DROPPED fact -- a missing `noalias`, scope or tag, a fence the hazard needs: the kernel is
    correct and LLVM knows less than BCIR declared, which is the gap this slice closed.

Both readers are total: text they cannot read is a finding, never a traceback (L1). The LLVM
reader sees every `load` and `store` in any spelling the IR parser accepts (with or without an
alignment, atomic or not), and a fence narrowed to a sync scope is not the barrier: a
`syncscope("singlethread")` fence orders a thread against its own signal handlers and nothing
else. The C reader reads the declarations, and holds the body to the one shape that makes a
declaration the whole story: every operand is used only by subscript, and no address is taken,
so C carries each parameter's qualifiers to every access (a cast or a derived pointer can shed
them -- `volatile` among them).
"""

from __future__ import annotations

import re

_POSITIONS = ("A", "B", "C")
_DEFINE = re.compile(r"^define\b[^@]*@[\w.$]+\((.*)\)[^{]*\{$")
_GEP = re.compile(r"^(%[\w.]+)\s*=\s*getelementptr\b[^,]*,\s*ptr\s+(%[\w.]+)\s*,")
_ACCESS = re.compile(r"^(?:%[\w.]+\s*=\s*)?(load|store)\b(.*)$")
_POINTER = re.compile(r"^ptr\s+(%[\w.]+)(?:\s|$)")
_MD = re.compile(r"^!(\d+)\s*=\s*(distinct\s+)?!\{(.*)\}$")
_ATTACH = re.compile(r"^!([\w.]+)\s+!(\d+)$")
_FENCE = re.compile(r'^fence\s+(?:syncscope\("([^"]*)"\)\s+)?(\w+)\s*(?:,.*)?$')
_BRACKETS = {"(": 1, "<": 1, "{": 1, "[": 1, ")": -1, ">": -1, "}": -1, "]": -1}


def _split_top(text: str) -> list[str]:
    """`text` split at the commas outside brackets (`captures(...)`, `<8 x float>`, `{...}`)."""
    out, depth, cur = [], 0, []
    for ch in text:
        depth += _BRACKETS.get(ch, 0)
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


def _ref(tok: str) -> int | None:
    m = re.fullmatch(r"!(\d+)", tok)
    return int(m.group(1)) if m else None


def _string(tok: str) -> str | None:
    m = re.fullmatch(r'!"([^"]*)"', tok)
    return m.group(1) if m else None


def _fence_ordering(ins: str) -> str | None:
    """A fence's ordering, or None when `ins` is not a fence. A fence narrowed to a sync scope
    names it (`seq_cst in syncscope("singlethread")`): only the system scope -- no `syncscope`,
    or `syncscope("")`, which is its spelled-out name -- orders the kernel against other threads
    and devices, so only a system fence is the barrier a hazard contract declares."""
    toks = ins.split()
    if not toks or toks[0] != "fence":
        return None
    m = _FENCE.match(ins)
    if m is None:
        return "unreadable"
    scope, order = m.group(1), m.group(2)
    return order if not scope else f'{order} in syncscope("{scope}")'


class _Kernel:
    """The facts one LLVM function carries, read from its text."""

    def __init__(self, text: str) -> None:
        self.params: dict[int, list[str]] = {}
        self.accesses: list[tuple] = []  # (position or None, pointer, volatile, attachments)
        self.metadata: dict[int, tuple[bool, list[str]]] = {}
        self.instrs: list[str] = []
        self.functions = 0
        pointers: dict[str, int] = {}
        inside = False
        for raw in text.splitlines():
            line = raw.strip()
            md = _MD.match(line)
            if md:
                body = md.group(3).strip()
                self.metadata[int(md.group(1))] = (
                    bool(md.group(2)),
                    [tok.strip() for tok in body.split(",")] if body else [],
                )
                continue
            line = line.split(";", 1)[0].strip()
            d = _DEFINE.match(line)
            if d:
                self.functions += 1
                inside = self.functions == 1
                for param in _split_top(d.group(1)):
                    toks = param.split()
                    if toks and toks[-1] in ("%A", "%B", "%C"):
                        self.params[_POSITIONS.index(toks[-1][1:])] = toks
                continue
            if not inside or not line:
                continue
            if line == "}":
                inside = False
                continue
            if line.endswith(":"):
                continue
            self.instrs.append(line)
            g = _GEP.match(line)
            if g:
                base = g.group(2)
                if base in ("%A", "%B", "%C"):
                    pointers[g.group(1)] = _POSITIONS.index(base[1:])
                elif base in pointers:
                    pointers[g.group(1)] = pointers[base]
            a = _ACCESS.match(line)
            if a:
                # `load [atomic] [volatile] <ty>, ptr %p[, <ordering>...]` and `store [atomic]
                # [volatile] <ty> <v>, ptr %p...`: the pointer is the second operand, the
                # alignment and the attachments follow it -- whichever of them are spelled.
                parts = _split_top(a.group(2))
                p = _POINTER.match(parts[1]) if len(parts) > 1 else None
                ptr = p.group(1) if p else "an unreadable operand"
                position = pointers.get(ptr)
                if position is None and ptr in ("%A", "%B", "%C"):
                    position = _POSITIONS.index(ptr[1:])
                attach = {}
                for part in parts[2:]:
                    m = _ATTACH.match(part)
                    if m:
                        attach[m.group(1)] = int(m.group(2))
                self.accesses.append((position, ptr, "volatile" in parts[0].split(), attach))

    def refs(self, node: int | None) -> frozenset | None:
        """The nodes a metadata list names, or None when `node` is not such a list."""
        if node is None or node not in self.metadata:
            return None
        refs = [_ref(tok) for tok in self.metadata[node][1]]
        return None if any(r is None for r in refs) else frozenset(refs)

    def domain(self, scope: int) -> int | None:
        """A scope's domain: `distinct !{!s, !d, ...}` over `distinct !{!d, ...}`, else None."""
        entry = self.metadata.get(scope)
        if entry is None or not entry[0] or len(entry[1]) < 2 or _ref(entry[1][0]) != scope:
            return None
        d = _ref(entry[1][1])
        dom = self.metadata.get(d) if d is not None else None
        if dom is None or not dom[0] or not dom[1] or _ref(dom[1][0]) != d:
            return None
        return d

    def tbaa(self, tag: int | None, root: str, char: str) -> str | None:
        """The scalar type a struct-path access tag names under `root`, or None."""
        entry = self.metadata.get(tag) if tag is not None else None
        if entry is None or entry[0] or len(entry[1]) != 3 or entry[1][2] != "i64 0":
            return None
        base, access = _ref(entry[1][0]), _ref(entry[1][1])
        scalar = self.metadata.get(base) if base is not None and base == access else None
        if scalar is None or scalar[0] or len(scalar[1]) != 3 or scalar[1][2] != "i64 0":
            return None
        parent = self.metadata.get(_ref(scalar[1][1]))
        if parent is None or len(parent[1]) != 3 or _string(parent[1][0]) != char:
            return None
        top = self.metadata.get(_ref(parent[1][1]))
        if top is None or len(top[1]) != 1 or _string(top[1][0]) != root:
            return None
        return _string(scalar[1][0])


def ll_alias_diagnostics(facts, text: str) -> list[str]:
    """Every way the LLVM kernel's alias facts differ from `facts` (the claim's declaration)."""
    from ..lower.alias_facts import TBAA_CHAR, TBAA_ROOT

    k = _Kernel(text)
    out: list[str] = []
    if k.functions != 1:
        return [f"alias facts unreadable: the kernel text defines {k.functions} functions, not 1"]
    for p in range(3):
        name, rid = _POSITIONS[p], facts.rids[p]
        toks = k.params.get(p)
        if toks is None:
            out.append(f"alias facts unreadable: no pointer parameter %{name}")
            continue
        sharers = [f"%{_POSITIONS[q]}" for q in range(3) if q != p and facts.rids[q] == rid]
        if "noalias" in toks and sharers:
            out.append(
                f"false alias fact: %{name} carries noalias, but RID {rid} is also named by "
                f"{', '.join(sharers)}"
            )
        elif "noalias" not in toks and not sharers:
            out.append(
                f"alias fact dropped: %{name} is the only pointer to RID {rid} and carries no noalias"
            )
    if not k.accesses:
        return out + ["alias facts unreadable: the kernel performs no memory access"]
    stray = [ptr for position, ptr, _vol, _att in k.accesses if position is None]
    if stray:
        out.append(
            f"alias facts unreadable: {len(stray)} accesses through pointers no declared operand "
            f"explains ({', '.join(sorted(set(stray)))})"
        )
    known = [acc for acc in k.accesses if acc[0] is not None]

    # Volatility: every access of a volatile claim, and no access of any other.
    plain = sum(1 for acc in known if not acc[2])
    marked = len(known) - plain
    if facts.volatile and plain:
        out.append(
            f"false fact: {plain} accesses of a volatile claim are not volatile (LLVM may elide, merge or reorder them)"
        )
    if not facts.volatile and marked:
        out.append(f"volatile not declared: {marked} accesses of a non-volatile claim are volatile")

    # The alias scopes: one per resource, in one domain; each access its own in !alias.scope
    # and every other resource's in !noalias.
    scopes: dict[int, set] = {}
    missing = 0
    for position, _ptr, _vol, attach in known:
        refs = k.refs(attach.get("alias.scope"))
        if refs is None:
            missing += 1
            continue
        scopes.setdefault(facts.rids[position], set()).update(refs)
    if missing:
        out.append(f"alias fact dropped: {missing} accesses carry no !alias.scope list")
    owner: dict[int, int] = {}
    for rid, refs in scopes.items():
        if len(refs) != 1:
            out.append(
                f"alias scopes not preserved: RID {rid}'s accesses carry {len(refs)} scopes, not one"
            )
            continue
        s = next(iter(refs))
        if s in owner:
            out.append(f"alias fact dropped: RIDs {owner[s]} and {rid} share one scope")
        owner.setdefault(s, rid)
    per_rid = {rid: next(iter(refs)) for rid, refs in scopes.items() if len(refs) == 1}
    domains = {k.domain(s) for s in per_rid.values()}
    if None in domains:
        out.append("alias scopes not preserved: a scope is not a distinct scope node in a domain")
    elif len(domains) > 1:
        out.append(f"alias fact dropped: the scopes span {len(domains)} domains, not one")
    if len(per_rid) == len(facts.resources) and len(set(per_rid.values())) == len(per_rid):
        false = dropped = 0
        for position, _ptr, _vol, attach in known:
            rid = facts.rids[position]
            want = {per_rid[r] for r in facts.resources if r != rid}
            got = k.refs(attach["noalias"]) if "noalias" in attach else frozenset()
            if got is not None and per_rid[rid] in got:
                false += 1  # its own resource declared disjoint from itself
            elif got != want:
                dropped += 1
        if false:
            out.append(
                f"false alias fact: {false} accesses name their own resource's scope in !noalias"
            )
        if dropped:
            out.append(
                f"alias fact dropped: {dropped} accesses' !noalias does not name exactly the other "
                f"resources' scopes"
            )

    # TBAA: the declared element type's tag, under clang's C/C++ root.
    wrong = sum(
        1 for acc in known if k.tbaa(acc[3].get("tbaa"), TBAA_ROOT, TBAA_CHAR) != facts.tbaa
    )
    if wrong:
        out.append(
            f"TBAA not preserved: {wrong} accesses carry no {facts.tbaa!r} tag under {TBAA_ROOT!r} "
            f"(the declared element type)"
        )

    # The hazard's fences: a barriered claim's first instruction and the one before each `ret`.
    ordering = [_fence_ordering(ins) for ins in k.instrs]
    fences = sum(o is not None for o in ordering)
    rets = [i for i, ins in enumerate(k.instrs) if ins.startswith("ret")]
    if facts.fence is None:
        if fences:
            out.append(
                f"hazard contract 'unique' declares no ordering, and the kernel carries {fences} fences"
            )
    else:
        entry = bool(ordering) and ordering[0] == facts.fence
        exits = bool(rets) and all(i > 0 and ordering[i - 1] == facts.fence for i in rets)
        if not (entry and exits and fences == 1 + len(rets)):
            out.append(
                f"hazard contract 'barriered' requires a fence (>= {facts.fence}) before the "
                f"kernel's first access and after its last (entry fence: {'yes' if entry else 'no'}, "
                f"every exit fenced: {'yes' if exits else 'no'}, fences: {fences})"
            )
    return out


_C_DEF = re.compile(r"\bvoid\s+\w+\s*\(([^)]*)\)\s*\{", re.S)


def c_alias_diagnostics(facts, text: str, ctype: str | None = None) -> list[str]:
    """Every way the C kernel's pointer qualifiers and fences differ from `facts`."""
    from ..lower.alias_facts import C_FENCE

    m = _C_DEF.search(text)
    if m is None:
        return ["alias facts unreadable: no kernel definition"]
    params = m.group(1).replace("*", " * ").split(",")
    if len(params) < 3:
        return [
            f"alias facts unreadable: the kernel takes {len(params)} parameters, not 3 pointers and n"
        ]
    out: list[str] = []
    for p in range(3):
        toks = params[p].split()
        name, rid = _POSITIONS[p], facts.rids[p]
        if not toks or toks[-1] != name or "*" not in toks:
            out.append(f"alias facts unreadable: parameter {p} is not the pointer {name}")
            continue
        sharers = [_POSITIONS[q] for q in range(3) if q != p and facts.rids[q] == rid]
        if "restrict" in toks and sharers:
            out.append(
                f"false alias fact: {name} is restrict-qualified, but RID {rid} is also named by "
                f"{', '.join(sharers)}"
            )
        elif "restrict" not in toks and not sharers:
            out.append(
                f"aliasing contract not preserved: {name} is the only pointer to RID {rid} and is "
                f"not restrict-qualified"
            )
        if facts.volatile and "volatile" not in toks:
            out.append(
                f"false fact: {name} points to plain {ctype or facts.ctype}, and the claim is volatile"
            )
        if not facts.volatile and "volatile" in toks:
            out.append(
                f"volatile not declared: {name} points to volatile data in a non-volatile claim"
            )
    depth, j = 1, m.end()
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    body = text[m.end() : j - 1]
    code = re.sub(r"//[^\n]*", " ", re.sub(r"/\*.*?\*/", " ", body, flags=re.S))
    # The declarations are the facts only while every access goes through them: an operand used
    # other than by subscript (a cast, a derived pointer) or an address taken can shed the
    # qualifiers the parameter declares -- a plain access to a volatile operand, a write through
    # a read one.
    for name in _POSITIONS:
        if re.search(rf"\b{name}\b(?!\s*\[)", code):
            out.append(
                f"alias facts unreadable: {name} is used other than as a subscript in the kernel "
                f"body (a cast or a derived pointer sheds the qualifiers its declaration carries)"
            )
    if "&" in code:
        out.append(
            "alias facts unreadable: the kernel body takes an address, which a cast can strip of "
            "the qualifiers the parameters declare"
        )
    stmts = [
        ln.strip()
        for ln in body.splitlines()
        if ln.strip() and not ln.strip().startswith(("/*", "//"))
    ]
    fences = sum(C_FENCE in s for s in stmts)
    if facts.fence is None:
        if fences:
            out.append(
                f"hazard contract 'unique' declares no ordering, and the kernel carries {fences} fences"
            )
    elif not (
        stmts and stmts[0].startswith(C_FENCE) and stmts[-1].startswith(C_FENCE) and fences == 2
    ):
        out.append(
            f"hazard contract 'barriered' requires `{C_FENCE}` as the kernel's first and last statement"
        )
    elif re.search(r"\breturn\b", code):
        out.append(
            f"hazard contract 'barriered' requires every exit fenced, and a `return` leaves the "
            f"kernel before its last `{C_FENCE}`"
        )
    return out


def c_param_types(text: str) -> list[list[str]] | None:
    """The type tokens of the C kernel's first three parameters, or None."""
    m = _C_DEF.search(text)
    if m is None:
        return None
    params = m.group(1).replace("*", " * ").split(",")
    return [p.split() for p in params[:3]] if len(params) >= 3 else None
