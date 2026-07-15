"""Governed registry inspection and mutation baseline (MC2).

This is the in-process oracle/operator surface, not a hardware peek/poke transport. A write is scoped to
one declared resource element, replaces the immutable Resource with a `data_gen + 1` successor, and
therefore makes already-hydrated StreamPacks fail R11 until they are rehydrated. A future resident driver
may bind the same commands to RuntimeChannel; it must not bypass these generation semantics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import re
import shlex
import sys

from .examples import PROGRAMS
from .model import Module, Resource

_REF = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*|[0-9]+)(?:\[(?P<index>[0-9]+)\])?$")


@dataclass(frozen=True)
class RegistryAddress:
    rid: int
    index: int


@dataclass(frozen=True)
class PokeResult:
    address: RegistryAddress
    previous: int | None
    value: int
    previous_data_gen: int
    data_gen: int


class RegistryInspector:
    """Deterministic value overlay over a BCIR module's immutable resource registry."""

    def __init__(self, module: Module):
        self.module = module
        self._values: dict[RegistryAddress, int] = {}

    def resolve(self, reference: str) -> RegistryAddress:
        match = _REF.fullmatch(reference)
        if match is None:
            raise ValueError(f"invalid resource reference {reference!r}; use RID|NAME[index]")
        token = match.group("name")
        if token.isdigit():
            rid = int(token)
        else:
            matches = [res.rid for res in self.module.resources.values() if res.name == token]
            if not matches:
                raise KeyError(f"unknown resource name {token!r}")
            if len(matches) != 1:
                raise ValueError(f"ambiguous resource name {token!r}")
            rid = matches[0]
        resource = self.module.resource(rid)
        if resource is None:
            raise KeyError(f"unknown RID {rid}")
        explicit_index = match.group("index")
        if explicit_index is None and resource.count != 1:
            raise ValueError(
                f"resource {resource.name or rid} has {resource.count} elements; specify [index]")
        index = int(explicit_index or 0)
        if index >= resource.count:
            raise IndexError(
                f"resource {resource.name or rid}[{index}] is outside [0, {resource.count})")
        return RegistryAddress(rid, index)

    def getp(self, reference: str) -> int | None:
        """Read an operator value. `None` means no resident binding/value has supplied it."""
        return self._values.get(self.resolve(reference))

    def setp(self, reference: str, value: int) -> PokeResult:
        """Govern one value write and advance the resource's R11 data generation exactly once."""
        address = self.resolve(reference)
        resource = self.module.resources[address.rid]
        previous = self._values.get(address)
        new_value = int(value)
        if resource.data_gen >= 0xFFFFFFFF:
            raise OverflowError(
                f"resource {resource.name or address.rid} data_gen cannot advance past u32")
        successor = replace(resource, data_gen=resource.data_gen + 1)
        # Complete every operation that can invoke user conversion or allocation before
        # publishing either half of the state transition. Both dictionary keys already
        # exist after these assignments, so rollback is deterministic if publication is
        # interrupted by an ordinary exception.
        had_previous = address in self._values
        try:
            self.module.resources[address.rid] = successor
            self._values[address] = new_value
        except BaseException:
            self.module.resources[address.rid] = resource
            if had_previous:
                assert previous is not None
                self._values[address] = previous
            else:
                self._values.pop(address, None)
            raise
        return PokeResult(address, previous, new_value, resource.data_gen, successor.data_gen)

    def resources_text(self) -> str:
        lines = []
        for resource in sorted(self.module.resources.values(), key=lambda item: item.rid):
            populated = sum(1 for address in self._values if address.rid == resource.rid)
            lines.append(
                f"rid={resource.rid} name={resource.name or '-'} domain={resource.domain.name.lower()} "
                f"shape={list(resource.shape)!r} access={resource.access} priority={resource.priority} "
                f"map_gen={resource.map_gen} data_gen={resource.data_gen} values={populated}")
        return "\n".join(lines) + ("\n" if lines else "")

    def claims_text(self) -> str:
        lines = []
        for phase in sorted(self.module.phases, key=lambda item: item.phase_id):
            for claim in sorted(phase.claims, key=lambda item: item.id):
                lines.append(
                    f"claim={claim.id} phase={phase.phase_id} op={claim.op or claim.opcode.name.lower()} "
                    f"rd={list(claim.rd)!r} wr={list(claim.wr)!r} count={claim.count} "
                    f"hazard={claim.hazard} bounds={claim.bounds}")
        return "\n".join(lines) + ("\n" if lines else "")

    def phases_text(self) -> str:
        lines = [
            f"phase={phase.phase_id} deps={list(phase.deps)!r} event={phase.event or '-'} "
            f"claims={len(phase.claims)}"
            for phase in sorted(self.module.phases, key=lambda item: item.phase_id)
        ]
        return "\n".join(lines) + ("\n" if lines else "")


def execute_command(inspector: RegistryInspector, command: str) -> tuple[bool, str]:
    """Execute one bounded operator command; return `(keep_running, output)`."""
    words = shlex.split(command)
    if not words:
        return True, ""
    if words[0] in {"quit", "exit"}:
        return False, ""
    if words == ["show", "resources"]:
        return True, inspector.resources_text()
    if words == ["show", "claims"]:
        return True, inspector.claims_text()
    if words == ["show", "phases"]:
        return True, inspector.phases_text()
    if len(words) == 2 and words[0] == "getp":
        address = inspector.resolve(words[1])
        value = inspector.getp(words[1])
        return True, f"rid={address.rid} index={address.index} value={value!r}\n"
    if len(words) == 3 and words[0] == "setp":
        result = inspector.setp(words[1], int(words[2], 0))
        return True, (
            f"rid={result.address.rid} index={result.address.index} value={result.value} "
            f"data_gen={result.previous_data_gen}->{result.data_gen}\n")
    raise ValueError(
        "command must be: show resources|claims|phases, getp RID|NAME[index], "
        "setp RID|NAME[index] VALUE, or quit")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bcir-registry", description="governed in-process BCIR registry inspector")
    parser.add_argument("program", choices=sorted(PROGRAMS))
    parser.add_argument("-c", "--command", action="append", default=[],
                        help="execute a command; repeat to preserve state across commands")
    args = parser.parse_args(argv)
    inspector = RegistryInspector(PROGRAMS[args.program]())

    try:
        if args.command:
            for command in args.command:
                keep_running, output = execute_command(inspector, command)
                sys.stdout.write(output)
                if not keep_running:
                    break
            return 0
        while True:
            try:
                command = input("bcir-registry> ")
            except EOFError:
                return 0
            keep_running, output = execute_command(inspector, command)
            sys.stdout.write(output)
            if not keep_running:
                return 0
    except (KeyError, IndexError, OverflowError, ValueError) as exc:
        print(f"bcir-registry: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
