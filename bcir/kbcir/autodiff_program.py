"""B3 admission, functionalization, tracing, rematerialization, and AD-order evidence.

BCIR chooses ``differentiate high, optimize low`` for the first deployable strategy.
Programs outside the explicitly supported subset are quarantined with machine-readable
reasons; they never silently fall through to a foreign AD engine.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .autodiff import (GradResult, Tape, _accumulate_adjoints, _forward_values, _reverse_order,
                       _topo_order, evaluate, grad_graph)


class ADQuarantineReason(str, Enum):
    ALIASED_MUTATION = "aliased-mutation"
    DYNAMIC_LOOP_WITHOUT_BUDGET = "dynamic-loop-without-budget"
    DYNAMIC_LOOP_BUDGET_EXHAUSTED = "dynamic-loop-budget-exhausted"
    DYNAMIC_HIGHER_ORDER_CALL = "dynamic-higher-order-call"
    RECURSION = "recursion"
    UNKNOWN_CALL = "unknown-call"


class ADQuarantinedError(ValueError):
    def __init__(self, reason: ADQuarantineReason, message: str):
        self.reason = reason
        super().__init__(f"{reason.value}: {message}")


@dataclass(frozen=True)
class ADProgramFeatures:
    mutation: str = "none"
    dynamic_loop: bool = False
    loop_max_steps: int | None = None
    higher_order: str = "none"
    recursion: bool = False
    unknown_calls: bool = False

    def __post_init__(self) -> None:
        if self.mutation not in ("none", "local_scalar", "aliased"):
            raise ValueError("mutation must be none, local_scalar, or aliased")
        if type(self.dynamic_loop) is not bool or type(self.recursion) is not bool \
                or type(self.unknown_calls) is not bool:
            raise ValueError("AD feature flags must be bool")
        if self.loop_max_steps is not None \
                and (type(self.loop_max_steps) is not int or self.loop_max_steps < 1):
            raise ValueError("loop_max_steps must be None or a positive integer")
        if self.higher_order not in ("none", "finite", "dynamic"):
            raise ValueError("higher_order must be none, finite, or dynamic")


@dataclass(frozen=True)
class ADAdmission:
    admitted: bool
    rewrites: tuple[str, ...]
    quarantine: tuple[ADQuarantineReason, ...]


def admit_ad_program(features: ADProgramFeatures) -> ADAdmission:
    if not isinstance(features, ADProgramFeatures):
        raise ValueError("features must be ADProgramFeatures")
    rewrites = []
    reasons = []
    if features.mutation == "local_scalar":
        rewrites.append("functionalize-local-scalar-mutation")
    elif features.mutation == "aliased":
        reasons.append(ADQuarantineReason.ALIASED_MUTATION)
    if features.dynamic_loop:
        if features.loop_max_steps is None:
            reasons.append(ADQuarantineReason.DYNAMIC_LOOP_WITHOUT_BUDGET)
        else:
            rewrites.append("trace-and-unroll-bounded-dynamic-loop")
    if features.higher_order == "finite":
        rewrites.append("defunctionalize-finite-call-table")
    elif features.higher_order == "dynamic":
        reasons.append(ADQuarantineReason.DYNAMIC_HIGHER_ORDER_CALL)
    if features.recursion:
        reasons.append(ADQuarantineReason.RECURSION)
    if features.unknown_calls:
        reasons.append(ADQuarantineReason.UNKNOWN_CALL)
    return ADAdmission(not reasons, tuple(rewrites), tuple(reasons))


@dataclass(frozen=True)
class MutationStatement:
    target: str
    operation: str
    operand: str | float

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("mutation target must be nonempty")
        if self.operation not in ("assign", "add", "sub", "mul"):
            raise ValueError("mutation operation is unsupported")
        if not isinstance(self.operand, str) \
                and (isinstance(self.operand, bool) or not isinstance(self.operand, (int, float))
                     or not math.isfinite(float(self.operand))):
            raise ValueError("mutation operand must be a variable name or finite scalar")


@dataclass(frozen=True)
class FunctionalizedProgram:
    tape: Tape
    output: int
    versions: tuple[tuple[str, int], ...]


def functionalize_scalar_mutation(inputs, statements, output: str) -> FunctionalizedProgram:
    if not isinstance(inputs, (tuple, list)) or not inputs \
            or any(not isinstance(name, str) or not name for name in inputs) \
            or len(set(inputs)) != len(inputs):
        raise ValueError("functionalization inputs must be unique nonempty names")
    if not isinstance(statements, (tuple, list)) \
            or any(not isinstance(statement, MutationStatement) for statement in statements):
        raise ValueError("statements must be MutationStatement values")
    tape = Tape(); current = {name: tape.var(name) for name in inputs}
    versions = {name: 0 for name in inputs}; history = []
    for statement in statements:
        if statement.target not in current:
            raise ADQuarantinedError(ADQuarantineReason.ALIASED_MUTATION,
                                     f"unknown local target {statement.target!r}")
        operand = current.get(statement.operand) if isinstance(statement.operand, str) \
            else tape.const(float(statement.operand))
        if operand is None:
            raise ADQuarantinedError(ADQuarantineReason.UNKNOWN_CALL,
                                     f"unknown mutation operand {statement.operand!r}")
        previous = current[statement.target]
        current[statement.target] = {
            "assign": lambda: operand,
            "add": lambda: tape.add(previous, operand),
            "sub": lambda: tape.sub(previous, operand),
            "mul": lambda: tape.mul(previous, operand),
        }[statement.operation]()
        versions[statement.target] += 1
        history.append((statement.target, versions[statement.target]))
    if output not in current:
        raise ValueError("functionalized output is not an input/local variable")
    return FunctionalizedProgram(tape, current[output], tuple(history))


@dataclass(frozen=True)
class LoopTrace:
    iterations: int
    condition_nodes: tuple[int, ...]
    carry_nodes: tuple[int, ...]
    max_steps: int


def trace_bounded_while(tape: Tape, initial: int, condition, body, env: dict, *,
                        max_steps: int) -> tuple[int, LoopTrace]:
    if not isinstance(tape, Tape) or type(initial) is not int or not callable(condition) \
            or not callable(body) or not isinstance(env, dict) \
            or type(max_steps) is not int or max_steps < 1:
        raise ValueError("bounded while has invalid arguments")
    carry = initial; conditions = []; carries = [carry]
    for iteration in range(max_steps):
        predicate = condition(tape, carry)
        if type(predicate) is not int:
            raise ValueError("loop condition must build and return a Tape node")
        conditions.append(predicate)
        if evaluate(tape, predicate, env) <= 0.0:
            return carry, LoopTrace(iteration, tuple(conditions), tuple(carries), max_steps)
        carry = body(tape, carry)
        if type(carry) is not int:
            raise ValueError("loop body must build and return a Tape node")
        carries.append(carry)
    predicate = condition(tape, carry)
    if evaluate(tape, predicate, env) > 0.0:
        raise ADQuarantinedError(ADQuarantineReason.DYNAMIC_LOOP_BUDGET_EXHAUSTED,
                                 f"loop remained live after {max_steps} traced steps")
    conditions.append(predicate)
    return carry, LoopTrace(max_steps, tuple(conditions), tuple(carries), max_steps)


def defunctionalize_call(tag: str, call_table: dict, tape: Tape, args: tuple) -> int:
    if not isinstance(tag, str) or not isinstance(call_table, dict) or not isinstance(tape, Tape) \
            or not isinstance(args, tuple) or any(type(arg) is not int for arg in args):
        raise ValueError("defunctionalized call has invalid arguments")
    builder = call_table.get(tag)
    if not callable(builder):
        raise ADQuarantinedError(ADQuarantineReason.DYNAMIC_HIGHER_ORDER_CALL,
                                 f"call tag {tag!r} is absent from the finite call table")
    result = builder(tape, *args)
    if type(result) is not int:
        raise ValueError("defunctionalized builder must return a Tape node")
    return result


@dataclass(frozen=True)
class RematerializationPlan:
    checkpoint_nodes: tuple[int, ...]
    max_saved_intermediates: int
    reachable_nodes: int


def plan_rematerialization(tape: Tape, output: int,
                           max_saved_intermediates: int) -> RematerializationPlan:
    if not isinstance(tape, Tape) or type(max_saved_intermediates) is not int \
            or max_saved_intermediates < 1:
        raise ValueError("rematerialization budget must be a positive integer")
    topo = _topo_order(tape, output)
    intermediates = [nid for nid in topo if tape.node(nid).op not in ("const", "var")]
    if len(intermediates) <= max_saved_intermediates:
        checkpoints = tuple(intermediates)
    else:
        stride = math.ceil(len(intermediates) / max_saved_intermediates)
        checkpoints = tuple(intermediates[index] for index in range(stride - 1, len(intermediates), stride))
        if output not in checkpoints:
            checkpoints = (*checkpoints[:-1], output) if len(checkpoints) >= max_saved_intermediates \
                else (*checkpoints, output)
    return RematerializationPlan(checkpoints, max_saved_intermediates, len(topo))


class _RematerializedValues(Mapping):
    def __init__(self, tape: Tape, env: dict, checkpoints):
        self.tape = tape; self.env = env; self.checkpoints = set(checkpoints)
        self.cache = {}; self.evaluated_ops = 0

    def __iter__(self):
        return iter(self.cache)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, nid):
        if nid in self.cache:
            return self.cache[nid]
        node = self.tape.node(nid); op = node.op
        if op == "const": value = node.const
        elif op == "var": value = float(self.env[node.const])
        elif op == "neg": value = -self[node.args[0]]
        elif op == "add": value = self[node.args[0]] + self[node.args[1]]
        elif op == "sub": value = self[node.args[0]] - self[node.args[1]]
        elif op == "mul": value = self[node.args[0]] * self[node.args[1]]
        elif op == "div": value = self[node.args[0]] / self[node.args[1]]
        elif op == "exp": value = math.exp(self[node.args[0]])
        elif op == "log": value = math.log(self[node.args[0]])
        elif op == "sqrt": value = math.sqrt(self[node.args[0]])
        elif op == "tanh": value = math.tanh(self[node.args[0]])
        elif op == "sin": value = math.sin(self[node.args[0]])
        elif op == "cos": value = math.cos(self[node.args[0]])
        elif op == "select":
            cond, left, right = node.args; value = self[left] if self[cond] > 0 else self[right]
        elif op == "dot":
            width = node.const[1]; left, right = node.args[:width], node.args[width:]
            value = sum(self[a] * self[b] for a, b in zip(left, right))
        else:  # pragma: no cover - Tape's registry owns this boundary
            raise ValueError(f"cannot rematerialize op {op!r}")
        if op not in ("const", "var"):
            self.evaluated_ops += 1
        if nid in self.checkpoints or op in ("const", "var"):
            self.cache[nid] = value
        return value


@dataclass(frozen=True)
class RematerializedGrad:
    result: GradResult
    plan: RematerializationPlan
    evaluated_ops: int


def grad_with_rematerialization(tape: Tape, output: int, env: dict, *,
                                max_saved_intermediates: int) -> RematerializedGrad:
    plan = plan_rematerialization(tape, output, max_saved_intermediates)
    topo = _topo_order(tape, output); values = _RematerializedValues(tape, env, plan.checkpoint_nodes)
    value = values[output]
    adjoints, firings = _accumulate_adjoints(tape, output, _reverse_order(topo), values)
    names = {tape.node(nid).const: nid for nid in topo if tape.node(nid).op == "var"}
    result = GradResult({name: adjoints.get(names.get(name, -1), 0.0) for name in env}, value,
                        sum(tape.node(nid).op not in ("const", "var") for nid in topo), firings)
    return RematerializedGrad(result, plan, values.evaluated_ops)


def _constant(tape: Tape, nid: int):
    node = tape.node(nid)
    return node.const if node.op == "const" else None


def optimize_tape(tape: Tape, output: int) -> tuple[Tape, int]:
    """Rebuild a reachable DAG with deterministic local algebraic/constant rewrites."""
    optimized = Tape(); mapping = {}
    for nid in _topo_order(tape, output):
        node = tape.node(nid); args = tuple(mapping[arg] for arg in node.args)
        if node.op == "const": result = optimized.const(node.const)
        elif node.op == "var": result = optimized.var(node.const)
        elif node.op == "dot":
            width = node.const[1]; result = optimized.dot(args[:width], args[width:])
        elif node.op == "select": result = optimized.select(*args)
        elif node.op in ("neg", "exp", "log", "sqrt", "tanh", "sin", "cos"):
            value = _constant(optimized, args[0])
            if value is not None:
                function = {"neg": lambda x: -x, "exp": math.exp, "log": math.log,
                            "sqrt": math.sqrt, "tanh": math.tanh, "sin": math.sin,
                            "cos": math.cos}[node.op]
                result = optimized.const(function(value))
            elif node.op == "neg" and optimized.node(args[0]).op == "neg":
                result = optimized.node(args[0]).args[0]
            else:
                result = getattr(optimized, node.op)(args[0])
        else:
            left, right = args; lv, rv = _constant(optimized, left), _constant(optimized, right)
            if lv is not None and rv is not None:
                value = {"add": lv + rv, "sub": lv - rv, "mul": lv * rv,
                         "div": lv / rv}[node.op]
                result = optimized.const(value)
            elif node.op == "add" and rv == 0.0 or node.op == "sub" and rv == 0.0 \
                    or node.op == "mul" and rv == 1.0 or node.op == "div" and rv == 1.0:
                result = left
            elif node.op == "add" and lv == 0.0 or node.op == "mul" and lv == 1.0:
                result = right
            elif node.op == "mul" and (lv == 0.0 or rv == 0.0):
                result = optimized.const(0.0)
            else:
                result = getattr(optimized, node.op)(left, right)
        mapping[nid] = result
    return optimized, mapping[output]


@dataclass(frozen=True)
class ADOrderingReport:
    selected_strategy: str
    forward_nodes: int
    optimized_forward_nodes: int
    differentiate_high_optimize_low_nodes: int
    optimize_high_differentiate_low_nodes: int
    differentiate_high_optimize_low_ns: int
    optimize_high_differentiate_low_ns: int
    max_gradient_error: float


def compare_ad_ordering(tape: Tape, output: int, env: dict) -> ADOrderingReport:
    names = tuple(env)
    forward_nodes = len(_topo_order(tape, output))
    start = time.perf_counter_ns(); high_gradients = grad_graph(tape, output, names)
    high_values = {}; high_nodes = 0
    for name in names:
        gradient_tape, gradient = optimize_tape(tape, high_gradients[name])
        high_values[name] = evaluate(gradient_tape, gradient, env)
        high_nodes += gradient_tape.node_count()
    high_ns = max(1, time.perf_counter_ns() - start)

    start = time.perf_counter_ns(); optimized, optimized_output = optimize_tape(tape, output)
    optimized_forward_nodes = optimized.node_count()
    low_gradients = grad_graph(optimized, optimized_output, names)
    low_values = {name: evaluate(optimized, node, env) for name, node in low_gradients.items()}
    low_ns = max(1, time.perf_counter_ns() - start)
    error = max((abs(high_values[name] - low_values[name]) for name in names), default=0.0)
    return ADOrderingReport(
        "differentiate-high-optimize-low", forward_nodes, optimized_forward_nodes,
        high_nodes, optimized.node_count(), high_ns, low_ns, error)
