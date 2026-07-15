"""MC2 governed registry inspector/poke tests."""

from dataclasses import replace

from bcir.examples import vector_add
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.registry_tool import RegistryInspector, execute_command
from bcir.verify import verify_pack


def _planned():
    module = vector_add(8)
    result = optimize(module, TargetProfile.x86_avx512(), Theta.cool())
    return module, hydrate(module, result)


def test_governed_poke_bumps_generation_and_stales_existing_pack():
    module, pack = _planned()
    inspector = RegistryInspector(module)
    result = inspector.setp("A[3]", 0x2A)
    assert result.previous is None and result.value == 42
    assert result.previous_data_gen == 0 and result.data_gen == 1
    assert inspector.getp("10[3]") == 42
    assert any(diag.law == "R11" and "data_gen" in diag.message
               for diag in verify_pack(module, pack))


def test_failed_poke_does_not_mutate_value_or_generation():
    module, _pack = _planned()
    inspector = RegistryInspector(module)
    before = module.resources[10]
    try:
        inspector.setp("A[99]", 1)
        assert False, "expected bounds refusal"
    except IndexError:
        pass
    assert module.resources[10] == before
    assert inspector.getp("A[0]") is None


def test_generation_overflow_refuses_before_mutation():
    module, _pack = _planned()
    module.resources[10] = replace(module.resources[10], data_gen=0xFFFFFFFF)
    inspector = RegistryInspector(module)
    try:
        inspector.setp("A[0]", 1)
        assert False, "expected a u32 generation-overflow refusal"
    except OverflowError:
        pass
    assert module.resources[10].data_gen == 0xFFFFFFFF
    assert inspector.getp("A[0]") is None


def test_poke_converts_the_value_once_before_publication():
    class ConvertOnce:
        calls = 0

        def __int__(self):
            self.calls += 1
            if self.calls != 1:
                raise RuntimeError("value converted more than once")
            return 7

    module, _pack = _planned()
    inspector = RegistryInspector(module)
    value = ConvertOnce()
    result = inspector.setp("A[0]", value)  # type: ignore[arg-type]
    assert value.calls == 1 and result.value == 7
    assert inspector.getp("A[0]") == 7 and module.resources[10].data_gen == 1


def test_array_requires_an_explicit_index_and_unknown_rids_refuse():
    inspector = RegistryInspector(vector_add(8))
    for reference in ("A", "999[0]"):
        try:
            inspector.resolve(reference)
            assert False, f"expected refusal for {reference}"
        except (KeyError, ValueError):
            pass


def test_operator_commands_are_deterministic_and_stateful():
    inspector = RegistryInspector(vector_add(8))
    keep, text = execute_command(inspector, "show resources")
    assert keep and text.splitlines()[0].startswith("rid=10 name=A")
    keep, text = execute_command(inspector, "setp A[0] 0x10")
    assert keep and "value=16 data_gen=0->1" in text
    keep, text = execute_command(inspector, "getp 10[0]")
    assert keep and "value=16" in text
