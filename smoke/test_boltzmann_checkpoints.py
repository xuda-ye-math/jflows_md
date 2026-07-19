#!/usr/bin/env python
"""Complete-stage persistence and post-sharpen resume checks."""

import inspect
import json
from functools import partial, wraps
from pathlib import Path
from tempfile import TemporaryDirectory

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import jflows_md.boltzmann as compute
import jflows_md.boltzmann.load as store
import jflows_md.boltzmann.write as write
from jflows_md.core.domain import Mixed_Domain
from jflows_md.flow import Mixed_NSF


class Probe_Flow(eqx.Module):
    shift: jax.Array
    tag: int = eqx.field(static=True)


class Activation_Flow(eqx.Module):
    shift: jax.Array
    activation: object = eqx.field(static=True)


class Aliased_Flow(eqx.Module):
    shift: jax.Array
    first: object = eqx.field(static=True)
    second: object = eqx.field(static=True)
    selected: object = eqx.field(static=True)


def shifted_activation(value, offset):
    return value + offset


GLOBAL_OFFSET = 0.0


def global_activation(value):
    return value + GLOBAL_OFFSET


class Scaler:
    def __init__(self, scale):
        self.scale = scale

    def apply(self, value):
        return self.scale * value


class Unsupported:
    __slots__ = ()


class Slotted_Activation:
    __slots__ = ("offset",)

    def __init__(self, offset):
        self.offset = offset

    def __call__(self, value):
        return value + self.offset


class Class_State_Activation:
    __slots__ = ()
    offset = 0.0

    def __call__(self, value):
        return value + self.offset


def base_activation(value):
    return value


def decorated_activation(offset):
    def decorate(function):
        @wraps(function)
        def wrapped(value):
            return function(value) + offset

        return wrapped

    return decorate(base_activation)


def attributed_activation(value):
    return value + attributed_activation.offset


attributed_activation.offset = 0.0


RG0 = np.asarray((100.0, 0.2))
RG1 = np.asarray((10.0, 0.0))


def _record(t_start, t_end, flow, objective="forward_klx"):
    rg_start = RG0 + t_start * (RG1 - RG0)
    rg_end = RG0 + t_end * (RG1 - RG0)
    continuation = Probe_Flow(flow.shift + 1.0, flow.tag)
    return {
        "t": t_end,
        "t_start": t_start,
        "rg_start": tuple(rg_start),
        "rg_end": tuple(rg_end),
        "flow_rg": tuple(rg_start),
        "population_rg": tuple(rg_end),
        "flow_endpoint": "pre_sharpen",
        "valid_selected_ess": 0.8,
        "valid_trained_ess": 0.8,
        "valid_identity_ess": 0.5,
        "valid_sample_count": 4,
        "initialized_from_identity": True,
        "selected": "trained",
        "flow": continuation,
        "continuation_flow": continuation,
        "t_hist": jnp.asarray((t_end,)),
        "batch_ess_hist": jnp.asarray(((0.7, 0.8),)),
        "valid_trained_ess_hist": jnp.asarray((0.8,)),
        "valid_identity_ess_hist": jnp.asarray((0.5,)),
        "sharpen_ess_hist": jnp.asarray((0.6,)),
        "attempt_status_hist": ("accepted",),
        "selection_history": ({"decision": "accepted"},),
        "smc_ess": jnp.asarray((0.9,)),
        "smc_acceptance": jnp.asarray(((0.75,),)),
        "mala_acceptance": jnp.asarray((0.7,)),
        "hat_mala_acceptance": (
            jnp.asarray((0.55,)) if objective == "forward_klxx" else None
        ),
        "sharpen_ess": 0.6,
        "sharpen_mala_acceptance": jnp.asarray((0.65,)),
        "objective": objective,
        "elapsed_seconds": 1.0,
    }, continuation


def _iterator(calls):
    def iterate(samples, flow, accepted, number):
        calls.append((np.asarray(samples).copy(), float(flow.shift[0]), accepted, number))
        for t_end in (0.5, 1.0):
            if t_end <= accepted[-1]:
                continue
            record, continuation = _record(accepted[-1], t_end, flow)
            samples = samples + t_end
            yield samples, record, continuation
            accepted = (*accepted, t_end)
            flow = continuation

    return iterate


def main() -> None:
    assert compute.__all__ == [
        "boltzmann_forward_KLX_G",
        "boltzmann_forward_KLXX_G",
        "iterate_boltzmann",
    ]
    assert write.__all__ == ["create", "finish", "stage"]
    assert store.__all__ == [
        "fork",
        "fork_run",
        "inspect_run",
        "load",
        "load_stage_flow",
        "load_training_history",
        "load_validation_samples",
        "manifest",
        "run",
        "validate",
        "validate_run",
    ]
    assert "run_dir" not in inspect.signature(
        compute.boltzmann_forward_KLX_G
    ).parameters
    assert "resume" not in inspect.signature(
        compute.boltzmann_forward_KLXX_G
    ).parameters

    samples = jnp.arange(8.0).reshape(4, 2)
    flow = Probe_Flow(jnp.zeros((1,)), 0)
    config = {"rg_param_0": tuple(RG0), "rg_param_1": tuple(RG1)}
    calls = []

    def first_stage(*args):
        yield next(iter(_iterator(calls)(*args)))

    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        unsupported_root = base / "unsupported"
        try:
            write.create(
                unsupported_root,
                "unsupported-template",
                config,
                samples,
                Activation_Flow(jnp.zeros((1,)), Unsupported()),
            )
        except TypeError:
            pass
        else:
            raise AssertionError("unsupported static flow value was accepted")
        assert not unsupported_root.exists()

        slotted_root = base / "slotted"
        slotted = Activation_Flow(
            jnp.zeros((1,)), Slotted_Activation(0.0)
        )
        write.create(
            slotted_root, "slotted-template", config, samples, slotted
        )
        store.load(slotted_root, slotted)
        try:
            store.load(
                slotted_root,
                Activation_Flow(
                    jnp.zeros((1,)), Slotted_Activation(1.0)
                ),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("slotted callable state was ignored")

        class_state_root = base / "class-state"
        class_state = Activation_Flow(
            jnp.zeros((1,)), Class_State_Activation()
        )
        write.create(
            class_state_root,
            "class-state-template",
            config,
            samples,
            class_state,
        )
        store.load(class_state_root, class_state)
        Class_State_Activation.offset = 1.0
        try:
            store.load(class_state_root, class_state)
        except ValueError:
            pass
        else:
            raise AssertionError("callable class state was ignored")
        finally:
            Class_State_Activation.offset = 0.0

        mixed_root = base / "mixed"
        mixed_flow = Mixed_NSF(
            jax.random.key(1),
            Mixed_Domain(2, 1),
            bins=4,
            transforms=2,
            hidden_features=(4,),
        )
        mixed_samples = jnp.arange(12.0).reshape(4, 3)
        mixed_run = write.create(
            mixed_root, "mixed-template", config, mixed_samples, mixed_flow
        )
        mixed_record, _ = _record(0.0, 1.0, flow)
        mixed_record["flow"] = mixed_flow
        mixed_record["continuation_flow"] = mixed_flow
        write.stage(
            mixed_root, mixed_run, mixed_record, mixed_samples
        )
        write.finish(mixed_root, mixed_run, "complete")
        restored_samples, _, restored_records = store.load(
            mixed_root, mixed_flow
        )
        assert restored_samples.shape == (4, 3)
        assert len(restored_records) == 1

        alias_root = base / "alias"
        first = Scaler(1.0)
        second = Scaler(2.0)
        aliased = Aliased_Flow(
            jnp.zeros((1,)), first, second, first
        )
        write.create(
            alias_root, "alias-template", config, samples, aliased
        )
        store.load(alias_root, aliased)
        try:
            store.load(
                alias_root,
                Aliased_Flow(
                    jnp.zeros((1,)), first, second, second
                ),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("static object aliases were conflated")

        occupied = base / "occupied"
        occupied.mkdir()
        (occupied / "keep.txt").write_text("keep\n", encoding="utf-8")
        try:
            write.create(occupied, "no-overwrite", config, samples, flow)
        except FileExistsError:
            pass
        else:
            raise AssertionError("nonempty run directory was overwritten")

        mismatch = base / "mismatch"
        bad_config = {**config, "rg_param_1": (20.0, 0.0)}
        try:
            store.run(
                mismatch,
                "bad-schedule",
                bad_config,
                samples,
                flow,
                first_stage,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("stage regularization ignored the run config")
        assert store.manifest(mismatch)["stages"] == []

        activation_root = base / "activation"
        activation = Activation_Flow(
            jnp.zeros((1,)), partial(shifted_activation, offset=0.0)
        )
        write.create(
            activation_root,
            "activation-template",
            config,
            samples,
            activation,
        )
        store.load(activation_root, activation)
        try:
            store.load(
                activation_root,
                Activation_Flow(
                    jnp.zeros((1,)), partial(shifted_activation, offset=1.0)
                ),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("partial activation arguments were ignored")

        decorated_root = base / "decorated"
        decorated = Activation_Flow(
            jnp.zeros((1,)), decorated_activation(0.0)
        )
        write.create(
            decorated_root, "decorated-template", config, samples, decorated
        )
        try:
            store.load(
                decorated_root,
                Activation_Flow(
                    jnp.zeros((1,)), decorated_activation(1.0)
                ),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("decorated activation closure was ignored")

        attributed_root = base / "attributed"
        attributed = Activation_Flow(jnp.zeros((1,)), attributed_activation)
        write.create(
            attributed_root, "attributed-template", config, samples, attributed
        )
        attributed_activation.offset = 1.0
        try:
            store.load(attributed_root, attributed)
        except ValueError:
            pass
        else:
            raise AssertionError("activation function attributes were ignored")
        attributed_activation.offset = 0.0

        bound_root = base / "bound"
        bound = Activation_Flow(jnp.zeros((1,)), Scaler(1.0).apply)
        write.create(bound_root, "bound-template", config, samples, bound)
        try:
            store.load(
                bound_root,
                Activation_Flow(jnp.zeros((1,)), Scaler(2.0).apply),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("bound activation receiver state was ignored")

        primitive_root = base / "primitive"
        primitive = Probe_Flow(jnp.zeros((1,)), True)
        write.create(
            primitive_root, "primitive-template", config, samples, primitive
        )
        try:
            store.load(primitive_root, Probe_Flow(jnp.zeros((1,)), 1))
        except ValueError:
            pass
        else:
            raise AssertionError("primitive static types were conflated")

        global GLOBAL_OFFSET
        global_root = base / "global"
        global_flow = Activation_Flow(jnp.zeros((1,)), global_activation)
        write.create(global_root, "global-template", config, samples, global_flow)
        GLOBAL_OFFSET = 1.0
        try:
            store.load(global_root, global_flow)
        except ValueError:
            pass
        else:
            raise AssertionError("activation global state was ignored")
        GLOBAL_OFFSET = 0.0

        klxx_root = base / "klxx"
        klxx_run = write.create(
            klxx_root, "klxx-history", config, samples, flow
        )
        klxx_record, _ = _record(0.0, 1.0, flow, "forward_klxx")
        write.stage(klxx_root, klxx_run, klxx_record, samples)
        write.finish(klxx_root, klxx_run, "complete")
        assert "hat_mala_acceptance" in store.load_training_history(klxx_root, 1)

        root = base / "run"
        partial_samples, records = store.run(
            root, "sharpen-resume", config, samples, flow, first_stage
        )
        assert len(records) == 1
        assert store.manifest(root)["format"] == "jflows-md-stage-resume-1"
        assert store.manifest(root)["status"] == "exhausted"
        np.testing.assert_array_equal(partial_samples, samples + 0.5)

        orphan = root / "stages" / "stage_000002"
        outside = base / "outside"
        outside.mkdir()
        orphan.symlink_to(outside, target_is_directory=True)
        try:
            store.run(
                root,
                "sharpen-resume",
                config,
                None,
                flow,
                _iterator([]),
                resume=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("stage symlink was followed")
        assert not any(outside.iterdir())
        assert len(store.manifest(root)["stages"]) == 1
        orphan.unlink()

        orphan.mkdir()
        (orphan / "unpublished.txt").write_text("ignored\n", encoding="utf-8")
        loaded, _, loaded_records = store.load(root, flow)
        np.testing.assert_array_equal(loaded, partial_samples)
        assert len(loaded_records) == 1

        final, records = store.run(
            root,
            "sharpen-resume",
            config,
            None,
            flow,
            _iterator(calls),
            resume=True,
        )
        np.testing.assert_array_equal(calls[-1][0], partial_samples)
        assert calls[-1][1:] == (1.0, (0.0, 0.5), 2)
        np.testing.assert_array_equal(final, samples + 1.5)
        assert len(records) == 2 and records[-1]["t"] == 1.0
        assert store.manifest(root)["status"] == "complete"

        loaded, continuation, records = store.load(root, flow)
        np.testing.assert_array_equal(loaded, final)
        assert float(continuation.shift[0]) == 2.0
        assert records[-1]["rg_end"] == tuple(RG1)
        assert records[-1]["population_rg"] == tuple(RG1)
        assert records[-1]["flow_endpoint"] == "pre_sharpen"
        assert records[-1]["sharpen_ess"] == 0.6
        assert records[-1]["hat_mala_acceptance"] is None
        assert store.load_validation_samples(root, 2).shape == samples.shape
        assert float(store.load_stage_flow(root, 2, "selected", flow).shift[0]) == 2.0
        assert float(store.load_stage_flow(root, 2, "continuation", flow).shift[0]) == 2.0
        history = store.load_training_history(root, 2)
        assert history["batch_ess_hist"].shape == (1, 2)
        np.testing.assert_allclose(history["sharpen_ess_hist"], [0.6])
        np.testing.assert_allclose(
            history["sharpen_mala_acceptance"], [0.65], rtol=1e-6, atol=0.0
        )
        assert "hat_mala_acceptance" not in history
        assert (orphan / "unpublished.txt").is_file()
        try:
            store.load_validation_samples(root, 0)
        except IndexError:
            pass
        else:
            raise AssertionError("zero silently selected the final stage")

        for problem_id, resume_config in (
            ("different", config),
            ("sharpen-resume", {**config, "rg_param_1": (20.0, 0.0)}),
        ):
            try:
                store.run(
                    root,
                    problem_id,
                    resume_config,
                    None,
                    flow,
                    _iterator(calls),
                    resume=True,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("mismatched resume was accepted")
        try:
            store.load(root, Probe_Flow(jnp.zeros((1,)), 1))
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched static flow template was accepted")
        try:
            store.fork(root, root / "recursive")
        except ValueError:
            pass
        else:
            raise AssertionError("recursive fork destination was accepted")

        forked = base / "forked"
        forked_manifest = store.fork(root, forked, "forked-run")
        assert forked_manifest["problem_id"] == "forked-run"
        assert forked_manifest["status"] == "running"
        escaped = store.manifest(forked)
        escaped["initial_samples_path"] = "../outside.npy"
        (forked / "run.json").write_text(
            json.dumps(escaped), encoding="utf-8"
        )
        try:
            store.validate(forked)
        except ValueError:
            pass
        else:
            raise AssertionError("escaping manifest path was accepted")

    print("PASS molecular post-sharpen stage resume")


if __name__ == "__main__":
    main()
