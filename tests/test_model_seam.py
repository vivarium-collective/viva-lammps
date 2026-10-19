"""Tests for the LAMMPS model-plugin seam (LammpsModel / LammpsModelProcess).

The seam code must import cleanly whether or not the ``lammps`` python binding
is installed; tests that actually build a simulation skip gracefully when it is
not available.
"""

import importlib.util

import pytest
from process_bigraph import allocate_core

from pbg_lammps import (
    LammpsModel,
    LammpsModelProcess,
    LennardJonesBox,
    register_model,
    get_model,
    list_models,
    register_pbg_lammps,
)


HAVE_LAMMPS = importlib.util.find_spec('lammps') is not None
requires_lammps = pytest.mark.skipif(
    not HAVE_LAMMPS, reason='lammps python binding not installed')


@pytest.fixture
def core():
    c = allocate_core()
    register_pbg_lammps(c)
    return c


# ── Registry / protocol (no lammps needed) ──────────────────────────

def test_demo_model_registered():
    assert 'lj-box' in list_models()
    assert get_model('lj-box') is LennardJonesBox
    assert issubclass(LennardJonesBox, LammpsModel)


def test_get_model_unknown_raises():
    with pytest.raises(KeyError):
        get_model('does-not-exist')


def test_register_model_rejects_non_model():
    class NotAModel:
        pass
    with pytest.raises(TypeError):
        register_model('bad', NotAModel)


def test_register_and_retrieve_roundtrip():
    class TmpModel(LammpsModel):
        def build_data(self, params):
            return None, ''
        def build_script(self, params, data_path):
            return ''
        def ports(self):
            return {}, {}
        def read_observables(self, lmp):
            return {}
    register_model('tmp-model', TmpModel)
    try:
        assert get_model('tmp-model') is TmpModel
    finally:
        # keep the shared registry clean for other tests
        from pbg_lammps import model as model_mod
        model_mod._MODEL_REGISTRY.pop('tmp-model', None)


def test_on_step_default_is_noop():
    # The base on_step is a no-op and must not require a live simulation.
    assert LennardJonesBox().on_step(lmp=None, inputs={}, interval=1.0) is None


def test_demo_model_build_script_and_ports():
    model = LennardJonesBox()
    data, wd = model.build_data({'n': 3})
    assert data is None
    script = model.build_script({'n': 3, 'density': 0.5}, data)
    assert 'pair_style lj/cut' in script
    assert 'fix integ all nve' in script
    inputs, outputs = model.ports()
    assert inputs == {}
    for port in ('temperature', 'total_energy', 'num_atoms'):
        assert port in outputs


def test_process_requires_model(core):
    with pytest.raises(ValueError):
        LammpsModelProcess(config={}, core=core)


def test_process_exposes_model_ports(core):
    proc = LammpsModelProcess(
        config={'model': 'lj-box', 'params': {'n': 3}}, core=core)
    assert proc.inputs() == {}
    assert proc.outputs()['temperature'].startswith('overwrite')


# ── Live simulation through the seam (needs lammps) ─────────────────

@requires_lammps
def test_demo_model_initial_state(core):
    proc = LammpsModelProcess(
        config={'model': 'lj-box', 'params': {'n': 3, 'density': 0.5}},
        core=core)
    state = proc.initial_state()
    assert state['num_atoms'] == 27
    assert isinstance(state['total_energy'], float)
    proc.close()


@requires_lammps
def test_demo_model_update_produces_observable(core):
    proc = LammpsModelProcess(
        config={'model': 'lj-box', 'params': {'n': 3, 'density': 0.5}},
        core=core)
    proc.initial_state()
    result = None
    for _ in range(3):
        result = proc.update({}, interval=0.5)
    assert isinstance(result['temperature'], float)
    assert result['temperature'] > 0
    assert result['num_atoms'] == 27
    proc.close()


@requires_lammps
def test_seam_composite_runs(core):
    """The lj-box-model composite runs through the declarative loader."""
    from process_bigraph import Composite, gather_emitter_results
    from process_bigraph.emitter import RAMEmitter
    from pbg_lammps.composites import build_composite

    core.register_link('ram-emitter', RAMEmitter)
    sim = build_composite('lj-box-model', overrides={'interval': 0.5}, core=core)
    assert isinstance(sim, Composite)
    sim.run(1.0)
    raw = gather_emitter_results(sim)
    series = raw[('emitter',)]
    assert len(series) >= 1
    assert 'total_energy' in series[0]
