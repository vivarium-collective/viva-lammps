"""LAMMPS composite documents + composite-spec discovery.

Two flavors of composite construction live in this package:

1. **Hand-coded factories** — `make_lammps_document(input_script=…)` builds a
   PBG state-dict programmatically for callers that want full control over
   the LAMMPS script + wiring. Used by `demo/demo_report.py`.

2. **Declarative `*.composite.yaml`** — sibling files in this directory
   follow the pbg-superpowers composite-spec convention.
   `build_composite()` loads one by name and instantiates
   `process_bigraph.Composite` with parameter substitution. The dashboard's
   composite explorer discovers these automatically once the package is
   installed in a workspace.

Both flavors are equivalent — pick the one that fits your use case.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import yaml
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_lammps.processes import LAMMPSProcess, PIDController, OscillatingForce
from pbg_lammps.model import LammpsModelProcess


# ---------------------------------------------------------------------------
# Hand-coded composite factories (bidirectional-port / control-loop API)
# ---------------------------------------------------------------------------

_SENSOR_PORTS = (
    'temperature', 'potential_energy', 'kinetic_energy', 'total_energy',
    'pressure', 'num_atoms', 'positions', 'velocities', 'atom_types',
    'volume', 'pxx', 'pyy', 'pzz', 'box_dimensions',
)


def _lammps_node(input_file, input_script, working_directory, interval,
                 thermostat_fix='', thermostat_style='nvt',
                 thermostat_group='all', thermostat_damping=0.5,
                 barostat_fix='', barostat_style='iso',
                 barostat_group='all', barostat_damping=1.0,
                 force_fix='', force_group='all'):
    if not input_file and not input_script:
        raise ValueError(
            'LAMMPS document requires input_file or input_script')

    config = {
        'input_file': input_file,
        'input_script': input_script,
        'working_directory': working_directory,
        'thermostat_fix': thermostat_fix,
        'thermostat_style': thermostat_style,
        'thermostat_group': thermostat_group,
        'thermostat_damping': thermostat_damping,
        'barostat_fix': barostat_fix,
        'barostat_style': barostat_style,
        'barostat_group': barostat_group,
        'barostat_damping': barostat_damping,
        'force_fix': force_fix,
        'force_group': force_group,
    }
    inputs = {
        'target_temperature': ['controls', 'target_temperature'],
        'target_pressure': ['controls', 'target_pressure'],
        'external_force': ['controls', 'external_force'],
    }
    outputs = {p: ['stores', p] for p in _SENSOR_PORTS}
    return {
        '_type': 'process',
        'address': 'local:LAMMPSProcess',
        'config': config,
        'interval': interval,
        'inputs': inputs,
        'outputs': outputs,
    }


def _emitter_node(scalar_ports, controls=()):
    emit_schema = {p: 'float' for p in scalar_ports}
    emit_schema['time'] = 'float'
    inputs = {p: ['stores', p] for p in scalar_ports}
    inputs['time'] = ['global_time']
    for c in controls:
        emit_schema[c] = 'float'
        inputs[c] = ['controls', c]
    return {
        '_type': 'step',
        'address': 'local:RAMEmitter',
        'config': {'emit': emit_schema},
        'inputs': inputs,
    }


def make_lammps_document(
    input_file='',
    input_script='',
    working_directory='',
    interval=1.0,
    initial_target_temperature=1.0,
    initial_target_pressure=1.0,
    initial_external_force=(0.0, 0.0, 0.0),
    thermostat_fix='',
    barostat_fix='',
    force_fix='',
    thermostat_damping=0.5,
):
    """Composite document with LAMMPS control ports exposed as stores.

    By default the bridge does not own any thermostat/barostat/addforce
    fix — pass ``thermostat_fix='integ'`` (etc.) to let the bridge
    take ownership of those fixes and let upstream code write to the
    matching control store.
    """
    return {
        'lammps': _lammps_node(
            input_file=input_file,
            input_script=input_script,
            working_directory=working_directory,
            interval=interval,
            thermostat_fix=thermostat_fix,
            thermostat_damping=thermostat_damping,
            barostat_fix=barostat_fix,
            force_fix=force_fix),
        'controls': {
            'target_temperature': float(initial_target_temperature),
            'target_pressure': float(initial_target_pressure),
            'external_force': list(initial_external_force),
        },
        'stores': {},
        'emitter': _emitter_node(
            scalar_ports=('temperature', 'potential_energy', 'kinetic_energy',
                          'total_energy', 'pressure', 'volume',
                          'pxx', 'pyy', 'pzz'),
            controls=('target_temperature',)),
    }


def make_pid_controlled_document(
    input_script,
    target_temperature,
    interval=1.0,
    thermostat_fix='integ',
    thermostat_damping=0.5,
    initial_target_temperature=None,
    kp=1.0,
    ki=0.0,
    kd=0.0,
    controller_interval=None,
):
    """Closed-loop document: PIDController.setpoint -> LAMMPS.target_temperature.

    The controller observes the current measured temperature from the
    bigraph state and adjusts the LAMMPS thermostat setpoint each step.
    Both processes share the ``controls/target_temperature`` store as
    a single source of truth.
    """
    initial = (initial_target_temperature
               if initial_target_temperature is not None
               else target_temperature)
    controller_interval = controller_interval or interval

    doc = make_lammps_document(
        input_script=input_script,
        interval=interval,
        thermostat_fix=thermostat_fix,
        thermostat_damping=thermostat_damping,
        initial_target_temperature=initial,
    )

    doc['controller'] = {
        '_type': 'process',
        'address': 'local:PIDController',
        'config': {
            'target': float(target_temperature),
            'kp': kp, 'ki': ki, 'kd': kd,
            'initial_setpoint': float(initial),
        },
        'interval': controller_interval,
        'inputs': {'measurement': ['stores', 'temperature']},
        'outputs': {'setpoint': ['controls', 'target_temperature']},
    }
    return doc


def make_force_driven_document(
    input_script,
    interval=1.0,
    force_fix='ext',
    amplitude=0.5,
    frequency=0.1,
    direction=(1.0, 0.0, 0.0),
):
    """Document where an OscillatingForce drives LAMMPS.external_force."""
    doc = make_lammps_document(
        input_script=input_script,
        interval=interval,
        force_fix=force_fix,
    )
    doc['driver'] = {
        '_type': 'process',
        'address': 'local:OscillatingForce',
        'config': {
            'amplitude': amplitude,
            'frequency': frequency,
            'direction': list(direction),
        },
        'interval': interval,
        'inputs': {},
        'outputs': {'force': ['controls', 'external_force']},
    }
    return doc


def register_lammps(core=None):
    """Return a core with LAMMPSProcess, PIDController, OscillatingForce,
    the RAM emitter, and the Visualization Step(s) registered."""
    if core is None:
        core = allocate_core()
    core.register_link('LAMMPSProcess', LAMMPSProcess)
    core.register_link('PIDController', PIDController)
    core.register_link('OscillatingForce', OscillatingForce)
    core.register_link('LammpsModelProcess', LammpsModelProcess)
    core.register_link('ram-emitter', RAMEmitter)
    # Register Visualization Steps so composites can wire them by name.
    from pbg_lammps.visualizations import LAMMPSThermoPlots
    core.register_link('LAMMPSThermoPlots', LAMMPSThermoPlots)
    return core


# ---------------------------------------------------------------------------
# Declarative composite-spec loader (*.composite.yaml)
# ---------------------------------------------------------------------------

_COMPOSITES_DIR = Path(__file__).parent

_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type == "int":
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    if declared_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _substitute(state: Any, params: dict, overrides: dict) -> Any:
    if isinstance(state, dict):
        return {k: _substitute(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [_substitute(v, params, overrides) for v in state]
    if isinstance(state, str):
        m = _FULL_PLACEHOLDER.match(state)
        if m:
            pname = m.group(1)
            pdef = params.get(pname, {})
            raw = overrides.get(pname, pdef.get("default"))
            return _cast(raw, pdef.get("type"))
        if _INLINE_PLACEHOLDER.search(state):
            return _INLINE_PLACEHOLDER.sub(
                lambda mm: str(overrides.get(mm.group(1), params.get(mm.group(1), {}).get("default", ""))),
                state,
            )
    return state


def list_composite_specs() -> list[str]:
    """Return short names of every `*.composite.yaml` shipped in this package."""
    out: list[str] = []
    for path in sorted(_COMPOSITES_DIR.glob("*.composite.yaml")):
        out.append(path.name[: -len(".composite.yaml")])
    return out


def load_composite_spec(name: str) -> dict:
    """Load and parse a named composite spec. `name` is the stem (no suffix)."""
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a *.composite.yaml by name and instantiate process_bigraph.Composite.

    overrides: parameter overrides (keys must match spec.parameters)
    core:      optional pre-built core; otherwise register_lammps() is used
    """
    from process_bigraph import Composite

    spec = load_composite_spec(name)
    if not isinstance(spec, dict) or "state" not in spec or "name" not in spec:
        raise ValueError(f"composite '{name}' missing required keys (name, state)")

    if core is None:
        core = register_lammps()

    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, overrides or {})
    return Composite({"state": state}, core=core)
