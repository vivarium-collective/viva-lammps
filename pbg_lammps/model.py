"""Model-plugin seam for LAMMPS over a single general Process.

Instead of writing a bespoke :class:`~process_bigraph.Process` for every new
molecular LAMMPS model (an actomyosin cortex, a collagen network, a lipid
bilayer, …), a model is written once as a :class:`LammpsModel` *plugin* and
driven by the single general :class:`LammpsModelProcess`. The process owns the
LAMMPS lifecycle (instantiate, feed the input script, integrate, read back,
close) and delegates everything model-specific to the plugin selected by
``config['model']``.

A plugin implements five hooks:

    build_data(params)            -> (data_path_or_str, working_dir)
    build_script(params, data)    -> str  (the LAMMPS input script)
    ports()                       -> (inputs: dict, outputs: dict)
    read_observables(lmp)         -> dict  (this model's observables)
    on_step(lmp, inputs, interval)-> None  (optional; per-step actions)

Plugins register themselves under a name via :func:`register_model`, and
``LammpsModelProcess(config={'model': <name>, 'params': {...}})`` wires them in.

This module imports cleanly whether or not the ``lammps`` python binding is
installed; ``lammps`` is imported lazily inside ``LammpsModelProcess`` only when
a simulation is actually built.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from process_bigraph import Process

from pbg_lammps.processes import _strip_active_command


# ---------------------------------------------------------------------------
# Plugin protocol
# ---------------------------------------------------------------------------


class LammpsModel(ABC):
    """Base class for a LAMMPS model plugin.

    A plugin is stateless with respect to the live simulation: it is handed the
    live ``lammps`` instance (``lmp``) by the process whenever it needs to read
    or act on it. All model-specific knowledge — how to lay out atoms, which
    pair styles and fixes to use, which observables to expose, how to respond to
    per-step inputs — lives here, so the process itself stays general.
    """

    @abstractmethod
    def build_data(self, params: dict) -> tuple[Any, str]:
        """Produce the LAMMPS data (atoms / box / topology).

        Return ``(data, working_dir)`` where ``data`` is either a path to a
        LAMMPS data file this method wrote, an inline data string, or ``None``
        when the input script creates its own atoms (e.g. ``create_atoms``).
        ``working_dir`` is the directory LAMMPS should run in (``''`` = cwd);
        it is where any written data file lives and where ``build_script``
        should reference it from.
        """

    @abstractmethod
    def build_script(self, params: dict, data_path: Any) -> str:
        """Return the LAMMPS input script for this model.

        ``data_path`` is the first element returned by :meth:`build_data`. The
        script sets up units, box, pair styles, integrator/fixes and any base
        run setup. ``run``/``rerun`` lines are optional — the process strips
        them and drives integration itself — so they may be left in or out.
        """

    @abstractmethod
    def ports(self) -> tuple[dict, dict]:
        """Return ``(inputs, outputs)`` port schemas for this model.

        Both are process-bigraph port-schema dicts. ``outputs`` typically use
        ``overwrite[T]`` because each emission reports current state, not a
        delta. ``inputs`` may be empty for a passive model.
        """

    @abstractmethod
    def read_observables(self, lmp) -> dict:
        """Pull this model's observables from the live LAMMPS instance.

        ``lmp`` is a ``lammps.lammps`` object. Return a dict keyed by the
        output ports declared in :meth:`ports`. Use ``lmp.get_thermo(...)``,
        ``lmp.numpy.extract_atom(...)``, ``lmp.extract_compute(...)``, etc.
        """

    def on_step(self, lmp, inputs: dict, interval: float) -> None:
        """Apply per-step inputs / model actions before integrating.

        Called every ``update`` with the incoming bigraph ``inputs`` (the ports
        from :meth:`ports`) and the integration ``interval``. Use it to push
        setpoints into LAMMPS (``lmp.command(...)``) or to perform dynamic
        actions such as making/breaking bonds. Default is a no-op, which is
        correct for a passive model that only reports observables.
        """
        return None


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------


_MODEL_REGISTRY: dict[str, type[LammpsModel]] = {}


def register_model(name: str, cls: type[LammpsModel]) -> None:
    """Register a :class:`LammpsModel` subclass under ``name``."""
    if not (isinstance(cls, type) and issubclass(cls, LammpsModel)):
        raise TypeError(
            f"model {name!r} must be a LammpsModel subclass, got {cls!r}")
    _MODEL_REGISTRY[name] = cls


def get_model(name: str) -> type[LammpsModel]:
    """Return the registered :class:`LammpsModel` subclass for ``name``."""
    try:
        return _MODEL_REGISTRY[name]
    except KeyError:
        known = ', '.join(sorted(_MODEL_REGISTRY)) or '(none)'
        raise KeyError(
            f"no LAMMPS model registered as {name!r}; known models: {known}")


def list_models() -> list[str]:
    """Return the names of all registered models."""
    return sorted(_MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# General model-driven Process
# ---------------------------------------------------------------------------


class LammpsModelProcess(Process):
    """One general LAMMPS Process driven by a :class:`LammpsModel` plugin.

    Configure with the registered name of a model plus its params::

        LammpsModelProcess(config={'model': 'lj-box', 'params': {'n': 4}})

    The process owns the LAMMPS lifecycle and delegates all model-specific
    behavior to the plugin:

      * **init** — ``build_data`` + ``build_script`` → ``commands_string``
      * **initial_state** — ``run 0`` then ``read_observables``
      * **update** — ``on_step`` (apply inputs) → ``run N`` → ``read_observables``
      * **close** — release the LAMMPS instance

    Ports are whatever the plugin's ``ports()`` declares. ``LAMMPSProcess``,
    ``PIDController`` and ``OscillatingForce`` are untouched by this class.
    """

    config_schema = {
        'model': {'_type': 'string', '_default': ''},
        'params': {'_type': 'quote', '_default': {}},
        'working_directory': {'_type': 'string', '_default': ''},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        name = self.config['model']
        if not name:
            raise ValueError(
                "LammpsModelProcess requires config['model'] (a registered "
                "model name); known models: " + (', '.join(list_models()) or '(none)'))
        self._model = get_model(name)()
        self._params = self.config['params'] or {}
        self._inputs, self._outputs = self._model.ports()
        self._lmp = None
        self._dt = None
        self._first_run = True

    def inputs(self):
        return dict(self._inputs)

    def outputs(self):
        return dict(self._outputs)

    def _build_simulation(self):
        if self._lmp is not None:
            return

        from lammps import lammps

        data_path, wd = self._model.build_data(self._params)
        wd = self.config['working_directory'] or wd or ''
        script = self._model.build_script(self._params, data_path)
        # The process owns integration — strip any run/rerun the script carries.
        script = _strip_active_command(script, 'run', 'rerun')

        original_cwd = os.getcwd()
        if wd:
            os.chdir(wd)
        try:
            self._lmp = lammps(
                cmdargs=['-nocite', '-log', 'none', '-screen', 'none'])
            self._lmp.commands_string(script)
        finally:
            if wd:
                os.chdir(original_cwd)

        self._dt = self._lmp.extract_global('dt')

    def initial_state(self):
        self._build_simulation()
        self._lmp.command('run 0')
        self._first_run = False
        return self._model.read_observables(self._lmp)

    def update(self, state, interval):
        self._build_simulation()

        inputs = state or {}
        self._model.on_step(self._lmp, inputs, interval)

        n_steps = max(1, int(round(interval / self._dt)))
        if self._first_run:
            self._lmp.command(f'run {n_steps}')
            self._first_run = False
        elif inputs:
            # Inputs may have added/modified fixes in on_step; `pre yes`
            # re-initializes them before integrating.
            self._lmp.command(f'run {n_steps} pre yes post no')
        else:
            self._lmp.command(f'run {n_steps} pre no post no')

        return self._model.read_observables(self._lmp)

    def close(self):
        lmp = getattr(self, '_lmp', None)
        if lmp is not None:
            lmp.close()
            self._lmp = None

    def __del__(self):
        try:
            self.close()
        except (ImportError, TypeError):
            pass


# ---------------------------------------------------------------------------
# Reference / demo plugin — a tiny Lennard-Jones box
# ---------------------------------------------------------------------------


class LennardJonesBox(LammpsModel):
    """Minimal self-contained reference plugin: a small LJ box in NVE.

    Creates a single-type Lennard-Jones fluid on a simple-cubic lattice and
    integrates it microcanonically. It writes no data file (atoms come from
    ``create_atoms``), exposes no inputs, and reports the standard thermodynamic
    observables. Intended to prove the seam and to serve as a template for real
    models (collagen, actomyosin, …).

    params (all optional):
        n          : lattice replications per side (default 3 → n**3 atoms)
        density    : LJ number density (default 0.5)
        temperature: initial Maxwell-Boltzmann temperature (default 1.0)
        cutoff     : LJ pair cutoff (default 2.5)
        timestep   : integration timestep (default 0.005)
        seed       : velocity-init RNG seed (default 87287)
    """

    def build_data(self, params):
        # Atoms are created inline via `create_atoms`; no data file needed.
        return None, params.get('working_directory', '')

    def build_script(self, params, data_path):
        n = int(params.get('n', 3))
        density = float(params.get('density', 0.5))
        temperature = float(params.get('temperature', 1.0))
        cutoff = float(params.get('cutoff', 2.5))
        timestep = float(params.get('timestep', 0.005))
        seed = int(params.get('seed', 87287))
        lines = [
            'units lj',
            'atom_style atomic',
            'dimension 3',
            'boundary p p p',
            f'lattice sc {density}',
            f'region box block 0 {n} 0 {n} 0 {n}',
            'create_box 1 box',
            'create_atoms 1 box',
            'mass 1 1.0',
            f'pair_style lj/cut {cutoff}',
            'pair_coeff 1 1 1.0 1.0',
            'pair_modify shift yes',
            f'velocity all create {temperature} {seed} dist gaussian',
            f'timestep {timestep}',
            'fix integ all nve',
        ]
        return '\n'.join(lines) + '\n'

    def ports(self):
        inputs = {}
        outputs = {
            'temperature': 'overwrite[float]',
            'potential_energy': 'overwrite[float]',
            'kinetic_energy': 'overwrite[float]',
            'total_energy': 'overwrite[float]',
            'num_atoms': 'overwrite[integer]',
        }
        return inputs, outputs

    def read_observables(self, lmp):
        return {
            'temperature': float(lmp.get_thermo('temp')),
            'potential_energy': float(lmp.get_thermo('pe')),
            'kinetic_energy': float(lmp.get_thermo('ke')),
            'total_energy': float(lmp.get_thermo('etotal')),
            'num_atoms': int(lmp.get_natoms()),
        }


register_model('lj-box', LennardJonesBox)
