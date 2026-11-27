# viva-lammps

[![Tests](https://github.com/vivarium-collective/viva-lammps/actions/workflows/tests.yml/badge.svg)](https://github.com/vivarium-collective/viva-lammps/actions/workflows/tests.yml)
[![license](https://img.shields.io/github/license/vivarium-collective/viva-lammps)](./LICENSE)

Process-bigraph **engine provider** for the [LAMMPS](https://www.lammps.org/) molecular dynamics simulator.

This repository is the installable Python package `pbg_lammps` (distribution name `viva-lammps`). It wraps LAMMPS as process-bigraph Processes and Composites that other workspaces import and compose. It is **not** a research workspace: the demo studies and investigations that used to live here have moved to the shared showcase workspace at **[vivarium-collective/viva-showcase](https://github.com/vivarium-collective/viva-showcase)**, which pins this package by tag.

`LAMMPSProcess` is a bidirectional bridge: it advances a LAMMPS simulation each step, emits the thermodynamic state as sensor outputs, and consumes per-step control inputs (target temperature, target pressure, applied force) so a sibling process — a controller, scheduler, or environment — can drive the simulation through standard process-bigraph wiring.

## Installation

```bash
pip install -e ".[dev]"
# LAMMPS Python bindings (engine; may require a system/conda build)
pip install lammps
```

The `pbg_lammps` package imports without the `lammps` engine installed; the engine is only needed to actually run a simulation.

## Usage

```python
from process_bigraph import allocate_core
from pbg_lammps import register_pbg_lammps  # wire the package's processes into a core

core = allocate_core()
register_pbg_lammps(core)
```

See `demo/` for a self-contained Lennard-Jones example, and the composites under `pbg_lammps/composites/` for reusable building blocks.

## License

MIT — see [LICENSE](./LICENSE).
