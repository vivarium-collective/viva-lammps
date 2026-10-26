# viva-lammps

Process-bigraph wrapper for the [LAMMPS](https://www.lammps.org/) molecular dynamics simulator.

`LAMMPSProcess` is a bidirectional bridge: it advances a LAMMPS simulation each step, emits the thermodynamic state as sensor outputs, and consumes per-step control inputs (target temperature, target pressure, applied force) so a sibling process — a controller, scheduler, or environment — can drive the simulation through standard process-bigraph wiring.

## Installation

```bash
git clone https://github.com/vivarium-collective/viva-lammps.git
cd viva-lammps
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
# LAMMPS Python bindings
uv pip install lammps
```

## Quick Start

A LAMMPSProcess is configured with a standard LAMMPS input script (`.in` file or inline). `run` / `rerun` commands are stripped at load time — the process-bigraph orchestrator drives integration via `update(interval=...)`. Any LAMMPS fix the bridge owns at runtime (the thermostat, barostat, or `addforce` fix) is also stripped from the script and re-issued each step from the matching input port.

### Open-loop run

```python
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_lammps import register_pbg_lammps, make_lammps_document

core = allocate_core()
register_pbg_lammps(core)
core.register_link('ram-emitter', RAMEmitter)

doc = make_lammps_document(input_script="""
units lj
atom_style atomic
dimension 3
boundary p p p
lattice fcc 0.85
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_modify shift yes
velocity all create 1.0 87287 dist gaussian
timestep 0.005
fix integ all nvt temp 1.0 1.0 0.5
""", interval=1.0)

sim = Composite({'state': doc}, core=core)
sim.run(10.0)
print(sim.state['stores']['temperature'])
```

### Closed-loop temperature control

```python
from pbg_lammps import make_pid_controlled_document

doc = make_pid_controlled_document(
    input_script=NVT_SCRIPT,
    target_temperature=2.5,
    interval=0.5,
    kp=1.4,
    initial_target_temperature=0.8,
    thermostat_damping=0.3,
)
sim = Composite({'state': doc}, core=core)
sim.run(20.0)
# controls/target_temperature was driven by the PID controller toward 2.5;
# stores/temperature tracks it.
```

## API Reference

### LAMMPSProcess ports

| Port | Direction | Type | Notes |
|---|---|---|---|
| `target_temperature` | input | `float` | Setpoint for the thermostat fix the bridge owns. Ignored if `thermostat_fix` is `''`. |
| `target_pressure` | input | `float` | Setpoint for the barostat fix the bridge owns. Ignored if `barostat_fix` is `''`. |
| `external_force` | input | `list[float]` | Three-component force applied via `fix addforce`. Ignored if `force_fix` is `''`. |
| `temperature`, `potential_energy`, `kinetic_energy`, `total_energy`, `pressure`, `pxx`, `pyy`, `pzz`, `volume` | output | `overwrite[float]` | Instantaneous sensor readings. |
| `num_atoms` | output | `overwrite[integer]` | |
| `positions`, `velocities`, `atom_types`, `box_dimensions` | output | `overwrite[list]` | |

Sensor outputs use `overwrite[T]` because each emission reports the *current* state of the simulator, not a delta. Inputs use bare types so a sibling process can compute and write them with normal arithmetic semantics.

### LAMMPSProcess config

| Field | Default | Description |
|---|---|---|
| `input_file` / `input_script` | `''` | Path to a `.in` file, or inline LAMMPS script. Exactly one is required. |
| `working_directory` | `''` | For resolving relative paths in commands like `read_data`. |
| `thermostat_fix` | `''` | Name of the LAMMPS fix the bridge owns for the thermostat. When set, lines starting with `fix <name>` or `unfix <name>` are stripped from the script, and the bridge re-issues `fix <name> <group> <style> temp T T damping` whenever `target_temperature` changes. |
| `thermostat_style` | `nvt` | Fix style passed after the group name (`nvt`, `nvh`, ...). |
| `thermostat_group` | `all` | LAMMPS group the thermostat acts on. |
| `thermostat_damping` | `0.5` | Tdamp in LJ time units. |
| `barostat_fix` / `barostat_style` / `barostat_group` / `barostat_damping` | `''` / `iso` / `all` / `1.0` | Same shape, but for `target_pressure`. |
| `force_fix` / `force_group` | `''` / `all` | When set, the bridge re-issues `fix <name> <group> addforce Fx Fy Fz` each time `external_force` changes. |

### Helper processes

* **`PIDController`** — observes a scalar measurement and emits a clamped setpoint. `inputs: measurement: float`, `outputs: setpoint: overwrite[float]`. Config: `target`, `kp`, `ki`, `kd`, `initial_setpoint`, `setpoint_min`, `setpoint_max`.
* **`OscillatingForce`** — emits `A * sin(2 pi f t + phi) * direction` as a `list[float]` of length 3. `outputs: force: overwrite[list]`.

### Composite factories

* `make_lammps_document(...)` — wraps LAMMPSProcess with `controls` and `stores` substores; control input ports are pre-populated so a downstream factory or test can write to them.
* `make_pid_controlled_document(...)` — adds a `PIDController` whose `setpoint` writes back to `controls/target_temperature`. Closes a feedback loop on temperature.
* `make_force_driven_document(...)` — adds an `OscillatingForce` whose `force` writes to `controls/external_force`.

### `register_pbg_lammps(core)`

Convenience that registers `LAMMPSProcess`, `PIDController`, and `OscillatingForce` against a process-bigraph core in one call.

## Architecture

```
                   ┌──────────────────┐
                   │  PIDController   │
                   │  (or scheduler,  │
                   │   driver, ...)   │
                   └────┬─────────┬───┘
                        │ measurement
                        ▼         │ setpoint
              ┌──────────┐        │
              │  stores  │◄──────┐│
              │  /T,...  │       ││
              └────┬─────┘       ▼▼
                   │       ┌──────────────┐
        outputs    │       │   controls   │
        ──────────►│       │  /target_T,  │
                   │       │  /target_P,  │
                   │       │  /ext_force  │
                   │       └─────┬────────┘
                   │             │ inputs
                   ▼             ▼
              ┌──────────────────────────┐
              │      LAMMPSProcess       │
              │  (bridges to LAMMPS C++) │
              └──────────────────────────┘
```

Each step the LAMMPS bridge: (1) reads the `controls/*` substores, (2) issues `unfix`/`fix` LAMMPS commands to update the thermostat, barostat, or addforce fix it owns, (3) advances LAMMPS by `interval / timestep` integration steps with `pre yes` whenever a managed fix changed, (4) reads back thermodynamic and atomic state and writes it to `stores/*`.

## Demo

```bash
source .venv/bin/activate
python demo/demo_report.py
```

Generates `demo/report.html` with three molecular-dynamics scenarios and opens it in the default browser:

1. **Spinodal decomposition** — 50:50 binary Lennard-Jones mixture (2,048 atoms) quenched below its consolute temperature, demixing into A-rich and B-rich domains. Pure NVT physics observed via the bigraph emitter; no control input perturbs the simulation.
2. **Kremer-Grest polymer melt** — 36 FENE bead-spring chains × 20 beads each (720 atoms) relaxing from straight-rod initial conditions into a disordered melt. Pure NVT, observed.
3. **Reversible pressure cycle (NPT)** — a `ScheduledSetpoint` process drives `target_pressure` through a closed schedule (1.0 → 4.5 → 1.0). LAMMPS runs an NPT barostat that re-issues itself from the current target each step, so the simulation box visibly contracts at the apex and re-expands. The schedule closes, so the macroscopic state returns close to its starting point — a closed orbit in P-V space that exercises the new input port without permanently disturbing equilibrium.

Each section embeds the LAMMPS `.in` file used, an interactive 3D particle viewer (Three.js with InstancedMesh), Plotly charts (energy components, temperature, pressure-volume trajectory), a colored bigraph-viz architecture diagram, and a collapsible PBG composite-document tree.
