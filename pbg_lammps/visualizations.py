"""Visualization Step subclasses for pbg-lammps.

Visualizations follow the pbg-superpowers new-style contract: each
subclass implements ``accumulate(state)`` to buffer per-step numeric
data (like an Emitter) and ``render()`` to build the Plotly figure once
at end-of-run, returning the HTML string. The base orchestrator owns the
per-tick path (it calls ``accumulate`` each step and ``render`` once), so
subclasses do NOT override ``update()``. The composite spec wires the
input ports to store paths.

See viva_superpowers.visualization for the base-class contract.
"""
from __future__ import annotations

from viva_superpowers.visualization import Visualization


class LAMMPSThermoPlots(Visualization):
    """Time-series HTML plot of LAMMPSProcess thermodynamic outputs.

    Buffers the core LAMMPS scalar thermo outputs (temperature,
    potential_energy, kinetic_energy, total_energy, pressure) at each step,
    then renders a single Plotly HTML figure at end-of-run. Downstream
    consumers (dashboards, notebook viewers) read the rendered 'html' from
    the wired store.
    """

    config_schema = {
        'title': {'_type': 'string', '_default': 'LAMMPS thermodynamics'},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One list per consumed scalar; aligned by index across all signals.
        self.times: list[float] = []
        self.history: dict[str, list[float]] = {
            'temperature': [],
            'potential_energy': [],
            'kinetic_energy': [],
            'total_energy': [],
            'pressure': [],
        }

    def inputs(self):
        return {
            'temperature': 'float',
            'potential_energy': 'float',
            'kinetic_energy': 'float',
            'total_energy': 'float',
            'pressure': 'float',
            'time': 'float',
        }

    def accumulate(self, state):
        self.times.append(float(state.get('time', len(self.times))))
        for key in self.history:
            v = state.get(key)
            self.history[key].append(float(v) if v is not None else 0.0)

    def render(self):
        title = (self.config or {}).get('title', 'LAMMPS thermodynamics')
        traces = []
        for key, ys in self.history.items():
            traces.append(
                '{"x":' + repr(self.times) + ',"y":' + repr(ys) +
                ',"type":"scatter","mode":"lines","name":"' + key + '"}'
            )
        html = (
            f'<div id="ltp" style="height:380px"></div>'
            f'<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            f'<script>Plotly.newPlot("ltp",[{",".join(traces)}],'
            f'{{title:"{title}",margin:{{l:55,r:15,t:35,b:40}},'
            f'xaxis:{{title:"time"}},'
            f'legend:{{orientation:"h",y:-0.2}}}},'
            f'{{responsive:true,displayModeBar:false}});</script>'
        )
        return html
