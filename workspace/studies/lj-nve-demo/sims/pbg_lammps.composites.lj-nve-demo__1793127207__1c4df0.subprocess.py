import json, sys, traceback
try:
    from pbg_lammps.core import build_core
    from process_bigraph import Composite, gather_emitter_results
    try:
        from viva_emitters.sqlite_emitter import SQLiteEmitter
    except ImportError:  # process-bigraph < 1.4.17 (legacy location)
        from process_bigraph.emitter import SQLiteEmitter
    from bigraph_schema.json_codec import bigraph_json_hook
    from vivarium_workbench.lib import composite_runs as cr
    core = build_core()
    core.register_link('SQLiteEmitter', SQLiteEmitter)
    with open('/var/folders/vy/vr0_ytms6m95qrnk7xnh0bth0000gp/T/vivarium-run-ervmu13p.state.json') as _sf:
        _state = json.load(_sf, object_hook=bigraph_json_hook)
    composite = Composite({'state': _state}, core=core)
    _steps_run = cr.run_with_division(composite, 20)
    results = gather_emitter_results(composite)

    # Flatten tuple keys to JSON-friendly dotted strings
    out = {}
    for path_tuple, entries in results.items():
        key = '.'.join(str(p) for p in path_tuple)
        out[key] = entries
    # Gather rendered viz HTML, if viva_superpowers is importable.
    viz_html = {}
    try:
        from process_bigraph.visualization import render_results
        rendered = render_results(composite)
        for path_tuple, payload in rendered.items():
            key = '.'.join(str(p) for p in path_tuple)
            viz_html[key] = payload
    except Exception:
        viz_html = {}
    # reproducible-rerun-spine Task 3 (G4), fix round 1: snapshot this
    # run's declared fingerprint_fields from the just-completed
    # composite's live state — only available HERE, in the child;
    # the parent (after this subprocess exits) reads it back via
    # result_fingerprint.fingerprint_run(). Best-effort: swallowed so
    # a snapshot failure never turns a successful run into a reported
    # @@@ERROR@@@.
    try:
        from vivarium_workbench.lib import result_fingerprint as _rfp
        _rfp.write_snapshot('/Users/eranagmon/code/viva-lammps--workbench/.pbg/runs/pbg_lammps.composites.lj-nve-demo__1793127207__1c4df0', composite.state, [])
    except Exception:
        pass
    from bigraph_schema.json_codec import BigraphJSONEncoder as _BJE
    print('@@@RESULTS@@@')
    print(json.dumps({'results': out, 'viz_html': viz_html,
                      'steps_run': _steps_run}, cls=_BJE))
except Exception as e:
    print('@@@ERROR@@@')
    print(traceback.format_exc())
