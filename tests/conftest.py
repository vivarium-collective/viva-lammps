"""Skip tests that drive the real LAMMPS engine when its native binary isn't present.

The `lammps` engine (native binary / Python bindings) is not pip-installable, so a
vanilla CI runner can import `pbg_lammps` and check the wrapper's API but cannot run a
simulation. These tests pass on a runner with the engine installed; elsewhere the
engine-driving modules are skipped so the structural/API tests still gate the PR.
"""
import pytest

_ENGINE_MODULES = ("test_composites", "test_processes")


def _lammps_available() -> bool:
    try:
        from lammps import lammps  # noqa: F401
        lammps(cmdargs=["-screen", "none", "-log", "none", "-nocite"])
        return True
    except Exception:
        return False


_AVAILABLE = _lammps_available()


def pytest_collection_modifyitems(config, items):
    if _AVAILABLE:
        return
    skip = pytest.mark.skip(reason="LAMMPS engine (native binary) not available on this runner")
    for item in items:
        if any(mod in item.nodeid for mod in _ENGINE_MODULES):
            item.add_marker(skip)
