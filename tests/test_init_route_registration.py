import logging
import sys


def test_init_module_already_imported_without_raising():
    # conftest.py's pre-import (run once, automatically, the moment any
    # test under tests/ is collected) is itself the primary thing this task
    # protects: if route registration could still crash module import on a
    # collision, this assertion would never even be reached -- the whole
    # test session would already have failed at collection time.
    assert "__init__" in sys.modules
    assert hasattr(sys.modules["__init__"], "comfy_entrypoint")


def test_defensive_wrapper_logs_a_warning_instead_of_raising(caplog):
    init_module = sys.modules["__init__"]

    def _always_raises(routes):
        raise RuntimeError("simulated path collision with another custom node")

    with caplog.at_level(logging.WARNING):
        init_module._register_route_defensively(_always_raises, "/fake/path/for/this/test")

    assert any("/fake/path/for/this/test" in record.message for record in caplog.records)


def test_defensive_wrapper_does_not_log_when_registration_succeeds(caplog):
    init_module = sys.modules["__init__"]
    calls = []

    def _succeeds(routes):
        calls.append(routes)

    with caplog.at_level(logging.WARNING):
        init_module._register_route_defensively(_succeeds, "/fake/successful/path")

    assert len(calls) == 1
    assert not any("/fake/successful/path" in record.message for record in caplog.records)
