"""Lets pytest's package-collection machinery (pytest >=8's Package.setup(),
which unconditionally imports this repo's root __init__.py the moment any
test under tests/ is collected) succeed without a live ComfyUI process.

Two independent pytest-side issues, neither requiring changes to the real
__init__.py (which must stay byte-for-byte what ComfyUI's own importlib
loader expects -- see __init__.py's own docstring and Task 7):

1. This repo's directory name ("comfyui-realtime") contains a hyphen, so
   pytest's resolve_package_path() refuses to treat it as a dotted package
   name and falls back to importing __init__.py as a *bare* top-level
   module with no __package__ set -- so its `from .comfyui_realtime...`
   relative imports fail with "attempted relative import with no known
   parent package". ComfyUI's real loader (nodes.py:load_custom_node) does
   not hit this: it uses importlib.util.spec_from_file_location with a
   synthetic module name, which (because the loaded file is literally named
   __init__.py and the synthetic name's last dotted component isn't
   "__init__") gets correctly auto-detected as a package, with
   submodule_search_locations set so relative imports resolve.

2. __init__.py does `from server import PromptServer` -- `server` is
   ComfyUI's own top-level module, only importable inside a running
   ComfyUI process, never during a standalone pytest run.

Fix: pre-import the real __init__.py here, the same way ComfyUI's loader
does (a synthetic spec name whose last component isn't "__init__", so
SourceFileLoader.is_package() returns True and relative imports resolve),
against a stubbed `server` module, and seed the result into sys.modules
under the bare name "__init__" that pytest's Package.setup() asks for
(import_path()'s CouldNotResolvePathError fallback uses path.stem, which is
"__init__" for any __init__.py). That makes Package.setup()'s own import a
sys.modules cache hit instead of a failure.

CAUTION: _StubRouteTableDef.get() is a no-op decorator -- it never inspects
the registered path, method, or handler. It lets __init__.py's module-level
route registration *execute* under pytest without error; it gives zero
protection against a mistake made in that registration (wrong path string,
duplicate route, wrong handler). Routing correctness is covered separately,
against the real aiohttp.web.RouteTableDef, by tests/test_rest_routes.py and
tests/test_websocket_handler.py -- not by this stub. A routing bug
introduced directly in __init__.py will only surface by booting a real
ComfyUI server (see Task 7's live-server verification steps), not by
`pytest` passing.
"""
import importlib.util
import os
import pathlib
import sys
import types

_ROOT = pathlib.Path(__file__).parent


def _add_comfyui_root_to_sys_path() -> None:
    """comfy_api.latest (ComfyUI's v3 node API) is not a standalone pip
    package -- it's a module living inside a real ComfyUI install,
    exactly like server.PromptServer already is (which is why this file
    already stubs that one instead of installing it). Unlike
    server.PromptServer, comfy_api.latest is too large a surface
    (io.Schema/ComfyNode/NodeOutput/Custom, each with real internal
    behavior) to hand-stub faithfully without risking silent drift from
    the real API -- so this points pytest at the real module on disk
    instead. append(), not insert(0, ...): ComfyUI's own modules must
    never take import priority over this project's own modules or
    properly-installed pip packages.
    """
    comfyui_root = os.environ.get(
        "COMFYUI_ROOT",
        "/Users/kale/Development/comfyui_installs/ComfyUI_v0.22.0_comfyui-realtime-dev",
    )
    if comfyui_root not in sys.path:
        sys.path.append(comfyui_root)


def _install_stub_server_module() -> None:
    """Stand in for ComfyUI's top-level `server` module, which only exists
    inside a running ComfyUI process. The real __init__.py only touches
    `PromptServer.instance.routes.get(...)` at import time (to register
    routes) -- a no-op decorator stub is all that's needed for that line to
    execute without error during a standalone test run."""
    if "server" in sys.modules:
        return
    stub = types.ModuleType("server")

    class _StubRouteTableDef:
        def get(self, *_args, **_kwargs):
            def _decorator(fn):
                return fn

            return _decorator

        def delete(self, *_args, **_kwargs):
            def _decorator(fn):
                return fn

            return _decorator

    class _StubPromptServerInstance:
        routes = _StubRouteTableDef()

    stub.PromptServer = types.SimpleNamespace(instance=_StubPromptServerInstance())
    sys.modules["server"] = stub


def _preimport_root_init_for_pytest() -> None:
    if "__init__" in sys.modules:
        return
    _add_comfyui_root_to_sys_path()
    _install_stub_server_module()
    # Last dotted component must not be "__init__" itself, or
    # SourceFileLoader.is_package() returns False and the relative imports
    # inside __init__.py fail exactly like pytest's own fallback does.
    spec = importlib.util.spec_from_file_location(
        "comfyui_realtime_root_init", _ROOT / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Must be registered under its own name *before* exec_module() runs --
    # the relative imports inside __init__.py resolve via
    # sys.modules[spec.parent], which module_from_spec() does not set.
    sys.modules["comfyui_realtime_root_init"] = module
    spec.loader.exec_module(module)
    # Also seed "__init__": that's the bare name pytest's Package.setup()
    # looks up (see import_path()'s CouldNotResolvePathError fallback in
    # _pytest/pathlib.py, which uses path.stem == "__init__").
    sys.modules["__init__"] = module


_preimport_root_init_for_pytest()
