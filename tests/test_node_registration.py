"""Guards against the recurring bug class (found 3x across Phases 1-2): a
node class gets built and unit-tested but never wired into the root
__init__.py's node list, only caught later by an expensive live-ComfyUI
end-to-end test.

This test enumerates node classes *programmatically* from each node-defining
module (any class satisfying ComfyUI's v3 node convention -- a classmethod
named define_schema -- that is actually defined in that module, not merely
imported into it) and asserts each one is returned by comfy_entrypoint()'s
ComfyExtension.get_node_list() (comfy-org v3 migration; this test's
mechanism previously read the v1 NODE_CLASS_MAPPINGS dict directly -- v3's
entrypoint is async and returns a list rather than exposing a static dict
attribute, so discovery now goes through tests/conftest.py's
discover_v3_node_classes() helper, which calls comfy_entrypoint() the same
way ComfyUI's real loader does). It deliberately does not hardcode a list
of node class names: the point is to automatically catch the *next* node
someone adds and forgets to register, without anyone needing to remember to
update this test too.

Getting comfy_entrypoint(): the root __init__.py can't be imported normally
outside a running ComfyUI process (it does `from server import
PromptServer` and, since the v3 migration, `from comfy_api.latest import
ComfyExtension, io`). The repo's root conftest.py already solves both --
it stubs `server` and points `sys.path` at a real ComfyUI install for
`comfy_api`, then pre-imports the real __init__.py and seeds the result
into sys.modules["__init__"] (the bare name pytest's own Package.setup()
looks up). Reusing that same sys.modules entry here means this test
exercises the *actual* comfy_entrypoint() assembled by the real
__init__.py, not a re-implementation of it.

Caveat: conftest.py's synthetic-spec trick (needed so __init__.py's relative
imports resolve) loads the *entire* comfyui_realtime package a second time
under a different top-level module name (e.g.
"comfyui_realtime_root_init.comfyui_realtime.nodes.pipeline_node"), so the
node classes returned by comfy_entrypoint()'s node list are distinct
*objects* from the ones this test imports directly via
`comfyui_realtime.nodes.*` -- identity/`in` comparisons on the class objects
themselves would always report everything missing. Comparing by qualified
name (the class's own module/class name with the leading synthetic-package
component stripped) instead correctly treats both copies as "the same" node.
"""
from __future__ import annotations

import inspect

from comfyui_realtime.nodes import dev_nodes, pipeline_node, provider_nodes

from .conftest import discover_v3_node_classes

_NODE_MODULES = (pipeline_node, provider_nodes, dev_nodes)


def _node_classes_defined_in(module):
    """Classes defined in `module` or its submodules (not merely imported
    from an unrelated package) that follow ComfyUI's v3 node convention of
    exposing a define_schema classmethod. The submodule check handles the
    case where a module is a package whose classes live in per-file submodules
    (e.g. provider_nodes/llama_cpp_llm.py)."""
    module_name = module.__name__
    return [
        cls
        for _name, cls in inspect.getmembers(module, inspect.isclass)
        if (cls.__module__ == module_name or cls.__module__.startswith(module_name + "."))
        and hasattr(cls, "define_schema")
    ]


def test_every_node_class_is_discoverable_via_define_schema():
    # Sanity check on the enumeration mechanism itself: if this starts
    # returning zero across all three modules (e.g. a refactor renames
    # define_schema or moves classes elsewhere), the real assertion below
    # would pass vacuously and stop protecting anything.
    discovered = [cls for module in _NODE_MODULES for cls in _node_classes_defined_in(module)]
    assert len(discovered) >= 7, (
        f"Expected to discover at least the 7 known node classes via define_schema, "
        f"found {len(discovered)}: {[c.__name__ for c in discovered]}"
    )


def _qualified_name(cls):
    """(module-path-after-the-comfyui_realtime-component, class name), so a
    class loaded twice under two different top-level synthetic module names
    (see module docstring) still compares equal."""
    module_parts = cls.__module__.split(".")
    if "comfyui_realtime" in module_parts:
        module_parts = module_parts[module_parts.index("comfyui_realtime") :]
    return (".".join(module_parts), cls.__qualname__)


def test_every_node_class_is_registered_via_comfy_entrypoint():
    registered_classes = discover_v3_node_classes()
    registered_names = {_qualified_name(cls) for cls in registered_classes.values()}

    missing = []
    for module in _NODE_MODULES:
        for cls in _node_classes_defined_in(module):
            if _qualified_name(cls) not in registered_names:
                missing.append(f"{module.__name__}.{cls.__name__}")

    assert not missing, (
        "Node class(es) defined but not wired into __init__.py's "
        f"comfy_entrypoint() node list: {missing}"
    )
