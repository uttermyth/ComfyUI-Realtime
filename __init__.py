from __future__ import annotations

import logging
import os

from comfy_api.latest import ComfyExtension, io
from server import PromptServer

from .comfyui_realtime.nodes.dev_nodes import BusyWorkNode
from .comfyui_realtime.nodes.pipeline_node import EchoLLMProviderNode, RealtimePipelineNode
from .comfyui_realtime.nodes.provider_nodes import (
    LlamaCppLLMProviderNode,
    PiperTTSProviderNode,
    PocketTTSProviderNode,
    SileroVADProviderNode,
    WhisperCppSTTProviderNode,
    FasterWhisperSTTProviderNode,
)
from .comfyui_realtime.server.rest_routes import (
    register_health_route,
    register_models_route,
    register_pipelines_routes,
    register_sessions_route,
)
from .comfyui_realtime.server.websocket_handler import realtime_websocket_handler

logger = logging.getLogger("comfyui_realtime")

# Bundled React reference client
WEB_DIRECTORY = "dist"

routes = PromptServer.instance.routes


def _register_route_defensively(register_fn, route_description: str) -> None:
    """aiohttp raises on duplicate route registration at startup, which
    would prevent ComfyUI from loading cleanly if another custom node
    already claimed one of our paths. Log a clear warning naming the
    conflicting route instead of crashing node load (spec section 4.4)."""
    try:
        register_fn(routes)
    except Exception as exc:
        logger.warning(
            "comfyui-realtime: failed to register %s -- likely a path collision with another custom node: %r",
            route_description,
            exc,
        )


_register_route_defensively(register_health_route, "/realtime/health")
_register_route_defensively(register_pipelines_routes, "/realtime/pipelines (list/detail/delete)")
_register_route_defensively(register_sessions_route, "/realtime/sessions")

if os.environ.get("COMFYUI_REALTIME_DISABLE_MODELS_ROUTE") != "1":
    _register_route_defensively(register_models_route, "/v1/models")


def _register_websocket_route(routes) -> None:
    @routes.get("/v1/realtime")
    async def _realtime_websocket_route(request):
        return await realtime_websocket_handler(request)


_register_route_defensively(_register_websocket_route, "/v1/realtime")


class ComfyUIRealtimeExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EchoLLMProviderNode,
            RealtimePipelineNode,
            BusyWorkNode,
            LlamaCppLLMProviderNode,
            PiperTTSProviderNode,
            PocketTTSProviderNode,
            SileroVADProviderNode,
            WhisperCppSTTProviderNode,
            FasterWhisperSTTProviderNode,
        ]


async def comfy_entrypoint() -> ComfyUIRealtimeExtension:
    return ComfyUIRealtimeExtension()


__all__ = ["comfy_entrypoint"]
