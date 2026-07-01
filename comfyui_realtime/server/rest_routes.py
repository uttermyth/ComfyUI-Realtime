"""REST surface."""
from __future__ import annotations

import time

from aiohttp import web

from ..registry import pipeline_registry
from ..session_registry import derive_session_status, session_registry


def register_health_route(routes: web.RouteTableDef) -> None:
    @routes.get("/realtime/health")
    async def health(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "pipelines": len(pipeline_registry.list()),
                "active_sessions": session_registry.count(),
            }
        )


def _format_registered_at(unix_timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(unix_timestamp))


def _pipeline_to_dict(config) -> dict:
    providers = {}
    if config.llm is not None:
        providers["llm"] = type(config.llm).__name__
    if config.vad is not None:
        providers["vad"] = type(config.vad).__name__
    if config.stt is not None:
        providers["stt"] = type(config.stt).__name__
    if config.tts is not None:
        providers["tts"] = type(config.tts).__name__
    voices = [v.id for v in config.tts.list_voices()] if config.tts is not None else []
    return {
        "name": config.name,
        "modalities": {"input": config.modalities_input, "output": config.modalities_output},
        "providers": providers,
        "voices": voices,
        "registered_at": _format_registered_at(config.registered_at),
    }


def register_pipelines_routes(routes: web.RouteTableDef) -> None:
    @routes.get("/realtime/pipelines")
    async def list_pipelines(request: web.Request) -> web.Response:
        return web.json_response({"pipelines": [_pipeline_to_dict(c) for c in pipeline_registry.list()]})

    @routes.get("/realtime/pipelines/{name}")
    async def get_pipeline(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        config = pipeline_registry.get(name)
        if config is None:
            return web.json_response(
                {"error": {"code": "pipeline_not_found", "message": f"No pipeline registered as '{name}'"}},
                status=404,
            )
        return web.json_response(_pipeline_to_dict(config))

    @routes.delete("/realtime/pipelines/{name}")
    async def delete_pipeline(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if pipeline_registry.unregister(name):
            return web.json_response({"status": "deleted", "name": name})
        return web.json_response(
            {"error": {"code": "pipeline_not_found", "message": f"No pipeline registered as '{name}'"}},
            status=404,
        )


def register_sessions_route(routes: web.RouteTableDef) -> None:
    @routes.get("/realtime/sessions")
    async def list_sessions(request: web.Request) -> web.Response:
        now = time.monotonic()
        sessions = [
            {
                "id": record.session_id,
                "pipeline": record.pipeline_name,
                "uptime_seconds": now - record.connected_at,
                "state": derive_session_status(record),
            }
            for record in session_registry.list()
        ]
        return web.json_response({"sessions": sessions})


def register_models_route(routes: web.RouteTableDef) -> None:
    @routes.get("/v1/models")
    async def list_models(request: web.Request) -> web.Response:
        data = [
            {
                "id": config.name,
                "object": "model",
                "created": int(config.registered_at),
                "owned_by": "comfyui-realtime",
            }
            for config in pipeline_registry.list()
        ]
        return web.json_response({"object": "list", "data": data})
