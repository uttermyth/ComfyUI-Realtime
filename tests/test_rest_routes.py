from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from comfyui_realtime.engine.session_state import SessionState
from comfyui_realtime.registry import PipelineConfig, pipeline_registry
from comfyui_realtime.server.rest_routes import (
    register_health_route,
    register_models_route,
    register_pipelines_routes,
    register_sessions_route,
)
from comfyui_realtime.session_registry import session_registry


class _StubLLM:
    async def generate(self, messages, options):
        yield  # pragma: no cover -- never actually called in these tests


async def test_health_route_returns_ok(aiohttp_client):
    app = web.Application()
    routes = web.RouteTableDef()
    register_health_route(routes)
    app.add_routes(routes)
    client = await aiohttp_client(app)
    resp = await client.get("/realtime/health")
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "ok"
    assert body["pipelines"] == 0


async def test_health_route_reports_real_active_session_count(aiohttp_client):
    # Directly exercise the route in isolation -- register a fake session
    # record and confirm the count reflects it, without needing a real
    # websocket connection.
    session_registry.register("fake-sess-for-health-test", "irrelevant", SessionState())
    try:
        app = web.Application()
        routes = web.RouteTableDef()
        register_health_route(routes)
        app.add_routes(routes)
        client = await aiohttp_client(app)
        resp = await client.get("/realtime/health")
        body = await resp.json()
        assert body["active_sessions"] >= 1
    finally:
        session_registry.unregister("fake-sess-for-health-test")


async def test_list_pipelines_route_reports_registered_pipelines(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="rest-list-test", llm=_StubLLM()))
    try:
        app = web.Application()
        routes = web.RouteTableDef()
        register_pipelines_routes(routes)
        app.add_routes(routes)
        client = await aiohttp_client(app)
        resp = await client.get("/realtime/pipelines")
        body = await resp.json()
        names = [p["name"] for p in body["pipelines"]]
        assert "rest-list-test" in names
        entry = next(p for p in body["pipelines"] if p["name"] == "rest-list-test")
        assert entry["modalities"] == {"input": ["text"], "output": ["text"]}
        assert entry["providers"] == {"llm": "_StubLLM"}
        assert "registered_at" in entry
    finally:
        pipeline_registry.unregister("rest-list-test")


async def test_list_pipelines_route_reports_voices_when_tts_present(aiohttp_client):
    class _MultiVoiceStubTTS:
        output_sample_rate = 24000
        output_format = "pcm16"

        async def synthesize(self, text_stream, voice=None):
            async for _ in text_stream:
                yield b""

        def list_voices(self):
            from comfyui_realtime.providers.base import VoiceInfo

            return [VoiceInfo(id="voice-a", name="voice-a"), VoiceInfo(id="voice-b", name="voice-b")]

    pipeline_registry.register(PipelineConfig(name="rest-voices-test", llm=_StubLLM(), tts=_MultiVoiceStubTTS()))
    try:
        app = web.Application()
        routes = web.RouteTableDef()
        register_pipelines_routes(routes)
        app.add_routes(routes)
        client = await aiohttp_client(app)
        resp = await client.get("/realtime/pipelines")
        body = await resp.json()
        entry = next(p for p in body["pipelines"] if p["name"] == "rest-voices-test")
        assert sorted(entry["voices"]) == ["voice-a", "voice-b"]
    finally:
        pipeline_registry.unregister("rest-voices-test")


async def test_list_pipelines_route_reports_empty_voices_when_no_tts(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="rest-no-voices-test", llm=_StubLLM()))
    try:
        app = web.Application()
        routes = web.RouteTableDef()
        register_pipelines_routes(routes)
        app.add_routes(routes)
        client = await aiohttp_client(app)
        resp = await client.get("/realtime/pipelines")
        body = await resp.json()
        entry = next(p for p in body["pipelines"] if p["name"] == "rest-no-voices-test")
        assert entry["voices"] == []
    finally:
        pipeline_registry.unregister("rest-no-voices-test")


async def test_get_pipeline_route_returns_404_for_unknown_name(aiohttp_client):
    app = web.Application()
    routes = web.RouteTableDef()
    register_pipelines_routes(routes)
    app.add_routes(routes)
    client = await aiohttp_client(app)
    resp = await client.get("/realtime/pipelines/does-not-exist")
    assert resp.status == 404
    body = await resp.json()
    assert body["error"]["code"] == "pipeline_not_found"


async def test_get_pipeline_route_returns_details_for_known_name(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="rest-detail-test", llm=_StubLLM()))
    try:
        app = web.Application()
        routes = web.RouteTableDef()
        register_pipelines_routes(routes)
        app.add_routes(routes)
        client = await aiohttp_client(app)
        resp = await client.get("/realtime/pipelines/rest-detail-test")
        assert resp.status == 200
        body = await resp.json()
        assert body["name"] == "rest-detail-test"
        assert body["providers"] == {"llm": "_StubLLM"}
    finally:
        pipeline_registry.unregister("rest-detail-test")


async def test_delete_pipeline_route_returns_404_for_unknown_name(aiohttp_client):
    app = web.Application()
    routes = web.RouteTableDef()
    register_pipelines_routes(routes)
    app.add_routes(routes)
    client = await aiohttp_client(app)
    resp = await client.delete("/realtime/pipelines/does-not-exist")
    assert resp.status == 404


async def test_delete_pipeline_route_removes_a_registered_pipeline(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="rest-delete-test", llm=_StubLLM()))
    app = web.Application()
    routes = web.RouteTableDef()
    register_pipelines_routes(routes)
    app.add_routes(routes)
    client = await aiohttp_client(app)
    resp = await client.delete("/realtime/pipelines/rest-delete-test")
    assert resp.status == 200
    body = await resp.json()
    assert body == {"status": "deleted", "name": "rest-delete-test"}
    assert pipeline_registry.get("rest-delete-test") is None


async def test_sessions_route_reports_active_sessions(aiohttp_client):
    session_registry.register("rest-sessions-test", "some-pipeline", SessionState())
    try:
        app = web.Application()
        routes = web.RouteTableDef()
        register_sessions_route(routes)
        app.add_routes(routes)
        client = await aiohttp_client(app)
        resp = await client.get("/realtime/sessions")
        body = await resp.json()
        entry = next(s for s in body["sessions"] if s["id"] == "rest-sessions-test")
        assert entry["pipeline"] == "some-pipeline"
        assert entry["state"] == "idle"
        assert entry["uptime_seconds"] >= 0
    finally:
        session_registry.unregister("rest-sessions-test")


async def test_models_route_lists_registered_pipelines_openai_style(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="rest-models-test", llm=_StubLLM()))
    try:
        app = web.Application()
        routes = web.RouteTableDef()
        register_models_route(routes)
        app.add_routes(routes)
        client = await aiohttp_client(app)
        resp = await client.get("/v1/models")
        body = await resp.json()
        assert body["object"] == "list"
        entry = next(m for m in body["data"] if m["id"] == "rest-models-test")
        assert entry["object"] == "model"
        assert entry["owned_by"] == "comfyui-realtime"
        assert isinstance(entry["created"], int)
    finally:
        pipeline_registry.unregister("rest-models-test")
