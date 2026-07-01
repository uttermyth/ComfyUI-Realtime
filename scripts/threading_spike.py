# scripts/threading_spike.py
"""Phase 0 threading spike (spec section 6.8, section 10).

Queues a BusyWorkNode workflow (blocking ComfyUI's prompt-executor thread
for 10s) and, concurrently, opens a websocket session against the echo
pipeline and round-trips a message every 200ms, recording each round-trip
latency. If section 6.8's threading model is correctly implemented, these
latencies stay low and don't spike with the busy-work duration -- because
the websocket session lives entirely on the aiohttp event loop, never on
the prompt-executor thread.

Prerequisites: ComfyUI running with the echo pipeline already registered
(Task 7, Steps 7-9). Run: python scripts/threading_spike.py
"""
import asyncio
import time

import aiohttp

COMFYUI_URL = "http://127.0.0.1:8188"
BUSY_WORK_WORKFLOW = {
    "prompt": {"1": {"class_type": "BusyWorkNode", "inputs": {"duration_seconds": 10.0}}},
    "client_id": "phase0-threading-spike",
}


async def queue_busy_work(session: aiohttp.ClientSession) -> None:
    async with session.post(f"{COMFYUI_URL}/prompt", json=BUSY_WORK_WORKFLOW) as resp:
        body = await resp.json()
        print(f"queued BusyWorkNode workflow: {body}")


async def ping_echo_session(duration_seconds: float) -> list[float]:
    latencies = []
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"{COMFYUI_URL}/v1/realtime?model=echo") as ws:
            await ws.receive()  # session.created
            deadline = time.perf_counter() + duration_seconds
            while time.perf_counter() < deadline:
                start = time.perf_counter()
                await ws.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {"content": [{"type": "input_text", "text": "ping"}]},
                    }
                )
                # Drain the four echo events (response.created, delta, done, response.done).
                for _ in range(4):
                    await ws.receive()
                latencies.append(time.perf_counter() - start)
                await asyncio.sleep(0.2)
    return latencies


async def main() -> None:
    async with aiohttp.ClientSession() as session:
        busy_work_task = asyncio.create_task(queue_busy_work(session))
        ping_task = asyncio.create_task(ping_echo_session(duration_seconds=12.0))
        await busy_work_task
        latencies = await ping_task

    print(f"\nRound trips completed: {len(latencies)}")
    print(f"max latency: {max(latencies) * 1000:.1f}ms")
    print(f"mean latency: {sum(latencies) / len(latencies) * 1000:.1f}ms")
    print(f"all latencies (ms): {[round(l * 1000, 1) for l in latencies]}")


if __name__ == "__main__":
    asyncio.run(main())
