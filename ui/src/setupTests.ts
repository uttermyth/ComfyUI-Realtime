import "@testing-library/jest-dom";

/** jsdom's native WebSocket polyfill does not resolve relative URLs
 * against the document's base the way real browsers do (confirmed: even
 * with an explicit testEnvironmentOptions.url configured, `new
 * WebSocket("/v1/realtime?model=x")` still throws SyntaxError) --
 * RealtimeConnection (protocol/connection.ts) deliberately uses a
 * relative URL by design (this extension is always served from
 * ComfyUI's own origin, the whole point of the "no special privileges"
 * constraint), so this is purely a test-environment gap, not a
 * production bug. Tests that need finer control over WebSocket behavior
 * already inject their own fake via RealtimeConnection's constructor
 * parameter (see protocol/connection.test.ts) and are unaffected by this
 * global stub; this only matters for tests exercising the real default
 * WebSocket path (e.g. App.test.tsx), which only care that the
 * constructor doesn't throw, not about real socket behavior. */
class StubWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  readyState = StubWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;

  constructor(public url: string) {}

  send(): void {
    // no-op -- tests that need to assert on sent messages inject their
    // own fake via RealtimeConnection's constructor parameter instead.
  }

  close(): void {
    this.readyState = StubWebSocket.CLOSED;
    this.onclose?.({ code: 1000 });
  }
}

(global as unknown as { WebSocket: typeof WebSocket }).WebSocket = StubWebSocket as unknown as typeof WebSocket;
