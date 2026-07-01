import { RealtimeConnection } from "./connection";

class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  readyState = 0;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.onclose?.({ code: 1000 });
  }
  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }
  receive(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}

beforeEach(() => {
  FakeWebSocket.instances = [];
});

test("connect() opens a websocket at the right URL with the pipeline name as a query param", () => {
  const conn = new RealtimeConnection(FakeWebSocket as unknown as typeof WebSocket);
  conn.connect("my-pipeline");
  expect(FakeWebSocket.instances).toHaveLength(1);
  expect(FakeWebSocket.instances[0].url).toBe("/v1/realtime?model=my-pipeline");
});

test("dispatches parsed events to registered listeners", () => {
  const conn = new RealtimeConnection(FakeWebSocket as unknown as typeof WebSocket);
  const received: string[] = [];
  conn.addEventListener((event) => received.push(event.type));
  conn.connect("my-pipeline");
  FakeWebSocket.instances[0].receive({ type: "session.created", session: { id: "x", model: "m", modalities: { input: [], output: [] }, turn_detection: null } });
  expect(received).toEqual(["session.created"]);
});

test("sendTextMessage sends conversation.item.create then response.create", () => {
  const conn = new RealtimeConnection(FakeWebSocket as unknown as typeof WebSocket);
  conn.connect("my-pipeline");
  const ws = FakeWebSocket.instances[0];
  ws.open();
  conn.sendTextMessage("hello");
  const sent = ws.sent.map((s) => JSON.parse(s));
  expect(sent[0]).toEqual({
    type: "conversation.item.create",
    item: { role: "user", content: [{ type: "input_text", text: "hello" }] },
  });
  expect(sent[1]).toEqual({ type: "response.create" });
});

test("sendAudioChunk sends input_audio_buffer.append with the given base64 payload", () => {
  const conn = new RealtimeConnection(FakeWebSocket as unknown as typeof WebSocket);
  conn.connect("my-pipeline");
  const ws = FakeWebSocket.instances[0];
  ws.open();
  conn.sendAudioChunk("BASE64DATA");
  expect(JSON.parse(ws.sent[0])).toEqual({ type: "input_audio_buffer.append", audio: "BASE64DATA" });
});

test("sendSessionUpdate sends session.update with the given voice", () => {
  const conn = new RealtimeConnection(FakeWebSocket as unknown as typeof WebSocket);
  conn.connect("my-pipeline");
  const ws = FakeWebSocket.instances[0];
  ws.open();
  conn.sendSessionUpdate("voice-b");
  expect(JSON.parse(ws.sent[0])).toEqual({ type: "session.update", session: { voice: "voice-b" } });
});

test("disconnect() closes the underlying websocket", () => {
  const conn = new RealtimeConnection(FakeWebSocket as unknown as typeof WebSocket);
  conn.connect("my-pipeline");
  const ws = FakeWebSocket.instances[0];
  let closed = false;
  ws.onclose = () => {
    closed = true;
  };
  conn.disconnect();
  expect(closed).toBe(true);
});
