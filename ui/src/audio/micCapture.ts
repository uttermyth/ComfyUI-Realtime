import { floatTo16BitPCM, int16ToBase64 } from "./pcm";

const WIRE_SAMPLE_RATE = 24000;
// ~170ms per outgoing chunk -- small enough for responsive VAD, large
// enough not to spam the socket with a message every single 128-sample
// (5.3ms) AudioWorklet callback. Matches manual_voice_test.html exactly.
const SEND_CHUNK_SAMPLES = 4096;

const WORKLET_SOURCE = `
class MicCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channelData = inputs[0] && inputs[0][0];
    if (channelData) {
      this.port.postMessage(channelData.slice());
    }
    return true;
  }
}
registerProcessor("mic-capture-processor", MicCaptureProcessor);
`;

/** Captures mic audio and delivers pcm16/24kHz base64-encoded chunks via
 * onChunk, at roughly SEND_CHUNK_SAMPLES intervals -- the same technique
 * already proven working in scripts/manual_voice_test.html. */
export class MicCapture {
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private sendBuffer: number[] = [];

  async start(onChunk: (base64Audio: string) => void): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.ctx = new AudioContext({ sampleRate: WIRE_SAMPLE_RATE });

    const blob = new Blob([WORKLET_SOURCE], { type: "application/javascript" });
    const workletUrl = URL.createObjectURL(blob);
    await this.ctx.audioWorklet.addModule(workletUrl);

    const source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "mic-capture-processor");
    this.node.port.onmessage = (e: MessageEvent<Float32Array>) => {
      this.sendBuffer.push(...e.data);
      this.flush(onChunk, false);
    };
    source.connect(this.node);
  }

  stop(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    void this.ctx?.close();
    this.stream = null;
    this.ctx = null;
    this.node = null;
    this.sendBuffer = [];
  }

  private flush(onChunk: (base64Audio: string) => void, force: boolean): void {
    while (this.sendBuffer.length >= SEND_CHUNK_SAMPLES || (force && this.sendBuffer.length > 0)) {
      const take = this.sendBuffer.splice(0, SEND_CHUNK_SAMPLES);
      const pcm16 = floatTo16BitPCM(Float32Array.from(take));
      onChunk(int16ToBase64(pcm16));
    }
  }
}
