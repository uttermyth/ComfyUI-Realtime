import { base64ToInt16, int16ToFloat32 } from "./pcm";

const WIRE_SAMPLE_RATE = 24000;

/** Schedules pcm16/24kHz base64 audio chunks for sequential gapless
 * playback, and supports immediate barge-in cancellation -- the same
 * technique already proven working in scripts/manual_voice_test.html. */
export class AudioPlayer {
  private ctx: AudioContext | null = null;
  private nextPlayTime = 0;
  private scheduledSources: AudioBufferSourceNode[] = [];

  private ensureContext(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext({ sampleRate: WIRE_SAMPLE_RATE });
      this.nextPlayTime = this.ctx.currentTime;
    }
    return this.ctx;
  }

  play(base64Audio: string): void {
    const int16 = base64ToInt16(base64Audio);
    if (int16.length === 0) return;
    const floatSamples = int16ToFloat32(int16);

    const ctx = this.ensureContext();
    const buffer = ctx.createBuffer(1, floatSamples.length, WIRE_SAMPLE_RATE);
    // floatSamples is always backed by a plain ArrayBuffer (it's freshly
    // allocated by int16ToFloat32), never a SharedArrayBuffer -- this cast
    // just satisfies TS's stricter typed-array generics for the DOM lib.
    buffer.copyToChannel(floatSamples as Float32Array<ArrayBuffer>, 0);

    const sourceNode = ctx.createBufferSource();
    sourceNode.buffer = buffer;
    sourceNode.connect(ctx.destination);

    const startAt = Math.max(ctx.currentTime, this.nextPlayTime);
    sourceNode.start(startAt);
    this.nextPlayTime = startAt + buffer.duration;

    this.scheduledSources.push(sourceNode);
    sourceNode.onended = () => {
      this.scheduledSources = this.scheduledSources.filter((s) => s !== sourceNode);
    };
  }

  /** Real barge-in: the server already cancelled generation server-side;
   * this stops whatever audio THIS client had already queued for
   * playback, so the user doesn't keep hearing the old response. */
  stopAll(): void {
    for (const s of this.scheduledSources) {
      try {
        s.stop();
      } catch {
        // may already have finished
      }
    }
    this.scheduledSources = [];
    this.nextPlayTime = this.ctx?.currentTime ?? 0;
  }
}
