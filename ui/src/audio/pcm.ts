export function floatTo16BitPCM(floatSamples: Float32Array): Int16Array {
  const out = new Int16Array(floatSamples.length);
  for (let i = 0; i < floatSamples.length; i++) {
    const s = Math.max(-1, Math.min(1, floatSamples[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out;
}

export function int16ToBase64(int16Samples: Int16Array): string {
  const bytes = new Uint8Array(int16Samples.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

export function base64ToInt16(b64: string): Int16Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

export function int16ToFloat32(int16Samples: Int16Array): Float32Array {
  const out = new Float32Array(int16Samples.length);
  for (let i = 0; i < int16Samples.length; i++) {
    out[i] = int16Samples[i] / (int16Samples[i] < 0 ? 32768 : 32767);
  }
  return out;
}
