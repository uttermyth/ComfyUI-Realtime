import { base64ToInt16, floatTo16BitPCM, int16ToBase64, int16ToFloat32 } from "./pcm";

test("floatTo16BitPCM clamps and scales float samples to int16 range", () => {
  const result = floatTo16BitPCM(new Float32Array([0, 1, -1, 0.5, -0.5, 2, -2]));
  expect(Array.from(result)).toEqual([0, 32767, -32768, 16383, -16384, 32767, -32768]);
});

test("int16ToFloat32 is the inverse of floatTo16BitPCM at the extremes", () => {
  const result = int16ToFloat32(new Int16Array([32767, -32768, 0]));
  expect(result[0]).toBeCloseTo(1, 4);
  expect(result[1]).toBeCloseTo(-1, 4);
  expect(result[2]).toBe(0);
});

test("int16ToBase64 and base64ToInt16 round-trip", () => {
  const original = new Int16Array([0, 1, -1, 12345, -12345]);
  const roundTripped = base64ToInt16(int16ToBase64(original));
  expect(Array.from(roundTripped)).toEqual(Array.from(original));
});
