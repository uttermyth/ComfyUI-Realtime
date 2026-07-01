import { useCallback, useRef, useReducer } from "react";
import { AudioPlayer } from "../audio/audioPlayer";
import { MicCapture } from "../audio/micCapture";
import { isKnownRealtimeEvent } from "../protocol/events";
import { RealtimeConnection } from "../protocol/connection";
import { initialRealtimeSessionState, realtimeSessionReducer } from "./reducer";

export function useRealtimeSession() {
  const [state, dispatch] = useReducer(realtimeSessionReducer, initialRealtimeSessionState);
  const connectionRef = useRef<RealtimeConnection | null>(null);
  const micRef = useRef<MicCapture | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);

  const connect = useCallback((pipelineName: string) => {
    dispatch({ type: "local.connecting" });
    if (!playerRef.current) playerRef.current = new AudioPlayer();

    const connection = new RealtimeConnection();
    connection.addEventListener((event) => {
      dispatch(event);
      if (!isKnownRealtimeEvent(event)) return;
      if (event.type === "input_audio_buffer.speech_started") {
        playerRef.current?.stopAll();
      }
      if (event.type === "response.output_audio.delta") {
        playerRef.current?.play(event.delta);
      }
    });
    connection.connect(pipelineName);
    connectionRef.current = connection;
  }, []);

  const disconnect = useCallback(() => {
    connectionRef.current?.disconnect();
    micRef.current?.stop();
    playerRef.current?.stopAll();
    connectionRef.current = null;
    micRef.current = null;
    dispatch({ type: "local.disconnected" });
  }, []);

  const startMic = useCallback(async () => {
    const mic = new MicCapture();
    await mic.start((chunk) => connectionRef.current?.sendAudioChunk(chunk));
    micRef.current = mic;
  }, []);

  const stopMic = useCallback(() => {
    micRef.current?.stop();
    micRef.current = null;
  }, []);

  const sendText = useCallback((text: string) => {
    dispatch({ type: "local.user_text_sent", text });
    connectionRef.current?.sendTextMessage(text);
  }, []);

  const setVoice = useCallback((voice: string) => {
    connectionRef.current?.sendSessionUpdate(voice);
  }, []);

  return { state, connect, disconnect, startMic, stopMic, sendText, setVoice };
}
