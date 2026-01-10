import os
import sysconfig
import glob
import pathlib
import socket
import struct
import time
import threading
import queue

import webrtcvad
import requests
import numpy as np

# ============================================================
# CUDA DLL PATH FIX (must run BEFORE importing faster_whisper)
# ============================================================
# IMPORTANT: keep add_dll_directory handles alive, or Windows may
# drop the directory later and CUDA loads will fail at transcribe().
CUDA_DLL_HANDLES = []

try:
    sp = sysconfig.get_paths()["purelib"]  # reliable venv site-packages
    nvidia_bins = sorted(
        set(
            d
            for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin"))
            if os.path.isdir(d)
        )
    )

    for d in nvidia_bins:
        CUDA_DLL_HANDLES.append(os.add_dll_directory(d))

    # Extra safety: prepend bins to PATH for subprocesses / late loads
    os.environ["PATH"] = ";".join(nvidia_bins) + ";" + os.environ.get("PATH", "")

    print(f"[WIN] NVIDIA CUDA DLL bins added ({len(nvidia_bins)}):")
    for d in nvidia_bins:
        print("   ", d)

except Exception as e:
    print("[WIN] Failed to add NVIDIA DLL dirs:", e)

# Now safe to import GPU libs
from faster_whisper import WhisperModel
from kokoro import KPipeline

# --------------------------
# Network config
# --------------------------
LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5555

PI_IP = "192.168.68.76"
PI_OPUS_PORT = 5557  # Pi UDP Opus listener port

# --------------------------
# Mic stream format (Pi -> Windows)
# --------------------------
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 320
FRAME_BYTES = FRAME_SAMPLES * 2  # 640 bytes int16 mono

# --------------------------
# Opus stream format (Windows -> Pi)
# --------------------------
OPUS_SR = 24000
OPUS_CH = 1
OPUS_FRAME_MS = 20
OPUS_FRAME_SAMPLES = int(OPUS_SR * OPUS_FRAME_MS / 1000)  # 480 samples
OPUS_APPLICATION_AUDIO = 2049  # Opus "audio"

# --------------------------
# Ollama
# --------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b"

# --------------------------
# DLL search paths (opus.dll must be findable)
# --------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
for d in [SCRIPT_DIR, SCRIPT_DIR / "dll", SCRIPT_DIR / "dlls", SCRIPT_DIR / "deps"]:
    if d.exists():
        try:
            os.add_dll_directory(str(d))
        except Exception:
            pass

# Optional: if PyOgg is installed, add its dirs too
try:
    import pyogg

    pyogg_dir = pathlib.Path(pyogg.__file__).resolve().parent
    for d in [pyogg_dir, pyogg_dir / "libs", pyogg_dir / "bin", pyogg_dir / "lib"]:
        if d.exists():
            try:
                os.add_dll_directory(str(d))
            except Exception:
                pass
except Exception:
    pass

# Import Opus library (requires opus.dll to be loadable)
try:
    import opuslib_next
except Exception as e:
    raise SystemExit(
        "\n[WIN] opuslib_next could not load the Opus library.\n"
        f"Make sure a 64-bit opus.dll is available (best: put it in {SCRIPT_DIR}).\n"
        f"Original error: {e}\n"
    )

# --------------------------
# Load models
# --------------------------
print("[WIN] Loading Whisper (GPU)...")
stt = WhisperModel("small", device="cuda", compute_type="float16")
print("[WIN] Whisper loaded.")

print("[WIN] Loading Kokoro...")
tts = KPipeline(lang_code="a")
KOKORO_VOICE = "af_heart"
print("[WIN] Kokoro loaded.")

# Sanity check Opus encoder
try:
    _enc_test = opuslib_next.Encoder(OPUS_SR, OPUS_CH, OPUS_APPLICATION_AUDIO)
    _ = _enc_test.encode(b"\x00\x00" * OPUS_FRAME_SAMPLES, OPUS_FRAME_SAMPLES)
    print("[WIN] Opus encoder OK.")
except Exception as e:
    raise SystemExit(f"[WIN] Opus encoder init failed: {e}")

messages = [
    {
        "role": "system",
        "content": "You are a helpful voice assistant. Keep responses concise and natural.",
    }
]

# --------------------------
# Helpers
# --------------------------
def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def transcribe_pcm(pcm: bytes) -> str:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _info = stt.transcribe(
        audio,
        language="en",
        vad_filter=False,
        beam_size=1,
        best_of=1,
        temperature=0.0,
    )
    return "".join(seg.text for seg in segments).strip()


def ollama_chat(user_text: str) -> str:
    messages.append({"role": "user", "content": user_text})
    r = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    reply = r.json()["message"]["content"].strip()
    messages.append({"role": "assistant", "content": reply})
    return reply


# --------------------------
# Opus streaming (PACED) so long audio doesn't get dropped on Pi
# --------------------------
def stream_kokoro_opus_udp_paced(text: str) -> float:
    """
    Sends UDP packets to Pi:
      header: >HB (seq:uint16, flags:uint8) + opus payload
      flags bit0 = end marker

    Paced at real-time (20ms/packet) so the Pi queue doesn't overflow.
    Returns duration seconds (based on frames sent).
    """
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    enc = opuslib_next.Encoder(OPUS_SR, OPUS_CH, OPUS_APPLICATION_AUDIO)

    seq = 0
    carry = np.zeros((0,), dtype=np.int16)

    frame_period = OPUS_FRAME_MS / 1000.0
    next_send = time.perf_counter()

    frames_sent = 0

    def send_packet(flags: int, payload: bytes):
        nonlocal seq, next_send, frames_sent

        # Pace sending (only audio frames need pacing)
        now = time.perf_counter()
        if flags == 0 and now < next_send:
            time.sleep(next_send - now)

        hdr = struct.pack(">HB", seq & 0xFFFF, flags)
        udp.sendto(hdr + payload, (PI_IP, PI_OPUS_PORT))
        seq += 1

        if flags == 0:
            frames_sent += 1
            next_send += frame_period

    for _i, (_gs, _ps, audio_f32) in enumerate(tts(text, voice=KOKORO_VOICE)):
        if audio_f32 is None:
            continue

        audio_f32 = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
        audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)

        carry = audio_i16 if carry.size == 0 else np.concatenate([carry, audio_i16])

        while carry.shape[0] >= OPUS_FRAME_SAMPLES:
            frame = carry[:OPUS_FRAME_SAMPLES]
            carry = carry[OPUS_FRAME_SAMPLES:]

            pkt = enc.encode(frame.tobytes(), OPUS_FRAME_SAMPLES)
            send_packet(0, pkt)

    # pad final frame if needed
    if carry.shape[0] > 0:
        pad = np.zeros((OPUS_FRAME_SAMPLES - carry.shape[0],), dtype=np.int16)
        frame = np.concatenate([carry, pad])
        pkt = enc.encode(frame.tobytes(), OPUS_FRAME_SAMPLES)
        send_packet(0, pkt)

    # End marker (no payload)
    send_packet(1, b"")

    udp.close()
    return frames_sent * frame_period


# --------------------------
# Main server
# --------------------------
def run_server():
    vad = webrtcvad.Vad(3)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((LISTEN_IP, LISTEN_PORT))
    sock.listen(1)
    print(f"[WIN] Listening for Pi mic on {LISTEN_IP}:{LISTEN_PORT}")

    conn, addr = sock.accept()
    print(f"[WIN] Connected from {addr}")

    # TRUE while bot is speaking; mic reader discards frames
    speaking = threading.Event()
    post_speak_ignore_until = 0.0

    # Mic frames queue (only when not speaking)
    mic_q: queue.Queue[bytes] = queue.Queue(maxsize=2000)

    def mic_reader():
        nonlocal post_speak_ignore_until
        try:
            while True:
                frame = recv_exact(conn, FRAME_BYTES)

                now = time.time()
                if speaking.is_set() or now < post_speak_ignore_until:
                    continue

                try:
                    mic_q.put_nowait(frame)
                except queue.Full:
                    # drop oldest to stay responsive
                    try:
                        mic_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        mic_q.put_nowait(frame)
                    except queue.Full:
                        pass
        except Exception:
            pass

    threading.Thread(target=mic_reader, daemon=True).start()

    ring = []
    voiced = []
    triggered = False
    speech_count = 0
    silence_count = 0

    START_FRAMES = 8
    END_FRAMES = 12

    MIN_UTTERANCE_SECONDS = 0.6
    MIN_UTTERANCE_BYTES = int(SAMPLE_RATE * MIN_UTTERANCE_SECONDS) * 2

    SMALL_TALK_SET = {
        "hello",
        "hello?",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "thank you.",
        "thanks.",
    }

    while True:
        frame = mic_q.get()

        is_speech = vad.is_speech(frame, SAMPLE_RATE)

        if not triggered:
            ring.append(frame)
            if len(ring) > 40:
                ring.pop(0)

            if is_speech:
                speech_count += 1
                if speech_count >= START_FRAMES:
                    triggered = True
                    voiced = ring[:]
                    ring = []
                    silence_count = 0
                    print("[WIN] Speech started...")
            else:
                speech_count = 0

        else:
            voiced.append(frame)

            if is_speech:
                silence_count = 0
            else:
                silence_count += 1

                if silence_count >= END_FRAMES:
                    pcm = b"".join(voiced)

                    # reset capture state
                    voiced = []
                    triggered = False
                    speech_count = 0
                    silence_count = 0
                    ring = []
                    print("[WIN] Speech ended. Transcribing...")

                    if len(pcm) < MIN_UTTERANCE_BYTES:
                        print("[WIN] Too short, skipping.")
                        continue

                    try:
                        user_text = transcribe_pcm(pcm)
                    except Exception as e:
                        print("[WIN] Transcribe error:", e)
                        continue

                    if not user_text:
                        print("[WIN] Empty transcription.")
                        continue

                    print("[USER]", user_text)

                    if user_text.strip().lower() in SMALL_TALK_SET:
                        print("[WIN] Small-talk ignored (anti-loop).")
                        continue

                    try:
                        reply = ollama_chat(user_text)
                    except Exception as e:
                        print("[WIN] Ollama error:", e)
                        continue

                    print("[BOT]", reply)

                    # HARD BLOCK mic input until bot is finished speaking
                    speaking.set()
                    try:
                        dur = stream_kokoro_opus_udp_paced(reply)
                        post_speak_ignore_until = time.time() + 0.35
                        print(f"[WIN] Sent Opus (paced) to Pi. Speak time: {dur:.2f}s")
                    except Exception as e:
                        print("[WIN] TTS/send error:", e)
                        post_speak_ignore_until = time.time() + 0.5
                    finally:
                        speaking.clear()


if __name__ == "__main__":
    run_server()
