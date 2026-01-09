import socket, threading, struct, subprocess, queue, time
import sounddevice as sd
import numpy as np
import opuslib_next  # pip install opuslib_next

WINDOWS_IP = "192.168.68.61"   # Windows IP
WINDOWS_PORT = 5555            # Mic stream -> Windows (TCP)

# Old WAV playback (TCP) - optional to keep
PI_PLAYBACK_PORT = 5556

# Opus playback (UDP) from Windows
OPUS_LISTEN_PORT = 5557

# Mic stream format (to match VAD on Windows)
MIC_SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
FRAME_MS = 20
FRAME_SAMPLES = int(MIC_SAMPLE_RATE * FRAME_MS / 1000)  # 320 samples
FRAME_BYTES = FRAME_SAMPLES * 2  # 640 bytes

# Opus playback format (Kokoro native output sample rate)
OPUS_SR = 24000
OPUS_CHANNELS = 1
OPUS_FRAME_MS = 20
OPUS_FRAME_SAMPLES = int(OPUS_SR * OPUS_FRAME_MS / 1000)  # 480 samples @ 24kHz

# ---------------------------------------------------------------------
# MIC MUTE WHILE TTS PLAYS (Echo prevention)
# ---------------------------------------------------------------------
MIC_MUTE_HANGOVER_MS = 350  # keep mic muted a bit after audio ends

_talking_until = 0.0
_talking_lock = threading.Lock()

def mark_talking():
    """Extend the 'talking' window into the near future."""
    global _talking_until
    with _talking_lock:
        _talking_until = max(
            _talking_until,
            time.monotonic() + MIC_MUTE_HANGOVER_MS / 1000.0
        )

def is_talking() -> bool:
    """True if we are currently in the 'talking' window."""
    with _talking_lock:
        return time.monotonic() < _talking_until


def playback_server_wav_tcp():
    """Legacy TCP WAV playback on port 5556 (optional)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("", PI_PLAYBACK_PORT))
    srv.listen(5)
    print(f"[PI] WAV playback server listening on TCP :{PI_PLAYBACK_PORT}")

    while True:
        conn, addr = srv.accept()
        try:
            hdr = conn.recv(4)
            if len(hdr) < 4:
                continue
            (n,) = struct.unpack(">I", hdr)

            buf = b""
            while len(buf) < n:
                chunk = conn.recv(min(65536, n - len(buf)))
                if not chunk:
                    break
                buf += chunk

            if len(buf) == n:
                path = "/tmp/tts.wav"
                with open(path, "wb") as f:
                    f.write(buf)

                # Mute mic during playback + hangover
                mark_talking()
                subprocess.run(["aplay", "-q", path], check=False)
                mark_talking()
        finally:
            conn.close()


def opus_playback_server_udp():
    """
    Low-latency Opus playback:
    - Receives UDP packets from Windows on port 5557
    - Packet format: >HB header (seq:uint16, flags:uint8) + opus payload
      flags bit0 = end-of-stream marker
    - Decodes with opuslib_next.Decoder and plays with sounddevice in ~20ms frames.
    """
    print(f"[PI] Opus playback listening on UDP :{OPUS_LISTEN_PORT}")

    # Decoder matches Windows encoder settings
    dec = opuslib_next.Decoder(OPUS_SR, OPUS_CHANNELS)

    q = queue.Queue(maxsize=300)  # ~6 seconds of 20ms frames

    def audio_cb(outdata, frames, time_info, status):
        try:
            pcm = q.get_nowait()
            # Still actively playing buffered audio -> mute mic
            mark_talking()
            out = np.frombuffer(pcm, dtype=np.int16)
        except queue.Empty:
            out = np.zeros((frames,), dtype=np.int16)

        # Ensure exact frame count
        if out.shape[0] < frames:
            out = np.pad(out, (0, frames - out.shape[0]))
        elif out.shape[0] > frames:
            out = out[:frames]

        outdata[:] = out.reshape(-1, 1)

    stream = sd.OutputStream(
        samplerate=OPUS_SR,
        channels=1,
        dtype="int16",
        blocksize=OPUS_FRAME_SAMPLES,
        callback=audio_cb,
    )
    stream.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", OPUS_LISTEN_PORT))

    while True:
        data, addr = sock.recvfrom(4096)
        if len(data) < 3:
            continue

        _seq, flags = struct.unpack(">HB", data[:3])
        payload = data[3:]

        if flags & 1:
            # End marker; let queue drain (audio_cb will keep marking talking)
            mark_talking()
            continue

        if not payload:
            continue

        # We are about to enqueue audio -> mute mic
        mark_talking()

        try:
            # Decode to PCM16 bytes; frame_size must match what Windows encodes (20ms @ 24k = 480)
            pcm = dec.decode(payload, OPUS_FRAME_SAMPLES)
            try:
                q.put_nowait(pcm)
            except queue.Full:
                # Drop frames if overloaded to keep latency low
                pass
        except Exception as e:
            print("[PI] Opus decode error:", e)


def mic_stream_to_windows():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((WINDOWS_IP, WINDOWS_PORT))
    print(f"[PI] Connected to Windows at {WINDOWS_IP}:{WINDOWS_PORT}")

    # Pre-allocated silence frame (exactly one 20ms frame)
    silence = np.zeros((FRAME_SAMPLES, CHANNELS), dtype=np.int16)

    def callback(indata, frames, time_info, status):
        # fixed 20ms frames -> 640 bytes (matches VAD requirements)
        try:
            if is_talking():
                s.sendall(silence.tobytes())
            else:
                s.sendall(indata.tobytes())
        except Exception:
            raise sd.CallbackStop

    with sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=FRAME_SAMPLES,
        callback=callback,
    ):
        print("[PI] Streaming mic... Ctrl+C to stop")
        threading.Event().wait()


if __name__ == "__main__":
    # Start Opus playback (fast)
    threading.Thread(target=opus_playback_server_udp, daemon=True).start()

    # Optional: keep old WAV TCP playback too
    # threading.Thread(target=playback_server_wav_tcp, daemon=True).start()

    # Start mic stream to Windows
    mic_stream_to_windows()
