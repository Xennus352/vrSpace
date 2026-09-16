"""Push-to-talk voice input for FRIDAY.

Captures the microphone with `sounddevice`, splits speech into utterances
with an adaptive voice-activity detector (noise floor + silence gap), and
sends each utterance to a transcriber (Groq Whisper) in a background
thread. Non-blocking and self-contained so the game loop never stalls.

The main loop arms/disarms the listener with a walkie-talkie pattern
(press L to talk; listening stops automatically after one utterance) so
FRIDAY's own reply through the speakers can't feed back into the mic.
"""

import struct
import threading
import time

import numpy as np

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:  # pragma: no cover - env dependent
    sd = None
    SD_AVAILABLE = False

NOISE_SEED = 0.02       # initial normalized RMS floor (0..1)
VOICE_MIN = 0.03        # absolute RMS floor for very quiet rooms
VOICE_RATIO = 3.5       # voice onset when rms > VOICE_MIN * floor


def _wav_bytes(samples, sample_rate):
    """Build a 16-bit mono WAV file from an int16 numpy array."""
    n = len(samples)
    block = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + n * 2, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", n * 2,
    )
    return block + samples.tobytes()


class VoiceListener:
    """Mic -> VAD -> per-utterance WAV -> transcriber text callback."""

    def __init__(self, transcriber, on_text, sample_rate=16000,
                 blocksize=1024, silence_sec=0.55, max_sec=8.0,
                 min_utterance_ms=300):
        self.transcriber = transcriber   # callable(wav_bytes) -> str
        self.on_text = on_text           # callable(str) called on transcript
        self.sr = sample_rate
        self.silence = silence_sec
        self.max_sec = max_sec
        self.min_wav_bytes = int(sample_rate * min_utterance_ms / 1000)

        self.armed = False
        self._floor = NOISE_SEED
        self._seg = []                   # list of int16 blocks while speaking
        self._seg_len = 0
        self._speaking = False
        self._last_voice = 0.0
        self._seg_start = 0.0
        self._closed = False
        self._lock = threading.Lock()
        self._stream = None

    # ------------------------------------------------------------------
    @property
    def sd_ok(self):
        return SD_AVAILABLE

    def start(self):
        if not SD_AVAILABLE:
            print("[mic] sounddevice not installed - voice chat disabled.")
            return None
        try:
            self._stream = sd.InputStream(
                samplerate=self.sr, channels=1, dtype="int16",
                blocksize=1024, callback=self._on_audio)
            self._stream.start()
        except Exception as exc:
            print("[mic] could not open microphone:", exc)
            self._stream = None
        return self._stream

    def arm(self):
        self.armed = True

    def disarm(self, flush=True):
        """Stop listening. Optionally transcribe a trailing partial
        utterance that was still being captured."""
        self.armed = False
        if flush:
            self._flush_utterance(force=True)

    def set_armed(self, on):
        if on:
            self.arm()
        else:
            self.disarm()

    def close(self):
        self._closed = True
        self.disarm(flush=False)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ------------------------------------------------------------------
    def _on_audio(self, indata, _frames, _time, _status):
        if self._closed or not self.armed:
            return
        try:
            rms = float(np.sqrt(
                np.mean(indata.astype(np.float64) ** 2))) / 32768.0
        except Exception:
            return
        now = time.time()

        threshold = max(VOICE_MIN, self._floor * VOICE_RATIO)
        if not self._speaking:
            # idle: keep the ambient noise floor warm; start on voice onset
            self._floor += (rms - self._floor) * 0.05
            if rms > threshold:
                self._speaking = True
                self._seg = [indata.copy()]
                self._seg_len = len(indata)
                self._last_voice = now
                self._seg_start = now
        else:
            # speech: accumulate, end after a quiet gap or max length
            self._seg.append(indata.copy())
            self._seg_len += len(indata)
            if rms > threshold:
                self._last_voice = now
            if self._seg_len >= int(self.sr * self.max_sec):
                self._speaking = False
                self._dispatch()
            elif now - self._last_voice > self.silence:
                self._speaking = False
                self._dispatch()

    def _dispatch(self):
        """Ship the captured utterance off to STT in a worker thread."""
        if self._seg_len < self.min_wav_bytes:
            self._seg = []
            self._seg_len = 0
            return
        audio = np.concatenate(self._seg, axis=0)
        self._seg = []
        self._seg_len = 0
        wav = _wav_bytes(audio, self.sr)
        threading.Thread(target=self._handle, args=(wav,), daemon=True).start()
        self.armed = False  # walkie-talkie: one utterance per L press

    def _handle(self, wav):
        try:
            text = self.transcriber(wav) if self.transcriber else None
        except Exception as exc:
            text = None
            print("[mic] STT error:", exc)
        if text and text.strip():
            self.on_text(text.strip())

    def _flush_utterance(self, force=False):
        """Called from the main thread on disarm - capture any trailing
        speech that did not reach the silence gap yet."""
        if not self._speaking:
            return
        self._speaking = False
        if force and self._seg_len >= self.min_wav_bytes:
            audio = np.concatenate(self._seg, axis=0)
            self._seg = []
            self._seg_len = 0
            wav = _wav_bytes(audio, self.sr)
            threading.Thread(target=self._handle, args=(wav,),
                             daemon=True).start()
        else:
            self._seg = []
            self._seg_len = 0


def default_transcriber_session(ai):
    """Convenience: return a VoiceListener bound to a FridayAI instance
    with its Groq Whisper transcription."""
    return VoiceListener(
        transcriber=ai.transcribe,
        on_text=lambda text: print("You:", text),
    )