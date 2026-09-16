"""FRIDAY — AI companion for Iron Man VR.

Uses Groq (OpenAI-compatible API) to generate in-character banter and
ElevenLabs text-to-speech to speak it, fully non-blocking so the 60 FPS
game loop never stalls. Falls back to pyttsx3 offline speech and, as a
last resort, to silent text-only.

API keys are read from the environment; a small .env loader is included
so `python main.py` works without extra tooling. Keys are never
hardcoded in source.
"""

import io
import os
import queue
import re
import threading
import time

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path=None):
    """Minimal .env parser: fill os.environ only for unset keys."""
    path = path or os.path.join(_DIR, ".env")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        pass


_load_dotenv()


# ----------------------------------------------------------------------
#  Groq LLM
# ----------------------------------------------------------------------
_GROQ_BASE = "https://api.groq.com/openai/v1"
_GROQ_PREFERRED = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-specdec",
    "meta-llama/llama-3.3-70b-versatile",
    "meta-llama/llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
    "qwen-2.5-32b",
]
_GROQ_BANNED = ("tts", "audio", "whisper", "stt", "embed", "asr", "transcribe",
                "canopylabs", "image", "vision")


def _pick_model(ids):
    chat = [m for m in ids if not any(b in m.lower() for b in _GROQ_BANNED)]
    for name in _GROQ_PREFERRED:
        if name in chat:
            return name
    return chat[0] if chat else (ids[0] if ids else None)


class _Llm:
    def __init__(self):
        self._client = None
        self.model = None
        self.error = None

    def ensure(self):
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            self.error = "no GROQ_API_KEY"
            return False
        if self._client is not None:
            return True
        try:
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=_GROQ_BASE, timeout=10.0)
            ids = [m.id for m in client.models.list()]
            self.model = _pick_model(ids)
            if not self.model:
                return False
            self._client = client
            return True
        except Exception as exc:  # offline or bad key
            self.error = str(exc)
            self._client = None
            return False

    def complete(self, system, user, timeout=10.0, max_tokens=70):
        if self._client is None and not self.ensure():
            return None
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=False,
                temperature=0.9,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # rate limit / timeout / offline
            self.error = str(exc)
            return None
        try:
            text = resp.choices[0].message.content or ""
            return _tighten(text)
        except Exception:
            return None


def _tighten(text):
    text = re.sub(r"[*#\\\-`]", "", text)
    text = " ".join(text.split())
    if not text:
        return text
    text = text[0:1].upper() + text[1:]
    if not text.endswith((".", "!", "?")):
        text += "."
    return text[:200]


# ----------------------------------------------------------------------
#  Speech
# ----------------------------------------------------------------------
def _elevenlabs_speech(text, voice_id, timeout=25.0):
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    try:
        import requests

        resp = requests.post(
            url,
            headers={
                "xi-api-key": key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": "eleven_turbo_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.7},
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


class FridayAI:
    """Async FRIDAY: generate a line, speak it, and handle voice commands."""

    ACTIONS = {
        "ironman:on":    "Activate Iron Man combat mode.",
        "ironman:off":   "Deactivate Iron Man combat mode.",
        "holo:toggle":   "Toggle the holographic interface on/off.",
        "scene:next":    "Switch to the next holographic scene.",
        "scene:prev":    "Go back to the previous scene.",
        "materials:next": "Cycle the glove material (graphite/white/skin/hologram).",
        "glove:toggle":  "Toggle glove between transparent and opaque.",
        "stream:url":    "Speak the phone view/stream URL aloud.",
        "recenter":      "Re-center the view to face forward.",
        "status":        "Report score, wave, drones, health, energy.",
        "help":          "Say what FRIDAY can control.",
    }

    _CONV_SYSTEM = (
        "You are FRIDAY, Tony Stark's calm, sharp AI assistant inside an "
        "Iron Man suit. You can also control the suit's systems via voice "
        "commands. Keep replies to one short sentence, casual, dry, witty. "
        "No markdown, no emoji. Never break character.\n\n"
        "If the user asks you to DO something from this list, append one "
        "line at the end of your reply containing EXACTLY `ACT:<key>` "
        "(for example `ACT:ironman:on`). Never append ACT otherwise.\n\n"
        "Available actions:\n{actions}"
    )

    def __init__(self, voice_id=None):
        self.voice_id = voice_id or os.environ.get(
            "FRIDAY_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"
        )
        self.llm = _Llm()
        self.available = self.llm.ensure()
        self.latest = None          # most recent generated/spoken line
        self.last_scope = None
        self._q = queue.Queue()
        self._worker = None
        self._last_event_t = 0.0
        self._min_interval = 20.0   # seconds between AI broadcasts
        self._silent = os.environ.get("FRIDAY_SILENT", "") == "1"
        self._history = []          # (role, text) turns for conversation
        self._init_mixer()

    # ---- audio ----
    @staticmethod
    def _init_mixer():
        try:
            import pygame

            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=44100, size=-16, channels=2)
        except Exception:
            pass

    def _play_on_mixer(self, audio_bytes, wait=False):
        import pygame

        if pygame.mixer.get_init() is None:
            return False
        if wait:
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        elif pygame.mixer.music.get_busy():
            return False
        pygame.mixer.music.load(io.BytesIO(audio_bytes))
        pygame.mixer.music.play()
        return True

    def _speaker(self):
        try:
            from voice_engine import VoiceEngine
        except Exception:
            VoiceEngine = None

        while True:
            text = self._q.get()
            if text is None:
                return
            try:
                audio = _elevenlabs_speech(text, self.voice_id)
                if audio and self._play_on_mixer(audio, wait=True):
                    continue
                if VoiceEngine is not None:
                    VoiceEngine().speak(text)
                else:
                    print("FRIDAY:", text)
            except Exception as exc:
                print("FRIDAY speech error:", exc)

    def speak(self, text):
        """Non-blocking: queue a line for TTS + playback."""
        if self._silent or not text:
            return
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._speaker, daemon=True, name="friday-voice"
            )
            self._worker.start()
        self._q.put(text)

    # ---- STT ----
    def transcribe(self, audio_bytes, filename="voice.wav"):
        """Transcribe audio via Groq Whisper. Returns text or None."""
        if self.llm._client is None and not self.llm.ensure():
            return None
        try:
            resp = self.llm._client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(filename, audio_bytes, "audio/wav"),
            )
            return (getattr(resp, "text", "") or "").strip() or None
        except Exception as exc:
            print("[friday] STT error:", exc)
            return None

    # ---- conversation (voice command + chat) ----
    def conversation(self, user_text, situation=""):
        """Talk with FRIDAY. Returns (reply_text, action_key_or_None).
        Speaks the reply via TTS automatically."""
        if not user_text:
            return None, None
        actions_str = "\n".join(
            f"  {k}: {v}" for k, v in self.ACTIONS.items()
        )
        system = self._CONV_SYSTEM.format(actions=actions_str)
        history = self._history[-6:]  # last 3 turns
        messages = "\n".join(f"{r}: {t}" for r, t in history)
        if situation:
            messages += f"\n[situation: {situation}]"
        messages += f"\nuser: {user_text}"

        reply = self.llm.complete(system, messages, max_tokens=180)
        if not reply:
            reply = "I'm a little slow, boss. Try again."

        action = None
        m = re.search(r"\bACT\s*:\s*([\w:]+)", reply)
        if m:
            tag = m.group(1)
            if tag in self.ACTIONS:
                action = tag
            reply = re.sub(r"\s*\bACT\s*:\s*[\w:]+\s*$", "", reply).strip()

        self._history = history + [("user", user_text[:300]),
                                   ("assistant", reply[:300])]
        self.latest = reply
        print("FRIDAY:", reply)
        self.speak(reply)
        return reply, action

    # ---- generation ----
    _SYSTEM = (
        "You are FRIDAY, Tony Stark's calm, sharp AI assistant inside an "
        "Iron Man suit during aerial drone combat over New York. Respond in "
        "one short sentence. No markdown, no emoji. Keep it casual, dry, "
        "just a little witty. Never break character."
    )

    _CANON = {
        "boot": "F.R.I.D.A.Y. online. Repulsors are ready, boss.",
        "suit_up": "Suit up complete. Repulsors online and, frankly, they look great.",
        "wave_start": "Incoming hostiles from the north. Try not to let them scratch the paint.",
        "boss": "Big target on the radar. I would say enjoy, but you know.",
        "unibeam_ready": "Unibeam is charged and waiting. Try not to leave scorch marks on the skyscrapers.",
        "game_over": "Suit down. Landing gear deployed. Some repairs overdue, boss.",
        "suit_critical": "Hull integrity critical. I strongly suggest evasion.",
        "low_health": "Structural damage rising. Recommend retreat and a very expensive repair.",
        "kill": "Target neutralized. Score: {score}.",
        "ask": "All systems report green. {drones} hostiles on screen, score {score}.",
    }

    def _canon(self, scope, **kw):
        if scope in self._CANON:
            return self._CANON[scope].format(**kw)
        return (self.latest or "All systems nominal, boss.").capitalize()

    def broadcast(self, scope, elapsed=0.0, **kw):
        """Fire a FRIDAY line for a game event. Returns the HUD line
        immediately; the AI version arrives (and speaks) async when ready."""
        line = self._canon(scope, **kw)
        if self._silent or not self.available:
            return line
        now = time.monotonic()
        if now - self._last_event_t < self._min_interval:
            return line
        self._last_event_t = now
        self.last_scope = scope
        self.latest = line

        system = self._SYSTEM
        user = self._describe(scope, **kw)

        def _gen():
            gen = self.llm.complete(system, user)
            if not gen:
                gen = self._canon(scope, **kw)
            self.latest = gen
            self.speak(gen)

        threading.Thread(target=_gen, daemon=True, name="friday-gen").start()
        return line

    @staticmethod
    def _describe(scope, **kw):
        base = {
            "suit_up": "The suit has just finished powering up and the user is about to fly over New York for the first time.",
            "wave_start": f"A new enemy wave just started (wave {kw.get('wave', 1)}). Say one short combat line.",
            "boss": "A boss enemy just appeared on radar. React in one line.",
            "unibeam_ready": "The chest unibeam just finished charging. Announce it in one short line.",
            "game_over": f"The suit just went down (final score {kw.get('score', 0)}). One calm line before shutdown.",
"suit_critical": "The suit is taking heavy damage. Urge evasive action in one line.",
        "ask": f"The user asked for a status report. Current situation: wave {kw.get('wave', 1)}, {kw.get('drones', 0)} hostiles remaining, score {kw.get('score', 0)}. Report it in one short, dry, casual line.",
    }.get(scope)
        return base or "Comment on the situation in one short line."


__all__ = ["FridayAI", "_load_dotenv"]

if __name__ == "__main__":
    ai = FridayAI()
    print("FRIDAY available:", ai.available, "| model:", ai.llm.model)
    ai.broadcast("suit_up")
    time.sleep(4.0)