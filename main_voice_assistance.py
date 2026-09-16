"""Kazumi — standalone space voice assistant (Groq + ElevenLabs).

Formerly hardcoded the ElevenLabs key in source; now reads API keys from
the environment / .env (see .env.example). AI: Groq first, Ollama local
fallback. TTS: ElevenLabs. STT: SpeechRecognition (Google) — requires
`pip install SpeechRecognition pyaudio` for the microphone path.
"""

import io
import os
import queue
import re
import threading

import pygame

from friday_ai import _load_dotenv, _pick_model, _tighten

_load_dotenv()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = "gpt-oss:120b-cloud"

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "").strip()
VOICE_ID = os.environ.get("KAZUMI_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

if not ELEVENLABS_API_KEY:
    print(
        "WARNING: ELEVENLABS_API_KEY not set — voice disabled. "
        "Copy .env.example to .env and add your key."
    )

eleven_client = None
try:
    from elevenlabs.client import ElevenLabs

    if ELEVENLABS_API_KEY:
        eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
except Exception:
    pass

pygame.mixer.init()

# ---- optional speech recognition (only if installed) ----
try:
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    MIC_AVAILABLE = True
except Exception as e:
    recognizer = None
    MIC_AVAILABLE = False
    print("Speech recognition unavailable:", e)

try:
    import pandas as pd

    planets_df = pd.read_csv("extracted_data/planets_updated.csv")
    exoplanets_df = pd.read_csv("extracted_data/space_objects_combined.csv")
    print("Space data loaded successfully.")
except Exception as e:
    print(f"Error reading CSV: {e}")

    class _Empty:
        empty = True

    planets_df = _Empty()
    exoplanets_df = _Empty()


# ---- Groq backend ----
GROQ_BASE = "https://api.groq.com/openai/v1"
_groq_client = None
_groq_model = None


def _groq_ready():
    global _groq_client, _groq_model
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return False
    if _groq_client is not None:
        return True
    try:
        from openai import OpenAI

        _groq_client = OpenAI(api_key=key, base_url=GROQ_BASE, timeout=15.0)
        _groq_model = _pick_model([m.id for m in _groq_client.models.list()])
        return _groq_model is not None
    except Exception:
        _groq_client = None
        return False


def ask_groq(prompt, system_prompt):
    if not _groq_ready():
        return None
    try:
        resp = _groq_client.chat.completions.create(
            model=_groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=90,
        )
        return _tighten(resp.choices[0].message.content or "")
    except Exception:
        return None


# ---- local Ollama backend (fallback) ----
def ask_ollama(prompt, system_prompt):
    import requests

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_prompt}\nUser: {prompt}\nAssistant:",
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        return _tighten(response.json()["response"]) if response.status_code == 200 else None
    except Exception:
        return None


def get_space_context(query):
    from rapidfuzz import fuzz, process

    query = query.lower()
    if not planets_df.empty:
        planet_names = planets_df["Planet"].tolist()
        best_match = process.extractOne(query, planet_names, scorer=fuzz.partial_token_sort_ratio)
        if best_match and best_match[1] > 65:
            row = planets_df[planets_df["Planet"] == best_match[0]].iloc[0]
            return (
                f"Data for {row['Planet']}: {row['Color']} color, "
                f"features: {row['Surface Features']}, atmosphere: {row['Atmospheric Composition']}."
            )

    if not exoplanets_df.empty:
        exo_match = process.extractOne(query, exoplanets_df["name"].dropna().tolist(), scorer=fuzz.WRatio)
        if exo_match and exo_match[1] > 80:
            row = exoplanets_df[exoplanets_df["name"] == exo_match[0]].iloc[0]
            return f"{row['name']} is a {row['planet_type']} located {row['distance']} light-years away."

    return "Use general space knowledge."


def ask_ai(prompt):
    context = get_space_context(prompt)
    system_prompt = (
        "You are 'Kazumi', a cute female Space AI.\n"
        f"Context: {context}\n"
        "Rules: 1. English only. 2. No markdown. 3. Short (max 2 sentences). 4. Child-friendly."
    )
    return ask_groq(prompt, system_prompt) or ask_ollama(prompt, system_prompt) or "I'm having trouble thinking right now."


def speak_text_eleven(text):
    if not text or eleven_client is None:
        return
    try:
        print(f"Kazumi is speaking: {text}")
        audio_generator = eleven_client.text_to_speech.convert(
            voice_id=VOICE_ID,
            model_id="eleven_turbo_v2",
            text=text,
        )
        audio_bytes = b"".join(audio_generator)
        pygame.mixer.music.load(io.BytesIO(audio_bytes))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print("ElevenLabs Error:", e)


class VoiceAssistantService:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._action_queue = queue.Queue()

    def _extract_navigation_action(self, text):
        match = re.search(r"\b(show me|go to|fly to|take me to)\b\s+(.+)", text.lower().strip())
        if not match:
            return None
        raw = re.sub(r"\b(please|now|kazumi)\b", "", match.group(2).strip(" .!?")).strip()
        if not raw:
            return None
        targets = {
            "sun": "Sun", "earth": "Earth", "moon": "Moon",
            "mercury": "Mercury", "venus": "Venus", "mars": "Mars",
            "jupiter": "Jupiter", "saturn": "Saturn",
            "uranus": "Uranus", "neptune": "Neptune",
            "new york": "New York", "yangon": "Yangon",
            "tokyo": "Tokyo", "london": "London", "paris": "Paris",
        }
        return {"type": "VOICE_ACTION", "action": "FLY_TO", "target": targets.get(raw, raw.title())}

    def _extract_simulation_action(self, text):
        l = text.lower()
        if "lunar eclipse" in l:
            return {"type": "SIMULATION", "action": "LUNAR_ECLIPSE"}
        if "solar eclipse" in l:
            return {"type": "SIMULATION", "action": "SOLAR_ECLIPSE"}
        if "reset all" in l or "reset" in l:
            return {"type": "SIMULATION", "action": "RESET_UNIVERSE"}
        return None

    def pop_pending_action(self):
        try:
            return self._action_queue.get_nowait()
        except queue.Empty:
            return None

    def _run(self):
        if not MIC_AVAILABLE or recognizer is None:
            print("Kazumi: STT unavailable — install SpeechRecognition + PyAudio to enable the mic.")
            return
        print("Kazumi English (ElevenLabs + Groq) Started")
        speak_text_eleven("Hi! I'm Kazumi. Ask me anything about space!")
        while not self._stop_event.is_set():
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.8)
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                text = recognizer.recognize_google(audio, language="en-US")
                print("You:", text)
                if any(w in text.lower() for w in ["stop", "exit", "bye"]):
                    speak_text_eleven("Goodbye! See you later!")
                    break
                sim = self._extract_simulation_action(text)
                if sim:
                    self._action_queue.put(sim)
                    speak_text_eleven("Simulation command received.")
                    continue
                nav = self._extract_navigation_action(text)
                if nav:
                    self._action_queue.put(nav)
                    speak_text_eleven(f"Flying to {nav['target']}.")
                    continue
                reply = ask_ai(text)
                print("Kazumi:", reply)
                speak_text_eleven(reply)
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                continue
            except Exception as e:
                print("System Error:", e)

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        pygame.mixer.music.stop()


if __name__ == "__main__":
    service = VoiceAssistantService()
    service.start()
    try:
        while True:
            threading.Event().wait(1.0)
    except KeyboardInterrupt:
        service.stop()