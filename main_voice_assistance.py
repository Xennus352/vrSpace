import speech_recognition as sr
import requests
import io
import pygame
import re
import threading
import queue
import pandas as pd
from rapidfuzz import process, fuzz
from elevenlabs.client import ElevenLabs

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gpt-oss:120b-cloud"

ELEVENLABS_API_KEY = "sk_c1af9769b5d87ccbe06e4f13bdef045c6b8ed172a9dfbb24"
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

pygame.mixer.init()
recognizer = sr.Recognizer()

try:
    planets_df = pd.read_csv("extracted_data/planets_updated.csv")
    exoplanets_df = pd.read_csv("extracted_data/space_objects_combined.csv")
    print("Space data loaded successfully.")
except Exception as e:
    print(f"Error reading CSV: {e}")
    planets_df = pd.DataFrame()


def get_space_context(query):
    query = query.lower()
    if not planets_df.empty:
        planet_names = planets_df["Planet"].tolist()
        best_match = process.extractOne(query, planet_names, scorer=fuzz.partial_token_sort_ratio)
        if best_match and best_match[1] > 65:
            match_name = best_match[0]
            row = planets_df[planets_df["Planet"] == match_name].iloc[0]
            return (
                f"Data for {row['Planet']}: {row['Color']} color, "
                f"features: {row['Surface Features']}, atmosphere: {row['Atmospheric Composition']}."
            )

    if "exoplanets_df" in globals() and not exoplanets_df.empty:
        exo_names = exoplanets_df["name"].dropna().tolist()
        exo_match = process.extractOne(query, exo_names, scorer=fuzz.WRatio)
        if exo_match and exo_match[1] > 80:
            match_name = exo_match[0]
            row = exoplanets_df[exoplanets_df["name"] == match_name].iloc[0]
            return f"{row['name']} is a {row['planet_type']} located {row['distance']} light-years away."

    return "Use general space knowledge."


def clean_text(text):
    text = re.sub(r"[*#\\-]", "", text)
    text = re.sub(r"\d+\.", "", text)
    return text.strip()


def ask_ollama(prompt):
    context = get_space_context(prompt)
    system_prompt = f"""
    You are 'Kazumi', a cute female Space AI.
    Context: {context}
    Rules: 1. English only. 2. No markdown. 3. Short (max 2 sentences). 4. Child-friendly.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_prompt}\nUser: {prompt}\nAssistant:",
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        return clean_text(response.json()["response"]) if response.status_code == 200 else "Server error."
    except Exception:
        return "I'm having trouble thinking right now."


def speak_text_eleven(text):
    if not text:
        return

    try:
        print(f"Kazumi is speaking: {text}")
        audio_generator = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            model_id="eleven_turbo_v2",
            text=text,
        )
        audio_bytes = b"".join(audio_generator)
        audio_stream = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(audio_stream)
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
        lowered = text.lower().strip()
        match = re.search(r"\b(show me|go to|fly to|take me to)\b\s+(.+)", lowered)
        if not match:
            return None

        raw_target = match.group(2).strip(" .!?")
        raw_target = re.sub(r"\b(please|now|kazumi)\b", "", raw_target).strip()
        if not raw_target:
            return None

        canonical_map = {
            "sun": "Sun",
            "earth": "Earth",
            "moon": "Moon",
            "mercury": "Mercury",
            "venus": "Venus",
            "mars": "Mars",
            "jupiter": "Jupiter",
            "saturn": "Saturn",
            "uranus": "Uranus",
            "neptune": "Neptune",
            "new york": "New York",
            "yangon": "Yangon",
            "tokyo": "Tokyo",
            "london": "London",
            "paris": "Paris",
        }
        target = canonical_map.get(raw_target, raw_target.title())
        return {"type": "VOICE_ACTION", "action": "FLY_TO", "target": target}

    def _extract_simulation_action(self, text):
        lowered = text.lower()
        if "lunar eclipse" in lowered:
            return {"type": "SIMULATION", "action": "LUNAR_ECLIPSE"}
        if "solar eclipse" in lowered:
            return {"type": "SIMULATION", "action": "SOLAR_ECLIPSE"}
        if "reset all" in lowered or "reset universe" in lowered or "reset" in lowered:
            return {"type": "SIMULATION", "action": "RESET_UNIVERSE"}
        return None

    def pop_pending_action(self):
        try:
            return self._action_queue.get_nowait()
        except queue.Empty:
            return None

    def _run(self):
        print("Kazumi English (ElevenLabs) Started")
        speak_text_eleven("Hi! I'm Kazumi. Ask me anything about space!")

        while not self._stop_event.is_set():
            try:
                with sr.Microphone() as source:
                    print("\nListening...")
                    recognizer.adjust_for_ambient_noise(source, duration=0.8)
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)

                text = recognizer.recognize_google(audio, language="en-US")
                print("You:", text)

                if any(word in text.lower() for word in ["stop", "exit", "bye"]):
                    speak_text_eleven("Goodbye! See you later!")
                    break

                sim_action = self._extract_simulation_action(text)
                if sim_action is not None:
                    self._action_queue.put(sim_action)
                    speak_text_eleven("Simulation command received.")
                    continue

                nav_action = self._extract_navigation_action(text)
                if nav_action is not None:
                    self._action_queue.put(nav_action)
                    speak_text_eleven(f"Flying to {nav_action['target']}.")
                    continue

                ai_response = ask_ollama(text)
                print("Kazumi:", ai_response)
                speak_text_eleven(ai_response)

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
