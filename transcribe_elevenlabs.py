from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import json
import os
import requests
from config_parser import parse_opera_config
import sys
import librosa
from dataclasses import asdict
from classes import TranscriptionVerbose, TranscriptionWord

def get_audio_duration(file_path):
    audio, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=audio, sr=sr)
    return duration

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

if len(sys.argv) != 2:
    print("Usage: python transcribe_elevenlabs.py <config.md>")
    sys.exit(1)

config = parse_opera_config(sys.argv[1])
opera_name = config.file_prefix
language = config.language
end_idx = config.end_idx

# Map language codes to ElevenLabs format
language_map = {
    "en": "eng",  # English
    "it": "ita",  # Italian
    "de": "deu",  # German
    "fr": "fra",  # French
    "es": "spa"   # Spanish
}

elevenlabs_language = language_map.get(language, None)  # None will trigger auto-detection
# elevenlabs_language = language

in_dir = f"sep/{opera_name}_sep"
out_dir = f"transcribed/{opera_name}_transcribed"

for i in range(1, end_idx):
    if os.path.exists(f"{out_dir}/{str(i).zfill(2)}.json"):
        print(f"Skipping {i}")
        continue

    i_string = str(i).zfill(2)
    print(f"Transcribing {i_string}")

    # Read the audio file
    audio_file_path = f"{in_dir}/htdemucs/{i_string}/vocals.m4a"
    
    # Get audio duration
    duration = get_audio_duration(audio_file_path)
    
    # ElevenLabs expects file data, not a file path
    with open(audio_file_path, "rb") as audio_file:
        audio_data = audio_file.read()
    
    # Perform the transcription using ElevenLabs with retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            transcript = client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1",  # More advanced model
                language_code=elevenlabs_language,
                tag_audio_events=False,
                diarize=False,
            )
        except Exception as e:
            # retry it
            print(e)
            continue
        
        # Check if we got the expected language or if words are missing
        transcript_language = getattr(transcript, "language_code", None)
        has_words = False
        
        if hasattr(transcript, "words") and transcript.words:
            has_words = True
        
        # If expected language doesn't match detected language, or if no words were found, retry
        if (elevenlabs_language and transcript_language != elevenlabs_language) or not has_words:
            retry_count += 1
            print(f"Retry {retry_count}/{max_retries} - Detected language: {transcript_language}, Has words: {has_words}")
            if retry_count >= max_retries:
                print(f"Warning: Max retries reached for {i_string}. Skipping file.")
                break
        else:
            break  # Successful transcription
    
    # Only save if we didn't hit max retries
    if retry_count < max_retries:
        # Create TranscriptionVerbose object with OpenAI format
        transcription_obj = TranscriptionVerbose(
            text=transcript.text,
            words=[
                TranscriptionWord(
                    start=word.start,
                    end=word.end,
                    word=word.text,
                    confidence=getattr(word, "confidence", None),
                )
                for word in transcript.words
            ],
            language=language,
            duration=duration
        )
        
        # if transcribed directory does not exist, make it
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # dump transcript to json using OpenAI format
        with open(f"{out_dir}/{i_string}.json", "w") as f:
            json_str = json.dumps(asdict(transcription_obj), indent=2)
            f.write(json_str)