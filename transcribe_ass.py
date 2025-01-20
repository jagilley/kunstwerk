from openai import OpenAI
from classes import TranscriptionVerbose, TranscriptionWord
from dotenv import load_dotenv
import assemblyai as aai
from pdb import set_trace
from dataclasses import asdict
import json
import os
import librosa

def get_audio_duration(file_path):
    audio, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=audio, sr=sr)
    return duration

opera_name = "tristan"
language = "de"
end_idx = 33

load_dotenv()

config = aai.TranscriptionConfig(
    speech_model=aai.SpeechModel.best,
    language_code=language,
)
aai.settings.api_key = "d91d43fd2f754946859238338166242d"
transcriber = aai.Transcriber(config=config)

in_dir = f"sep/{opera_name}_sep"
out_dir = f"transcribed/{opera_name}_ass_transcribed"

for i in range(1, end_idx):
    if os.path.exists(f"{out_dir}/{str(i).zfill(2)}.json"):
        print(f"Skipping {i}")
        continue

    i_string = str(i).zfill(2)
    print(f"Transcribing {i_string}")

    # audio_file = open(f"{in_dir}/htdemucs/{i_string}/vocals.m4a", "rb")
    # get duration of the audio file
    duration = get_audio_duration(f"{in_dir}/htdemucs/{i_string}/vocals.m4a")

    transcript = transcriber.transcribe(f"{in_dir}/htdemucs/{i_string}/vocals.m4a")

    print(transcript.text)

    # set_trace()

    transcription_obj = TranscriptionVerbose(
        text=transcript.text,
        words=[
            TranscriptionWord(
                start=word.start / 1000,
                end=word.end / 1000,
                word=word.text,
                confidence=word.confidence,
            )
            for word in transcript.words
        ],
        language=language,
        duration=duration
    )

    # if transcribed directory does not exist, make it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # dump transcript to json
    with open(f"{out_dir}/{i_string}.json", "w") as f:
        json_str = json.dumps(asdict(transcription_obj), indent=2)
        f.write(json_str)
