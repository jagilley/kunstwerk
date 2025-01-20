from openai import OpenAI
from openai.types.audio import TranscriptionVerbose, TranscriptionWord
from dotenv import load_dotenv
import openai
import json
import os

load_dotenv()

client = OpenAI()

opera_name = "tristan"
language = "de"
end_idx = 33

in_dir = f"sep/{opera_name}_sep"
out_dir = f"transcribed/{opera_name}_transcribed"

for i in range(1, end_idx):
    if os.path.exists(f"{out_dir}/{str(i).zfill(2)}.json"):
        print(f"Skipping {i}")
        continue

    i_string = str(i).zfill(2)
    print(f"Transcribing {i_string}")

    audio_file = open(f"{in_dir}/htdemucs/{i_string}/vocals.m4a", "rb")
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        language=language,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )

    # if transcribed directory does not exist, make it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # dump transcript to json
    with open(f"{out_dir}/{i_string}.json", "w") as f:
        json_str = transcript.model_dump_json()
        formatted_json_str = json.dumps(json.loads(json_str), indent=2)
        f.write(formatted_json_str)
