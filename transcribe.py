from openai import OpenAI
from openai.types.audio import TranscriptionVerbose, TranscriptionWord
from dotenv import load_dotenv
import json
import os
from config_parser import parse_opera_config
import sys

load_dotenv()

use_v3 = False

if use_v3:
    client = OpenAI(
        api_key=os.getenv("DEEPINFRA_API_KEY"),
        base_url="https://api.deepinfra.com/v1/openai",
)
else:
    client = OpenAI()

if len(sys.argv) != 2:
    print("Usage: python transcribe.py <config.md>")
    sys.exit(1)

config = parse_opera_config(sys.argv[1])
opera_name = config.file_prefix
language = config.language
end_idx = config.end_idx

in_dir = f"sep/{opera_name}_sep"
out_dir = f"transcribed/{opera_name}_transcribed"

for i in range(1, end_idx):
    if os.path.exists(f"{out_dir}/{str(i).zfill(2)}.json"):
        print(f"Skipping {i}")
        continue

    i_string = str(i).zfill(2)
    print(f"Transcribing {i_string}")

    audio_file = open(f"{in_dir}/htdemucs/{i_string}/vocals.m4a", "rb")
    try:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            language=language,
            model="whisper-1" if not use_v3 else "openai/whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "insufficient_quota" in msg or "credit_balance_exhausted" in msg or "no credits" in msg.lower():
            print(f"ERROR: the OpenAI key has no credits, so {i_string} (which ElevenLabs did not transcribe) "
                  f"cannot be transcribed. Add credits at https://platform.openai.com/settings/organization/billing/ "
                  f"or re-run transcribe_elevenlabs.py; make_video.py needs a JSON for every track.", file=sys.stderr)
            sys.exit(1)
        raise

    # if transcribed directory does not exist, make it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # dump transcript to json
    with open(f"{out_dir}/{i_string}.json", "w") as f:
        json_str = transcript.model_dump_json()
        formatted_json_str = json.dumps(json.loads(json_str), indent=2)
        f.write(formatted_json_str)
