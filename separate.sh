#!/bin/bash
# separate.sh <config.yaml>
#   1. download_album.py  →  audio/<file_prefix>/NN.m4a + tracks.json
#                            (resolves album_url / album_query / playlist_url; fails on any missing track)
#   2. demucs two-stems   →  sep/<file_prefix>_sep/htdemucs/NN/{vocals,no_vocals}.m4a
# Run from the repo root with the venv active.
set -euo pipefail
shopt -s nullglob

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config.yaml>" >&2
    exit 1
fi
CONFIG="$1"
FILE_PREFIX=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["file_prefix"])' "$CONFIG")
AUDIO_DIR="audio/$FILE_PREFIX"
SEP_DIR="sep/${FILE_PREFIX}_sep"

echo "=== Downloading audio for $FILE_PREFIX ==="
python download_album.py "$CONFIG"

echo "=== Separating vocals (demucs) ==="
# Only run demucs on tracks that don't already have a separated vocals file.
TODO=()
for f in "$AUDIO_DIR"/[0-9][0-9].m4a; do
    n=$(basename "${f%.m4a}")
    if [ ! -s "$SEP_DIR/htdemucs/$n/vocals.m4a" ]; then
        TODO+=("$f")
    fi
done
if [ "${#TODO[@]}" -eq 0 ]; then
    echo "all tracks already separated in $SEP_DIR"
else
    demucs -d cpu -j 2 --two-stems=vocals -o "$SEP_DIR" "${TODO[@]}"
fi

# demucs writes wav; convert to m4a and drop the wav.
# (-nostdin: otherwise ffmpeg eats the rest of find's output from the pipe.)
find "$SEP_DIR/htdemucs" -type f -name "*.wav" -print0 | while IFS= read -r -d '' f; do
    ffmpeg -nostdin -loglevel error -y -i "$f" "${f%.wav}.m4a" && rm "$f"
done
echo "=== Done: $SEP_DIR/htdemucs/NN/vocals.m4a ==="
