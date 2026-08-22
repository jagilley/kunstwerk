#!/bin/bash
# separate.sh <config.yaml>
#   1. download_album.py  →  audio/<file_prefix>/NN.m4a + tracks.json
#                            (resolves album_url / album_query / playlist_url; fails on any missing track)
#   2. demucs two-stems   →  sep/<file_prefix>_sep/htdemucs/NN/vocals.m4a
#                            (on Modal GPUs via separate_modal.py by default; KUNSTWERK_SEPARATOR=local for CPU demucs)
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
# Prefer Modal GPUs (separate_modal.py: minutes and cents per opera); fall back to local
# CPU demucs (~1x realtime) if Modal isn't installed/configured (exit code 3) or if
# KUNSTWERK_SEPARATOR=local. Both only touch tracks that lack a vocals stem.
SEPARATOR="${KUNSTWERK_SEPARATOR:-modal}"
if [ "$SEPARATOR" = "modal" ]; then
    rc=0
    python separate_modal.py "$CONFIG" || rc=$?
    if [ "$rc" -eq 3 ]; then
        echo "Modal unavailable; falling back to local demucs"
        SEPARATOR=local
    elif [ "$rc" -ne 0 ]; then
        exit "$rc"
    fi
fi
if [ "$SEPARATOR" = "local" ]; then
    TODO=()
    for f in "$AUDIO_DIR"/[0-9][0-9].m4a; do
        n=$(basename "${f%.m4a}")
        if [ ! -s "$SEP_DIR/htdemucs/$n/vocals.m4a" ] && [ ! -s "$SEP_DIR/htdemucs/$n/vocals.wav" ]; then
            TODO+=("$f")
        fi
    done
    if [ "${#TODO[@]}" -eq 0 ]; then
        echo "all tracks already separated in $SEP_DIR"
    else
        demucs -d cpu -j 2 --two-stems=vocals -o "$SEP_DIR" "${TODO[@]}"
    fi
fi

# demucs writes wav; convert to m4a and drop the wav.
# (-nostdin: otherwise ffmpeg eats the rest of find's output from the pipe.)
find "$SEP_DIR/htdemucs" -type f -name "*.wav" -print0 | while IFS= read -r -d '' f; do
    ffmpeg -nostdin -loglevel error -y -i "$f" "${f%.wav}.m4a" && rm "$f"
done
echo "=== Done: $SEP_DIR/htdemucs/NN/vocals.m4a ==="
