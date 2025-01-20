#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config.yaml>"
    exit 1
fi

# Extract file prefix and playlist URL from yaml config
CONFIG_OUTPUT=$(python parse_yaml.py "$1")
read -r FILE_PREFIX PLAYLIST_URL <<< "$CONFIG_OUTPUT"

yt-dlp -x --audio-format m4a -o "audio/$FILE_PREFIX/%(playlist_index)s.m4a" "$PLAYLIST_URL"
demucs -d cpu -j 2 --two-stems=vocals audio/"$FILE_PREFIX"/*.m4a -o sep/"$FILE_PREFIX"_sep
# find . -type f -name "$FILE_PREFIX"_sep/htdemucs/*/*.wav -exec sh -c 'ffmpeg -i "$FILE_PREFIX"_sep "${1%.wav}.m4a" && rm "$FILE_PREFIX"_sep' _ {} \;

# do this 50 times
for i in {1..20}; do
    find . -type f -name "*.wav" -path "*/sep/"$FILE_PREFIX"_sep/htdemucs/*" | while read -r file; do
        ffmpeg -i "$file" "${file%.wav}.m4a" && rm "$file"
    done
done
