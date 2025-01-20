#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <config.yaml> <youtube_url>"
    exit 1
fi

# Extract file prefix from yaml config
FILE_PREFIX=$(python parse_yaml.py "$1")

yt-dlp -x --audio-format m4a -o "audio/$FILE_PREFIX/%(playlist_index)s.m4a" "$2"
demucs -d cpu -j 2 --two-stems=vocals audio/"$FILE_PREFIX"/*.m4a -o sep/"$FILE_PREFIX"_sep
find . -type f -name "$FILE_PREFIX"_sep/htdemucs/*/*.wav -exec sh -c 'ffmpeg -i "$1"_sep "${1%.wav}.m4a" && rm "$1"_sep' _ {} \;

# do this 50 times
for i in {1..20}; do
    find . -type f -name "*.wav" -path "*/sep/$1_sep/htdemucs/*" | while read -r file; do
        ffmpeg -i "$file" "${file%.wav}.m4a" && rm "$file"
    done
done
