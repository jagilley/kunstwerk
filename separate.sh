#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config.yaml>"
    exit 1
fi

# Extract file prefix and playlist URL from yaml config
CONFIG_OUTPUT=$(python parse_yaml.py "$1")
# parse_yaml now outputs: file_prefix playlist_url spotify_url
read -r FILE_PREFIX PLAYLIST_URL SPOTIFY_URL <<< "$CONFIG_OUTPUT"

# Download audio: either via Spotify album/playlist link or fallback to YouTube playlist
if [ -n "$SPOTIFY_URL" ]; then
    echo "\n=== Downloading audio via Spotify link: $SPOTIFY_URL ==="
    # Use Python script to fetch track list, resolve YouTube URLs, and download sequentially
    python download_spotify.py "$1"
else
    echo "\n=== Downloading audio via YouTube playlist: $PLAYLIST_URL ==="
    yt-dlp -x --audio-format m4a -o "audio/$FILE_PREFIX/%(playlist_index)s.m4a" "$PLAYLIST_URL"
fi
demucs -d cpu -j 2 --two-stems=vocals audio/"$FILE_PREFIX"/*.m4a -o sep/"$FILE_PREFIX"_sep
# find . -type f -name "$FILE_PREFIX"_sep/htdemucs/*/*.wav -exec sh -c 'ffmpeg -i "$FILE_PREFIX"_sep "${1%.wav}.m4a" && rm "$FILE_PREFIX"_sep' _ {} \;

# do this 50 times
for i in {1..20}; do
    find . -type f -name "*.wav" -path "*/sep/"$FILE_PREFIX"_sep/htdemucs/*" | while read -r file; do
        ffmpeg -i "$file" "${file%.wav}.m4a" && rm "$file"
    done
done
