yt-dlp -x --audio-format m4a -o "audio/$1/%(playlist_index)s.m4a" "$2"
demucs -d cpu -j 2 --two-stems=vocals audio/"$1"/*.m4a -o sep/"$1"_sep
find . -type f -name "$1"_sep/htdemucs/*/*.wav -exec sh -c 'ffmpeg -i "$1"_sep "${1%.wav}.m4a" && rm "$1"_sep' _ {} \;

# do this 50 times
for i in {1..20}; do
    find . -type f -name "*.wav" -path "*/sep/$1_sep/htdemucs/*" | while read -r file; do
        ffmpeg -i "$file" "${file%.wav}.m4a" && rm "$file"
    done
done