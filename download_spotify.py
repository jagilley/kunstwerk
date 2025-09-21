#!/usr/bin/env python3
"""
download_spotify.py

Fetches track list from a Spotify album or playlist, resolves each track to the auto-generated YouTube art-track URL via the Song.link (Odesli) API,
and downloads the audio with yt-dlp sequentially into audio/<file_prefix>/.
"""
import os
import sys
import re
import time
import yaml
import requests
import subprocess
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_spotify_client():
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    if not client_id or not client_secret:
        print('Error: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in the environment.', file=sys.stderr)
        sys.exit(1)
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return Spotify(auth_manager=auth_manager)


def fetch_track_ids(sp: Spotify, spotify_url: str) -> list[str]:
    # Extract kind and ID from URL or URI
    m = re.search(r"(album|playlist)[/:]([0-9A-Za-z]+)", spotify_url)
    if not m:
        print(f"Error: Couldn't parse Spotify album or playlist ID from '{spotify_url}'", file=sys.stderr)
        sys.exit(1)
    kind, coll_id = m.groups()
    ids = []
    print(f"Fetching Spotify {kind} tracks for ID: {coll_id}")
    if kind == 'album':
        results = sp.album_tracks(coll_id, limit=50)
        items = results.get('items', [])
        ids.extend(item.get('id') for item in items if item.get('id'))
        # pagination
        while results.get('next'):
            results = sp.next(results)
            items = results.get('items', [])
            ids.extend(item.get('id') for item in items if item.get('id'))
    else:
        results = sp.playlist_tracks(coll_id, limit=50)
        items = results.get('items', [])
        for item in items:
            track = item.get('track')
            if track and track.get('id'):
                ids.append(track.get('id'))
        # pagination
        while results.get('next'):
            results = sp.next(results)
            items = results.get('items', [])
            for item in items:
                track = item.get('track')
                if track and track.get('id'):
                    ids.append(track.get('id'))
    return ids


def resolve_youtube_url(track_id: str) -> str | None:
    odesli_api = f"https://api.song.link/v1-alpha.1/links?url=https://open.spotify.com/track/{track_id}"
    try:
        r = requests.get(odesli_api, timeout=10)
        r.raise_for_status()
        info = r.json()
        yt_url = info.get('linksByPlatform', {})
        yt_url = yt_url.get('youtube', {})
        url = yt_url.get('url')
        if not url:
            print(f"Warning: No YouTube link found for track {track_id}", file=sys.stderr)
        return url
    except Exception as e:
        print(f"Warning: Failed to resolve YouTube URL for {track_id}: {e}", file=sys.stderr)
        return None


def download_audio(prefix: str, youtube_urls: list[str]) -> None:
    out_dir = os.path.join('audio', prefix)
    os.makedirs(out_dir, exist_ok=True)
    for idx, url in enumerate(youtube_urls, start=1):
        if not url:
            print(f"Skipping track {idx}: no URL", file=sys.stderr)
            continue
        out_file = os.path.join(out_dir, f"{str(idx).zfill(2)}.m4a")
        print(f"Downloading track {idx}/{len(youtube_urls)}: {url}")
        cmd = [
            'yt-dlp',
            '-x',
            '--audio-format', 'm4a',
            '-o', out_file,
            url
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Error: yt-dlp failed for {url}", file=sys.stderr)
            sys.exit(res.returncode)
        # small delay to be polite
        time.sleep(0.5)


def main():
    if len(sys.argv) != 2:
        print('Usage: python download_spotify.py <config.yaml>', file=sys.stderr)
        sys.exit(1)
    config_path = sys.argv[1]
    config = load_config(config_path)
    prefix = config.get('file_prefix')
    spotify_url = config.get('spotify_url')
    if not prefix or not spotify_url:
        print('Error: config must include file_prefix and spotify_url', file=sys.stderr)
        sys.exit(1)
    # Spotify client
    sp = get_spotify_client()
    # Fetch all track IDs
    track_ids = fetch_track_ids(sp, spotify_url)
    print(f"Found {len(track_ids)} track(s) in Spotify collection.")
    # Resolve YouTube URLs
    youtube_urls = []
    for tid in track_ids:
        url = resolve_youtube_url(tid)
        youtube_urls.append(url)
        # avoid hitting rate limits
        time.sleep(0.2)
    # Download audio
    download_audio(prefix, youtube_urls)
    print('All tracks downloaded successfully.')


if __name__ == '__main__':
    main()