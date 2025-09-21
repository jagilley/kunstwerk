#!/usr/bin/env python3
import yaml
import sys

def get_config_values(yaml_path):
    """Extract file_prefix, playlist_url, and spotify_url from YAML config."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    # spotify_url is optional; playlist_url may be blank when using Spotify flow
    return (
        config.get('file_prefix', ''),
        config.get('playlist_url', ''),
        config.get('spotify_url', '')
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_yaml.py <config.yaml>")
        sys.exit(1)
    file_prefix, playlist_url, spotify_url = get_config_values(sys.argv[1])
    # Output file_prefix, playlist_url, and spotify_url for downstream scripts
    print(f"{file_prefix} {playlist_url} {spotify_url}")
