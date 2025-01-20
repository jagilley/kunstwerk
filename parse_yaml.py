#!/usr/bin/env python3
import yaml
import sys

def get_config_values(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['file_prefix'], config['playlist_url']

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_yaml.py <config.yaml>")
        sys.exit(1)
    file_prefix, playlist_url = get_config_values(sys.argv[1])
    print(f"{file_prefix} {playlist_url}")
