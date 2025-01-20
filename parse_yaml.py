#!/usr/bin/env python3
import yaml
import sys

def get_file_prefix(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['file_prefix']

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_yaml.py <config.yaml>")
        sys.exit(1)
    print(get_file_prefix(sys.argv[1]))
