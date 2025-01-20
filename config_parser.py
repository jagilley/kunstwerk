from dataclasses import dataclass
from typing import List
import re

@dataclass
class OperaConfig:
    title: str
    file_prefix: str
    language: str
    secondary_language: str
    start_idx: int
    end_idx: int
    overture_indices: List[int]
    video_width: int
    video_height: int
    font_size: int
    secondary_color: str
    font: str
    fps: int
    text_timeout: float
    character_names: List[str]
    timestamp_markers: List[tuple]

def parse_md_config(filename: str) -> OperaConfig:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse sections using regex
    metadata_match = re.search(r'## Metadata\n(.*?)(?=\n##)', content, re.DOTALL)
    video_match = re.search(r'## Video Settings\n(.*?)(?=\n##)', content, re.DOTALL)
    characters_match = re.search(r'## Character Names\n(.*?)(?=\n##)', content, re.DOTALL)
    timestamps_match = re.search(r'## Timestamp Markers\n(.*?)(?=$)', content, re.DOTALL)
    
    # Parse metadata
    metadata_lines = metadata_match.group(1).strip().split('\n')
    metadata = {}
    for line in metadata_lines:
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip('- ')] = value.strip()
    
    # Parse resolution
    video_settings = {}
    video_lines = video_match.group(1).strip().split('\n')
    for line in video_lines:
        if ':' in line:
            key, value = line.split(':', 1)
            video_settings[key.strip('- ')] = value.strip()
    
    width, height = map(int, video_settings['Resolution'].split('x'))
    
    # Parse character names
    character_lines = characters_match.group(1).strip().split('\n')
    characters = [line.strip('- ') for line in character_lines if line.strip()]
    
    # Parse timestamps
    timestamp_lines = timestamps_match.group(1).strip().split('\n')
    timestamps = []
    for line in timestamp_lines:
        if '-' in line:
            time, marker = line.strip('- ').split('-', 1)
            timestamps.append((time.strip(), marker.strip()))
    
    return OperaConfig(
        title=metadata['Title'],
        file_prefix=metadata['File Prefix'],
        language=metadata['Language'],
        secondary_language=metadata['Secondary Language'],
        start_idx=int(metadata['Start Index']),
        end_idx=int(metadata['End Index']),
        overture_indices=eval(metadata['Overture Indices']),
        video_width=width,
        video_height=height,
        font_size=int(video_settings['Font Size']),
        secondary_color=video_settings['Secondary Color'],
        font=video_settings['Font'],
        fps=int(video_settings['FPS']),
        text_timeout=float(video_settings['Text Timeout']),
        character_names=characters,
        timestamp_markers=timestamps
    )
