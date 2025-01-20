import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class OperaConfig:
    title: str
    file_prefix: str
    language: str
    start_idx: int
    end_idx: int
    overture_indices: List[int]
    character_names: List[str]
    secondary_color: str
    video_width: int = 3840
    video_height: int = 2160
    font_size: int = 96
    res_divisor: int = 1

def parse_opera_config(yaml_path: str) -> OperaConfig:
    """Parse opera configuration from YAML file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return OperaConfig(
        title=config['title'],
        file_prefix=config['file_prefix'],
        language=config['language'],
        start_idx=config['start_idx'],
        end_idx=config['end_idx'],
        overture_indices=config['overture_indices'],
        character_names=config['characters'],
        secondary_color=config.get('secondary_color', 'Silver'),
        video_width=config.get('video_width', 3840),
        video_height=config.get('video_height', 2160),
        font_size=config.get('font_size', 96),
        res_divisor=config.get('res_divisor', 1)
    )
