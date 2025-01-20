import re
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

def parse_opera_config(md_path: str) -> OperaConfig:
    """Parse opera configuration from markdown file."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract basic metadata
    title = re.search(r'^# (.+)$', content, re.MULTILINE).group(1)
    file_prefix = re.search(r'^File Prefix: (.+)$', content, re.MULTILINE).group(1)
    language = re.search(r'^Language: (.+)$', content, re.MULTILINE).group(1)
    
    # Extract numeric values
    start_idx = int(re.search(r'^Start Index: (\d+)$', content, re.MULTILINE).group(1))
    end_idx = int(re.search(r'^End Index: (\d+)$', content, re.MULTILINE).group(1))
    
    # Extract overture indices
    overture_str = re.search(r'^Overture Indices: \[([\d,\s]+)\]$', content, re.MULTILINE).group(1)
    overture_indices = [int(x.strip()) for x in overture_str.split(',') if x.strip()]
    
    # Extract character names
    char_section = re.search(r'## Characters\n((?:.+\n)+)', content, re.MULTILINE).group(1)
    character_names = [line.strip('- \n') for line in char_section.split('\n') if line.strip('- \n')]
    
    # Extract optional parameters
    secondary_color_match = re.search(r'^Secondary Color: (.+)$', content, re.MULTILINE)
    secondary_color = secondary_color_match.group(1) if secondary_color_match else "Silver"
    
    video_width_match = re.search(r'^Video Width: (\d+)$', content, re.MULTILINE)
    video_width = int(video_width_match.group(1)) if video_width_match else 3840
    
    video_height_match = re.search(r'^Video Height: (\d+)$', content, re.MULTILINE)
    video_height = int(video_height_match.group(1)) if video_height_match else 2160
    
    font_size_match = re.search(r'^Font Size: (\d+)$', content, re.MULTILINE)
    font_size = int(font_size_match.group(1)) if font_size_match else 96
    
    res_divisor_match = re.search(r'^Resolution Divisor: (\d+)$', content, re.MULTILINE)
    res_divisor = int(res_divisor_match.group(1)) if res_divisor_match else 1

    return OperaConfig(
        title=title,
        file_prefix=file_prefix,
        language=language,
        start_idx=start_idx,
        end_idx=end_idx,
        overture_indices=overture_indices,
        character_names=character_names,
        secondary_color=secondary_color,
        video_width=video_width,
        video_height=video_height,
        font_size=font_size,
        res_divisor=res_divisor
    )
