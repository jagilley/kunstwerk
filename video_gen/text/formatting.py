from typing import List, Tuple
from moviepy.editor import TextClip, CompositeVideoClip
from ..config.video_config import VideoConfig

def split_text_for_formatting(text: str, character_names: List[str]) -> List[Tuple[str, str]]:
    """Split text into segments with formatting.
    Returns list of (text, format_type) tuples.
    format_type can be 'regular', 'italic', or 'bold'"""
    segments = []
    current_pos = 0
    
    # First check if the line starts with a character name
    for name in character_names:
        if text.startswith(name + "\n"):
            name_end = len(name) + 1
            segments.append((text[:name_end], 'bold'))
            current_pos = name_end
            break
    
    while True:
        start = text.find('(', current_pos)
        if start == -1:
            if current_pos < len(text):
                segments.append((text[current_pos:], 'regular'))
            break
            
        if start > current_pos:
            segments.append((text[current_pos:start], 'regular'))
            
        end = text.find(')', start)
        if end == -1:
            segments.append((text[start:], 'regular'))
            break
            
        segments.append((text[start:end+1], 'italic'))
        current_pos = end + 1
        
    return segments

def create_formatted_text(
    text: str, 
    color: str, 
    column_width: int, 
    max_height: int, 
    config: VideoConfig,
    character_names: List[str]
) -> TextClip:
    """Creates a composite clip of regular, italic, and bold text segments with height constraints."""
    segments = split_text_for_formatting(text, character_names)
    clips = []
    y_position = 0
    current_font_size = config.font_size
    
    while True:
        clips = []
        y_position = 0
        
        for segment_text, format_type in segments:
            font = config.font_name
            if format_type == 'italic':
                font = f"{config.font_name}-Italic"
            elif format_type == 'bold':
                font = f"{config.font_name}-Bold"
                
            clip = TextClip(
                segment_text,
                font=font,
                fontsize=current_font_size,
                color=color,
                size=(column_width, None),
                method='caption',
                align='center'
            )
            clips.append(clip.set_position((0, y_position)))
            y_position += clip.h
            
        if y_position <= max_height or current_font_size <= 20:
            break
            
        current_font_size -= 2
        for clip in clips:
            clip.close()
    
    if not clips:
        return None
        
    composite = CompositeVideoClip(
        clips, 
        size=(column_width, y_position)
    )
    
    for clip in clips:
        clip.close()
        
    return composite
