from typing import List, Tuple, Dict
from moviepy.editor import AudioFileClip, ColorClip, CompositeVideoClip
import numpy as np
from tqdm import tqdm
import imageio
from ..config.video_config import VideoConfig
from ..text.formatting import create_formatted_text
from .frame_data import FrameData

def create_title_clip(config: VideoConfig, title: str) -> np.ndarray:
    """Creates a title frame for the video."""
    background = ColorClip(size=(config.video_width, config.video_height), color=(0, 0, 0))
    
    title_text = TextClip(
        title,
        font=f"{config.font_name}-Bold",
        fontsize=config.font_size + 20,
        color=config.secondary_color,
        size=(config.video_width // 2 - 80, None),
        method='caption',
        align='center'
    )
    
    composed = CompositeVideoClip([
        background,
        title_text.set_position((40, config.video_height // 2 - title_text.h // 2)),
        title_text.set_position((config.video_width//2 + 40, config.video_height // 2 - title_text.h // 2))
    ])
    
    frame = composed.get_frame(0)
    
    background.close()
    title_text.close()
    composed.close()
    
    return frame

def create_frames(
    aligned_words: List[AlignedWord],
    line_pairs: List[Tuple[str, str]],
    character_names: List[str],
    audio_files: List[str],
    title: str,
    config: VideoConfig = VideoConfig()
) -> FrameData:
    """Creates all frame data needed for video creation"""
    
    # Audio handling
    audio_clips = [AudioFileClip(f) for f in audio_files]
    total_duration = sum(clip.duration for clip in audio_clips)
    
    # Create background
    background = ColorClip(size=(config.video_width, config.video_height), color=(0, 0, 0))
    background = background.set_duration(total_duration)
    
    # Create word-to-line mapping
    word_to_line_idx = {}
    current_word_idx = 0
    for line_idx, (sung_line, _) in enumerate(line_pairs):
        words_in_line = sung_line.split()
        for _ in words_in_line:
            if current_word_idx < len(aligned_words):
                word_to_line_idx[current_word_idx] = line_idx
                current_word_idx += 1

    # Pre-compute text clips
    line_pair_clips = {}
    frame_order = []  # Track the order of frames
    column_width = (config.video_width // 2) - 80
    vertical_margin = 80
    max_text_height = config.video_height - (2 * vertical_margin)

    print("Pre-computing text clips...")
    for idx, (sung_text, translated_text) in tqdm(enumerate(line_pairs), total=len(line_pairs)):
        left_text = create_formatted_text(
            sung_text, config.text_1_color, column_width, max_text_height, 
            config, character_names
        )
        right_text = create_formatted_text(
            translated_text, config.text_2_color, column_width, max_text_height,
            config, character_names
        )
        
        if left_text is None or right_text is None:
            continue
            
        max_height = max(left_text.h, right_text.h)
        y_position = vertical_margin + (max_text_height - max_height) // 2
        
        composed = CompositeVideoClip([
            background,
            left_text.set_position((40, y_position)),
            right_text.set_position((config.video_width//2 + 40, y_position))
        ])
        
        line_pair_clips[idx] = composed.get_frame(0)
        frame_order.append(idx)  # Add the index to frame_order
        
        left_text.close()
        right_text.close()
        composed.close()

    line_pair_clips[-1] = create_title_clip(config, title)
    frame_order.insert(0, -1)  # Add title frame at the beginning

    def get_active_line_idx(time: float) -> Optional[int]:
        if time < 10:
            return -1
        
        for word_idx, word in enumerate(aligned_words):
            if word.start is not None and word.end is not None:
                if word.start <= time <= word.end:
                    return word_to_line_idx.get(word_idx)
        
        last_active_time = -float('inf')
        last_active_idx = None
        
        for word_idx, word in enumerate(aligned_words):
            if word.end is not None and word.end <= time:
                if word.end > last_active_time:
                    last_active_time = word.end
                    last_active_idx = word_idx
        
        if last_active_idx is not None and time - last_active_time <= config.text_timeout:
            return word_to_line_idx.get(last_active_idx)
        
        return None
    
    time_to_line_idx = {}
    print("Computing frame timings...")
    for t in tqdm(np.arange(0, total_duration, 1/config.fps), total=int(total_duration*config.fps)):
        time_to_line_idx[t] = get_active_line_idx(t)

    return FrameData(
        time_to_line_idx,
        line_pair_clips,
        audio_clips,
        total_duration,
        frame_order
    )
