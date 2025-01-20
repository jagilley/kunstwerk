from dataclasses import dataclass
from typing import Dict, List, Optional
from moviepy.editor import AudioFileClip
import numpy as np

@dataclass
class FrameData:
    """Container for all frame-related data needed for video creation."""
    time_to_line_idx: Dict[float, Optional[int]]
    line_pair_clips: Dict[int, np.ndarray]
    audio_clips: List[AudioFileClip]
    total_duration: float
    frame_order: List[int]

def enforce_monotonicity(frame_data: FrameData) -> FrameData:
    """
    Enforces monotonicity in frame display order by replacing backwards-going frames
    with the most recently displayed valid frame.
    """
    frame_positions = {frame_idx: pos for pos, frame_idx in enumerate(frame_data.frame_order)}
    
    last_valid_idx = None
    last_valid_position = -1
    monotonic_mapping = {}
    
    for time in sorted(frame_data.time_to_line_idx.keys()):
        current_idx = frame_data.time_to_line_idx[time]
        
        if current_idx is None:
            monotonic_mapping[time] = None
            continue
            
        current_position = frame_positions.get(current_idx, -1)
        
        if last_valid_idx is None or current_position >= last_valid_position:
            monotonic_mapping[time] = current_idx
            last_valid_idx = current_idx
            last_valid_position = current_position
        else:
            monotonic_mapping[time] = last_valid_idx
    
    return FrameData(
        time_to_line_idx=monotonic_mapping,
        line_pair_clips=frame_data.line_pair_clips,
        audio_clips=frame_data.audio_clips,
        total_duration=frame_data.total_duration,
        frame_order=frame_data.frame_order
    )
