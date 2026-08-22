import time as _time
from typing import Optional
import numpy as np
from moviepy.editor import VideoClip, ColorClip, concatenate_audioclips
from ..config.video_config import VideoConfig
from ..frame.frame_data import FrameData

def create_parallel_text_video(
    frame_data: FrameData,
    output_filename: str = "parallel_text.mp4",
    config: VideoConfig = VideoConfig(),
    max_duration: Optional[float] = None,
):
    """Creates the final video from pre-computed frame data.

    `max_duration` (seconds) truncates the written video — a dev convenience so a
    render can be checked without waiting for the whole opera.
    """
    duration = frame_data.total_duration
    if max_duration is not None:
        duration = min(duration, float(max_duration))

    background = ColorClip(size=(config.video_width, config.video_height), color=(0, 0, 0))
    background = background.set_duration(duration)
    background_frame = background.get_frame(0)

    combined_audio = concatenate_audioclips(frame_data.audio_clips)
    if duration < combined_audio.duration:
        combined_audio = combined_audio.subclip(0, duration)

    # Look frames up by frame number rather than by exact float timestamp: the
    # keys of time_to_line_idx are k / fps, so round(t * fps) recovers k.
    fps = config.fps
    n_frames = int(round(frame_data.total_duration * fps)) + 1
    line_idx_by_frame: list = [None] * n_frames
    for t, idx in frame_data.time_to_line_idx.items():
        k = int(round(t * fps))
        if 0 <= k < n_frames:
            line_idx_by_frame[k] = idx

    frame_store = frame_data.line_pair_clips

    def make_frame(t: float):
        k = int(round(t * fps))
        idx = line_idx_by_frame[k] if 0 <= k < n_frames else None
        return frame_store.get(idx, background_frame)

    video = VideoClip(make_frame, duration=duration).set_fps(fps)
    final_video = video.set_audio(combined_audio)

    t0 = _time.time()
    final_video.write_videofile(
        output_filename,
        fps=fps,
        codec='libx264',
        audio_codec='aac'
    )
    print(f"Wrote {output_filename} ({duration:.0f}s of video) in {_time.time() - t0:.0f}s")

    video.close()
    final_video.close()
    combined_audio.close()
