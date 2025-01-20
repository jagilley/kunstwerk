from moviepy.editor import VideoClip, ColorClip, concatenate_audioclips
from ..config.video_config import VideoConfig
from ..frame.frame_data import FrameData

def create_parallel_text_video(
    frame_data: FrameData,
    output_filename: str = "parallel_text.mp4",
    config: VideoConfig = VideoConfig()
):
    """Creates the final video from pre-computed frame data"""
    background = ColorClip(size=(config.video_width, config.video_height), color=(0, 0, 0))
    background = background.set_duration(frame_data.total_duration)

    combined_audio = concatenate_audioclips(frame_data.audio_clips)

    def make_frame(t: float):
        return frame_data.line_pair_clips.get(
            frame_data.time_to_line_idx.get(t), 
            background.get_frame(0)
        )

    video = VideoClip(make_frame, duration=frame_data.total_duration).set_fps(config.fps)
    final_video = video.set_audio(combined_audio)
    
    final_video.write_videofile(
        output_filename,
        fps=config.fps,
        codec='libx264',
        audio_codec='aac'
    )
    
    video.close()
    final_video.close()
    combined_audio.close()
