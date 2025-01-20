from dataclasses import dataclass

@dataclass
class VideoConfig:
    """Configuration for video generation."""
    font_name: str = "Arial"
    font_size: int = 40
    text_1_color: str = "AliceBlue"
    text_2_color: str = "LightGoldenrod"
    video_width: int = 1920
    video_height: int = 1080
    fps: int = 24
    text_timeout: float = 5.0
