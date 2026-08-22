import time as _time
from typing import List, Tuple, Dict, Optional
from moviepy.editor import TextClip
from align import AlignedWord
from moviepy.editor import AudioFileClip, ColorClip, CompositeVideoClip
import numpy as np
from tqdm import tqdm
from ..config.video_config import VideoConfig
from ..text.formatting import create_formatted_text
from .frame_data import FrameData, FrameStore

def create_title_clip(config: VideoConfig, title: str) -> np.ndarray:
    """Creates a title frame for the video."""
    background = ColorClip(size=(config.video_width, config.video_height), color=(0, 0, 0))

    title_text = TextClip(
        title,
        font=f"{config.font_name}-Bold",
        fontsize=config.font_size + 20,
        color=config.text_2_color,
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


def compute_time_to_line_idx(
    aligned_words: List[AlignedWord],
    word_to_line_idx: Dict[int, int],
    times: np.ndarray,
    text_timeout: float,
    title_duration: float = 10.0,
) -> List[Optional[int]]:
    """For each timestamp in `times`, the index of the line pair to show (None for
    nothing, -1 for the title card). Vectorised equivalent of the old per-timestamp
    scan over every aligned word:

      * t < title_duration -> -1 (title card)
      * else the line of the first word (in libretto order) with start <= t <= end
      * else, if the most recently finished word (largest end <= t; ties -> first
        in libretto order) ended within `text_timeout`, that word's line
      * else None

    `word_to_line_idx` may not cover every word (trailing words past the last line
    pair); those words resolve to None, exactly as `dict.get` did.
    """
    n_times = len(times)
    NO_WORD = -1
    NO_LINE = -2  # sentinel for "word has no line" (-1 is the title card)

    line_of_word = np.full(len(aligned_words), NO_LINE, dtype=np.int64)
    for word_idx, line_idx in word_to_line_idx.items():
        line_of_word[word_idx] = line_idx

    # Words with both timings: candidates for "currently sung".
    both_idx = np.array(
        [i for i, w in enumerate(aligned_words) if w.start is not None and w.end is not None],
        dtype=np.int64,
    )
    active_word = np.full(n_times, NO_WORD, dtype=np.int64)
    if len(both_idx):
        starts = np.array([aligned_words[i].start for i in both_idx], dtype=np.float64)
        ends = np.array([aligned_words[i].end for i in both_idx], dtype=np.float64)
        # Frame range [k0, k1) with start <= times[k] <= end, using the same float
        # comparisons as the original scan.
        k0 = np.searchsorted(times, starts, side="left")
        k1 = np.searchsorted(times, ends, side="right")
        # Assign in reverse libretto order so the earliest word wins overlaps.
        for j in range(len(both_idx) - 1, -1, -1):
            if k1[j] > k0[j]:
                active_word[k0[j]:k1[j]] = both_idx[j]

    # Words with an end time: candidates for "recently finished".
    end_idx = np.array([i for i, w in enumerate(aligned_words) if w.end is not None], dtype=np.int64)
    recent_word = np.full(n_times, NO_WORD, dtype=np.int64)
    recent_ok = np.zeros(n_times, dtype=bool)
    if len(end_idx):
        ends = np.array([aligned_words[i].end for i in end_idx], dtype=np.float64)
        # Sort by end ascending; ties by descending word index so that the *last*
        # entry with a given end value is the earliest word (the original kept the
        # first word in libretto order on ties via a strict '>' comparison).
        order = np.lexsort((-end_idx, ends))
        sorted_ends = ends[order]
        sorted_words = end_idx[order]
        pos = np.searchsorted(sorted_ends, times, side="right") - 1
        has_prev = pos >= 0
        pos_clipped = np.clip(pos, 0, None)
        recent_word = np.where(has_prev, sorted_words[pos_clipped], NO_WORD)
        recent_end = sorted_ends[pos_clipped]
        recent_ok = has_prev & ((times - recent_end) <= text_timeout)

    result_line = np.full(n_times, NO_LINE, dtype=np.int64)
    use_active = active_word != NO_WORD
    result_line[use_active] = line_of_word[active_word[use_active]]
    use_recent = (~use_active) & recent_ok
    result_line[use_recent] = line_of_word[recent_word[use_recent]]
    result_line[times < title_duration] = -1

    return [None if v == NO_LINE else int(v) for v in result_line]


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

    # Pre-compute text clips. Each rendered frame is compressed into the store
    # right away so we never hold more than one or two raw frames at a time.
    line_pair_clips = FrameStore()
    frame_order = []  # Track the order of frames
    column_width = (config.video_width // 2) - 80
    vertical_margin = 80
    max_text_height = config.video_height - (2 * vertical_margin)

    print("Pre-computing text clips...")
    t0 = _time.time()
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
    print(f"Pre-computed {len(line_pair_clips)} frames in {_time.time() - t0:.0f}s "
          f"({line_pair_clips.nbytes / 1e6:.0f} MB compressed)")

    print("Computing frame timings...")
    t0 = _time.time()
    times = np.arange(0, total_duration, 1 / config.fps)
    line_idxs = compute_time_to_line_idx(aligned_words, word_to_line_idx, times, config.text_timeout)
    time_to_line_idx = dict(zip(times.tolist(), line_idxs))
    print(f"Computed {len(time_to_line_idx)} frame timings in {_time.time() - t0:.1f}s")

    return FrameData(
        time_to_line_idx,
        line_pair_clips,
        audio_clips,
        total_duration,
        frame_order
    )
