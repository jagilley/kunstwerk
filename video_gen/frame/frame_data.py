import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple
from moviepy.editor import AudioFileClip
import numpy as np


class FrameStore:
    """Dict-like store of pre-rendered RGB frames, kept zlib-compressed.

    Frames are almost entirely black with some text, so raw zlib (level 3) shrinks
    a 4K frame from ~25 MB to ~0.2-0.6 MB in ~40 ms, and decodes in ~10 ms — about
    5x faster to encode and 15x faster to decode than PNG. A small LRU cache of
    decoded frames (keyed on line index) means consecutive video timestamps, which
    nearly always show the same line, pay the decode only when the line changes.

    `store[idx] = frame` compresses; `store[idx]` / `store.get(idx, default)` decode.
    Decoded frames are read-only views over the decompressed buffer; copy before
    mutating.
    """

    def __init__(self, cache_size: int = 4, level: int = 3):
        self._blobs: Dict[int, bytes] = {}
        self._shapes: Dict[int, Tuple[int, ...]] = {}
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size
        self._level = level

    def __setitem__(self, idx: int, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        self._blobs[idx] = zlib.compress(frame.tobytes(), self._level)
        self._shapes[idx] = frame.shape
        self._cache.pop(idx, None)

    def __getitem__(self, idx: int) -> np.ndarray:
        cached = self._cache.get(idx)
        if cached is not None:
            self._cache.move_to_end(idx)
            return cached
        blob = self._blobs[idx]  # KeyError like a dict
        frame = np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(self._shapes[idx])
        self._cache[idx] = frame
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frame

    def get(self, idx: Optional[int], default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if idx is None or idx not in self._blobs:
            return default
        return self[idx]

    def __contains__(self, idx: object) -> bool:
        return idx in self._blobs

    def __len__(self) -> int:
        return len(self._blobs)

    def __iter__(self) -> Iterator[int]:
        return iter(self._blobs)

    def keys(self):
        return self._blobs.keys()

    @property
    def nbytes(self) -> int:
        """Total compressed size in bytes."""
        return sum(len(b) for b in self._blobs.values())


@dataclass
class FrameData:
    """Container for all frame-related data needed for video creation."""
    time_to_line_idx: Dict[float, Optional[int]]
    line_pair_clips: FrameStore  # line idx (-1 = title card) -> RGB frame, decoded on access
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
