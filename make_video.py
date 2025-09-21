
import os
from typing import List, Dict, Optional, Tuple
import os
import copy
from openai.types.audio import TranscriptionVerbose, TranscriptionWord
from align import AlignedWord, deserialize_transcription_from_file, convert_file_times_to_absolute_times, word_similarity
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from align import AlignedWord, deserialize_transcription_from_file, convert_file_times_to_absolute_times
from config_parser import parse_opera_config
from video_gen.config.video_config import VideoConfig
from video_gen.frame.generator import create_frames
from video_gen.video.creator import create_parallel_text_video
import sys

if len(sys.argv) != 2:
    print("Usage: python make_video.py <config.md>")
    sys.exit(1)

show_plots = True

config = parse_opera_config(sys.argv[1])

# Ensure all case variations of character names are included
CHARACTER_NAMES = [
    *config.character_names,
    *[name.lower() for name in config.character_names],
    *[name.upper() for name in config.character_names]
]

def pair_libretto_lines_simple(source_text, target_text):
    """Pair corresponding lines from source and target texts."""
    lines_source = [line for line in source_text.split("\n\n") if line.strip()]
    lines_target = [line for line in target_text.split("\n\n") if line.strip()]

    if len(lines_source) != len(lines_target):
        raise ValueError(
            f"Number of lines doesn't match: {len(lines_source)} source lines vs {len(lines_target)} target lines"
        )

    return list(zip(lines_source, lines_target))


with open(f"libretti/{config.file_prefix}_{config.language}.txt", "r", encoding="utf-8") as f:
    libretto_de = f.read()

### Align libretto with transcription

def align_transcription_with_libretto(
    transcription: List[TranscriptionWord],
    libretto: List[str],
    ground_truth_timestamps: Dict[int, float] = None,  # Maps libretto index to timestamp
    ground_truth_duration: float = 1.0,
    min_similarity: float = 0.5
) -> List[AlignedWord]:
    """
    Align transcription with libretto using dynamic programming.
    Returns list of aligned words with timing information where available.
    Assumes ground truth words are 1 second in duration.
    
    Args:
        transcription: List of TranscriptionWord objects
        libretto: List of ground truth words
        ground_truth_timestamps: Dictionary mapping libretto indices to known timestamps
        min_similarity: Minimum similarity score to consider words as matching
    """
    
    ground_truth_timestamps = ground_truth_timestamps or {}
    
    # Initialize scoring matrix
    m, n = len(transcription), len(libretto)
    score_matrix = [[0.0] * (n + 1) for _ in range(m + 1)]
    backtrack = [[None] * (n + 1) for _ in range(m + 1)]
    
    gap_penalty = -0.5
    timestamp_bonus = 2.0  # Bonus score for matching known timestamps
    
    # Fill scoring matrix
    for i in range(m + 1):
        score_matrix[i][0] = i * gap_penalty
    for j in range(n + 1):
        score_matrix[0][j] = j * gap_penalty
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            similarity = word_similarity(transcription[i-1].word, libretto[j-1])
            
            # Add bonus if this alignment matches a known timestamp
            if j-1 in ground_truth_timestamps:
                known_time = ground_truth_timestamps[j-1]
                trans_time = transcription[i-1].start
                # If transcription time is close to known time, add bonus
                if abs(known_time - trans_time) < 1.0:  # Within 1 second
                    similarity += timestamp_bonus
            
            match_score = score_matrix[i-1][j-1] + similarity
            delete_score = score_matrix[i-1][j] + gap_penalty
            insert_score = score_matrix[i][j-1] + gap_penalty
            
            best_score = max(match_score, delete_score, insert_score)
            score_matrix[i][j] = best_score
            
            if best_score == match_score:
                backtrack[i][j] = 'match'
            elif best_score == delete_score:
                backtrack[i][j] = 'delete'
            else:
                backtrack[i][j] = 'insert'
    
    # Backtrack to build alignment
    aligned_words: List[AlignedWord] = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and backtrack[i][j] == 'match':
            similarity = word_similarity(transcription[i-1].word, libretto[j-1])
            
            # If we have a ground truth timestamp for this word
            if j-1 in ground_truth_timestamps:
                start_time = ground_truth_timestamps[j-1]
                aligned_words.append(AlignedWord(
                    word=libretto[j-1],
                    start=start_time,
                    end=start_time + 1.0  # Assume 1 second duration
                ))
            elif similarity >= min_similarity:
                # Regular good match - use transcription timing
                aligned_words.append(AlignedWord(
                    word=libretto[j-1],
                    start=transcription[i-1].start,
                    end=transcription[i-1].end
                ))
            else:
                # Poor match - include word without timing
                aligned_words.append(AlignedWord(
                    word=libretto[j-1],
                    start=None,
                    end=None
                ))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or backtrack[i][j] == 'delete'):
            i -= 1
        else:
            # For inserted words, if we have a ground truth timestamp, use it
            start_time = ground_truth_timestamps.get(j-1)
            if start_time is not None:
                aligned_words.append(AlignedWord(
                    word=libretto[j-1],
                    start=start_time,
                    end=start_time + ground_truth_duration
                ))
            else:
                aligned_words.append(AlignedWord(
                    word=libretto[j-1],
                    start=None,
                    end=None
                ))
            j -= 1
    
    return list(reversed(aligned_words))

def enforce_monotonicity(aligned_words: List[AlignedWord]) -> List[AlignedWord]:
    """
    Enforce temporal monotonicity in aligned words by adjusting timestamps.
    """
    result = copy.deepcopy(aligned_words)
    last_end_time = float('-inf')
    
    for word in result:
        if word.start is not None:
            if word.start < last_end_time:
                # Adjust timing to maintain monotonicity
                duration = word.end - word.start
                word.start = last_end_time
                word.end = last_end_time + duration
            last_end_time = word.end
    
    return result

def interpolate_word_timings(
    aligned_words: List[AlignedWord],
    max_interpolation_window: float = 8.0
) -> List[AlignedWord]:
    """
    Interpolate timing for words between known timestamps within a maximum window.
    
    Args:
        aligned_words: List of AlignedWord objects
        max_interpolation_window: Maximum time window in seconds for interpolation
        
    Returns:
        New list of AlignedWord objects with interpolated timings
    """
    result = copy.deepcopy(aligned_words)
    
    # First pass: identify sequences of words to interpolate
    sequences = []
    current_sequence = []
    last_known_end = None
    
    for i, word in enumerate(result):
        if word.start is not None and word.end is not None:
            # Found a word with known timing
            if current_sequence and last_known_end is not None:
                # Check if this word is within the interpolation window
                if word.start - last_known_end <= max_interpolation_window:
                    # Add the current word as the end anchor of the sequence
                    current_sequence.append(i)
                    sequences.append(current_sequence)
                current_sequence = []
            last_known_end = word.end
            current_sequence = [i]  # Start new sequence with this word
        elif current_sequence:
            # Add word without timing to current sequence
            current_sequence.append(i)
    
    # Second pass: perform interpolation for each valid sequence
    for sequence in sequences:
        if len(sequence) < 2:
            continue
        
        start_idx = sequence[0]
        end_idx = sequence[-1]
        start_word = result[start_idx]
        end_word = result[end_idx]
        
        # Skip if either anchor point doesn't have timing
        if (start_word.start is None or start_word.end is None or 
            end_word.start is None or end_word.end is None):
            continue
        
        # Calculate time distribution
        total_words = len(sequence)
        if total_words <= 1:
            continue
            
        # For the first word in sequence, keep its original end time
        # For the last word in sequence, keep its original start time
        total_time = end_word.start - start_word.end
        words_to_interpolate = total_words - 1  # excluding first word
        
        if words_to_interpolate <= 0:
            continue
            
        # Calculate time per word
        time_per_word = total_time / words_to_interpolate
        
        # Set timings for words in between
        current_time = start_word.end
        for i in range(1, len(sequence)):
            idx = sequence[i]
            word = result[idx]
            
            if i == len(sequence) - 1:
                # Last word in sequence - keep its original timing
                word.start = end_word.start
                word.end = end_word.end
            else:
                # Interpolated word
                word.start = current_time
                word.end = current_time + time_per_word
                current_time += time_per_word
    
    return result

def parse_timestamp_and_phrase(
    timestamp_str: str,
    phrase: str,
    libretto: List[str]
) -> Dict[int, float]:
    """
    Convert a human-readable timestamp and phrase into ground truth timestamp dict.
    
    Args:
        timestamp_str: Timestamp in format "H:M:S" or "M:S" or "S"
        phrase: Text phrase to locate in libretto
        libretto: List of ground truth words
    
    Returns:
        Dictionary mapping libretto index to timestamp in seconds
    
    Raises:
        ValueError: If phrase not found or found multiple times, or invalid timestamp
    """
    # Parse timestamp to seconds
    def parse_timestamp(ts: str) -> float:
        parts = ts.split(':')
        if len(parts) == 3:  # H:M:S
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:  # M:S
            m, s = map(float, parts)
            return m * 60 + s
        elif len(parts) == 1:  # S
            return float(parts[0])
        else:
            raise ValueError(f"Invalid timestamp format: {ts}")
    
    # Convert phrase to list of words and clean
    phrase_words = phrase.lower().split()
    
    # Find all occurrences of the phrase in libretto
    matches = []
    for i in range(len(libretto) - len(phrase_words) + 1):
        window = libretto[i:i + len(phrase_words)]
        if [w.lower() for w in window] == phrase_words:
            matches.append(i)
    
    # Verify unique match
    if len(matches) == 0:
        raise ValueError(f"Phrase '{phrase}' not found in libretto")
    if len(matches) > 1:
        raise ValueError(
            f"Phrase '{phrase}' found multiple times in libretto at indices {matches}"
        )
    
    # Convert timestamp to seconds
    start_time = parse_timestamp(timestamp_str)
    
    # Return dict mapping the starting index to the timestamp
    return {matches[0]: start_time}


# Load transcriptions
transcriptions: List[TranscriptionVerbose] = []
for i in range(config.start_idx, config.end_idx):
    i_string = str(i).zfill(2)
    transcription = deserialize_transcription_from_file(f'transcribed/{config.file_prefix}_transcribed/{i_string}.json')
    transcriptions.append(transcription)

for idx in config.overture_indices:
    zero_idx = 0 if idx == 0 else idx - 1
    transcriptions[zero_idx].words = []
    transcriptions[zero_idx].text = ""
    transcriptions[zero_idx].segments = []

transcriptions = convert_file_times_to_absolute_times(transcriptions)

all_words: List[TranscriptionWord] = [word for transcription in transcriptions for word in transcription.words]

# Load libretto
with open(f'libretti/{config.file_prefix}_{config.language}.txt', 'r') as f:
    libretto = f.read()

libretto = libretto.split()

markers = [
]

ground_truth = {}
for timestamp, phrase in markers:
    ground_truth.update(parse_timestamp_and_phrase(timestamp, phrase, libretto))

# Align texts
aligned_words = align_transcription_with_libretto(
    transcription=all_words,
    libretto=libretto,
    ground_truth_timestamps=ground_truth,
    ground_truth_duration=5,
    min_similarity=0.3
)

# enforce monotonicity
# aligned_words = enforce_monotonicity(aligned_words)

# give the percentage of AlignedWords that have a start and end time
percentage_aligned = len([word for word in aligned_words if word.start is not None and word.end is not None]) / len(aligned_words)
print(f"Percentage of aligned words: {percentage_aligned}")

import matplotlib.pyplot as plt
import numpy as np

def detect_low_alignment(smoothed: np.ndarray, overall_avg: float, threshold: float = 0.2, window: int = 500) -> List[tuple]:
    low_periods = []
    current_start = None
    
    for i, val in enumerate(smoothed):
        if val < (overall_avg - threshold):
            if current_start is None:
                current_start = i
        elif current_start is not None:
            if i - current_start >= window:
                low_periods.append((current_start, i))
            current_start = None
            
    if current_start is not None and len(smoothed) - current_start >= window:
        low_periods.append((current_start, len(smoothed)))
        
    return low_periods

def plot_aligned_words(aligned_words: List[AlignedWord]):
    aligned = [i.start is not None and i.end is not None for i in aligned_words]
    aligned_int = [1 if x else 0 for x in aligned]
    overall_avg = np.mean(aligned_int)

    window_size = 20
    smoothed = np.convolve(aligned_int, np.ones(window_size)/window_size, mode='valid')

    low_periods = detect_low_alignment(smoothed, overall_avg)

    plt.figure(figsize=(12, 6))
    plt.plot(smoothed, label='Alignment Rate (Moving Average)')
    plt.axhline(y=overall_avg, color='r', linestyle='--', label='Overall Average')
    
    for start, end in low_periods:
        plt.axvspan(start, end, color='red', alpha=0.2)
    
    plt.fill_between(range(len(smoothed)), smoothed, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.xlabel('Word Position')
    plt.ylabel('Proportion Aligned')
    plt.title('Word Alignment Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if low_periods:
        print(f"Warning: Found {len(low_periods)} periods of low alignment (>500 words below average)")
        for start, end in low_periods:
            print(f"Low alignment period: words {start} to {end}")

            # print the first 20 words of the low alignment period
            print(f"First 20 words: {[aligned_words[i].word for i in range(start, min(start+20, end))]}")
    
    plt.show()

if show_plots:
    plot_aligned_words(aligned_words)

# write aligned words to csv

def write_aligned_words_to_csv(aligned_words: List[AlignedWord], filename: str):
    # Convert list of AlignedWord to list of dictionaries
    data = [{'word': w.word, 'start': w.start, 'end': w.end} for w in aligned_words]
    
    # Create DataFrame and write to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def read_edited_aligned_words_from_csv(filename: str) -> List[AlignedWord]:
    # Read CSV into DataFrame
    df = pd.read_csv(filename)

    # Convert DataFrame to list of AlignedWord
    aligned_words = [AlignedWord(row['word'], row['start'], row['end']) for _, row in df.iterrows()]

    # replace nans with None
    for word in aligned_words:
        if pd.isna(word.start):
            word.start = None
        if pd.isna(word.end):
            word.end = None

    return aligned_words

write_aligned_words_to_csv(aligned_words, f'aligned_words_{config.file_prefix}.csv')


# read aligned words from csv
aligned_words = read_edited_aligned_words_from_csv(f'aligned_words_{config.file_prefix}.csv')


aligned_words = interpolate_word_timings(aligned_words, max_interpolation_window=20)

# give the percentage of AlignedWords that have a start and end time
percentage_aligned = len([word for word in aligned_words if word.start is not None and word.end is not None]) / len(aligned_words)
print(f"Percentage of aligned words after interpolation: {percentage_aligned}")

if show_plots:
    plot_aligned_words(aligned_words)

### Add translation

# Get translation file path from config
if not config.translation_file:
    raise ValueError("No translation file specified in config")

translation_path = f"libretti/{config.translation_file}"
if not os.path.exists(translation_path):
    raise FileNotFoundError(f"Translation file not found: {translation_path}")

with open(translation_path, "r", encoding="utf-8") as f:
    libretto_en = f.read()

pairs = pair_libretto_lines_simple(libretto_de, libretto_en)

def is_safe_split_point(lines, index):
    """Check if splitting at this line index would break any parentheses pairs"""
    text_before = '\n'.join(lines[:index])
    open_count = text_before.count('(') - text_before.count(')')
    return open_count == 0

def find_safe_split_point(lines):
    """Find the closest safe split point to the middle"""
    mid = len(lines) // 2
    
    # Try points progressively further from the middle
    for offset in range(len(lines)):
        # Try point after middle
        if mid + offset < len(lines):
            if is_safe_split_point(lines, mid + offset):
                return mid + offset
        # Try point before middle
        if mid - offset >= 1:  # Ensure we don't split at 0
            if is_safe_split_point(lines, mid - offset):
                return mid - offset
    
    # If no safe point found, return middle as fallback
    return mid

def split_long_pairs(pairs, max_length=15):
    need_another_pass = True
    while need_another_pass:
        need_another_pass = False
        i = 0
        while i < len(pairs):
            de, en = pairs[i]
            de_lines = de.split("\n")
            en_lines = en.split("\n")
            
            if len(de_lines) > max_length:
                need_another_pass = True
                print(f"Splitting pair {i}")
                
                # Find safe split point based on German text
                split_point = find_safe_split_point(de_lines)
                
                # Split both German and English at this point
                pairs[i] = (
                    "\n".join(de_lines[:split_point]),
                    "\n".join(en_lines[:split_point])
                )
                pairs.insert(i+1, (
                    "\n".join(de_lines[split_point:]),
                    "\n".join(en_lines[split_point:])
                ))
            i += 1

    return pairs

# Apply the splitting
pairs = split_long_pairs(pairs)

# print the number of pairs
print(f"Final number of pairs: {len(pairs)}")

# print the first pair
print("First pair:", pairs[0])

# print the last pair
print("Last pair:", pairs[-1])

### Create the video

from moviepy.editor import (
    AudioFileClip, TextClip, CompositeVideoClip, 
    ColorClip, concatenate_audioclips, VideoClip
)
from typing import List, Tuple, Optional, Dict
import imageio
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class FrameData:
    time_to_line_idx: Dict[float, Optional[int]]
    line_pair_clips: Dict[int, np.ndarray]
    audio_clips: List[AudioFileClip]
    total_duration: float
    frame_order: List[int]

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

# Create video configuration
video_config = VideoConfig(
    font_name="Baskerville",
    text_2_color=config.secondary_color,
    font_size=config.font_size // config.res_divisor,
    video_width=config.video_width // config.res_divisor,
    video_height=config.video_height // config.res_divisor,
    fps=4,
    text_timeout=8.0
)

# Generate frames and data
frame_data = create_frames(
    aligned_words=aligned_words,
    line_pairs=pairs,
    character_names=CHARACTER_NAMES,
    audio_files=[f"audio/{config.file_prefix}/{str(i).zfill(2)}.m4a" for i in range(config.start_idx, config.end_idx)],
    title=config.title,
    config=video_config
)

def enforce_monotonicity(frame_data: FrameData) -> FrameData:
    """
    Enforces monotonicity in frame display order by replacing backwards-going frames
    with the most recently displayed valid frame.
    """
    # Create frame position lookup for O(1) order comparison
    frame_positions = {frame_idx: pos for pos, frame_idx in enumerate(frame_data.frame_order)}
    
    last_valid_idx = None
    last_valid_position = -1
    
    # Create new time_to_line_idx mapping
    monotonic_mapping = {}
    
    for time in sorted(frame_data.time_to_line_idx.keys()):
        current_idx = frame_data.time_to_line_idx[time]
        
        if current_idx is None:
            monotonic_mapping[time] = None
            continue
            
        current_position = frame_positions.get(current_idx, -1)
        
        # If this is our first frame or it maintains/advances the order
        if last_valid_idx is None or current_position >= last_valid_position:
            monotonic_mapping[time] = current_idx
            last_valid_idx = current_idx
            last_valid_position = current_position
        else:
            # Replace with last valid frame if it would go backwards
            monotonic_mapping[time] = last_valid_idx
    
    # Create new FrameData with monotonic mapping
    return FrameData(
        time_to_line_idx=monotonic_mapping,
        line_pair_clips=frame_data.line_pair_clips,
        audio_clips=frame_data.audio_clips,
        total_duration=frame_data.total_duration,
        frame_order=frame_data.frame_order
    )

frame_data = enforce_monotonicity(frame_data)


def interpolate_frames(times_to_idxs: Dict[float, Optional[int]]) -> Dict[float, Optional[int]]:
    """
    Given a dictionary mapping timestamps to frame indices, if the same index appears twice with only None values in between, fill in the indices in between to also contain that frame index.
    
    Example:
        Input: {0.0: None, 1.0: 5, 2.0: None, 3.0: None, 4.0: 5, 5.0: None}
        Output: {0.0: None, 1.0: 5, 2.0: 5, 3.0: 5, 4.0: 5, 5.0: None}
    """
    # Convert to sorted list of (time, idx) pairs
    sorted_times = sorted(times_to_idxs.items())
    result = dict(sorted_times)
    
    # Find sequences that should be interpolated
    last_idx = None
    start_time = None
    
    for time, idx in tqdm(sorted_times, total=len(sorted_times)):
        if idx is not None:
            if last_idx is not None and idx == last_idx:
                # Found matching indices - fill in all None values between start_time and current time
                for t, current_idx in sorted_times:
                    if start_time < t < time and current_idx is None:
                        result[t] = idx
            # Reset tracking
            start_time = time
            last_idx = idx
            
    return result

frame_data.time_to_line_idx = interpolate_frames(frame_data.time_to_line_idx)

# Create the final video when ready
create_parallel_text_video(
    frame_data=frame_data,
    output_filename=f'output/{config.file_prefix}-{config.res_divisor}.mp4',
    config=video_config
)

import librosa
from datetime import datetime, timedelta

def generate_audio_timestamps(audio_files, aligned_words=None, character_names=None):
    """
    Generate timestamps for a list of audio files showing their start and end times.
    
    Args:
        audio_files (list): List of paths to audio files
        aligned_words (list, optional): List of AlignedWord objects to extract first lines
        character_names (list, optional): List of character names to filter out from first lines
        
    Returns:
        str: Formatted string with timestamps and scene descriptions
    """
    
    def format_timestamp(seconds):
        """Convert seconds to HH:MM:SS format"""
        stamp = str(timedelta(seconds=int(seconds))).zfill(8)
        return stamp
    
    def get_first_sung_line(start_time, end_time, aligned_words, character_names):
        """Find the first line of text at the given timestamp range"""
        if aligned_words is None or not aligned_words:
            return None
            
        # Find the words within this timestamp range
        words_in_range = []
        for word in aligned_words:
            if word.start is not None and start_time <= word.start < end_time:
                words_in_range.append(word.word)
                
        if not words_in_range:
            return None
            
        # Build the first line (up to 8 words)
        first_line = " ".join(words_in_range[:8])
        
        # Remove character name at the beginning if present
        if character_names:
            for name in character_names:
                if first_line.startswith(f"{name}:") or first_line.startswith(f"{name.upper()}:"):
                    first_line = first_line.split(":", 1)[1].strip()
                    break
        
        return first_line if first_line else None
    
    current_time = 0
    result = []
    
    # Process each audio file
    for i, file_path in enumerate(audio_files, 1):
        # Get duration of audio file
        duration = librosa.get_duration(path=file_path)
        
        # Calculate start and end times
        start_time = current_time
        end_time = current_time + duration
        
        # Get the first line if possible
        scene_description = f"Scene {i}"
        first_line = get_first_sung_line(start_time, end_time, aligned_words, character_names)
        
        if first_line:
            scene_description = f"{first_line}"
        
        # Format the timestamp line
        timestamp_line = f"{format_timestamp(start_time)} - {scene_description}"
        while timestamp_line[0] == "0" or timestamp_line[0] == ":":
            timestamp_line = timestamp_line[1:]
        result.append(timestamp_line)
        
        # Update current time for next file
        current_time = end_time
            
    return "\n".join(result)

print(generate_audio_timestamps(
    [f"audio/{config.file_prefix}/{str(i).zfill(2)}.m4a" for i in range(config.start_idx, config.end_idx)],
    aligned_words=aligned_words,
    character_names=CHARACTER_NAMES
))
