from openai.types.audio import TranscriptionVerbose, TranscriptionWord
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from Levenshtein import distance
import json
from tqdm import tqdm

@dataclass
class AlignedWord:
    word: str
    start: Optional[float]
    end: Optional[float]

def word_similarity(word1: str, word2: str) -> float:
    """
    Calculate similarity score between two words using Levenshtein distance.
    Returns score between 0 and 1, where 1 is exact match.
    """
    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 1.0
    return 1 - (distance(word1.lower(), word2.lower()) / max_len)

def deserialize_transcription_from_file(file_path: str) -> TranscriptionVerbose:
    """
    Deserialize a transcription from a JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    if 'duration' in data.keys():
        data['duration'] = str(data['duration'])
    
    return TranscriptionVerbose.model_validate(data)

def convert_file_times_to_absolute_times(transcriptions: List[TranscriptionVerbose]) -> List[TranscriptionVerbose]:
    """
    Convert the start and end times of each word in each transcription to absolute times (seconds since the start of the first transcription).
    """
    time_elapsed_so_far = 0

    for i, transcription in enumerate(transcriptions):
        # Update the start and end times of each word
        for word in transcription.words:
            word.start += time_elapsed_so_far
            word.end += time_elapsed_so_far

        # Update the time elapsed so far
        time_elapsed_so_far += float(transcription.duration)

    return transcriptions

def align_transcription_with_libretto(
    transcription: List[TranscriptionWord],
    libretto: List[str],
    ground_truth_timestamps: Optional[Dict[int, float]] = None,  # libretto index -> known timestamp
    ground_truth_duration: float = 1.0,
    min_similarity: float = 0.5,
    gap_penalty: float = -0.5,
    timestamp_bonus: float = 2.0,
    chunk_rows: int = 512,
) -> List[AlignedWord]:
    """Needleman–Wunsch-style alignment of transcript words to libretto words.

    Scores: Levenshtein similarity for a match (plus `timestamp_bonus` when the
    libretto word has a ground-truth timestamp within 1 s of the transcript
    word), `gap_penalty` for skipping a word on either side. Libretto words get
    the timing of their matched transcript word when the similarity is at least
    `min_similarity`, otherwise None; ground-truth words get their known time.

    Vectorised: similarities come from rapidfuzz in row chunks, each DP row is a
    numpy prefix-max, and only an int8 backtrack matrix is kept — so a full
    opera (30k x 10k words) takes well under a minute and < 1 GB instead of the
    pure-Python version's tens of minutes and ~10 GB.
    """
    import numpy as np
    from rapidfuzz.distance import Levenshtein as _Lev
    from rapidfuzz.process import cdist

    ground_truth_timestamps = ground_truth_timestamps or {}
    m, n = len(transcription), len(libretto)
    trans_words = [w.word.lower() for w in transcription]
    trans_starts = np.array([w.start for w in transcription], dtype=np.float64)
    lib_words = [w.lower() for w in libretto]

    # Per-column ground-truth data (column j of the DP = libretto word j-1)
    gt_cols = np.array(sorted(ground_truth_timestamps.keys()), dtype=np.int64)
    gt_times = np.array([ground_truth_timestamps[int(j)] for j in gt_cols], dtype=np.float64)

    MATCH, DELETE, INSERT = 0, 1, 2
    backtrack = np.full((m + 1, n + 1), INSERT, dtype=np.int8)  # row 0 / col 0 are never read as 'match'
    backtrack[1:, 0] = DELETE
    j_idx = np.arange(n + 1, dtype=np.float64)
    prev = j_idx * gap_penalty  # score row 0
    gap_j = j_idx * gap_penalty  # for the prefix-max trick

    for i0 in range(1, m + 1, chunk_rows):
        i1 = min(i0 + chunk_rows, m + 1)
        # similarity[r, j-1] for transcript words i0-1 .. i1-2 (rows) vs all libretto words
        sim = cdist(trans_words[i0 - 1:i1 - 1], lib_words, scorer=_Lev.normalized_similarity,
                    dtype=np.float64, workers=-1)
        if len(gt_cols):
            # bonus where the transcript word is within 1 s of a ground-truth libretto word
            close = np.abs(trans_starts[i0 - 1:i1 - 1, None] - gt_times[None, :]) < 1.0
            sim[:, gt_cols] += timestamp_bonus * close
        for r, i in enumerate(range(i0, i1)):
            cur = np.empty(n + 1, dtype=np.float64)
            cur[0] = i * gap_penalty
            match = prev[:-1] + sim[r]          # diag
            delete = prev[1:] + gap_penalty      # up
            tmp = np.maximum(match, delete)
            # insert (left) dependency: cur[j] = max(tmp[j], cur[j-1] + gap) == gap*j + cummax(tmp - gap*j),
            # seeded with cur[0]
            shifted = np.concatenate(([cur[0]], tmp - gap_j[1:]))
            cur[1:] = (np.maximum.accumulate(shifted)[1:] + gap_j[1:])
            # tie-breaking as in the scalar version (match >= delete, insert only if strictly
            # better) — up to floating-point rounding of the prefix-max, which flips a handful
            # of exact ties per opera (~0.1% of words, never the coverage)
            left = cur[:-1] + gap_penalty
            row_bt = np.where(match >= delete, MATCH, DELETE).astype(np.int8)
            row_bt[left > tmp] = INSERT
            backtrack[i, 1:] = row_bt
            prev = cur

    # Backtrack
    aligned: List[AlignedWord] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and backtrack[i, j] == MATCH:
            jj = j - 1
            if jj in ground_truth_timestamps:
                t = ground_truth_timestamps[jj]
                aligned.append(AlignedWord(libretto[jj], t, t + 1.0))
            elif word_similarity(transcription[i - 1].word, libretto[jj]) >= min_similarity:
                aligned.append(AlignedWord(libretto[jj], transcription[i - 1].start, transcription[i - 1].end))
            else:
                aligned.append(AlignedWord(libretto[jj], None, None))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or backtrack[i, j] == DELETE):
            i -= 1
        else:
            jj = j - 1
            t = ground_truth_timestamps.get(jj)
            if t is not None:
                aligned.append(AlignedWord(libretto[jj], t, t + ground_truth_duration))
            else:
                aligned.append(AlignedWord(libretto[jj], None, None))
            j -= 1
    return list(reversed(aligned))


def align_texts(transcription: List[TranscriptionWord], libretto: str) -> List[AlignedWord]:
    """Align transcription with libretto and transfer timestamps."""
    # Split libretto into words
    libretto_words = libretto.split()
    
    # Create alignment matrix
    n, m = len(transcription), len(libretto_words)
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    
    # Fill dynamic programming matrix
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                # Use 1 - similarity because we want distance, not similarity
                cost = 1 - word_similarity(
                    transcription[i-1].word,
                    libretto_words[j-1]
                )
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # skip transcription word
                    dp[i][j-1] + 1,    # skip libretto word
                    dp[i-1][j-1] + cost # match words
                )
    
    # Backtrack to find alignment
    aligned_words = []
    i, j = n, m
    while i > 0 and j > 0:
        cost = word_similarity(
            transcription[i-1].word,
            libretto_words[j-1]
        )
        if dp[i][j] == dp[i-1][j-1] + cost:
            # Words are aligned
            aligned_words.append(AlignedWord(
                word=libretto_words[j-1],
                start=transcription[i-1].start,
                end=transcription[i-1].end
            ))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            # Skip transcription word
            i -= 1
        else:
            # Skip libretto word
            aligned_words.append(AlignedWord(
                word=libretto_words[j-1],
                start=None,
                end=None
            ))
            j -= 1
    
    # Add any remaining libretto words
    while j > 0:
        aligned_words.append(AlignedWord(
            word=libretto_words[j-1],
            start=None,
            end=None
        ))
        j -= 1
    
    return list(reversed(aligned_words))

if __name__=="__main__":
    # Load transcriptions
    transcriptions: List[TranscriptionVerbose] = []
    for i in range(1, 29):
        i_string = str(i).zfill(3)
        transcription = deserialize_transcription_from_file(f'transcribed/{i_string}.json')
        transcriptions.append(transcription)

    transcriptions = convert_file_times_to_absolute_times(transcriptions)

    all_words: List[TranscriptionWord] = [word for transcription in transcriptions for word in transcription.words]

    # Load libretto
    with open('libretti/rheingold_de.txt', 'r') as f:
        libretto = f.read()

    # Align texts
    aligned_words = align_texts(all_words, libretto)

