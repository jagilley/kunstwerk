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

