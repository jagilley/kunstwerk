from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TranscriptionWord:
    start: float
    end: float
    word: str
    confidence: Optional[float] = None


@dataclass
class TranscriptionVerbose:
    text: str
    language: str
    duration: float
    words: Optional[List[TranscriptionWord]] = None