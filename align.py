from openai.types.audio import TranscriptionVerbose, TranscriptionWord
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from Levenshtein import distance
import json
import re
import unicodedata
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
    row_bands: Optional["np.ndarray"] = None,
) -> List[AlignedWord]:
    """Needleman–Wunsch-style alignment of transcript words to libretto words.

    Scores: Levenshtein similarity for a match (plus `timestamp_bonus` when the
    libretto word has a ground-truth timestamp within 1 s of the transcript
    word), `gap_penalty` for skipping a word on either side. Libretto words get
    the timing of their matched transcript word when the similarity is at least
    `min_similarity`, otherwise None; ground-truth words get their known time.

    `row_bands` (optional, shape (m, 2), int) restricts transcript word i to
    matching libretto words in the inclusive index range row_bands[i] — see
    `bands_from_anchors`. Outside the band the match score is -inf, so the DP
    can only skip; gaps are always feasible, so the alignment always exists.

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
    col_idx = np.arange(n, dtype=np.int64)[None, :]
    if row_bands is not None:
        row_bands = np.asarray(row_bands, dtype=np.int64)
        assert row_bands.shape == (m, 2), f"row_bands must be (len(transcription), 2), got {row_bands.shape}"
    prev = j_idx * gap_penalty  # score row 0
    gap_j = j_idx * gap_penalty  # for the prefix-max trick

    for i0 in range(1, m + 1, chunk_rows):
        i1 = min(i0 + chunk_rows, m + 1)
        # similarity[r, j-1] for transcript words i0-1 .. i1-2 (rows) vs all libretto words
        sim = cdist(trans_words[i0 - 1:i1 - 1], lib_words, scorer=_Lev.normalized_similarity,
                    dtype=np.float64, workers=-1)
        if row_bands is not None:
            lo = row_bands[i0 - 1:i1 - 1, 0][:, None]
            hi = row_bands[i0 - 1:i1 - 1, 1][:, None]
            sim[(col_idx < lo) | (col_idx > hi)] = -np.inf
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

# ---------------------------------------------------------------------------
# Track anchors: "libretto position p is sung somewhere inside track k"
# ---------------------------------------------------------------------------
#
# Album track titles usually end in the number's incipit ("…Habanera. L'amour
# est un oiseau rebelle"), and the track boundaries are known from the audio
# durations. Locating each incipit in the libretto gives one anchor per track
# with a *band* of uncertainty (the whole track), which is turned into a hard
# column band for the DP: transcript words of track k may only match libretto
# words between track k's anchor and track k+1's anchor (plus slack). This pins
# drift — a section can no longer be matched to the wrong repeat three numbers
# away — without pretending to know the second.

@dataclass
class TrackAnchor:
    track: int            # 1-based track number
    position: int         # index into the (whitespace-split) libretto word list
    phrase: str           # the title fragment that matched
    matched: str          # the libretto words it matched
    score: float          # normalised Levenshtein similarity of the two
    source: str = "title" # "title" (from tracks.json) or "marker" (manual)


# Title fragments that are descriptions, not sung text. A candidate whose
# tokens are all in here (after dropping numerals and the opera title) is not
# looked up — "Scene & Chorus" must never anchor on a libretto "Scène et chœur"
# heading three acts away.
_GENERIC_TITLE_WORDS = {
    # structure / genre (en, fr, it, de)
    "act", "acte", "atto", "akt", "aufzug", "scene", "scène", "scena", "szene", "auftritt",
    "prelude", "prélude", "preludio", "vorspiel", "overture", "ouverture", "ouvertüre", "sinfonia",
    "intermezzo", "entracte", "entr'acte", "interlude", "introduction", "introduzione", "einleitung",
    "finale", "final", "chorus", "chœur", "choeur", "coro", "chor", "march", "marche", "marcia", "marsch",
    "duet", "duo", "duetto", "duett", "trio", "terzetto", "terzett", "quartet", "quatuor", "quartetto", "quartett",
    "quintet", "quintette", "quintetto", "quintett", "sextet", "sextuor", "sestetto", "sextett",
    "septet", "ensemble", "aria", "air", "arie", "arioso", "cavatina", "cabaletta", "recitative", "récitatif",
    "recitativo", "rezitativ", "song", "chanson", "canzone", "lied", "couplets", "romance", "romanza", "ballad",
    "ballade", "dance", "danse", "danza", "tanz", "melodrama", "mélodrame", "dialogue", "dialogo",
    "habanera", "seguidilla", "séguedille", "toreador", "toreador's", "flower", "letter", "drinking",
    "catalogue", "champagne", "serenade", "sérénade", "serenata", "humming", "love", "death", "of", "the",
    "and", "&", "a", "an", "de", "du", "des", "la", "le", "les", "il", "lo", "gli", "der", "die", "das",
    "allegro", "andante", "moderato", "adagio", "vivace", "presto", "largo", "lento", "giocoso", "maestoso",
    "tempo", "motive", "motif", "theme", "after", "before", "part", "no", "n°", "nr", "op", "wd", "k",
    "street", "boys", "cigarette", "girls", "gypsy", "bohème", "bohémienne", "card", "cards", "fate",
    "first", "second", "third", "fourth", "i", "ii", "iii", "iv", "v", "vi", "1", "2", "3", "4", "5", "6",
    "one", "two", "three", "four", "five", "six", "i.", "ii.", "iii.", "iv.",
}

_TITLE_SPLIT_RE = re.compile(r"""
    \s*:\s* | \s+[-–—]\s+ | \s*/\s* | \.\s+ | \.\.\.|… | \s*[()\[\]"“”„«»]\s* | \s*\|\s* | ;\s*
""", re.VERBOSE)


def normalise_for_matching(s: str) -> str:
    """Lowercase, strip accents, turn apostrophes/hyphens into spaces, drop
    other punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("œ", "oe").replace("æ", "ae").replace("ß", "ss")
    s = re.sub(r"[’'ʼ`´\-‐‑‒–—]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def incipit_candidates(title: str, opera_title: str = "") -> List[str]:
    """Fragments of an album track title that might be sung text, in title
    order. Splits on the usual separators (":", " - ", ". ", quotes, brackets)
    and drops fragments of fewer than two words or made only of numerals,
    genre words and the opera title ("Act I", "Scene & Chorus", "Flower Song")."""
    skip = set(_GENERIC_TITLE_WORDS)
    skip.update(normalise_for_matching(opera_title).split())
    out: List[str] = []
    for frag in _TITLE_SPLIT_RE.split(title):
        frag = (frag or "").strip(" .,;:!?¡¿")
        if not frag:
            continue
        toks = normalise_for_matching(frag).split()
        alpha = [t for t in toks if any(c.isalpha() for c in t)]
        if len(alpha) < 2:
            continue
        content = [t for t in toks if t not in skip and not re.fullmatch(r"[0-9]+|[ivxlc]+", t)]
        if not content:
            continue
        out.append(frag)
    return out


def _libretto_tokens(libretto: List[str]) -> Tuple[List[str], List[int]]:
    """Normalised libretto tokens and, for each, the index of the source word
    (punctuation-only words such as the French " !" vanish)."""
    toks, src = [], []
    for i, w in enumerate(libretto):
        t = normalise_for_matching(w)
        if t:
            for piece in t.split():
                toks.append(piece)
                src.append(i)
    return toks, src


def _phrase_hits(phrase: str, toks: List[str], min_score: float) -> List[Tuple[int, float, str]]:
    """All (token_index, score, matched_text) where `phrase` matches a window of
    the normalised libretto tokens with similarity >= min_score, after
    non-maximum suppression (one hit per phrase-length neighbourhood)."""
    import numpy as np
    from rapidfuzz.distance import Levenshtein as _Lev
    from rapidfuzz.process import cdist

    p = normalise_for_matching(phrase)
    L = len(p.split())
    if L == 0 or not toks:
        return []
    best_score = np.full(len(toks), -1.0)
    best_len = np.zeros(len(toks), dtype=np.int64)
    for w in sorted({max(1, L - 1), L, L + 1}):
        if w > len(toks):
            continue
        windows = [" ".join(toks[i:i + w]) for i in range(len(toks) - w + 1)]
        scores = cdist([p], windows, scorer=_Lev.normalized_similarity, dtype=np.float64, workers=-1)[0]
        better = scores > best_score[:len(scores)]
        best_score[:len(scores)][better] = scores[better]
        best_len[:len(scores)][better] = w
    hits = []
    order = np.argsort(-best_score, kind="stable")
    taken = np.zeros(len(toks), dtype=bool)
    for i in order:
        if best_score[i] < min_score:
            break
        if taken[max(0, i - L):i + L + 1].any():
            continue
        taken[i] = True
        hits.append((int(i), float(best_score[i]), " ".join(toks[i:i + best_len[i]])))
    return sorted(hits)

def collapse_transcript_loops(
    words: List[TranscriptionWord],
    max_repeats: int = 6,
    max_ngram: int = 30,
) -> Tuple[List[TranscriptionWord], int]:
    """Drop ASR hallucination loops: when the same 1..max_ngram-word phrase
    repeats more than `max_repeats` times back to back, keep the first
    `max_repeats` copies and drop the rest.

    ElevenLabs/Whisper sometimes lock up on long melismas or orchestral
    stretches and emit thousands of copies of "la", "Oh!" or "à chaque instant"
    with bogus timestamps (Carmen's 5-minute "Je vais danser" came back as
    5,761 words, 5,000 of them "la"). Left in, the DP happily matches every one
    of them to *something* in the libretto, so a whole number ends up timed
    inside a 30-second window. Real repeats in opera are a handful, not
    hundreds — `max_repeats` copies are enough for the DP to time them.

    Returns (kept_words, number_dropped).
    """
    # Work on content tokens only: ElevenLabs also emits whitespace/punctuation
    # "words", which would otherwise double every loop's period.
    content = [(k, normalise_for_matching(w.word)) for k, w in enumerate(words)]
    content = [(k, t) for k, t in content if t]
    norm = [t for _, t in content]
    keep = [True] * len(words)
    i = 0
    n_tok = len(norm)
    while i < n_tok:
        best_n, best_r = 0, 0
        for n in range(1, max_ngram + 1):
            if i + 2 * n > n_tok:
                break
            unit = norm[i:i + n]
            r = 1
            while i + (r + 1) * n <= n_tok and norm[i + r * n:i + (r + 1) * n] == unit:
                r += 1
            if r > max_repeats and r * n > best_r * best_n:
                best_n, best_r = n, r
        if best_n:
            for k in range(i + max_repeats * best_n, i + best_r * best_n):
                keep[content[k][0]] = False
            i += best_r * best_n
        else:
            i += 1
    kept = [w for w, k in zip(words, keep) if k]
    return kept, len(words) - len(kept)



def find_track_anchors(
    track_titles: Dict[int, str],
    libretto: List[str],
    opera_title: str = "",
    min_score: float = 0.8,
) -> Tuple[List[TrackAnchor], List[TrackAnchor]]:
    """Locate each track's title incipit in the libretto.

    Returns (chosen, rejected): `chosen` is the best monotone chain (libretto
    position non-decreasing with track number, several anchors per track
    allowed if they are in order) maximising total match score; `rejected`
    are hits that were found but dropped for being out of order.
    """
    toks, src = _libretto_tokens(libretto)
    items: List[TrackAnchor] = []
    for track in sorted(track_titles):
        for phrase in incipit_candidates(track_titles[track], opera_title):
            for tok_i, score, matched in _phrase_hits(phrase, toks, min_score):
                items.append(TrackAnchor(track, src[tok_i], phrase, matched, score))
    if not items:
        return [], []
    items.sort(key=lambda a: (a.track, a.position))
    # Weighted longest increasing chain over (track, position); ties prefer
    # the earlier libretto position (a band that starts early is cheap, one
    # that starts late hides the opening of the number).
    best = [0.0] * len(items)
    prev = [-1] * len(items)
    for i, a in enumerate(items):
        best[i] = a.score
        for j in range(i):
            b = items[j]
            if b.position < a.position or (b.position == a.position and b.track < a.track):
                if best[j] + a.score > best[i] + 1e-9:
                    best[i] = best[j] + a.score
                    prev[i] = j
    end = max(range(len(items)), key=lambda i: (best[i], -items[i].position))
    chosen_idx = []
    while end != -1:
        chosen_idx.append(end)
        end = prev[end]
    chosen_set = set(chosen_idx)
    chosen = [items[i] for i in sorted(chosen_idx)]
    rejected = [a for i, a in enumerate(items) if i not in chosen_set]
    return chosen, rejected


def bands_from_anchors(
    anchors: List[TrackAnchor],
    row_tracks: "np.ndarray",
    n_libretto: int,
    slack: int = 80,
    ground_truth_timestamps: Optional[Dict[int, float]] = None,
    row_times: Optional["np.ndarray"] = None,
    track_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
) -> Tuple["np.ndarray", Dict[int, Tuple[int, int]]]:
    """Per-transcript-word libretto column bands from track anchors.

    Track k's band runs from (its own anchor, or the nearest earlier anchored
    track's) - slack to (the next anchored track's anchor) + slack; unanchored
    tracks inherit the span between their anchored neighbours; tracks before
    the first / after the last anchor are open on that side. Manual
    `ground_truth_timestamps` (libretto index -> seconds) always win: the band
    of the track containing that second is widened to include the index.

    Returns (row_bands (m, 2), {track: (lo, hi)}).
    """
    import numpy as np
    row_tracks = np.asarray(row_tracks)
    tracks = sorted(set(int(t) for t in row_tracks.tolist()))
    first_pos: Dict[int, int] = {}
    for a in anchors:
        first_pos[a.track] = min(first_pos.get(a.track, a.position), a.position)
    anchored = sorted(first_pos)
    bands: Dict[int, Tuple[int, int]] = {}
    for k in tracks:
        earlier = [t for t in anchored if t <= k]
        later = [t for t in anchored if t > k]
        lo = first_pos[earlier[-1]] - slack if earlier else 0
        hi = first_pos[later[0]] + slack - 1 if later else n_libretto - 1
        bands[k] = (max(0, lo), min(n_libretto - 1, max(hi, 0)))
    if ground_truth_timestamps and row_times is not None and track_ranges:
        for j, t in ground_truth_timestamps.items():
            for k, (t0, t1) in track_ranges.items():
                if k in bands and t0 <= t < t1:
                    lo, hi = bands[k]
                    bands[k] = (min(lo, max(0, j - slack)), max(hi, min(n_libretto - 1, j + slack)))
    row_bands = np.empty((len(row_tracks), 2), dtype=np.int64)
    for k, (lo, hi) in bands.items():
        sel = row_tracks == k
        row_bands[sel, 0] = lo
        row_bands[sel, 1] = hi
    return row_bands, bands


def libretto_block_ids(libretto_text: str) -> List[int]:
    """Index of the blank-line-separated block each whitespace-split libretto
    word belongs to (same word order as `libretto_text.split()`)."""
    ids: List[int] = []
    b = 0
    for block in libretto_text.split("\n\n"):
        words = block.split()
        if not words:
            continue
        ids.extend([b] * len(words))
        b += 1
    return ids


def alignment_report(
    aligned_raw: List[AlignedWord],
    aligned_interp: List[AlignedWord],
    track_ranges: Dict[int, Tuple[float, float]],
    instrumental_tracks: List[int],
    transcript_words: List[TranscriptionWord],
    row_tracks: "np.ndarray",
    anchors: Optional[List[TrackAnchor]] = None,
    rejected_anchors: Optional[List[TrackAnchor]] = None,
    bands: Optional[Dict[int, Tuple[int, int]]] = None,
    track_titles: Optional[Dict[int, str]] = None,
    dropped: Optional[Dict[str, int]] = None,
    block_ids: Optional[List[int]] = None,
    text_timeout: float = 8.0,
    gap_threshold: float = 60.0,
) -> Dict:
    """Alignment quality numbers for the tripwire (see `judge_alignment`).

    "black" seconds = time the renderer would show no text, emulating
    video_gen: a timed (post-interpolation) word keeps its line on screen
    until `text_timeout` after it ends, and two timed words of the same
    libretto block (`block_ids`, see `libretto_block_ids`) keep it on screen
    in between (video_gen's interpolate_frames). Instrumental tracks are
    expected to be black and are excluded. "gaps" are between consecutive
    *raw* timed words (pre-interpolation). Per track, `transcript_gap_s` is
    the longest stretch with no transcript word at all — an ASR hole no
    alignment can fill.
    """
    import numpy as np

    def _starts(words):
        return np.array(sorted(w.start for w in words if w.start is not None and w.end is not None), dtype=np.float64)

    def _is_word(w):
        return any(c.isalnum() for c in w.word)

    n = len(aligned_raw)
    n_alnum = sum(1 for w in aligned_raw if _is_word(w))
    raw_starts = _starts(aligned_raw)
    int_starts = _starts(aligned_interp)
    raw_alnum_t = sum(1 for w in aligned_raw if w.start is not None and w.end is not None and _is_word(w))

    total_end = max(t1 for _, t1 in track_ranges.values()) if track_ranges else float(raw_starts[-1]) if len(raw_starts) else 0.0

    def _edges(starts):
        return np.concatenate(([0.0], starts, [total_end])) if len(starts) else np.array([0.0, total_end])

    raw_edges = _edges(raw_starts)
    raw_gaps = np.diff(raw_edges)
    gap_list = [(float(raw_edges[i]), float(raw_edges[i + 1])) for i in range(len(raw_gaps)) if raw_gaps[i] > gap_threshold]

    sung = {k: v for k, v in track_ranges.items() if k not in instrumental_tracks}

    # Visible intervals: [start, end + timeout] per timed word, plus the span
    # between consecutive timed words of the same block; merged.
    ivals = []
    prev_i = None
    for i, w in enumerate(aligned_interp):
        if w.start is None or w.end is None:
            continue
        ivals.append((w.start, w.end + text_timeout))
        if prev_i is not None and block_ids is not None and block_ids[i] == block_ids[prev_i]:
            ivals.append((aligned_interp[prev_i].end, w.start))
        prev_i = i
    ivals.sort()
    visible = []
    for a, b in ivals:
        if visible and a <= visible[-1][1]:
            visible[-1][1] = max(visible[-1][1], b)
        else:
            visible.append([a, b])
    vis_a = np.array([a for a, _ in visible]) if visible else np.zeros(0)
    vis_b = np.array([b for _, b in visible]) if visible else np.zeros(0)

    def _black_in(t0, t1):
        """Seconds in [t0, t1) with nothing on screen."""
        covered = np.clip(np.minimum(vis_b, t1) - np.maximum(vis_a, t0), 0, None).sum() if len(vis_a) else 0.0
        return max(0.0, (t1 - t0) - float(covered))

    row_tracks = np.asarray(row_tracks)
    trans_starts = np.array([w.start for w in transcript_words], dtype=np.float64)

    per_track = {}
    for k, (t0, t1) in sorted(track_ranges.items()):
        dur = t1 - t0
        n_timed = int(((raw_starts >= t0) & (raw_starts < t1)).sum())
        black = _black_in(t0, t1) if k in sung else dur
        tw = np.sort(trans_starts[row_tracks == k]) if len(trans_starts) else np.array([])
        if len(tw):
            tgaps = np.diff(np.concatenate(([t0], tw, [t1])))
            transcript_gap = float(tgaps.max())
        else:
            transcript_gap = dur
        entry = {
            "start_s": round(t0, 1), "duration_s": round(dur, 1),
            "instrumental": k in instrumental_tracks,
            "transcript_words": int(len(tw)),
            "transcript_gap_s": round(transcript_gap, 1),
            "timed_libretto_words": n_timed,
            "black_s": round(black, 1),
            "black_frac": round(black / dur, 3) if dur > 0 else 0.0,
        }
        if track_titles and k in track_titles:
            entry["title"] = track_titles[k]
        if bands and k in bands:
            entry["band"] = [int(bands[k][0]), int(bands[k][1])]
        if anchors:
            entry["anchors"] = [
                {"position": a.position, "phrase": a.phrase, "matched": a.matched, "score": round(a.score, 3), "source": a.source}
                for a in anchors if a.track == k
            ]
        per_track[k] = entry

    sung_dur = sum(t1 - t0 for t0, t1 in sung.values())
    black_total = sum(per_track[k]["black_s"] for k in sung)
    report = {
        "libretto_words": n,
        "libretto_words_alnum": n_alnum,
        "transcript_words": len(transcript_words),
        "transcript_dropped": dict(dropped or {}),
        "coverage_raw": round(len(raw_starts) / n, 4) if n else 0.0,
        "coverage_raw_alnum": round(raw_alnum_t / n_alnum, 4) if n_alnum else 0.0,
        "coverage_interp": round(len(int_starts) / n, 4) if n else 0.0,
        "gaps_over_threshold": len(gap_list),
        "gap_threshold_s": gap_threshold,
        "longest_gap_s": round(float(raw_gaps.max()), 1) if len(raw_gaps) else 0.0,
        "gaps": [[round(a, 1), round(b, 1)] for a, b in gap_list],
        "sung_duration_s": round(sung_dur, 1),
        "black_s": round(black_total, 1),
        "black_frac": round(black_total / sung_dur, 4) if sung_dur else 0.0,
        "anchors_found": len(anchors or []) + len(rejected_anchors or []),
        "anchors_used": len(anchors or []),
        "tracks_anchored": len({a.track for a in (anchors or [])}),
        "tracks_sung": len(sung),
        "rejected_anchors": [
            {"track": a.track, "position": a.position, "phrase": a.phrase, "matched": a.matched, "score": round(a.score, 3)}
            for a in (rejected_anchors or [])
        ],
        "per_track": per_track,
    }
    report["worst_tracks"] = sorted(
        ({"track": k, "black_frac": v["black_frac"], "duration_s": v["duration_s"], "transcript_gap_s": v["transcript_gap_s"]}
         for k, v in per_track.items() if k in sung and v["duration_s"] >= 60),
        key=lambda e: -e["black_frac"],
    )[:5]
    report["transcript_holes"] = [
        {"track": k, "transcript_gap_s": v["transcript_gap_s"], "duration_s": v["duration_s"]}
        for k, v in per_track.items() if k in sung and v["transcript_gap_s"] > REVIEW_MAX_TRANSCRIPT_GAP_S
    ]
    return report


# Tripwire thresholds (see judge_alignment for where they come from).
REVIEW_MAX_BLACK_FRAC = 0.20          # > 20 % of sung time with nothing on screen
REVIEW_MIN_COVERAGE_ALNUM = 0.40      # < 40 % of libretto words timed before interpolation
REVIEW_MAX_LONGEST_GAP_S = 300.0      # a single stretch > 5 min without a timed word
REVIEW_MAX_TRACK_BLACK_FRAC = 0.60    # informational: sung tracks (>= 60 s) more than 60 % black
REVIEW_MAX_TRANSCRIPT_GAP_S = 90.0    # informational: sung tracks with > 90 s of no transcript at all


def judge_alignment(report: Dict) -> List[str]:
    """Reasons the alignment needs a human look; empty list means OK.

    Calibrated on the operas on disk, measured with this module's transcript
    cleaning and the renderer-emulating black metric: Don Giovanni and
    Butterfly (published) are black 13 % / 13 % with 62 % / 53 % of words
    timed; Carmen — the one that prompted the tripwire — is black 24 % with
    36 % timed. Hence black > 20 % or coverage < 40 % means review. A single
    gap > 5 min is also flagged (Giovanni has one of 6.4 min: an ASR hole
    across two tracks, and it *is* six minutes of black screen). Per-track
    black and ASR holes are reported as information — nearly every opera has
    a couple, and they point at what to fix (re-transcribe that track).
    """
    reasons = []
    if report["black_frac"] > REVIEW_MAX_BLACK_FRAC:
        reasons.append(f"{report['black_frac']:.0%} of sung time has nothing on screen (limit {REVIEW_MAX_BLACK_FRAC:.0%})")
    if report["coverage_raw_alnum"] < REVIEW_MIN_COVERAGE_ALNUM:
        reasons.append(f"only {report['coverage_raw_alnum']:.0%} of libretto words timed (limit {REVIEW_MIN_COVERAGE_ALNUM:.0%})")
    if report["longest_gap_s"] > REVIEW_MAX_LONGEST_GAP_S:
        reasons.append(f"longest stretch without a timed word is {report['longest_gap_s'] / 60:.1f} min (limit {REVIEW_MAX_LONGEST_GAP_S / 60:.0f} min)")
    return reasons


def alignment_notes(report: Dict) -> List[str]:
    """Informational lines to print under the verdict: mostly-black tracks
    and ASR holes (tracks with a long stretch of no transcript at all)."""
    notes = []
    bad = [e for e in report.get("worst_tracks", []) if e["black_frac"] > REVIEW_MAX_TRACK_BLACK_FRAC]
    if bad:
        notes.append("mostly-black sung tracks: " + ", ".join(f"{e['track']} ({e['black_frac']:.0%})" for e in bad))
    holes = report.get("transcript_holes", [])
    if holes:
        notes.append(f"tracks with > {REVIEW_MAX_TRANSCRIPT_GAP_S:.0f} s of no transcript (ASR hole — re-transcribe): "
                     + ", ".join(f"{h['track']} ({h['transcript_gap_s']:.0f} s)" for h in holes))
    return notes


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

