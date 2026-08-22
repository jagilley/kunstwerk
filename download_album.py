#!/usr/bin/env python3
"""
download_album.py — headless audio acquisition for one opera.

    python download_album.py configs/<opera>.yaml [--dry-run] [--pick N] ...

Resolves a recording to an ordered YouTube tracklist, downloads every track to
audio/<file_prefix>/NN.m4a (1-based, 2-digit, contiguous) and writes
audio/<file_prefix>/tracks.json.  Uses nothing but yt-dlp — no API keys.

Config fields, in priority order (most explicitly pinned first):
  track_urls    YAML list of YouTube video URLs or bare 11-char ids, in track
                order.  For hand-assembled recordings: pins exact videos and
                survives playlist rot.
  album_url     music.youtube.com/browse/MPREb_… album URL, an OLAK5uy_… album
                playlist URL, an ordinary YouTube playlist URL, or a bare id.
  album_query   free text ("Bizet Carmen Abbado").  Searched on YouTube Music
                (albums only); every candidate album is fetched, scored for
                "looks like the complete opera named in `title`", the table is
                printed for auditing and the best one is picked.
  playlist_url  legacy: any YouTube playlist URL (rots as videos go away).

Exit status is non-zero if the tracklist cannot be resolved, any track is
unavailable, or the files on disk do not match the tracklist exactly — the
downstream stages assume a contiguous, complete set of tracks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import parse_qs, quote_plus, urlparse

import yaml
import yt_dlp

AUDIO_ROOT = "audio"
TRACKS_JSON = "tracks.json"
YTM_SEARCH = "https://music.youtube.com/search?q={q}#albums"
YTM_BROWSE = "https://music.youtube.com/browse/{id}"
YT_PLAYLIST = "https://www.youtube.com/playlist?list={id}"
YT_WATCH = "https://www.youtube.com/watch?v={id}"

UNAVAILABLE_TITLES = {"[private video]", "[deleted video]", "[unavailable video]"}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

@dataclass
class Track:
    index: int                # 1-based position in the album
    video_id: str
    title: str
    channel: str
    duration: float | None    # seconds; None when YouTube does not report it
    available: bool = True


@dataclass
class Album:
    source_url: str           # what we asked yt-dlp for
    playlist_id: str          # OLAK5uy_… / PL… — what yt-dlp reports as the id
    browse_id: str | None     # MPREb_… when known
    title: str
    tracks: list[Track]
    view_count: int | None = None
    reported_count: int | None = None   # YouTube's own track count; != len(tracks) means some were dropped
    # scoring bookkeeping (album_query mode only)
    score: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def playlist_url(self) -> str | None:
        if self.playlist_id.startswith("track_urls:"):
            return None
        return YT_PLAYLIST.format(id=self.playlist_id)

    @property
    def total_duration(self) -> float:
        return sum(t.duration or 0 for t in self.tracks)

    @property
    def n_unavailable(self) -> int:
        return sum(1 for t in self.tracks if not t.available)

    @property
    def n_dropped(self) -> int:
        """tracks YouTube counts but did not list (region-blocked / removed)"""
        if self.reported_count is None:
            return 0
        return max(0, self.reported_count - len(self.tracks))

    def query_tokens_missing(self, query: str) -> list[str]:
        hay = [fold(self.title)] + [fold(t.title) for t in self.tracks] + [fold(PERFORMER_RE.sub("", t.channel)) for t in self.tracks]
        return [tok for tok in _query_tokens(query) if not any(f" {tok} " in h for h in hay)]


class ResolveError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# yt-dlp helpers
# ----------------------------------------------------------------------------

def _quiet_opts(**extra):
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",   # resolve the album, keep tracks flat
        "noplaylist": False,
        "ignoreerrors": False,
    }
    opts.update(extra)
    return opts


def _strip_album_prefix(title: str | None) -> str:
    """yt-dlp reports YouTube Music albums as 'Album - <title>'."""
    title = title or ""
    return re.sub(r"^\s*album\s*-\s*", "", title, flags=re.I).strip()


def fetch_album(url: str, browse_id: str | None = None) -> Album:
    """Flat-extract one album/playlist into an Album (no downloads)."""
    with yt_dlp.YoutubeDL(_quiet_opts()) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info or info.get("_type") not in ("playlist", "multi_video"):
        raise ResolveError(f"{url} did not resolve to a playlist/album (got {info and info.get('_type')})")
    tracks: list[Track] = []
    for i, e in enumerate(info.get("entries") or [], start=1):
        if e is None:
            tracks.append(Track(i, "", "[unavailable]", "", None, available=False))
            continue
        title = e.get("title") or ""
        dur = e.get("duration")
        available = bool(e.get("id")) and title.strip().lower() not in UNAVAILABLE_TITLES
        tracks.append(Track(
            index=i,
            video_id=e.get("id") or "",
            title=title,
            channel=e.get("channel") or e.get("uploader") or "",
            duration=float(dur) if dur is not None else None,
            available=available,
        ))
    if not browse_id:
        m = re.search(r"(MPREb_[\w-]+)", url)
        browse_id = m.group(1) if m else None
    return Album(
        source_url=url,
        playlist_id=info.get("id") or "",
        browse_id=browse_id,
        title=_strip_album_prefix(info.get("title")),
        tracks=tracks,
        view_count=info.get("view_count"),
        reported_count=info.get("playlist_count"),
    )


def search_album_ids(query: str, limit: int) -> list[str]:
    """YouTube Music album search → ordered, de-duplicated MPREb_… ids."""
    url = YTM_SEARCH.format(q=quote_plus(query))
    with yt_dlp.YoutubeDL(_quiet_opts(extract_flat=True, playlistend=limit)) as ydl:
        info = ydl.extract_info(url, download=False)
    ids: list[str] = []
    for e in info.get("entries") or []:
        if e and e.get("id", "").startswith("MPREb_") and e["id"] not in ids:
            ids.append(e["id"])
    return ids


def fetch_albums_parallel(ids: list[str], workers: int) -> list[Album]:
    albums: dict[str, Album] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_album, YTM_BROWSE.format(id=i), i): i for i in ids}
        for fut in as_completed(futs):
            bid = futs[fut]
            try:
                albums[bid] = fut.result()
            except Exception as exc:  # noqa: BLE001 — one bad candidate must not kill the search
                print(f"  (skipping {bid}: {type(exc).__name__}: {str(exc).splitlines()[0][:120]})", file=sys.stderr)
    # keep search order, and drop duplicate albums that map to the same playlist
    out, seen = [], set()
    for i in ids:
        a = albums.get(i)
        if a and a.playlist_id not in seen:
            seen.add(a.playlist_id)
            out.append(a)
    return out


VIDEO_ID_RE = re.compile(r"^[\w-]{11}$")
VIDEO_URL_RE = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/|/embed/|/live/)([\w-]{11})")


def parse_video_id(value) -> str:
    """'https://www.youtube.com/watch?v=ID', 'https://youtu.be/ID', music.youtube.com/watch?v=ID&list=…, or bare ID."""
    v = str(value).strip()
    if VIDEO_ID_RE.match(v):
        return v
    m = VIDEO_URL_RE.search(v)
    if m and "youtu" in v:
        return m.group(1)
    raise ResolveError(f"track_urls entry is not a YouTube video URL or id: {value!r}")


def fetch_video(index: int, video_id: str) -> Track:
    """Metadata for one video (no download).  Unavailable → Track(available=False)."""
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True, "ignoreerrors": False}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(YT_WATCH.format(id=video_id), download=False)
    except yt_dlp.utils.DownloadError as exc:
        msg = str(exc).splitlines()[0]
        msg = re.sub(r"^ERROR:\s*(\[youtube\]\s*\S+:\s*)?", "", msg)
        return Track(index, video_id, f"[unavailable: {msg}]", "", None, available=False)
    dur = info.get("duration")
    return Track(
        index=index,
        video_id=video_id,
        title=info.get("title") or "",
        channel=info.get("channel") or info.get("uploader") or "",
        duration=float(dur) if dur is not None else None,
        available=True,
    )


def resolve_track_urls(entries, work_title: str | None, workers: int) -> Album:
    """A hand-assembled ordered list of videos, treated as its own 'album' keyed on the id list."""
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ResolveError("track_urls must be a non-empty YAML list of video URLs / ids")
    ids = [parse_video_id(e) for e in entries]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        raise ResolveError(f"track_urls lists the same video more than once: {', '.join(dups)}")
    print(f"Fetching metadata for {len(ids)} pinned videos with {workers} workers …")
    tracks: dict[int, Track] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_video, i, vid): i for i, vid in enumerate(ids, start=1)}
        for fut in as_completed(futs):
            t = fut.result()
            tracks[t.index] = t
    key = hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]
    return Album(
        source_url="track_urls",
        playlist_id=f"track_urls:{key}",
        browse_id=None,
        title=f"{work_title or 'untitled'} (hand-assembled, {len(ids)} tracks)",
        tracks=[tracks[i] for i in range(1, len(ids) + 1)],
        reported_count=len(ids),
    )


# ----------------------------------------------------------------------------
# candidate scoring
# ----------------------------------------------------------------------------

def fold(s: str) -> str:
    """lowercase, strip accents and punctuation → space-separated words"""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " " + re.sub(r"\s+", " ", s).strip() + " "


QUERY_STOPWORDS = {
    "the", "and", "der", "die", "das", "les", "le", "la", "il", "de", "von", "van", "opera", "oper",
    "complete", "full", "recording", "album", "cd", "in", "of", "con", "mit", "with", "by",
}

# (regex on folded album title, penalty, label).  Folded text is lowercase ascii words
# padded with single spaces, so patterns can anchor on " word ".
TITLE_PENALTIES = [
    (r" highlights? | excerpts? | extraits? | querschnitt | auszuge | auszug | selections? | selezione | brani scelti | pagine scelte | scenes? from | szenen aus | hojdpunkter | das beste ", 45, "highlights"),
    (r" suites? | suite no ", 35, "suite"),
    (r" arias? | arien | arie | duets? | duetti | duette | choruses | chore | cori | famous | beruhmte | celebri | love duets? ", 30, "arias/duets"),
    (r" best of | the best | very best | greatest | essential | ultimate | gold | collection | anthology | compilation | (20|25|30|40|50|75|100) (greatest|best|masterpieces|classics|tracks|hits) | masterpieces | classics | hits | favou?rites? | treasury | sampler | introduction to | discover ", 30, "compilation"),
    (r" overtures? | ouvertures? | ouverturen | preludes? | intermezz[oi] | entractes? ", 25, "orchestral excerpts"),
    (r" ballet | fantas(y|ia|ie)s? | transcri\w* | arr | arranged | arrangements? | paraphrase | for (piano|guitar|organ|brass|band|orchestra|strings|cello|violin|flute|wind) | piano (solo|version|reduction) | solo piano | symphonic | orchestral | instrumental | without words | brass band | wind band ", 35, "arrangement/instrumental"),
    (r" karaoke | ringtones? | lullab(y|ies) | music box | relax\w* | study | sleep | meditation | baby | kids | children | 8 bit | lofi | lo fi | remix | dance | edm | trance | cover | tribute | sound ?track | ost | film | movie | game ", 60, "novelty"),
    (r" vocal score | libretto | audiobook | interview | documentary | rehearsal | masterclass | lecture | spoken | narrat\w* ", 50, "not a recording"),
    (r" various artists | sampler | megamix | medley ", 30, "various"),
]

# sung-in-translation markers: penalised unless the config language matches
SUNG_IN = [
    (r" sung in english | in english | english version | english national opera | eno ", "en"),
    (r" sung in german | in german | auf deutsch | in deutscher sprache | deutsch gesungen | german version ", "de"),
    (r" sung in italian | in italian | in italiano | italian version ", "it"),
    (r" sung in french | in french | en francais | french version ", "fr"),
    (r" sung in russian | in russian | russian version ", "ru"),
    (r" sung in swedish | sung in hungarian | sung in czech | sung in spanish | en espanol ", "xx"),
]

COMPLETE_MARKERS = r" complete | complet | completa | integrale | gesamtaufnahme | full opera | opera in (two|three|four|five|2|3|4|5) acts | opera completa | komplett | recording of the opera "
ACT_MARKERS = r" act | acte | akt | atto | aufzug | scene | szene | scena | scene | tableau | bild "
PERFORMER_RE = re.compile(r"[-–—]\s*topic\s*$", re.I)
# two "Composer:" segments separated by - / ; –  (e.g. "Mascagni: Cavalleria rusticana - Leoncavallo: Pagliacci")
MULTI_WORK_RE = re.compile(r"\b[A-ZÀ-Ý][\w'.]+(?:,\s*[A-Z]\.)?\s*:[^:/;–—-]*[-/;–—]\s*[A-ZÀ-Ý][\w'.]+(?:,\s*[A-Z]\.)?\s*:")


def _norm_lang(code: str | None) -> str:
    return (code or "").strip().lower()[:2]


def _query_tokens(query: str) -> list[str]:
    toks = [t for t in fold(query).split() if len(t) >= 3 and t not in QUERY_STOPWORDS]
    # keep order, drop dups
    return list(dict.fromkeys(toks))


def score_album(album: Album, query: str, work_title: str | None, language: str | None) -> None:
    """Mutates album.score / album.notes.  Higher is better; see the notes for why."""
    title_f = fold(album.title)
    track_titles_f = [fold(t.title) for t in album.tracks]
    channels_f = [fold(PERFORMER_RE.sub("", t.channel)) for t in album.tracks]
    n = len(album.tracks)
    minutes = album.total_duration / 60.0
    score = 0.0
    notes: list[str] = []

    # 1. words that say "not the complete opera"
    for pat, penalty, label in TITLE_PENALTIES:
        if re.search(pat, title_f):
            score -= penalty
            notes.append(f"-{penalty} {label}")
    lang = _norm_lang(language)
    for pat, code in SUNG_IN:
        if re.search(pat, title_f) and code != lang:
            score -= 30
            notes.append("-30 sung in translation")
            break
    if re.search(COMPLETE_MARKERS, title_f):
        score += 10
        notes.append("+10 'complete'")

    # 2. size: duration is the strongest single signal.  ~60 min = one CD of
    #    highlights; complete operas run 90–300 min.
    dur_score = max(-1.0, min(1.0, (minutes - 65.0) / 45.0)) * 30.0
    score += dur_score
    notes.append(f"{dur_score:+.0f} duration {minutes:.0f}m")
    if minutes > 330:
        score -= 20
        notes.append("-20 >5.5h (box set?)")
    if n >= 12:
        cnt_score = min(n, 40) / 40.0 * 10.0
    else:
        cnt_score = -10.0 if minutes < 85 else 0.0
    score += cnt_score
    notes.append(f"{cnt_score:+.0f} {n} tracks")
    if n and minutes / n > 12:   # per-act mega-tracks: complete, but worse for transcription/alignment/chapters
        long_pen = min(20.0, minutes / n - 12)
        score -= long_pen
        notes.append(f"-{long_pen:.0f} long tracks (avg {minutes / n:.0f}m)")
    if album.n_unavailable:
        score -= 100
        notes.append(f"-100 {album.n_unavailable} unavailable")
    if album.n_dropped:
        score -= 100
        notes.append(f"-100 {album.n_dropped} track(s) not listed (of {album.reported_count})")
    if n == 0:
        score -= 100
        notes.append("-100 empty")

    # 3. is it the work we asked for?  (config `title`, e.g. "DON GIOVANNI")
    work_f = fold(work_title).strip() if work_title else ""
    if work_f:
        if f" {work_f} " in title_f:
            score += 15
            notes.append("+15 work in title")
        else:
            # last word of the work name ("giovanni", "tristan") as a weaker check
            last = work_f.split()[-1]
            if f" {last} " in title_f:
                score += 5
                notes.append("+5 work (partial) in title")
            else:
                score -= 30
                notes.append("-30 work not in title")
        if n:
            cov = sum(1 for t in track_titles_f if f" {work_f} " in t or f" {work_f.split()[-1]} " in t) / n
            cov_score = cov * 10.0
            score += cov_score
            notes.append(f"{cov_score:+.0f} work in {cov:.0%} of tracks")
            # the work is named in only *some* track titles → coupled with other works
            # (Cav/Pag, two-opera boxes).  0% just means the metadata never names it.
            if 0.15 < cov < 0.85:
                score -= 20
                notes.append("-20 mixed album (work in only part of the tracklist)")
    # "Mascagni: Cavalleria rusticana - Leoncavallo: Pagliacci" — two 'Composer:' segments
    if MULTI_WORK_RE.search(album.title):
        score -= 25
        notes.append("-25 several works in title")

    # 4. opera-ness of the tracklist: act/scene markers in track titles
    if n:
        act_cov = sum(1 for t in track_titles_f if re.search(ACT_MARKERS, t)) / n
        act_score = act_cov * 10.0
        score += act_score
        notes.append(f"{act_score:+.0f} act/scene in {act_cov:.0%} of tracks")

    # 5. how much of the query shows up (album title counts most; performers
    #    usually only appear as track channels — "Claudio Abbado - Topic")
    toks = _query_tokens(query)
    if toks:
        in_title = [t for t in toks if f" {t} " in title_f]
        rest = [t for t in toks if t not in in_title]
        elsewhere = [t for t in rest if any(f" {t} " in c for c in channels_f) or any(f" {t} " in tt for tt in track_titles_f)]
        q_score = len(in_title) / len(toks) * 20.0 + len(elsewhere) / len(toks) * 12.0
        score += q_score
        missing = [t for t in rest if t not in elsewhere]
        notes.append(f"{q_score:+.0f} query match" + (f" (missing: {' '.join(missing)})" if missing else ""))

    # 6. popularity as a small tie-breaker among otherwise-equal recordings
    if album.view_count:
        pop = min(math.log10(album.view_count + 1), 7) * 0.6
        score += pop
        notes.append(f"{pop:+.1f} views {album.view_count:,}")

    album.score = round(score, 1)
    album.notes = notes


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def print_candidates(albums: list[Album], pick: Album | None, width: int = 70) -> None:
    print("\n  #  score  trk  duration  id                 title")
    print("  -- ------ ---  --------  -----------------  " + "-" * width)
    for i, a in enumerate(albums, start=1):
        mark = "*" if a is pick else " "
        title = a.title if len(a.title) <= width else a.title[: width - 1] + "…"
        print(f"{mark}{i:3d} {a.score:6.1f} {len(a.tracks):4d}  {fmt_duration(a.total_duration):>8}  {a.browse_id or a.playlist_id:17}  {title}")
    print()
    for i, a in enumerate(albums[:5], start=1):
        print(f"  #{i}: " + "; ".join(a.notes))
    print()


def resolve_by_query(query: str, work_title: str | None, language: str | None,
                     max_candidates: int, workers: int, min_score: float,
                     forced_pick: int | None, strict: bool) -> tuple[Album, list[Album]]:
    """Search, score, pick.  Relaxes the query (drops trailing words) when the
    full query finds nothing convincing."""
    words = query.split()
    tried: list[str] = []
    all_albums: dict[str, Album] = {}
    ranked: list[Album] = []
    pick: Album | None = None
    while True:
        q = " ".join(words)
        tried.append(q)
        print(f"Searching YouTube Music albums for: {q!r} (top {max_candidates})")
        ids = search_album_ids(q, max_candidates)
        new_ids = [i for i in ids if i not in all_albums]
        print(f"  {len(ids)} results, {len(new_ids)} new; fetching tracklists with {workers} workers …")
        for a in fetch_albums_parallel(new_ids, workers):
            all_albums[a.browse_id or a.playlist_id] = a
        for a in all_albums.values():
            score_album(a, query, work_title, language)   # always score against the *full* query
        ranked = sorted(all_albums.values(), key=lambda a: a.score, reverse=True)
        if forced_pick is not None:
            break
        if ranked and ranked[0].score >= min_score:
            pick = ranked[0]
            break
        if len(words) <= 2:
            break
        words = words[:-1]
        print(f"  best score {ranked[0].score if ranked else 'n/a'} < {min_score}; relaxing query …")

    if forced_pick is not None:
        if not 1 <= forced_pick <= len(ranked):
            raise ResolveError(f"--pick {forced_pick} out of range (1..{len(ranked)})")
        pick = ranked[forced_pick - 1]
    print_candidates(ranked, pick)
    if pick is None:
        raise ResolveError(
            f"No candidate scored >= {min_score} for {tried}. Inspect the table above; set "
            f"`album_url` in the config, rerun with --pick N, or lower --min-score."
        )
    close = [a for a in ranked if a is not pick and pick.score - a.score < 8]
    if close:
        print(f"NOTE: {len(close)} other candidate(s) within 8 points of the pick "
              f"(e.g. {close[0].title!r} [{close[0].browse_id}]) — the query does not distinguish between "
              f"these recordings. Add performer surnames to `album_query`, or set `album_url`, to choose one.")
    missing = pick.query_tokens_missing(query)
    if missing:
        msg = (f"query words not found anywhere in the pick's title/track titles/channels: {' '.join(missing)!r}. "
               f"Either that recording is not on YouTube Music as an album, or its metadata does not name them "
               f"(orchestras often are not). The pick is a *different* recording of the same work.")
        if strict:
            raise ResolveError("--strict: " + msg + " Set `album_url`/`playlist_url` or drop those words from the query.")
        print("WARNING: " + msg + " Set `album_url`/`playlist_url` if that matters.")
    return pick, ranked


# ----------------------------------------------------------------------------
# url handling
# ----------------------------------------------------------------------------

def normalise_album_url(value: str) -> tuple[str, str | None]:
    """Return (url for yt-dlp, browse id or None) for the forms we accept."""
    v = value.strip()
    if re.fullmatch(r"MPREb_[\w-]+", v):
        return YTM_BROWSE.format(id=v), v
    if re.fullmatch(r"(OLAK5uy_|PL|RDCLAK5uy_|FL)[\w-]+", v):
        return YT_PLAYLIST.format(id=v), None
    u = urlparse(v)
    if "youtu" not in u.netloc:
        raise ResolveError(f"Not a YouTube / YouTube Music URL: {value}")
    m = re.search(r"/browse/(MPREb_[\w-]+)", u.path)
    if m:
        return YTM_BROWSE.format(id=m.group(1)), m.group(1)
    lst = parse_qs(u.query).get("list", [None])[0]
    if lst:
        return YT_PLAYLIST.format(id=lst), None
    raise ResolveError(f"Could not find an album id or list= parameter in {value}")


# ----------------------------------------------------------------------------
# downloading
# ----------------------------------------------------------------------------

def track_path(out_dir: str, index: int) -> str:
    return os.path.join(out_dir, f"{index:02d}.m4a")


def download_track(out_dir: str, track: Track) -> None:
    """Download one video as audio/<prefix>/NN.m4a.  Raises yt_dlp.utils.DownloadError."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "ignoreerrors": False,
        "retries": 10,
        "fragment_retries": 10,
        # prefer a native m4a stream (no transcode); fall back to anything, then extract to m4a
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": os.path.join(out_dir, f"{track.index:02d}.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([YT_WATCH.format(id=track.video_id)])


def write_tracks_json(path: str, cfg_prefix: str, source_kind: str, source_value: str,
                      album: Album, complete: bool) -> None:
    payload = {
        "file_prefix": cfg_prefix,
        "source": {"kind": source_kind, "value": source_value},   # value is the id list for track_urls
        "album_title": album.title,
        "album_id": album.browse_id,
        "playlist_id": album.playlist_id,
        "playlist_url": album.playlist_url,
        "track_count": len(album.tracks),
        "total_duration": round(album.total_duration, 1),
        "resolved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete": complete,
        "tracks": [
            {
                "index": t.index,
                "file": f"{t.index:02d}.m4a",
                "video_id": t.video_id,
                "title": t.title,
                "channel": t.channel,
                "duration": t.duration,
                "album_title": album.title,
            }
            for t in album.tracks
        ],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def existing_track_files(out_dir: str) -> list[str]:
    if not os.path.isdir(out_dir):
        return []
    return sorted(f for f in os.listdir(out_dir) if re.fullmatch(r"\d{2}\.m4a", f))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="configs/<opera>.yaml")
    ap.add_argument("--dry-run", action="store_true", help="resolve and print the tracklist, download nothing")
    ap.add_argument("--pick", type=int, metavar="N", help="album_query mode: force candidate #N from the table")
    ap.add_argument("--max-candidates", type=int, default=30, help="album_query mode: search results to inspect (default 30)")
    ap.add_argument("--workers", type=int, default=8, help="parallel tracklist fetches (default 8)")
    ap.add_argument("--min-score", type=float, default=25.0, help="album_query mode: refuse picks below this score (default 25)")
    ap.add_argument("--strict", action="store_true",
                    help="album_query mode: fail unless every word of the query appears in the picked album's metadata")
    ap.add_argument("--replace", action="store_true",
                    help="make audio/<prefix>/ match this resolution: re-download NN.m4a files whose recorded "
                         "video id differs (per tracks.json) and remove files beyond the tracklist")
    args = ap.parse_args(argv)
    # keep stdout/stderr interleaved sanely when logs are redirected to a file
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    prefix = cfg.get("file_prefix")
    if not prefix:
        print("config has no file_prefix", file=sys.stderr)
        return 2
    out_dir = os.path.join(AUDIO_ROOT, prefix)
    tracks_json = os.path.join(out_dir, TRACKS_JSON)

    # ---- resolve -----------------------------------------------------------
    try:
        if cfg.get("track_urls"):
            album = resolve_track_urls(cfg["track_urls"], cfg.get("title"), args.workers)
            source_kind, source_value = "track_urls", [t.video_id for t in album.tracks]
        elif cfg.get("album_url"):
            source_kind, source_value = "album_url", str(cfg["album_url"]).strip()
            url, bid = normalise_album_url(source_value)
            print(f"Fetching album {url}")
            album = fetch_album(url, bid)
        elif cfg.get("album_query"):
            source_kind, source_value = "album_query", str(cfg["album_query"]).strip()
            album, _ = resolve_by_query(source_value, cfg.get("title"), cfg.get("language"),
                                        args.max_candidates, args.workers, args.min_score, args.pick, args.strict)
        elif cfg.get("playlist_url"):
            source_kind, source_value = "playlist_url", str(cfg["playlist_url"]).strip()
            url, bid = normalise_album_url(source_value)
            print(f"Fetching playlist {url}")
            album = fetch_album(url, bid)
        else:
            print("config needs one of track_urls, album_url, album_query, playlist_url", file=sys.stderr)
            return 2
    except (ResolveError, yt_dlp.utils.DownloadError) as exc:
        print(f"ERROR: could not resolve a tracklist: {exc}", file=sys.stderr)
        return 1

    n = len(album.tracks)
    print(f"\nResolved: {album.title!r}  [{album.browse_id or album.playlist_id}]  {n} tracks, {fmt_duration(album.total_duration)}")
    if album.playlist_url:
        print(f"  {album.playlist_url}")
    for t in album.tracks:
        flag = "" if t.available else "  <-- UNAVAILABLE"
        print(f"  {t.index:02d}  {fmt_duration(t.duration or 0):>7}  {t.video_id:11}  {t.title}  [{t.channel}]{flag}")
    if n == 0:
        print("ERROR: empty tracklist", file=sys.stderr)
        return 1
    if album.n_dropped:
        print(f"ERROR: YouTube reports {album.reported_count} tracks for this album but lists only {n}; "
              f"{album.n_dropped} are unavailable (region-blocked or removed). Pick another recording.", file=sys.stderr)
        return 1
    bad = [t for t in album.tracks if not t.available]
    if bad:
        fix = "Replace those track_urls entries." if source_kind == "track_urls" else "Pick another recording."
        print(f"ERROR: {len(bad)} track(s) unavailable: " + ", ".join(f"{t.index:02d}" for t in bad) + ". " + fix,
              file=sys.stderr)
        return 1
    if args.dry_run:
        print("\n--dry-run: not downloading.")
        return 0

    # ---- guard against mixing two different resolutions in one directory ---
    # A previous tracks.json records which video each NN.m4a came from; any existing
    # file whose recorded video id differs from the new resolution is stale.
    # (Legacy dirs without tracks.json can't be checked: existing files are trusted.)
    os.makedirs(out_dir, exist_ok=True)
    have = existing_track_files(out_dir)
    prev: dict = {}
    if have and os.path.exists(tracks_json):
        try:
            with open(tracks_json, encoding="utf-8") as f:
                prev = json.load(f) or {}
        except (OSError, ValueError):
            prev = {}
    prev_ids = {t.get("index"): t.get("video_id") for t in prev.get("tracks", []) if isinstance(t, dict)}
    stale = [t for t in album.tracks
             if f"{t.index:02d}.m4a" in have and prev_ids.get(t.index) and prev_ids[t.index] != t.video_id]
    extra_now = [f for f in have if int(f[:2]) > n]
    if prev and prev.get("playlist_id") and prev.get("playlist_id") != album.playlist_id:
        print(f"NOTE: {out_dir} was last filled from {prev.get('album_title')!r} ({prev.get('playlist_id')}); "
              f"now resolving {album.title!r} ({album.playlist_id}).")
    if (stale or extra_now) and not args.replace:
        if stale:
            print(f"ERROR: {len(stale)} existing file(s) in {out_dir} came from different videos than now resolved: "
                  + ", ".join(f"{t.index:02d} ({prev_ids[t.index]} -> {t.video_id})" for t in stale), file=sys.stderr)
        if extra_now:
            print(f"ERROR: {out_dir} has files beyond the {n}-track resolution: {', '.join(extra_now)}", file=sys.stderr)
        print(f"Delete those files (or {out_dir}) or rerun with --replace to re-download/remove them.", file=sys.stderr)
        return 1
    if args.replace:
        for t in stale:
            print(f"--replace: removing stale {t.index:02d}.m4a (was {prev_ids[t.index]}, now {t.video_id})")
            os.remove(track_path(out_dir, t.index))
        for f in extra_now:
            print(f"--replace: removing extra {f}")
            os.remove(os.path.join(out_dir, f))
    write_tracks_json(tracks_json, prefix, source_kind, source_value, album, complete=False)

    # ---- download ----------------------------------------------------------
    failures: list[tuple[Track, str]] = []
    consecutive = 0
    for t in album.tracks:
        path = track_path(out_dir, t.index)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"[{t.index:02d}/{n}] exists, skipping: {os.path.basename(path)}")
            continue
        print(f"[{t.index:02d}/{n}] downloading {t.video_id}  {t.title}")
        t0 = time.time()
        try:
            download_track(out_dir, t)
            if not (os.path.exists(path) and os.path.getsize(path) > 0):
                raise RuntimeError(f"yt-dlp finished but {path} is missing")
            consecutive = 0
            print(f"          done in {time.time() - t0:.0f}s")
        except Exception as exc:  # noqa: BLE001 — record and keep going so the report is complete
            msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            print(f"          FAILED: {msg}", file=sys.stderr)
            failures.append((t, msg))
            consecutive += 1
            if consecutive >= 3:
                print("Three consecutive failures — probably systemic (network / bot check); stopping.", file=sys.stderr)
                break

    # ---- verify ------------------------------------------------------------
    have = existing_track_files(out_dir)
    expected = [f"{i:02d}.m4a" for i in range(1, n + 1)]
    missing = [f for f in expected if f not in have]
    extra = [f for f in have if f not in expected]
    ok = not failures and not missing and not extra
    write_tracks_json(tracks_json, prefix, source_kind, source_value, album, complete=ok)

    print()
    if failures:
        print(f"ERROR: {len(failures)} track(s) failed to download:", file=sys.stderr)
        for t, msg in failures:
            print(f"  {t.index:02d} {t.video_id} {t.title}: {msg}", file=sys.stderr)
    if missing:
        print(f"ERROR: missing on disk: {', '.join(missing)}", file=sys.stderr)
    if extra:
        print(f"ERROR: {out_dir} has files beyond the {n}-track album: {', '.join(extra)} — remove them "
              f"(downstream stages enumerate tracks by number).", file=sys.stderr)
    if not ok:
        return 1
    print(f"OK: {n}/{n} tracks in {out_dir} ({fmt_duration(album.total_duration)}); wrote {tracks_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
