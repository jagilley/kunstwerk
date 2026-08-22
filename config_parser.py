"""Opera config loading.

A config YAML only *needs* `title`, `file_prefix`, `language` and one audio
source (`album_url` / `album_query` / `playlist_url`). Everything the old
configs spelled out by hand — `end_idx`, `overture_indices`, `characters`,
`translation_file` — is derived lazily from what's on disk when it is not
given, so the same config works headlessly and can still be overridden.

Derivations happen on first attribute access (not at parse time) because the
stage scripts run in order: the libretto and audio don't exist yet when the
orchestrator first parses the config.
"""
import glob
import os
import re
import sys
import warnings
from functools import cached_property
from typing import List, Optional

import yaml


def _natural_track_indices(paths: List[str]) -> List[int]:
    idxs = []
    for p in paths:
        m = re.match(r"^(\d+)$", os.path.splitext(os.path.basename(p))[0])
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def derive_character_names(text: str) -> List[str]:
    """Speaker labels from a libretto in our text format: the first line of a
    blank-line-separated block when it is (essentially) all caps. Act headings
    like `PREMIER ACTE` come along too, which is harmless — they get bolded.
    """
    names = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip("\n")
        if not block:
            continue
        first = block.split("\n", 1)[0].strip()
        letters = [c for c in first if c.isalpha()]
        if not letters or len(first) > 60:
            continue
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)
        if upper_ratio < 0.7:
            continue
        if first[-1] in ".!?;:":
            continue
        if first not in names:
            names.append(first)
    return names


class OperaConfig:
    # ---- required ---------------------------------------------------------
    title: str
    file_prefix: str
    language: str

    def __init__(self, raw: dict, path: Optional[str] = None):
        self._raw = raw
        self.path = path
        missing = [k for k in ("title", "file_prefix", "language") if not raw.get(k)]
        if missing:
            raise ValueError(f"Config {path or ''} is missing required field(s): {', '.join(missing)}")

        self.title: str = raw["title"]
        self.file_prefix: str = raw["file_prefix"]
        self.language: str = raw["language"]
        self.translation_language: str = raw.get("translation_language", "en")

        # Audio sources (download_album.py picks the first one present)
        self.album_url: Optional[str] = raw.get("album_url")
        self.album_query: Optional[str] = raw.get("album_query")
        self.playlist_url: Optional[str] = raw.get("playlist_url")
        self.spotify_url: Optional[str] = raw.get("spotify_url")  # legacy, no longer functional

        # Libretto source (fetch_libretto.py); None = look the title up
        self.libretto_url: Optional[str] = raw.get("libretto_url")

        # Rendering
        self.secondary_color: str = raw.get("secondary_color", "Silver")
        self.video_width: int = raw.get("video_width", 3840)
        self.video_height: int = raw.get("video_height", 2160)
        self.font_size: int = raw.get("font_size", 96)
        self.res_divisor: int = raw.get("res_divisor", 1)

        self.start_idx: int = raw.get("start_idx", 1)

    # ---- paths --------------------------------------------------------------
    @property
    def audio_dir(self) -> str:
        return f"audio/{self.file_prefix}"

    @property
    def sep_dir(self) -> str:
        return f"sep/{self.file_prefix}_sep"

    @property
    def transcribed_dir(self) -> str:
        return f"transcribed/{self.file_prefix}_transcribed"

    @property
    def libretto_path(self) -> str:
        return f"libretti/{self.file_prefix}_{self.language}.txt"

    @property
    def translation_file(self) -> str:
        """Filename (relative to libretti/) of the translation."""
        return self._raw.get("translation_file") or f"{self.file_prefix}_{self.translation_language}.txt"

    @property
    def translation_path(self) -> str:
        return f"libretti/{self.translation_file}"

    def audio_files(self) -> List[str]:
        return [f"{self.audio_dir}/{str(i).zfill(2)}.m4a" for i in range(self.start_idx, self.end_idx)]

    # ---- derived ------------------------------------------------------------
    @cached_property
    def end_idx(self) -> int:
        """Exclusive upper bound of track indices (= number of tracks + 1).
        From the config if given, else counted from audio/ (or transcribed/ as
        a fallback for re-renders where the audio has been cleaned up)."""
        if self._raw.get("end_idx"):
            return int(self._raw["end_idx"])
        for pattern in (f"{self.audio_dir}/*.m4a", f"{self.transcribed_dir}/*.json"):
            idxs = _natural_track_indices(glob.glob(pattern))
            if idxs:
                expected = list(range(1, len(idxs) + 1))
                if idxs != expected:
                    raise ValueError(
                        f"Track numbering in {os.path.dirname(pattern)} is not contiguous "
                        f"(got {idxs}); refusing to guess end_idx"
                    )
                return len(idxs) + 1
        raise ValueError(
            f"end_idx is not set in {self.path} and nothing is downloaded yet under "
            f"{self.audio_dir}/ — run the download stage first"
        )

    @cached_property
    def overture_indices(self) -> List[int]:
        """1-based indices of purely instrumental tracks whose transcripts get
        blanked. From the config if given, else from detect_instrumental.py's
        cache (sep/<prefix>_sep/instrumental.json); empty with a warning if
        neither exists."""
        if "overture_indices" in self._raw and self._raw["overture_indices"] is not None:
            return list(self._raw["overture_indices"])
        try:
            from detect_instrumental import load_instrumental_indices
        except ImportError:
            load_instrumental_indices = None  # type: ignore
        if load_instrumental_indices is not None:
            cached = load_instrumental_indices(self.file_prefix)
            if cached is not None:
                return list(cached)
        warnings.warn(
            f"No overture_indices in {self.path} and no instrumental-track cache under "
            f"{self.sep_dir}/ — treating every track as sung. Run detect_instrumental.py "
            f"after separation to fix this.",
            stacklevel=2,
        )
        return []

    @cached_property
    def character_names(self) -> List[str]:
        """Speaker labels used for bold formatting. Derived from the source and
        translation libretti (both are displayed), plus anything listed under
        `characters:` in the config."""
        names: List[str] = list(self._raw.get("characters") or [])
        for p in (self.libretto_path, self.translation_path):
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    for n in derive_character_names(f.read()):
                        if n not in names:
                            names.append(n)
        if not names:
            warnings.warn(
                f"No character names: none in {self.path} and no libretto at {self.libretto_path}",
                stacklevel=2,
            )
        return names

    # ---- misc ---------------------------------------------------------------
    def __repr__(self) -> str:
        return f"OperaConfig({self.file_prefix!r}, {self.language!r} -> {self.translation_language!r})"


def parse_opera_config(yaml_path: str) -> OperaConfig:
    """Parse opera configuration from YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return OperaConfig(raw, path=str(yaml_path))


if __name__ == "__main__":
    # Quick introspection: python config_parser.py configs/carmen.yaml
    cfg = parse_opera_config(sys.argv[1])
    print(cfg)
    for attr in ("title", "language", "translation_file", "album_url", "album_query", "playlist_url", "libretto_url", "res_divisor"):
        print(f"  {attr:20s} {getattr(cfg, attr)!r}")
    for attr in ("end_idx", "overture_indices", "character_names"):
        try:
            print(f"  {attr:20s} {getattr(cfg, attr)!r}")
        except Exception as e:  # derivation needs on-disk artifacts
            print(f"  {attr:20s} <unavailable: {e}>")
