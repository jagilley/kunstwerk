#!/usr/bin/env python3
"""
fetch_libretto.py — pull an opera libretto (and, when available, its translation)
from librettoarchive.com (formerly murashev.com) into libretti/<prefix>_<lang>.txt
in the blank-line-separated block format that make_video.py / translate.py expect.

    python fetch_libretto.py configs/carmen.yaml
    python fetch_libretto.py configs/carmen.yaml --out-dir /tmp/x --force
    python fetch_libretto.py --list-catalog

Config fields used (read with yaml.safe_load, no dependency on config_parser.py):
    title                 fuzzy-matched against the site catalog (/Operas)
    file_prefix           output is libretti/<file_prefix>_<lang>.txt
    language              ISO-639-1 code of the sung language (source libretto)
    translation_language  ISO-639-1 code of the translation (default: en)
    libretto_url          optional explicit URL — either the opera landing page
                          (https://www.librettoarchive.com/Carmen) or a single-language
                          libretto page (…/Carmen_libretto_French); overrides title lookup

How the site is laid out (as of 2026-08):
  * /Operas is a table of operas; each row links to /<Opera_Title>.
  * /<Opera_Title> has a "Libretti" table with one row per libretto version:
      <language name> link, icons star.png (= original libretto) and
      blue-documents.png (= version used in the side-by-side edition), authors.
  * Single-language pages /<Opera>_libretto_<Language> come in two layouts:
      A. "lc-wrap": one page, <table class="lc-wrap">, one <tr> per text chunk.
      B. legacy per-act pages: /<Opera>_libretto_<Language> is the cast page
         (= _Act_0) with a <nav class="act-nav"> linking _Act_1.._Act_N; text sits
         in <tr valign=top><td valign=top> cells of a plain table.
  * The bilingual pages /<Opera>_libretto_<L1>_<L2> are now a gated preview
    (registration / payment for the full text), so this script never uses them.
    Instead it fetches the two single-language pages of the same edition and
    pairs them chunk-by-chunk (the site renders both from the same row structure),
    then block-by-block inside each chunk, verifying parity as it goes.

Blocks inside a chunk are separated by a blank line (<br /><br />), exactly like
the downstream text format. Bold/italic markup is dropped; line breaks are kept.

Parity: chunks pair 1:1 (same row structure) and blocks inside a chunk pair 1:1
when the counts match. Where the two editions disagree on where a blank line goes
(typically a stage direction glued to the previous block on one side, or an extra
note such as "Symphony No.38" on one side only), the blocks of that chunk are
aligned by a small DP on structural similarity and the odd block is merged into
its neighbour (newline-joined) — nothing is dropped, and every such repair is
logged as a WARNING with both sides' first lines. --strict-parity disables the
repair and writes only the source on any mismatch; translate.py is then the
fallback for the translation.

The cast list at the top (heading 'Personnages'/'Characters'/... plus the list) is
dropped by default because it is not sung; --keep-cast keeps it.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import html as htmllib
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup, Comment, NavigableString, Tag

SITE = "https://www.librettoarchive.com/"
CATALOG_URL = SITE + "Operas"
USER_AGENT = "kunstwerk-libretto-fetcher/0.1 (+https://www.youtube.com/@kunstwerk-opera)"
DEFAULT_CACHE_DIR = Path(os.environ.get("KUNSTWERK_CACHE_DIR", Path.home() / ".cache" / "kunstwerk")) / "librettoarchive"
DEFAULT_DELAY = 1.0  # seconds between live requests

# ISO-639-1 -> language name as used in the site's URLs / landing-page tables.
LANG_NAMES: Dict[str, str] = {
    "it": "Italian", "de": "German", "en": "English", "fr": "French", "ru": "Russian",
    "cs": "Czech", "cz": "Czech", "es": "Spanish", "la": "Latin", "hu": "Hungarian",
    "pl": "Polish", "pt": "Portuguese", "sv": "Swedish", "da": "Danish", "nl": "Dutch",
    "fi": "Finnish", "no": "Norwegian", "uk": "Ukrainian", "ja": "Japanese", "zh": "Chinese",
    "el": "Greek", "tr": "Turkish", "ro": "Romanian", "bg": "Bulgarian", "sk": "Slovak",
    "hr": "Croatian", "sl": "Slovene", "he": "Hebrew", "ca": "Catalan",
}

# Headings that introduce a cast list (see strip_cast); matched case-insensitively,
# trailing ':' ignored. The heading and the list that follows it are dropped unless
# --keep-cast is given.
CAST_HEADINGS = {
    "personaggi", "personnages", "personen", "characters", "cast", "dramatis personae",
    "действующие лица", "persons", "roles", "rôles", "osoby", "personajes", "die personen",
    "characters and cast", "cast of characters",
}


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- HTTP


class Fetcher:
    """requests wrapper with an on-disk HTML cache and a polite inter-request delay."""

    def __init__(self, cache_dir: Optional[Path], delay: float = DEFAULT_DELAY, refresh: bool = False):
        self.cache_dir = cache_dir
        self.delay = delay
        self.refresh = refresh
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last_request = 0.0
        self.live_requests = 0
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", urlparse(url).path.strip("/"))[:120] or "index"
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        return self.cache_dir / f"{slug}.{digest}.html"

    def get(self, url: str) -> str:
        cp = self._cache_path(url)
        if cp is not None and cp.exists() and not self.refresh:
            return cp.read_text(encoding="utf-8")
        wait = self.delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        log(f"  GET {url}")
        resp = self.session.get(url, timeout=60)
        self._last_request = time.monotonic()
        self.live_requests += 1
        resp.raise_for_status()
        resp.encoding = "utf-8"
        text = resp.text
        if cp is not None:
            cp.write_text(text, encoding="utf-8")
        return text


def site_url(path_or_url: str) -> str:
    """Absolute, percent-encoded URL on the site for a href/path that may contain accents."""
    u = urljoin(SITE, path_or_url)
    p = urlparse(u)
    # requote only the path; keep characters the site itself uses in slugs
    path = quote(p.path, safe="/()',_-.~!*:@&=+$;%")
    return f"{p.scheme}://{p.netloc}{path}" + (f"?{p.query}" if p.query else "")


# --------------------------------------------------------------------------- catalog


@dataclass
class CatalogEntry:
    title: str
    url: str
    year: str
    composer: str
    languages: List[str]
    side_by_side_url: Optional[str]


def parse_catalog(html: str) -> List[CatalogEntry]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        raise RuntimeError("Catalog page has no table — site layout changed?")
    entries: List[CatalogEntry] = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        a = tds[0].find("a")
        if a is None or not a.get("href"):
            continue
        title = a.get_text(" ", strip=True)
        langs_text = tds[3].get_text(" ", strip=True)
        langs_text = re.sub(r"side-by-side.*$", "", langs_text).strip()
        langs = [x.strip() for x in langs_text.split(",") if x.strip()]
        sbs = tds[3].find("a")
        entries.append(CatalogEntry(
            title=title,
            url=site_url(a["href"]),
            year=tds[1].get_text(strip=True),
            composer=tds[2].get_text(" ", strip=True),
            languages=langs,
            side_by_side_url=site_url(sbs["href"]) if sbs and sbs.get("href") else None,
        ))
    if not entries:
        raise RuntimeError("Catalog page parsed to zero operas — site layout changed?")
    return entries


def normalize_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def match_title(title: str, entries: Sequence[CatalogEntry]) -> CatalogEntry:
    q = normalize_title(title)
    if not q:
        raise SystemExit("Config 'title' is empty; set libretto_url instead.")
    scored: List[Tuple[float, CatalogEntry]] = []
    for e in entries:
        t = normalize_title(e.title)
        if t == q:
            score = 1.0
        elif t.startswith(q + " ") or q.startswith(t + " "):
            score = 0.95
        elif q in t.split(" , ") or f" {q} " in f" {t} ":
            score = 0.9
        else:
            score = difflib.SequenceMatcher(None, q, t).ratio()
        scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    best_score, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < 0.8:
        cands = ", ".join(f"{e.title!r} ({s:.2f})" for s, e in scored[:5])
        raise SystemExit(
            f"Could not find {title!r} in the librettoarchive.com catalog (best guesses: {cands}).\n"
            f"Set 'libretto_url' in the config to the opera's page, e.g. {SITE}Carmen "
            f"(run with --list-catalog to see all titles)."
        )
    if best_score < 1.0 and runner_up >= best_score - 0.05:
        cands = ", ".join(f"{e.title!r} ({s:.2f})" for s, e in scored[:5])
        raise SystemExit(
            f"Ambiguous catalog match for {title!r}: {cands}. Set 'libretto_url' in the config."
        )
    return best


# --------------------------------------------------------------------------- landing page


@dataclass
class LibrettoVersion:
    language: str           # language name as shown on the site ("French")
    url: Optional[str]      # None when the row has no online page
    is_original: bool       # star icon
    is_side_by_side: bool   # blue-documents icon: the version used in the bilingual edition
    authors: str


@dataclass
class OperaPage:
    title: str
    url: str
    versions: List[LibrettoVersion]
    side_by_side_url: Optional[str]


def parse_landing(html: str, url: str) -> OperaPage:
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else url
    title = re.sub(r"\s*♡\s*$", "", title)
    m = re.match(r"^[“\"](.+?)[”\"]", title)
    if m:
        title = m.group(1)

    heading = None
    for h in soup.find_all(["h2", "h3"]):
        if h.get_text(strip=True).lower().startswith("libretti"):
            heading = h
            break
    versions: List[LibrettoVersion] = []
    sbs_url = None
    if heading is not None:
        for sib in heading.next_siblings:
            if isinstance(sib, Tag) and sib.name in ("h2", "h3"):
                break
            if not isinstance(sib, Tag):
                continue
            for a in sib.find_all("a"):
                href = a.get("href") or ""
                if "_libretto_" in href and re.search(r"_libretto_[A-Za-z]+_[A-Za-z]+$", href) and sbs_url is None:
                    sbs_url = site_url(href)
            tables = [sib] if sib.name == "table" else sib.find_all("table")
            for table in tables:
                for tr in table.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) < 2:
                        continue
                    cell = tds[1]
                    a = cell.find("a")
                    icons = [os.path.basename(img.get("src", "")) for img in cell.find_all("img")]
                    versions.append(LibrettoVersion(
                        language=cell.get_text(" ", strip=True),
                        url=site_url(a["href"]) if a and a.get("href") else None,
                        is_original="star.png" in icons,
                        is_side_by_side="blue-documents.png" in icons,
                        authors=tds[2].get_text(" ", strip=True) if len(tds) > 2 else "",
                    ))
    if not versions:
        # Fall back to any libretto links on the page.
        seen = set()
        for a in soup.find_all("a"):
            href = a.get("href") or ""
            if "_libretto_" in href and not re.search(r"_libretto_[A-Za-z]+_[A-Za-z]+", href):
                u = site_url(href)
                if u not in seen:
                    seen.add(u)
                    versions.append(LibrettoVersion(a.get_text(" ", strip=True), u, False, False, ""))
    return OperaPage(title=title, url=url, versions=versions, side_by_side_url=sbs_url)


def pick_version(page: OperaPage, lang_name: str, role: str) -> Optional[LibrettoVersion]:
    """role = 'source' prefers the original-libretto row; 'translation' prefers the
    row flagged as part of the side-by-side edition. Returns None if the language is absent."""
    cands = [v for v in page.versions if v.url and v.language.casefold() == lang_name.casefold()]
    if not cands:
        return None
    if role == "source":
        cands.sort(key=lambda v: (not v.is_original, not v.is_side_by_side))
    else:
        cands.sort(key=lambda v: (not v.is_side_by_side, not v.is_original))
    return cands[0]


# --------------------------------------------------------------------------- libretto pages


@dataclass
class Chunk:
    """One site row (lc-wrap <tr>, or one legacy <td valign=top>); blocks are blank-line separated."""
    blocks: List[str]
    source: str = ""     # page url, for diagnostics
    kind: str = ""       # lc-act / lc-sec / lc-stage / ""


def cell_to_blocks(cell: Tag) -> List[str]:
    """Render a table cell to text the way a browser copy-paste would: <br> -> newline,
    inline tags dropped, entities decoded; then split into blank-line separated blocks."""
    cell = BeautifulSoup(str(cell), "lxml").find(cell.name)  # private copy
    for tag in cell.find_all(["img", "script", "style"]):
        tag.decompose()
    # HTML whitespace collapsing: newlines in the source text mean nothing, only <br> does.
    for node in list(cell.find_all(string=True)):
        if isinstance(node, Comment):
            node.extract()
        elif isinstance(node, NavigableString):
            node.replace_with(re.sub(r"\s+", " ", str(node)))
    for br in cell.find_all("br"):
        br.replace_with("\n")
    for tag in cell.find_all(["p", "div", "li", "tr", "h1", "h2", "h3", "h4", "act"]):
        # block-level elements (and the site's custom <act>) end a line
        tag.append("\n")
    text = cell.get_text("")
    text = htmllib.unescape(text).replace("\xa0", " ").replace("\r", "")
    lines = [re.sub(r"[ \t]+$", "", ln.strip(" \t")) for ln in text.split("\n")]
    blocks: List[str] = []
    cur: List[str] = []
    for ln in lines:
        if ln == "":
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def detect_layout(soup: BeautifulSoup) -> str:
    if soup.find("table", class_="lc-wrap"):
        return "lc-wrap"
    if soup.find("nav", class_="act-nav") or soup.find("act"):
        return "acts"
    return "unknown"


def is_gated_preview(html: str) -> bool:
    return "viewing a preview" in html or "Continue reading" in html


def parse_lc_wrap(soup: BeautifulSoup, url: str) -> List[Chunk]:
    table = soup.find("table", class_="lc-wrap")
    chunks: List[Chunk] = []
    for tr in table.find_all("tr", recursive=False) or table.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if not tds:
            continue
        td = tds[0]  # single-language page: one cell per row
        if td.find(class_="libretto-author") or "libretto-author" in (td.get("class") or []):
            continue
        blocks = cell_to_blocks(td)
        if not blocks:
            continue
        classes = td.get("class") or []
        kind = next((c for c in classes if c.startswith("lc-")), "")
        chunks.append(Chunk(blocks=blocks, source=url, kind=kind))
    return chunks


def parse_act_page(soup: BeautifulSoup, url: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    main = soup.find(class_="libretto-main") or soup
    for td in main.find_all("td"):
        if td.find("td"):
            continue  # nested table container
        classes = td.get("class") or []
        if "libretto-author" in classes or td.find(class_="libretto-author"):
            continue
        if td.find("nav", class_="act-nav"):
            continue
        blocks = cell_to_blocks(td)
        if not blocks:
            continue
        chunks.append(Chunk(blocks=blocks, source=url))
    return chunks


def act_urls(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Ordered list of per-act URLs from the first act-nav (the current act has no <a>)."""
    nav = soup.find("nav", class_="act-nav")
    if nav is None:
        return [base_url]
    urls: List[str] = []
    for item in nav.find_all(class_="act-nav__item"):
        if item.name == "a" and item.get("href"):
            urls.append(site_url(item["href"]))
        else:
            urls.append(base_url)  # the current page
    # de-dup while keeping order
    out: List[str] = []
    for u in urls:
        if u not in out:
            out.append(u)
    return out


def fetch_libretto(fetcher: Fetcher, url: str) -> Tuple[str, List[Chunk]]:
    """Returns (layout, chunks) for a single-language libretto page (all acts)."""
    html = fetcher.get(url)
    if is_gated_preview(html):
        raise SystemExit(f"{url} is a gated preview (bilingual pages need an account); use the single-language pages.")
    soup = BeautifulSoup(html, "lxml")
    layout = detect_layout(soup)
    if layout == "lc-wrap":
        return layout, parse_lc_wrap(soup, url)
    if layout == "acts":
        chunks: List[Chunk] = []
        for u in act_urls(soup, url):
            page_soup = soup if u == url else BeautifulSoup(fetcher.get(u), "lxml")
            chunks.extend(parse_act_page(page_soup, u))
        return layout, chunks
    raise SystemExit(f"Unrecognised libretto page layout at {url} (no lc-wrap table and no act navigation).")


_ACT_START_RE = re.compile(
    r"^(sinfonia|overture|ouverture|ouvertüre|vorspiel|prelude|prélude|preludio|introduzione|introduction|"
    r"einleitung|увертюра|вступление|интродукция|atto|act|akt|acte|aufzug|действие|prologo|prologue|prolog|пролог|"
    r"erster|zweiter|premier|primo|parte|teil|scena|scene|szene|картина)\b",
    re.IGNORECASE,
)


def strip_cast(chunks: List[Chunk]) -> Tuple[List[Chunk], List[str]]:
    """Drop the cast list at the top of a libretto (it isn't sung, and the hand-made
    libretti in libretti/ start at the first act or overture).

    The cast heading ('Personnages', 'Characters', ...) is looked for in the first few
    blocks. Everything before it (e.g. a premiere note) and the heading itself go; then
      * lc-wrap pages usually put the heading in a chunk of its own and the cast in the
        next chunk -> that whole next chunk goes;
      * otherwise the cast shares the heading's chunk -> the rest of that chunk goes,
        except a trailing run of act/overture headings ('Sinfonia') which is kept.
    Returns (chunks, previews of dropped blocks)."""
    flat = [(ci, bi) for ci, c in enumerate(chunks[:3]) for bi in range(len(c.blocks))][:6]
    hit = next(((ci, bi) for ci, bi in flat if chunks[ci].blocks[bi].strip().rstrip(":").casefold() in CAST_HEADINGS), None)
    if hit is None:
        return chunks, []
    ci, bi = hit
    out = [Chunk(list(c.blocks), c.source, c.kind) for c in chunks]
    dropped: List[str] = []
    for c in out[:ci]:
        dropped.extend(c.blocks)
        c.blocks = []
    head = out[ci]
    if bi == len(head.blocks) - 1:
        dropped.extend(head.blocks)
        head.blocks = []
        if ci + 1 < len(out):
            dropped.extend(out[ci + 1].blocks)
            out[ci + 1].blocks = []
    else:
        rest = head.blocks[bi + 1:]
        keep_from = len(rest)
        while keep_from > 0 and _ACT_START_RE.match(rest[keep_from - 1]) and "\n" not in rest[keep_from - 1]:
            keep_from -= 1
        dropped.extend(head.blocks[:bi + 1] + rest[:keep_from])
        head.blocks = rest[keep_from:]
    return [c for c in out if c.blocks], [b.splitlines()[0][:50] for b in dropped]


# --------------------------------------------------------------------------- pairing


class ParityError(Exception):
    pass


def block_kind(b: str) -> str:
    """Coarse structural class of a block, used as the alignment signature:
    P = parenthetical stage direction, S = speaker line + text, H = short one-line heading, T = other."""
    lines = b.split("\n")
    first = re.sub(r"\s*\(.*$", "", lines[0]).strip()
    if b.startswith("(") and b.rstrip().endswith(")"):
        return "P"
    if len(lines) > 1 and first and first == first.upper() and re.search(r"[^\W\d_]", first):
        return "S"
    if len(lines) == 1 and len(first) <= 60:
        return "H"
    return "T"


def _first_line_key(b: str) -> str:
    first = re.sub(r"\s*\(.*$", "", b.split("\n", 1)[0]).strip()
    return normalize_title(first)


def block_sim(src: List[str], tr: List[str]) -> float:
    """Similarity in [0,1] between a (possibly merged) source block and translation block:
    same structural kind, similar first line (speaker names are often identical or
    cognate across languages: FIGARO/FIGARO, CONTE/COUNT), similar line count."""
    # kind and first line come from the first block of each side (a merged run starts with
    # it); the line count is that of the whole run, so k:1 merges pay for being long.
    kind = 1.0 if block_kind(src[0]) == block_kind(tr[0]) else 0.0
    fs, ft = _first_line_key(src[0]), _first_line_key(tr[0])
    name = difflib.SequenceMatcher(None, fs, ft).ratio() if fs and ft else 0.5
    s, t = "\n".join(src), "\n".join(tr)
    ls, lt = s.count("\n") + 1, t.count("\n") + 1
    lines = 1.0 - abs(ls - lt) / max(ls, lt)
    return 0.35 * kind + 0.35 * name + 0.30 * lines


def chunk_sim(src: List[Chunk], tr: List[Chunk]) -> float:
    sb = [b for c in src for b in c.blocks]
    tb = [b for c in tr for b in c.blocks]
    count = 1.0 - abs(len(sb) - len(tb)) / max(len(sb), len(tb))
    return 0.5 * count + 0.5 * block_sim(sb[:1], tb[:1])


def dp_align(a: list, b: list, sim, max_merge: int = 3, merge_penalty: float = 0.25) -> List[Tuple[list, list]]:
    """Monotone alignment of two item lists maximising summed similarity, where one
    item may pair with up to max_merge consecutive items on the other side (k:1 or 1:k).
    Nothing is dropped. Returns groups (a_items, b_items), each side non-empty."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return [(list(a), list(b))] if (a or b) else []
    NEG = float("-inf")
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    back: List[List[Optional[Tuple[int, int]]]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            for di in range(1, max_merge + 1):
                for dj in range(1, max_merge + 1):
                    if (di > 1 and dj > 1) or i - di < 0 or j - dj < 0:
                        continue
                    prev = best[i - di][j - dj]
                    if prev == NEG:
                        continue
                    score = prev + sim(a[i - di:i], b[j - dj:j]) - merge_penalty * (di + dj - 2)
                    if score > best[i][j]:
                        best[i][j] = score
                        back[i][j] = (di, dj)
    if best[n][m] == NEG:  # too lopsided for max_merge: one group with everything
        return [(list(a), list(b))]
    groups: List[Tuple[list, list]] = []
    i, j = n, m
    while i > 0 or j > 0:
        di, dj = back[i][j]  # type: ignore[misc]
        groups.append((list(a[i - di:i]), list(b[j - dj:j])))
        i, j = i - di, j - dj
    groups.reverse()
    return groups


def _preview(blocks: List[str]) -> str:
    return repr(blocks[0].splitlines()[0][:40]) if blocks else "''"


def _describe_merges(groups: List[Tuple[list, list]]) -> List[str]:
    notes = []
    for g_src, g_tr in groups:
        if len(g_src) > 1 or len(g_tr) > 1:
            s = " + ".join(_preview([x] if isinstance(x, str) else x.blocks) for x in g_src)
            t = " + ".join(_preview([x] if isinstance(x, str) else x.blocks) for x in g_tr)
            notes.append(f"[{s}] <-> [{t}]")
    return notes


def pair_chunks(src_chunks: List[Chunk], tr_chunks: List[Chunk], strict: bool) -> Tuple[List[str], List[str], List[str]]:
    """Pair source and translation chunk lists into two equal-length block lists.
    Chunks pair 1:1 when counts match, otherwise by structural signature; inside a
    chunk pair, blocks pair 1:1 when counts match, otherwise by block_kind signature
    and leftover blocks are merged into their neighbour (newline-joined) so that
    parity holds without dropping text. Returns (src_blocks, tr_blocks, warnings);
    raises ParityError in strict mode on any mismatch."""
    warnings: List[str] = []
    ns = sum(len(c.blocks) for c in src_chunks)
    nt = sum(len(c.blocks) for c in tr_chunks)

    if len(src_chunks) == len(tr_chunks):
        chunk_pairs = [([s], [t]) for s, t in zip(src_chunks, tr_chunks)]
    else:
        msg = (f"chunk counts differ: {len(src_chunks)} source vs {len(tr_chunks)} translation "
               f"({ns} vs {nt} blocks)")
        if strict:
            raise ParityError(msg)
        chunk_pairs = dp_align(src_chunks, tr_chunks, chunk_sim)
        notes = _describe_merges(chunk_pairs)
        warnings.append(f"{msg} — aligned chunks by similarity; merged chunk groups: " + "; ".join(notes[:20]))

    src_out: List[str] = []
    tr_out: List[str] = []
    repaired = 0
    for idx, (s_chunks, t_chunks) in enumerate(chunk_pairs):
        sb = [b for c in s_chunks for b in c.blocks]
        tb = [b for c in t_chunks for b in c.blocks]
        if len(sb) == len(tb):
            src_out.extend(sb)
            tr_out.extend(tb)
            continue
        if strict:
            raise ParityError(f"chunk {idx} ({_preview(sb)}): {len(sb)} source blocks vs {len(tb)} translation blocks")
        repaired += 1
        groups = dp_align(sb, tb, block_sim)
        for g_src, g_tr in groups:
            src_out.append("\n".join(g_src))
            tr_out.append("\n".join(g_tr))
        warnings.append(f"chunk {idx} ({_preview(sb)}): {len(sb)} vs {len(tb)} blocks — merged " + "; ".join(_describe_merges(groups)))
    if repaired:
        warnings.append(f"{repaired} of {len(chunk_pairs)} chunks needed block-level repair (see above)")
    assert len(src_out) == len(tr_out)
    return src_out, tr_out, warnings


# --------------------------------------------------------------------------- main


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path} did not parse to a mapping")
    return cfg


def lang_name(code: str) -> str:
    code = (code or "").strip()
    if code.casefold() in LANG_NAMES:
        return LANG_NAMES[code.casefold()]
    if len(code) > 2:
        return code[:1].upper() + code[1:]  # assume a language name was given
    raise SystemExit(f"Unknown language code {code!r}; add it to LANG_NAMES in fetch_libretto.py")


def write_text(path: Path, blocks: List[str], force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite {path} (use --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks), encoding="utf-8")
    log(f"  wrote {path}  ({len(blocks)} blocks)")


def resolve_opera(fetcher: Fetcher, cfg: dict) -> Tuple[Optional[OperaPage], Optional[str]]:
    """Returns (landing page, explicit source-libretto url). Exactly one is non-None
    unless libretto_url pointed straight at a libretto page, in which case the landing
    page is still loaded (for the translation) when it can be inferred."""
    explicit = cfg.get("libretto_url")
    if explicit:
        url = site_url(str(explicit).strip())
        if "_libretto_" in url:
            landing_url = site_url(urlparse(url).path.split("_libretto_")[0])
            log(f"libretto_url points at a libretto page; landing page inferred as {landing_url}")
            try:
                page = parse_landing(fetcher.get(landing_url), landing_url)
            except requests.HTTPError:
                page = None
            return page, url
        log(f"Using libretto_url from config: {url}")
        return parse_landing(fetcher.get(url), url), None
    title = cfg.get("title")
    if not title:
        raise SystemExit("Config has neither 'title' nor 'libretto_url'")
    entries = parse_catalog(fetcher.get(CATALOG_URL))
    entry = match_title(str(title), entries)
    log(f"Catalog match for {title!r}: {entry.title!r} ({entry.composer}, {entry.year}) -> {entry.url}  languages: {', '.join(entry.languages)}")
    return parse_landing(fetcher.get(entry.url), entry.url), None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="configs/<opera>.yaml")
    ap.add_argument("--out-dir", default="libretti", help="where to write <prefix>_<lang>.txt (default: libretti/)")
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    ap.add_argument("--translation-language", help="override translation_language from the config (ISO-639-1)")
    ap.add_argument("--no-translation", action="store_true", help="only fetch the source-language libretto")
    ap.add_argument("--keep-cast", action="store_true", help="keep the cast list (heading + characters) at the top")
    ap.add_argument("--strict-parity", action="store_true",
                    help="never repair chunk-level block mismatches; on any mismatch write only the source")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help=f"raw HTML cache (default: {DEFAULT_CACHE_DIR})")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write the HTML cache")
    ap.add_argument("--refresh", action="store_true", help="ignore cached HTML (but still write it)")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="seconds between live requests")
    ap.add_argument("--list-catalog", action="store_true", help="print the site catalog and exit")
    args = ap.parse_args(argv)

    fetcher = Fetcher(None if args.no_cache else Path(args.cache_dir), delay=args.delay, refresh=args.refresh)

    if args.list_catalog:
        for e in parse_catalog(fetcher.get(CATALOG_URL)):
            print(f"{e.title} | {e.composer} | {e.year} | {', '.join(e.languages)} | {e.url}")
        return 0
    if not args.config:
        ap.error("config is required (or use --list-catalog)")

    cfg = load_config(args.config)
    prefix = cfg.get("file_prefix")
    src_code = cfg.get("language")
    if not prefix or not src_code:
        raise SystemExit("Config needs 'file_prefix' and 'language'")
    tr_code = args.translation_language or cfg.get("translation_language") or "en"
    src_name = lang_name(src_code)
    tr_name = lang_name(tr_code)
    out_dir = Path(args.out_dir)
    src_path = out_dir / f"{prefix}_{src_code}.txt"
    tr_path = out_dir / f"{prefix}_{tr_code}.txt"
    if not args.force:
        for p in ([src_path] + ([] if args.no_translation else [tr_path])):
            if p.exists():
                raise SystemExit(f"{p} already exists; use --force to overwrite")

    page, explicit_src_url = resolve_opera(fetcher, cfg)
    if page is not None:
        log(f"Opera page: {page.title!r} {page.url}")
        for v in page.versions:
            flags = ("original " if v.is_original else "") + ("side-by-side " if v.is_side_by_side else "")
            log(f"  version: {v.language:10s} {flags:22s} {v.url or '(no page)'}  {v.authors}")
        if page.side_by_side_url:
            log(f"  bilingual page (gated preview, not used): {page.side_by_side_url}")

    # ---- source
    if explicit_src_url:
        src_url = explicit_src_url
    else:
        v = pick_version(page, src_name, "source")
        if v is None:
            avail = ", ".join(sorted({x.language for x in page.versions if x.url})) or "none"
            raise SystemExit(f"No {src_name} libretto for {page.title!r} on the site (available: {avail}).")
        if not v.is_original:
            log(f"  note: the {src_name} version is not flagged as the original libretto")
        src_url = v.url
    log(f"Source libretto ({src_name}): {src_url}")
    layout, src_chunks = fetch_libretto(fetcher, src_url)
    log(f"  layout={layout}  chunks={len(src_chunks)}  blocks={sum(len(c.blocks) for c in src_chunks)}")
    if not args.keep_cast:
        src_chunks, dropped = strip_cast(src_chunks)
        if dropped:
            log(f"  dropped cast list ({len(dropped)} blocks: {dropped[0]!r} … {dropped[-1]!r}); --keep-cast to keep it")

    # ---- translation
    tr_blocks: Optional[List[str]] = None
    src_blocks = [b for c in src_chunks for b in c.blocks]
    if not args.no_translation and tr_code != src_code:
        tv = pick_version(page, tr_name, "translation") if page is not None else None
        if tv is None:
            avail = ", ".join(sorted({x.language for x in page.versions if x.url})) if page else "?"
            log(f"No {tr_name} translation on the site (available: {avail}); writing source only — use translate.py.")
        else:
            if not tv.is_side_by_side:
                log(f"  note: the {tr_name} version is not flagged as part of the side-by-side edition; parity is less likely")
            log(f"Translation ({tr_name}): {tv.url}")
            tr_layout, tr_chunks = fetch_libretto(fetcher, tv.url)
            log(f"  layout={tr_layout}  chunks={len(tr_chunks)}  blocks={sum(len(c.blocks) for c in tr_chunks)}")
            if not args.keep_cast:
                tr_chunks, dropped = strip_cast(tr_chunks)
                if dropped:
                    log(f"  dropped cast list ({len(dropped)} blocks: {dropped[0]!r} … {dropped[-1]!r})")
            try:
                src_blocks, tr_blocks, warnings = pair_chunks(src_chunks, tr_chunks, strict=args.strict_parity)
                for w in warnings:
                    log(f"  WARNING: {w}")
            except ParityError as e:
                log(f"  PARITY FAILURE: {e}")
                log(f"  Writing the source only; run translate.py {args.config} {tr_code} for the translation.")
                tr_blocks = None
                src_blocks = [b for c in src_chunks for b in c.blocks]

    if not src_blocks:
        raise SystemExit("Parsed zero blocks from the source libretto — refusing to write an empty file.")
    write_text(src_path, src_blocks, args.force)
    if tr_blocks is not None:
        assert len(tr_blocks) == len(src_blocks)
        write_text(tr_path, tr_blocks, args.force)
        log(f"Done: {len(src_blocks)} blocks in both {src_path.name} and {tr_path.name} (parity verified).")
    else:
        log(f"Done: {len(src_blocks)} blocks in {src_path.name}; no translation written.")
    log(f"({fetcher.live_requests} live requests; cache: {fetcher.cache_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
