You are the supervising art director for a single anime film adaptation of the opera '{opera_title}'. From the libretto excerpt below, derive a cohesive visual style guide that keeps all generated frames looking like they belong to ONE movie.

Output STRICTLY a JSON object with keys:
- short_suffix: a concise one-line style tail to append to every image prompt (avoid artist names).
- guide: a compact paragraph (3–6 lines) describing the film's art direction: palette, lighting, lens/shot feel, texture/grain, era/format (e.g., 16:9 1080p), production vibe (generic, not name-dropping living artists), and any continuity notes.

Constraints:
- Favor practical, production-plausible details (camera/lens feeling, color grading, film/scan texture, background style).
- Keep it applicable across the whole story; do not lock to a single scene.

Libretto excerpt for style inference:
---
{fragment}
---

