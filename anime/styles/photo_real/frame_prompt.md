You are composing frame-by-frame shot ideas for a single coherent live‑action film adaptation of this story. Given the fragment below, propose up to {max_per_window} plausible photographic film stills — frames that look like they were captured mid‑scene from one continuous production.

Global Photography Style — apply consistently across ALL frames in this session:
{global_style}

Requirements (photographic stills, not posters):
- Each idea is a single captured moment (no multi-beat sequences, no montage).
- Be literal to the described action; prefer stage directions and diegetic behavior over abstract symbolism.
- Favor production‑plausible cinematography and blocking; no impossible physics or floating cameras.
- Describe what the camera captures (framing, action, mood) rather than instructions.
- Keep continuity with the Global Photography Style (color science, lighting character, lens choice/DOF, era/format, texture/grain).
- Return ONLY a JSON array of objects with keys: title, description, shot, time_of_day, lighting, palette, characters (array), tags (array), nsfw (boolean).
- The 'description' must be ONE sentence that fully describes the frame. Do NOT include any show or story title.
- Avoid repeating essentially the same shot within this window.

Character cards (if provided):
- Available character reference images (by filename stem) will be listed below.
- When relevant, set the 'characters' array to the exact names from this list (filename stem).
- Downstream image generation will attach the matching references — keep designs consistent and avoid reinventing looks.

Story fragment (blocks, preserve line breaks):
---
{fragment}
---

