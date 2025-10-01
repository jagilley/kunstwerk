You are composing frame-by-frame shot ideas for a single coherent anime film adaptation of the opera '{opera_title}'. Given the libretto fragment, propose up to {max_per_window} plausible film stills — frames that look like they were captured mid-scene from one continuous production.

Global Film Style — apply consistently across ALL frames in this session:
{global_style}

Requirements (film stills, not posters):
- Each idea is a single captured frame (no multi-beat sequences, no montage).
- Be literal to the libretto; prefer stage directions and diegetic action over abstract symbolism.
- Favor grounded, production-plausible staging and camera work; avoid impossible physics or floating cameras.
- Describe the shot as it appears on screen (framing, action, mood), not as instructions.
- Keep continuity with the Global Film Style (palette, lighting character, lens feel, era, texture).
- Return ONLY a JSON array of objects with keys: title, description, shot, time_of_day, lighting, palette, characters (array), tags (array), nsfw (boolean).
- The 'description' must be ONE sentence that fully describes the frame. Do NOT include the opera title.
- Avoid repeating essentially the same shot within this window.

Character cards (if provided):
- Available character reference images (by filename stem) will be listed below.
- When relevant, set the 'characters' array to the exact names from this list (filename stem).
- Downstream image generation will attach the matching references — keep designs consistent and avoid reinventing looks.

Libretto fragment (blocks, preserve line breaks):
---
{fragment}
---

