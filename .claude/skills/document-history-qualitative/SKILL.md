---
description: Document the qualitative history of our work in this repo.
---

Consider the body of our work in this repo. I'd like you to use git to go through the entire history of our repo and document the broad-scale shape of our work across this entire work process. (We migrated to this monorepo from separate sub-repos, so the initial commit will be large but only important insofar as it informs the scope of our later work.)

There's a LOT of content to read that needs to be summarized, so please orchestrate as many subagents as is appropriate to recursively learn about the entire history of our work without losing important content and without overflowing your context. Our process generally involves documenting our work at regular check points to "main READMEs" and "auxiliary READMEs" per experiment, so you and/or the subagents ought to be able to get a pretty complete picture primarily by reading already-written docs rather than piecing things together from code, though looking at the code is certainly allowed.

Your job is not to document the raw process in its day-to-day, but more to read between the lines in tracing our thought process over time and the priors we currently have/have updated. Think of yourself less as a specialist historian of any particular thread in our work and more as a "Big History"-style historian who excels at finding the subtle shifts that were indicative of the big narratives.

When you're done, please write your history to a Markdown doc in the cwd (~/Code/research) called HISTORY.md. If this file already exists, feel free to overwrite it completely (this skill may be run from a cron job such that the history auto-updates.)