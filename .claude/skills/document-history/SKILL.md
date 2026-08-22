---
description: Document the history of our work in this repo.
---

Consider the body of our work in this repo. I'd like you to use git to go through the entire history of our repo or sub-folder within the repo and document the broad-scale shape of our work across this entire work process.

There's a LOT of content to read that needs to be summarized, so please orchestrate as many subagents as is appropriate to recursively learn about the entire history of our work without losing important content and without overflowing your context. Our process generally involves documenting our work at regular check points to "main READMEs" and "auxiliary READMEs" per experiment, so you and/or the subagents ought to be able to get a pretty complete picture primarily by reading already-written docs rather than piecing things together from code, though looking at the code is certainly allowed. Please be sure to cite references to READMEs and sub-READMEs via filepath links wherever appropriate so that future agents know where to look to get more info on certain developments.

Your job is not to document the raw process in its day-to-day, but more to read between the lines in tracing our thought process over time and the priors we currently have/have updated. Think of yourself less as a specialist historian of any particular thread in our work and more as a "Big History"-style historian who excels at finding the subtle shifts that were indicative of the big narratives.

When you're done, please write your history to a Markdown doc in the appropriate directory called HISTORY.md.

This skill may be run from a cron job such that the history auto-updates, and as such, a HISTORY.md file may already exist in the relevant directory. If this file already exists, do NOT read it: we want a fresh view of the big picture of this work uninfluenced by other agents' priors. Rather, first `rm` the file entirely before writing from scratch.

All things considered, your history doc should be no more than 200 lines. If it winds up being longer, try trimming any unnecessary narrative that may have been written.