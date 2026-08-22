---
description: Instructions for opening and merging PRs in this repo.
---

Open and merge a PR with all of the files that you would like to commit in this line of work. If there are other untracked files that are unrelated, please leave them alone and be sure that you don't accidentally delete any.

Once you have a PR open, go ahead and merge it right away, and then, if you're on Jasper's computer as opposed to a cloud environment, reset the local working directory to `main`. We generally use PRs as a tracking device rather than an actual review point.

The main possible gotcha if you're working on Jasper's computer is that other agents might possibly be working on the same working directory. This is fine, but let's treat git branches like a "file lock" of sorts where if another agent has set the local git branch to anything other than `main`, you should immediately halt your work while the other agent merges their own PR and clears out their portion of the work tree.

If you're merging from your branch to `main`, be very sure that you've pulled ToT before doing so, so you don't revert another agent's work accidentally.
