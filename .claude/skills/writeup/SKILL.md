---
description: Write up the results of an experiment or sequence of experiments to a README file
---

Follow the canonical structure in [STRUCTURE.md](../../../STRUCTURE.md) (folder shape) and [`experiments/CLAUDE.md`](../../../experiments/CLAUDE.md) (README/`FILES.md` content conventions). Before writing anything, you will need to have discussed results with Jasper — per the repo's `CLAUDE.md`, results are easy to misread, so agree on the interpretation first. If this skill is invoked after you've discussed already, you're good to go.

**Write the leaf.** A writeup is a README in a folder, not a sibling file: create `parent/<topic_snake_case>/README.md` and put the full writeup there — goal, method, tables, headline finding, reproduction command, and a pointer up to the parent README. Any Python files written in the process of this conversation should also live in this folder. (If you're converting an existing flat `TOPIC_README.md` into this form, do it as a [`/semantic-reorg`](../semantic-reorg/SKILL.md): `git mv`, fix inbound links repo-wide and outbound link depth, leave `/data/...` result paths alone.)

**Update the parent.** Add a per-experiment headline section (goal + one-line finding) to the parent `README.md` with a pointer down to `<topic>/README.md`, and add the row(s) to the parent `FILES.md` — the `## Code files` table for any new `.py`, the `## Children` table for the new folder. Keep the parent's addition to a headline, not the writeup.

**Offer to climb.** After the parent is updated, present the user the option to propagate the summary further up — the grandparent README, then the great-grandparent, and so on — for as many parent READMEs as exist. At each level up, the summary you add should roughly **halve** in length (parent = headline paragraph → grandparent = a sentence → above = a clause). Don't propagate silently or automatically; offer it, and only go as far up as the user wants. You can use the `AskUserQuestion` tool for this, but always include an 'end there and merge to main as-is' option.

If you're a subagent who doesn't directly interact with the user, you should not climb without prior authorization.

### Stylistic guidelines

Maintain appropriate epistemic humility when writing. In particular, it's easy to jump to deflationary conclusions beyond the scope of what's been tested. We should avoid doing this, and maintain a fairly narrative-agnostic record. It's certainly OK to record a result as epistemically suggestive, particularly when the evidence is strong and/or you've discussed with Jasper. The idea/beliefs system is more for recording belief updates based on the empirical results we document here.

### On super-agent/sub-agent setups

We often prompt "super-agents" as delegators to sub-agents which actually do the implementation, to increase the scope of intellectual updating which can occur in a single Claude convo (I, Jasper, want each convo to produce some possibly small but non-trivial epistemic update, and single-experiment-implementation convos sometimes don't do this.)

I generally want each write-up to also operate at the level of a single epistemic update. If you are a "super-agent," you should write a *single* write-up for the scope of the full conversation rather than for each implementer agent's work. This "super-writeup" can reference each implementer agent's work, of course, but the goal is to see the forest for the trees a bit. If your sub-agents have done multiple implementations in different folders, please move them to be children of a single super-folder, where the master super-agent README lives (it's not necessary to have individual READMEs for subagents' work.)
