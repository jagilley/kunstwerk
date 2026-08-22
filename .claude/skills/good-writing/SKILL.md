---
description: Our guide to good writing. Only needs to be invoked for external-facing communication, not internal. Don't invoke for writing idea docs, READMEs, beliefs docs, etc.
---

What follows is our accumulated taste about writing, aimed at technical communication. Treat it as a set of tools and named failure modes rather than a checklist to run top to bottom. The named patterns exist because naming them makes them faster to spot, not because every instance is a defect.

Three things to orient on before you start.

**Categorize the draft first.** The guide's opening move is the load-bearing one: decide whether you're looking at something 95% of the way there or something that needs substantial revision. Those call for very different responses, and making no edits is a valid outcome. Say which bucket you landed in.

**Weight the rules by medium.** Much of this was written for YouTube narration, where load-bearing punctuation genuinely fails because a listener can't re-read. For text that will be read on a page — a paper, a README, a slide deck — the em-dash and colon rules apply with much less force, and slides in particular are telegraphic by genre. The critical-thinking and precision rules apply everywhere, and they matter most. If you're scoring a piece, say which axis you're scoring on, because a draft can be thought well and shaped monotonously at the same time.

**Quote the line.** A finding should name the specific text, name the pattern, and say what's load-bearing in it that any rewrite has to preserve. "This is vague" isn't actionable. When you're unsure whether a bold claim is earned, ask rather than flattening it; over-cutting is the more common failure here, and the final notes below are explicit about that.

---

# Good Writing

This is a guide to writing well, with a specific focus on technical communication.

Some documents that you read will already be well-written, some will be less well-written. The first order of business is to briefly mentally categorize whether you're looking at a quality piece of writing that's 95% of the way there with a few edits to be made, or one that needs substantial revision and skepticism. Making no edits is okay, when none need to be made!

In general, writing clocks as slop when the writer doesn't think richly enough about what they're saying. The solution is simply to do more research or think harder about how to add more specificity and elegance to what's being written.

## Tools

### Critical thinking

This is the number one line of defense for our use cases. The worst form of slop is not stylistic; it's caused by the sort of imprecise language that arises when any writer, AI or human, doesn't fully understand what they're talking about but needs to produce an output regardless.

This can take the form of vagueness, triteness, imprecision, etc. Good writing is pithy and good writers can use very specific, evocative words because they understand the source material very well, whatever it may be.

Note that this doesn't mean cutting down what the video is trying to say! It's easy to flag statements that make a somewhat bold claim as being vague. Whether or not that bold claim is valid and appropriate depends on the context of whether it's been properly argued for in full and how much it resonates with the story that the user is trying to tell.

It's much better to pause the writing process entirely and flag that something is not well understood than it is to gloss over a lack in understanding with truisms.

### Good writing is conversational

Good writing, whether it takes the form of a speech, YouTube video, or essay, feels like a conversation with the author. "Just tell the reader what you think." That doesn't mean it's colloquial, necessarily, but it often means thinking of the piece as a direct conversation between the author and the reader rather than as an object in and of itself that gets constructed by the author.

### Named anti-patterns

These are the most common failure modes in narration text. Naming them makes them faster to spot.

- **This isn't X, it's Y**: The single most annoying pattern to me because this sort of truism almost always obscures real insight (why would one have thought X in the first place? can it *really* be said to be entirely Y?). With this one, it doesn't matter if the content is contextually appropriate. There is always a better way of phrasing it. Examples:
    - "This isn't just a research paper — it's a new paradigm."
    - "Grokking isn't a flash of insight. It's a sculptor chipping away stone — the statue was taking shape the whole time."
    - It hasn't learned how modular addition works. It's built a lookup table.
    - "This didn't just X, it Y"
- **Tricolons**: bad taste imo. There's got to be a better way of rephrasing this that's less hackneyed.
    - "No free parameters. No curve fitting. Just the structure of language itself."
- **Unnecessary rhetorical questions**: These don't read well when someone is narrating in a YouTube video since they disrupt the stream of speaking. Additionally, they're just poor taste.
    - "But test accuracy? One percent."
    - "Three plus five? Start at three, rotate five steps — land on eight. Ninety plus forty? Rotate past zero, land on thirty-three."
- **Evaluative tells**: "this is a striking result." Tells the audience how to feel instead of giving them grounds to feel it. Not the worst thing in the world, but can only be invoked once per essay/video. You've got to be sure this is the best place to use it. If not, don't necessarily just cut the content; find a better way to discuss what was trying to be conveyed.
- **Empty intensifiers**: "learning becomes dramatically harder", "is incredibly important." This is context-dependent. It's still helpful to have markers like this to tell uninformed audiences what to pay attention to, but if they're gratuitous it comes across poorly. As often as not, you can swap out the strong intensifier for a more appropriate, weaker one that still serves the same rhetorical function. (e.g., "learning becomes dramatically harder" -> "learning becomes much harder".)
- **Echo summaries**: Restating the previous sentence in slightly different words, disguised as a closing thought. Hardest to catch because each sentence reads fine in isolation — you only notice the problem reading them sequentially.
- **Evaluative vs. factual framing**: Compare: "this is a striking result" vs. "this is the first theory to derive scaling exponents from measurable properties of language." Both convey importance, but the first is the writer's opinion and the second is a verifiable claim that lets the audience draw their own conclusion. The factual version is almost always stronger. The fix is usually mechanical: replace the evaluative adjective with the specific factual basis for the evaluation.
- **Monotone-ness**: this can especially surface w.r.t. comma usage. "garbled, repetitive, or incoherent output" plus a bunch of other three-comma phrases starts to feel sleepy after a while. Vary the onomatopoeia of your writing! Otherwise, what's the point?
- **Unnecessary imprecision in the name of layperson legibility**:
    - "A small network, trained on a simple math problem, independently discovered that modular arithmetic is rotation on a circle, and it did so because that's what modular addition is." -> okay, modular arithmetic is not literally defined as rotation on a circle, it's isomorphic to that structure. Would be better phrased as "A small network, trained on modular addition, independently learned to represent it using rotations on a circle, which makes sense, because modular addition wraps around in exactly the same way that rotation does."
- **Load-bearing punctuation**: we're making videos for YouTube, so everything will be narrated. Punctuation with semantic meaning that strongly modifies its surroundsings just won't come through — think colons, em-dashes, rhetorical questions, semicolons, etc. If it can't be said aloud without confusing somebody who's listening on headphones, it's bad practice. There's a strong argument to be made that this is bad writing practice even for written language because it deviates from the morphology of oration, which remains the most compelling form of language to human ears. For content that's designed to be spoken aloud, we should probably avoid em-dashes entirely — but note that the fix needs to be structural (just replacing the em-dashes with a period doesn't change anything.)
    - "The V vector defines what it reads; the U vector defines what it writes. Together: 'when I see this pattern, I contribute this to the next computation.'" -> the colon following 'together' is load-bearing, as in you have to literally read the text to catch the meaning.
- **Em-dashes**: I find that there is almost always a better way of phrasing things than using an em-dash. It is almost invariably a form of load-bearing punctuation that only comes through in writing, and even then it weakens the persuasiveness of what's being written by its non-conversational nature. Em-dashes are almost always semantically loaded, meaning that you would usually be better off explicitly saying in words what the em-dash would otherwise tell the reader. This also means that just replacing em-dashes with periods is likely to create disjoint narration as well. You're almost always best off just thinking about what you're really trying to say and saying that instead.
- **Un-bridged phrases**: this can sometimes result from greedily removing "filler" content without regard to the natural flow of the oration. This is problematic because it effectively makes the next period a form of load-bearing punctuation (you have to see the period to understand that there is a semantic shift coming in the next sentence.) LLMs may be particularly vulnerable to this because they can make sense of words in context more easily than humans can. The human neocortex doesn't have O(n^2) attention - things should flow linearly.
    - "None of them needed new data. They needed a different way of looking at what was already there." -> should be "None of them needed new data. Rather, they needed a different way of looking at what was already there."

### What NOT to fix

Conveying importance and emotional intensity is sometimes a real part of good science communication — not every evaluative phrase is slop. "This is remarkable" can be exactly right when the result genuinely warrants it. The failure mode isn't emphasis itself; it's *unearned* emphasis — intensity that substitutes for specificity, or that claims more than the evidence supports. When the factual basis for the claim is already present in the surrounding sentences, the evaluative phrase is redundant. When it isn't — when the audience needs a signal that something is significant and the why is implicit or cumulative — a clear, proportionate statement of importance does real work. It's OK to say things like "deep structure", "the true insight", "a fundamental truth" etc. as long as these claims are backed up appropriately and aren't hand-wavey. We are, after all, trying to communicate important truths that very few people believe.

Also, slop-guard may sometimes flag words as being slop words, but it doesn't know if that word is genuinely appropriate in context or just being thrown around for the sake of being thrown around. It's up to you to decide.

### Good taste

There's a certain sense in which you know it when you see it when it comes to good writing. Use your subjective judgment at your discretion. You've seen all the text on the internet, so you almost certainly have a strong prior for what constitutes good writing, even if not all writing on the internet is good and it's hard to access that prior sometimes. 

### slop-guard

You can use the slop-guard CLI as an automated, hardcoded check against certain bad writing patterns. Invoke with e.g. `sg -v draft.md` for detailed advice.

**Run it after you've formed your own view, not before.** It only sees mechanical patterns. Put it first and it anchors you onto the mechanical axis, and you'll produce a critique that's all punctuation counts and never reaches the places where the writer didn't understand their own material. Those are the findings that matter most, and slop-guard is structurally incapable of finding them.

**Use the per-instance hits and ignore the aggregate score.** The score is dominated by density metrics that saturate, so a piece with tidy punctuation and nothing to say will outscore a dense, well-reasoned piece that leans on one construction too often. A low score tells you the prose runs one rhythm. It says nothing about whether the prose is any good.

Know its blind spots too. It matches the comma form of "X, not Y" but not the inversion ("Not X: Y"), so the most load-bearing instance of that pattern is often the one it misses. Work through its output as a checklist rather than acting on it as a verdict. Some of what it flags will be doing real conceptual work.

In the context of EG scripts, not all outputs matter. Many Markdown files will clock as slop simply because LLMs love to write in Markdown. The only thing that matters when desloppifying scripts is the text that will actually be spoken verbally.

### Video script-specific patterns

These apply specifically to narration scripts (as opposed to written prose).

- **Press-release framing**: "The researchers applied X" or "The team at Goodfire demonstrated Y" makes the video sound like it's reporting on someone else's work rather than teaching the viewer. Prefer "the paper" or "VPD" as the agent, or better, frame it as something we're exploring together ("Today we're going to break down..."). The video is a lesson, not a news segment.
- **Commanding the viewer**: "Think of it this way." "Notice how..." "Consider the following." These imperatives cast the narrator as a lecturer issuing instructions. Prefer invitations that acknowledge the viewer's agency: "You might draw the analogy that..." or just state the analogy directly. The difference is subtle but it's the difference between talking *at* someone and talking *with* them.
- **Unmarked epistemic shifts**: When narration moves from describing real results to a thought experiment or hypothetical, it needs an explicit marker ("Hypothetically," "Imagine a decomposition that..."). Without it, the viewer can't tell whether something actually happened or is being imagined for pedagogical purposes. Written text gets away with this more because readers can re-read; listeners can't rewind their ears.

### Final notes

- You should focus more on rephrasing content than cutting it! If it was in the original essay, chances are it was load-bearing in some fashion. Your job is to isolate that load-bearingness and emphasize it while removing the parts that aren't load-bearing.
- Make sure that your proposed edits don't suffer from the issues/anti-patterns described here. It's especially tempting to consolidate content while introducing new load-bearing punctuation and/or un-bridged phrases. Please be vigilant about avoiding this.
- If you have any doubt in your mind about what the subjective presentation and takeaways should be, please explicitly ask the user! They will likely have a pretty specific idea about the story they want to tell.