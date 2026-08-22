---
description: Communicate with another Claude Code session — send a message to, consult, or ask a question of a peer/parallel session and get its reply, using a forked `claude -p --resume` RPC. Use when the user wants to talk to, message, coordinate with, or borrow context from another Claude Code session (session-to-session / intra-Claude / cross-session communication). The user typically supplies the peer's session ID.
---

You are communicating with **another Claude Code session** — a peer running in its own process with its own accumulated context. The transport is the public `claude` CLI in headless mode (`-p`), resuming the peer's session.

**Why this and not transcript-reading.** This rides *documented CLI flags*, never the internal transcript (`.jsonl`) schema. That schema is undocumented and can change on any harness update, so parsing it is fine for passive, after-the-fact reading (what `conversations/parse_transcripts.py` does) but is the wrong foundation for live comms. The CLI's `-p/--resume/--fork-session/--output-format` surface is a stable public contract — build the channel on that.

**The core model: Claude Code is turn-based and idle between turns.** A session only computes when it has an active turn; there is no inbox interrupt. So "sending a message" means *spawning a turn in the peer's context from the outside* — which is exactly what `claude -p --resume` does.

## Default: fork the peer (non-destructive)

**Default to `--fork-session`.** It clones the peer's full context onto a **new** branch, runs your message there, and **leaves the peer's real transcript untouched** (verified: the peer's `.jsonl` does not grow; a new session id is returned). You get the peer's mind — all its context and reasoning — without mutating its state or interleaving with a human who may be typing into it.

```bash
claude -p --resume <PEER_ID> --fork-session --output-format json \
  "<your message>" < /dev/null 2>/dev/null \
  | jq -r '.result + "\n\n-- fork_id: " + .session_id + "  cost_usd: " + (.total_cost_usd|tostring)'
```

- `.result` — the peer's reply.
- `.session_id` — the **new fork id**; keep it to continue the thread.
- `< /dev/null` avoids the CLI's ~3s stdin wait; `2>/dev/null` drops the warning noise.
- No `jq`? `| python3 -c "import sys,json;o=json.loads(sys.stdin.read());print(o['result']);print('fork_id:',o['session_id'])"`

### Multi-turn with a fork
Continue by resuming the **fork id** (drop `--fork-session` now — you want to keep appending to the same branch, not re-fork each turn):

```bash
claude -p --resume <FORK_ID> --output-format json "<next message>" < /dev/null 2>/dev/null | jq -r .result
```

## Getting the session IDs
- **Yours:** `echo $CLAUDE_CODE_SESSION_ID`
- **Peer's:** the user usually gives it. To discover candidates in the current repo, list this project's sessions newest-first:
  ```bash
  d=~/.claude/projects/$(pwd | sed 's#/#-#g'); ls -t "$d"/*.jsonl | head
  ```
  Run the RPC from the **same working directory** as the peer so it loads the same `CLAUDE.md` / project context.

## Fork vs. write-through — when to override the default

| | **Fork** (default) | **Write-through** (`--resume`, no `--fork-session`) |
|---|---|---|
| Peer's real transcript | untouched | your message is appended into it |
| Safe on a human-occupied session | yes | **no** — you interleave turns with the human |
| Peer's own future work sees the exchange | no (private branch) | yes |
| Use for | asking, consulting, borrowing context, parallel fan-out | leaving a durable message, a persistent shared channel, coordinating state |

The key asymmetry: a fork is a *private clone* — the original peer never learns anything from the exchange. Only use **write-through** when you specifically need the peer's own session to carry the message forward, and only against a **dedicated** session no human is actively driving (e.g. a bridge/agent session spun up for the purpose). If in doubt, fork.

## Cost, latency, and the persistent upgrade
Each **cold** resume re-hydrates the peer's whole context as cache-creation — empirically **~$1** for a ~100k-token peer, **~15s/turn**. Within the ~5-min cache TTL, follow-ups drop to **~$0.06–0.08**. So:
- Batch a burst of messages while the cache is warm rather than spacing cold resumes out.
- For a *persistent* channel, don't repeat cold resumes. Hold one long-lived listener open with streaming I/O:
  `claude -p --resume <ID> --input-format stream-json --output-format stream-json` — one warm context, messages arrive on stdin as newline-delimited JSON, replies stream out.
- The durable, harness-proof endgame is a small **MCP "mailbox" server** both sessions `claude mcp add`, exposing `send` / `poll` / long-poll `wait` — real-time delivery on a stable, versioned protocol.

## Knobs
- `--model opus|sonnet|…` — pin the model the peer-turn runs on.
- `--append-system-prompt "<framing>"` — e.g. tell the peer it's talking to a peer Claude, not a human.
- `--allowedTools` / `--disallowedTools` — restrict what the peer-turn may do (keep a consult read-only or text-only, e.g. `--disallowedTools "Bash Edit Write"`).

## Etiquette
- Prefix messages so the peer knows they're machine-injected, not human-typed (e.g. `[intra-Claude channel]`), and include your own `$CLAUDE_CODE_SESSION_ID` if you want a reply.
- A fork inherits the peer's full context (can be several MB) and pays the cold cost above — don't fan out dozens of cold forks without budgeting for it.
- Tell the operator when you inject turns or create a fork, so the filesystem side-effects (new session files under `~/.claude/projects/…`) aren't a surprise — per this repo's "document what you touch" norm.
