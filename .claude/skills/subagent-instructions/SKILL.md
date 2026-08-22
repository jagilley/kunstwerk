---
description: What to do as a subagent while a long-running job is in flight — halting, waiting, and reporting back to your orchestrator. Invoke when you are a subagent about to wait on a Modal job.
---

**You cannot idle for free.** A top-level session ends its turn at no cost and the harness wakes it. You re-pay your entire context — usually 200-400k tokens by then — on every turn you spend waiting. So never poll, and never stay awake to wait.

Ending your turn is safe. It delivers a report to your orchestrator but does not kill you: you stay parked, and your own background notification revives you, even an hour later. Halt freely — just make each halt carry content.

While a job is in flight:

1. Launch detached, then arm **one** backgrounded `Bash` waiter that exits when the job ends. Cover the failure signatures too — silence is indistinguishable from "still running":
   ```bash
   until grep -qE "App completed|Traceback|Error|Killed|OOM" results/launch_<tag>.log 2>/dev/null; do sleep 60; done
   ```
   `grep -q` returns an exit code, not text, so this adds nothing to your context.
2. **End your turn with the handle** — app id, expected artifact path, what you'll do when woken. That halt is your orchestrator's status update; don't spend it on "I'll wait for the notification."
3. Your notification wakes you. Fetch from the volume, run the aggregation, make the figures, run the sanity checks, then **halt again with the reduced results** — summary and figures, not raw output. You reduce; your orchestrator interprets.

Never:

- burn a turn on `echo "awaiting ..."` to avoid ending one — end it.
- read a background task's output before it has signalled.
- arm a second waiter on a condition already being watched. (One subagent accumulated 41 and reported "stale watcher notification — already handled" as its findings.)

`Monitor` is the wrong tool here: it's built for a *stream* of events, and its own guidance says to use a backgrounded `until` loop for a single "X is ready" signal.

For a short job — a smoke test, under ~15 minutes — skip all of this and block in one foreground call. Keep any blocking call under ~8 minutes: the harness kills at 10m0s, and a no-progress watchdog fires at 600s (it does not touch parked agents).

Don't manipulate git branches or state; that's your orchestrator's job.
