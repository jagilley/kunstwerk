---
description: Instructions for how to run experiments on our Modal compute. Invoke before running anything on Modal.
---

**Instructions for running experiments on Modal**

- Please use `modal run --detach` for any Modal jobs that we expect to take longer than 5 minutes. It can be easy to kill jobs when running in attached mode.
- You need to run detached functions by invoking the function explicitly with e.g. `modal run --detach a2a_forward/permutation_test.py::a2a_permutation_test` rather than just `modal run --detach a2a_forward/permutation_test.py`. The `--detach` parameter only persists the most recently-created function, and normally you can't really control the order they get created in. So it's best to invoke explicitly.
- Likewise, running e.g. `modal run --detach a2a_forward/permutation_test.py::main` as a local entrypoint will also result in premature cancellations, so don't do this.
- The correct Modal workspace to use for most jobs is `chromatic` (we have credits applied to this workspace). However, there may be a few types of jobs that rely on existing volume results which may only be accessible from `jagilley`. If so, you're allowed to use this workspace as well.
- Almost all of our experiments will fit on the memory footprint of an L4 GPU. Please default to running on that (it has faster clock speed than a T4), and bump up to GPUs with more VRAM only if we encounter out-of-memory issues.

- If there is a specific reason to believe that it's epistemically important to replicate results across multiple seeds, it's ok to do so. However:
    - We should always start by running and waiting for a single-seed version of the full experiment to verify that things work as intended and get a sense of the described effect
    - *If and only if* it seems like the result is epistemically meaningful and may be sensitive to seed-dependence, *then* you should autonomously launch multiple seed-experiments in parallel. Running a multiple-seeds check is itself a statement you're making that "I think this is epistemically valuable".

- For long-running e.g. training jobs (long running = anything that takes more than 5 mins), please follow this procedure to manage things:
    1. Kick off the job, make sure it runs, etc. You can do this by using a detached Modal run with a 2-minute timeout and then auto-backgrounding the shell.
    2. Then, don't monitor any further, halt your work, and just wait for the background notification that the job has completed. Under no circumstances should you read logs in consecutive tool calls as a means of waiting for the run to finish.

- If you are a **subagent**, halting means something different — ending your turn reports to your orchestrator, and you cannot idle for free the way a top-level session can. You MUST read [`/subagent-instructions`](../subagent-instructions/SKILL.md) before you wait on anything.

It is very important that you halt your work when experiments are running so that you don't waste context by re-reading logs over and over or anything like that. Failure to halt your work will significantly degrade your efficacy as a research assistant. You're allowed to do unrelated work while experiments are running (e.g., writing an aggregation script to process the results when they come back), but you must still be cognizant to halt your work once this work is done.

### Opera-specific instructions

Please don't store any large files on Modal volumes or anything like that, at least permanently. You're welcome to temporarily store things such that they can be downloaded and used in a video though, provided they're subsequently cleaned up.