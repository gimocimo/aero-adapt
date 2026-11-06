# M3 literature review (6 Nov 2026)

This is the review the M3 spec ([docs/09](../../09_m3_technical_spec.md)) was built on: six
topic reports and a synthesis. It is here so readers can check the spec's **[U]** and **[D]**
claims against what the review actually found.

| File | Topic |
|---|---|
| [synthesis.md](synthesis.md) | the six reports combined into recommendations for M3, with the full reference list |
| [1-teacher-student.md](1-teacher-student.md) | teacher–student and privileged-information policies (RMA, Lee et al., RAPTOR) |
| [2-fixed-wing-rl.md](2-fixed-wing-rl.md) | deep RL for fixed-wing control, fault tolerance and upset recovery |
| [3-jax-ppo.md](3-jax-ppo.md) | massively parallel PPO in JAX, and running it on rented GPUs |
| [4-control-allocation.md](4-control-allocation.md) | control allocation with jammed, floating and reversed surfaces |
| [5-indi.md](5-indi.md) | INDI as the classical baseline (B2) |
| [6-evaluation.md](6-evaluation.md) | evaluating the controllers: statistics, splits, credibility |

## How it was made, and how far to trust it

- **AI research agents (Claude) wrote it**, in one session, from web searches and the
  repository's own files. Nothing was simulated. It is not peer-reviewed, and it was not
  checked line by line against the papers.
- **"Confirmed"** on a finding means the agent saw the claim in a search result or a source
  page in that session. 125 of the 136 findings are marked confirmed. The rest come from the
  agent's memory of a paper, or from a single snippet that conflicted with another; they are
  marked **not confirmed**. The shared search budget ran out during reports 1, 2, 3 and 6.
- **The spec's tags** come from here: **[U]** marks a claim resting on a finding that is not
  confirmed (or on a secondary summary), and **[D]** a derivation or estimate made by an agent
  or in the spec, which needs a numerical check before a report quotes it.
- **The spec holds where they differ.** A three-reviewer check of the spec (72 issues, on
  repository facts, internal consistency and the literature) corrected the spec, not these
  files. For example, the synthesis drops doc 03's L1-adaptive-control baseline, and the spec
  moves it to M6 as a Could instead; the synthesis's schedule (about 150 h) predates the spec's
  line-by-line costing (about 159 h).
- **Edits for publication:** the reports are rendered from the agents' structured output
  (JSON) into Markdown, and local file paths are made repository-relative. The wording is
  otherwise unchanged; the synthesis's "I" is its author agent.
