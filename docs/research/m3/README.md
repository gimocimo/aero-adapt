# M3 literature review (6 Nov 2026)

This is the review the M3 spec ([docs/09](../../09_m3_technical_spec.md)) was built on: six
topic reports and a synthesis. It is here so readers can check the spec's **[U]** and **[D]**
claims against what the review actually found.

| File | Topic |
|---|---|
| [synthesis.md](synthesis.md) | the six reports combined into recommendations for M3, with a reference list (incomplete: each report carries its own sources) |
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
- **"Confirmed"** is each agent's own flag. For a literature claim it means the agent saw the
  claim in a search result or a source page in that session; for a repository fact, that it
  read it from the files. 125 of the 136 findings are marked confirmed. The other 11 are marked
  **not confirmed**: claims from the agent's memory of a paper, from a single snippet that
  conflicted with another, or (5.8 and 5.13) the agent's own derivations.
- **Three findings are derivations, not sources:** 4.3 (marked confirmed, because its inputs
  were read from the Rascal's model in the session), 5.8 and 5.13. The spec tags what rests on
  them [D].
- **The shared search budget ran out in all six reports**, each of which says where it left
  claims unchecked.
- **The spec's tags** come from here: **[U]** marks a claim resting on a finding that is not
  confirmed (or on a secondary summary), and **[D]** a derivation or estimate made by an agent
  or in the spec, which needs a numerical check before a report quotes it.
- **The spec holds where they differ.** A three-reviewer check of the spec (72 issues, on
  repository facts, internal consistency and the literature) corrected the spec, not these
  files. For example, the synthesis drops doc 03's L1-adaptive-control baseline, and the spec
  moves it to M6 as a Could instead; the synthesis's schedule (about 150 h) predates the spec's
  line-by-line costing (about 159 h).
- **Edits for publication.** The reports are rendered from the agents' structured output
  (JSON) into Markdown, with the not-confirmed label's reason added and multiplication signs
  escaped so they do not render as italics. Absolute local paths are made
  repository-relative. Two quotations from abstracts that ran past 15 words are paraphrased
  (4.9, 4.10), and two pointers to the owner's private assistant notes now point to docs/09
  §8.6. One title in the synthesis's reference list is corrected (Saravanan et al. 2020). The
  wording is otherwise unchanged; the synthesis's "I" is its author agent.
