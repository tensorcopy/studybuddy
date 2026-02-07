---
name: research
description: Researches a topic in depth by spawning parallel sub-agents to cover foundational papers, modern advances, code implementations, and industry practice. Produces a structured research brief in _drafts/research/. Use when studying a new topic or preparing to write a blog post.
argument-hint: "<topic>"
---

# Research a Topic

Research **$ARGUMENTS** by orchestrating parallel sub-agents, then synthesize their findings into a structured brief.

## Step 1: Spawn 4 parallel research agents

Use the Task tool to launch ALL 4 agents **in a single message** so they run in parallel. Each should use `subagent_type: "general-purpose"`.

### Agent 1: "Research foundations"

Prompt:

> Research the foundational and seminal papers on "$ARGUMENTS".
>
> Find: original papers that established the field, key breakthroughs and paradigm shifts, most-cited papers and why they matter, a historical timeline of evolution.
>
> For each paper: authors, year, venue, core contribution, key equations/algorithms, and significance.
>
> Return structured markdown with a ## Foundational Papers section and a ## Timeline section.

### Agent 2: "Research modern advances"

Prompt:

> Research the latest advances and state-of-the-art in "$ARGUMENTS" (2023-2026).
>
> Find: recent papers pushing the frontier, new architectures/methods/paradigms, how deep learning/LLMs/transformers changed the field, current benchmarks and SOTA results, active research frontiers and open problems.
>
> For each source: what's new, what it improves over, whether it's validated in production or research-only.
>
> Return structured markdown with ## Recent Advances and ## Open Problems sections.

### Agent 3: "Research implementations"

Prompt:

> Research practical implementations and code resources for "$ARGUMENTS".
>
> Find: popular open-source libraries and frameworks (with GitHub URLs), tutorial implementations from scratch in Python/PyTorch, standard benchmark datasets, high-quality GitHub repos, runnable code examples demonstrating key concepts.
>
> For each: name, URL, what it covers, language/framework, quality assessment.
>
> Return structured markdown with ## Libraries, ## Tutorials, ## Datasets, and ## Code Examples sections.

### Agent 4: "Research industry practice"

Prompt:

> Research how "$ARGUMENTS" is used in industry and production systems.
>
> Find: engineering blog posts from major tech companies (Google, Meta, Netflix, Spotify, Amazon, etc.), real-world system architectures and design patterns, practical challenges (scale, latency, cold-start, evaluation), known failure modes and limitations, points where sources disagree.
>
> Return structured markdown with ## Industry Systems, ## Practical Challenges, and ## Controversies sections.

## Step 2: Synthesize

After ALL 4 agents return results:

1. Read all agent outputs
2. Deduplicate overlapping findings
3. Organize by concept hierarchy (foundational to advanced)
4. Identify the natural teaching order and conceptual leaps
5. Fill gaps with your own knowledge where needed

## Step 3: Write the brief

Save to `_drafts/research/YYYY-MM-DD-<topic-slug>.md` using today's date. Follow the template in [brief-template.md](brief-template.md).

## Step 4: Report

Tell the user:
- The file path where the brief was saved
- A 3-5 sentence summary of key findings
- The suggested post structure (section outline)
- How many papers/sources were collected
