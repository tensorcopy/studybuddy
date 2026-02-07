# StudyBuddy — Project Instructions for Claude

## What This Is

A technical blog hosted on GitHub Pages at `https://tensorcopy.github.io/studybuddy/`. Posts cover ML, AI, and LLM topics in depth.

## Directory Structure

```
_posts/          # Published posts (YYYY-MM-DD-title.md)
_drafts/         # Draft posts (not published)
_drafts/research/  # Research briefs from /research command
_layouts/        # Custom Jekyll layouts
_includes/       # Reusable HTML components
assets/css/      # Custom styles
assets/images/   # Post images and diagrams
.claude/commands/  # Custom slash commands
```

## Blog Style Guide

Posts follow Sebastian Raschka's writing style. Key principles:

### Structure
- **Length**: 4,000–10,000+ words. Go deep.
- **Flow**: Broad context → core concepts → technical details → practical takeaways
- **Sections**: Use clear H2 headers. Each section should be self-contained enough to skim.
- **Opening**: Start with *why* this topic matters. No fluff — jump into the substance within 2-3 paragraphs.

### Tone
- **Conversational authority**: Write like an expert explaining to a peer, not lecturing a student.
- **Transparent speculation**: Clearly mark what's established fact vs. your interpretation. Use phrases like "My intuition is...", "I suspect...", "One way to think about this...".
- **Honest about limits**: If you don't know something, say so. If results are surprising, discuss why.

### Technical Content
- **Math**: Use LaTeX via MathJax. Inline: `$...$`. Display: `$$...$$`. Always explain the intuition before/after formulas.
- **Code**: Use fenced code blocks with language tags. Code should be runnable and self-contained where possible. Prefer Python.
- **Figures**: Use descriptive alt text. Place figure caption in italics on the line after the image: `*Figure 1: Description*`
- **Tables**: Use for comparisons, benchmarks, hyperparameters.

### Formatting Rules
- Use blockquotes (`>`) for key takeaways or important insights
- Bold key terms on first introduction
- Use numbered lists for sequences/steps, bullet lists for unordered items
- Include a references section at the end with links

## Post Template

```markdown
---
title: "Your Title Here"
date: YYYY-MM-DD
categories: [category1, category2]
tags: [tag1, tag2, tag3]
mathjax: true
toc: true
description: "A 1-2 sentence description for SEO and social cards."
---

## Introduction

Why this topic matters. What you'll learn. Context.

## Background

Prerequisites and foundational concepts.

## [Core Topic Sections]

The meat of the post. Multiple H2 sections.

## Practical Takeaways

What the reader should do with this knowledge.

## Conclusion

Summary and forward-looking thoughts.

## References

- [Author, "Title", Year](url)
```

## Quality Checklist

Before publishing a post, verify:

- [ ] Title is specific and descriptive (not clickbait)
- [ ] Opening paragraph hooks the reader with *why*
- [ ] Structure flows from broad to specific
- [ ] All code blocks have language tags and are runnable
- [ ] Math notation is correct and renders properly
- [ ] Key terms are bolded on first use
- [ ] Figures have captions
- [ ] Blockquotes highlight key insights
- [ ] References section is complete
- [ ] Frontmatter is complete (title, date, categories, tags, description)
- [ ] No broken links
- [ ] Reading time is reasonable (check word count)
- [ ] Conclusion includes practical takeaways

## Custom Commands

- `/research <topic>` — Generate a research brief for a topic
- `/write-post <topic-or-file>` — Write a full blog post from a topic or research brief
- `/validate-post <file>` — Review a draft post for quality and correctness

## Build & Deploy

- Local preview: `bundle exec jekyll serve`
- Deploys automatically on push to `main` via GitHub Actions
- Site URL: `https://tensorcopy.github.io/studybuddy/`
