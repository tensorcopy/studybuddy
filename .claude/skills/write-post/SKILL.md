---
name: write-post
description: Writes a full 4,000-10,000+ word blog post in Sebastian Raschka's style. Automatically researches the topic first if no research brief exists, then writes, then validates. Use when creating a new blog post.
argument-hint: "<topic or path to research brief>"
---

# Write a Blog Post

Write a deep technical blog post on **$ARGUMENTS** following the StudyBuddy style guide in CLAUDE.md.

## Step 1: Gather source material

Check if a research brief already exists:

1. Use Glob to search `_drafts/research/*` for files matching the topic
2. If a brief exists, read it — this is your primary source material
3. If `$ARGUMENTS` is a file path, read that file directly

**If NO research brief exists**: Invoke the `/research` skill via the Skill tool with the topic as the argument. Wait for it to complete, then read the resulting brief.

## Step 2: Plan the structure

Read the style guide in `CLAUDE.md`, then create an outline. Follow this pattern:

1. **Introduction** — Why this topic matters (compelling hook, no "In this post...")
2. **Background** — Prerequisites and foundational concepts
3. **Core sections** (3-5 H2 sections) — The technical meat, broad to specific
4. **Practical Takeaways** — What the reader should do with this knowledge
5. **Conclusion** — Summary and forward-looking thoughts
6. **References** — All cited sources with links

## Step 3: Write sections in parallel

Use the Task tool to spawn **parallel sub-agents** for independent sections. Launch them in a single message with `subagent_type: "general-purpose"`.

For a typical post, split into 2-3 writing agents:

### Agent A: "Write early sections"

Prompt should include:
- The post outline (all H2 headings with 1-sentence descriptions)
- The research brief content relevant to their sections
- The style rules from [style-reference.md](style-reference.md)
- Which sections to write (e.g., Introduction + Background + first core section)

### Agent B: "Write middle sections"

Same context, but assigned the middle core technical sections.

### Agent C: "Write final sections"

Assigned the remaining core sections + Practical Takeaways + Conclusion + References.

Each agent should write complete, polished markdown for their assigned sections.

## Step 4: Assemble and edit

After all writing agents return:

1. Assemble sections in order into a single markdown file
2. Write smooth **transitions** between sections (the seams where agents joined)
3. Ensure consistent terminology and tone throughout
4. Add complete Jekyll frontmatter:

```yaml
---
title: "Descriptive Title Here"
date: YYYY-MM-DD
categories: [category1, category2]
tags: [tag1, tag2, tag3]
mathjax: true
toc: true
description: "1-2 sentence SEO description."
---
```

5. Save to `_drafts/YYYY-MM-DD-<topic-slug>.md`

## Step 5: Auto-validate

After saving the draft, invoke the `/validate-post` skill via the Skill tool with the draft file path as the argument.

If the validation returns critical issues, fix them immediately. For recommended issues, fix what you can.

## Step 6: Publish

After validation and fixes:

1. Copy the draft from `_drafts/` to `_posts/` (keep the draft as well)
2. Commit all changes (new post, research brief, any updated files) with a descriptive message
3. Push to `main` — this triggers GitHub Actions deployment automatically

## Step 7: Report

Tell the user:
- The published post file path
- Word count
- Summary of the validation results
- The site URL where the post will be live: `https://tensorcopy.github.io/studybuddy/`
