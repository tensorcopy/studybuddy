---
name: validate-post
description: Reviews a blog post draft for quality, accuracy, and style adherence. Runs code blocks to verify they execute. Checks technical accuracy, style guide compliance, and structural completeness. Returns actionable feedback with a score.
argument-hint: "<path to draft file>"
---

# Validate a Blog Post

Review **$ARGUMENTS** for quality, accuracy, and style adherence.

## Step 1: Locate the post

If `$ARGUMENTS` is a file path, read that file. If it's a topic name or empty, use Glob to find the most recent file in `_drafts/`.

Read the full post content. Also read `CLAUDE.md` for the style guide and quality checklist.

## Step 2: Spawn parallel review agents

Use the Task tool to launch ALL 3 agents **in a single message** with `subagent_type: "general-purpose"`.

Pass each agent the FULL post content in their prompt.

### Agent 1: "Verify code blocks"

Prompt:

> You are a code reviewer. Here is a blog post draft:
>
> <post>
> [FULL POST CONTENT]
> </post>
>
> For EACH fenced code block in the post:
> 1. Check it has a language tag
> 2. Check it is self-contained (includes imports)
> 3. Use the Bash tool to actually RUN the code with `uv run` (Python blocks only — save to a temp file and execute with `uv run python <file>`)
> 4. Verify output matches what the post text claims
> 5. Check comments explain non-obvious lines
>
> Return a markdown report:
> ```
> ## Code Verification
> - Block 1 (line ~N, language): [PASS/FAIL] — [details]
> - Block 2 (line ~N, language): [PASS/FAIL] — [details]
> ```
> Include the exact error message for any failures.

### Agent 2: "Review technical accuracy"

Prompt:

> You are a technical reviewer specializing in ML/AI. Here is a blog post draft:
>
> <post>
> [FULL POST CONTENT]
> </post>
>
> Review for:
> 1. **Math correctness**: Is the LaTeX notation valid? Are equations correct?
> 2. **Technical claims**: Are claims accurate and supported by references?
> 3. **Conceptual accuracy**: Are explanations correct? Any misleading simplifications?
> 4. **Completeness**: Are there important aspects of the topic that are missing?
>
> Return a markdown report:
> ```
> ## Technical Review
> ### Correct
> - [things that are accurate and well-explained]
> ### Issues
> - [specific technical errors with corrections]
> ### Missing
> - [important topics or nuances not covered]
> ```

### Agent 3: "Review style and structure"

Prompt:

> You are an editorial reviewer for a technical blog following Sebastian Raschka's style. Here is a blog post draft:
>
> <post>
> [FULL POST CONTENT]
> </post>
>
> Check against this quality checklist:
> - [ ] Title is specific and descriptive (not clickbait)
> - [ ] Opening hooks the reader with *why* (not "In this post...")
> - [ ] Structure flows from broad to specific
> - [ ] All code blocks have language tags
> - [ ] Key terms bolded on first use
> - [ ] Figures have captions
> - [ ] Blockquotes highlight key insights
> - [ ] References section is complete
> - [ ] Frontmatter has: title, date, categories, tags, description, mathjax, toc
> - [ ] Word count is 4,000+ words
> - [ ] Conclusion includes practical takeaways
> - [ ] Tone is conversational authority (not lecturing)
> - [ ] Speculation is clearly marked
>
> Also check: section transitions, paragraph length variety, reading flow.
>
> Return a markdown report with the filled checklist and specific suggestions.

## Step 3: Synthesize review

After ALL 3 agents return, combine their findings into a single review:

```markdown
# Post Review: <Post Title>

## Summary
1-2 sentence overall assessment. Is this ready to publish?

## Score: X/10

## Strengths
- What works well (be specific)

## Issues

### Critical (must fix before publishing)
- [issue + how to fix]

### Recommended (should fix)
- [issue + how to fix]

### Minor (nice to have)
- [issue + suggestion]

## Code Verification
[From Agent 1]

## Technical Review
[From Agent 2]

## Style Checklist
[From Agent 3]

## Word Count: N words
```

## Step 4: Auto-fix critical issues

If there are critical issues (broken code, wrong math, missing frontmatter), fix them directly using the Edit tool. Tell the user what you fixed.

For recommended and minor issues, list them for the user to decide.
