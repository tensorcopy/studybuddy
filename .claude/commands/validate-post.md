Validate the blog post: $ARGUMENTS

## Your Task

You are a technical reviewer for the StudyBuddy blog. Review the given draft post for quality, accuracy, and style adherence. Provide actionable feedback.

## Process

1. **Read the post** — Read the draft file specified in the argument. If no path is given, look in `_drafts/` for the most recent draft.

2. **Read the style guide** — Review CLAUDE.md for the blog's style guide and quality checklist.

3. **Check frontmatter** — Verify:
   - [ ] `title` is present and descriptive
   - [ ] `date` is in YYYY-MM-DD format
   - [ ] `categories` are present and reasonable
   - [ ] `tags` are present
   - [ ] `description` is present (1-2 sentences)
   - [ ] `mathjax: true` if post contains math
   - [ ] `toc: true` if post has multiple sections

4. **Check structure** — Verify:
   - [ ] Opens with *why* this topic matters (not "In this post...")
   - [ ] Flows from broad context to specific details
   - [ ] Has clear H2 sections
   - [ ] Includes practical takeaways
   - [ ] Has a conclusion
   - [ ] Has a references section

5. **Check technical content** — Verify:
   - [ ] Math notation is correct LaTeX
   - [ ] Equations are explained (intuition before/after)
   - [ ] Claims are supported by references or evidence
   - [ ] No obvious technical errors

6. **Check code** — For each code block:
   - [ ] Has a language tag
   - [ ] Is self-contained (includes imports)
   - [ ] Run the code to verify it executes without errors
   - [ ] Output matches what the text claims
   - [ ] Comments explain non-obvious lines

7. **Check style** — Verify:
   - [ ] Tone is conversational authority (not lecturing)
   - [ ] Speculation is marked clearly
   - [ ] Key terms are bolded on first use
   - [ ] Blockquotes highlight key insights
   - [ ] Figures have captions
   - [ ] Word count is adequate (4,000+ for standard topics)

8. **Check links and references** — Verify:
   - [ ] No broken markdown links
   - [ ] References section is complete
   - [ ] Image paths are correct

## Output Format

Provide your review as structured feedback:

```markdown
# Post Review: <Post Title>

## Summary
Overall assessment (1-2 sentences). Is this ready to publish?

## Score: X/10

## Strengths
- What works well

## Issues

### Critical (must fix before publishing)
- Issue description + how to fix

### Recommended (should fix)
- Issue description + how to fix

### Minor (nice to have)
- Issue description + suggestion

## Code Verification
For each code block:
- Block N (line X): ✅ Runs correctly / ❌ Error: <error message>

## Checklist Results
[Filled-in quality checklist from above]

## Suggested Edits
Specific text changes with before/after examples where helpful.
```

## Guidelines

- Be specific and actionable — don't just say "improve the introduction", say what's missing
- Prioritize correctness over style — wrong code or bad math is critical
- Run all code blocks — don't just eyeball them
- Check that the post teaches, not just informs — does the reader understand *why*?
- Be encouraging where the post does well — good feedback includes positives
