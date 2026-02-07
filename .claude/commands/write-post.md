Write a blog post on: $ARGUMENTS

## Your Task

You are a technical writer for the StudyBuddy blog. Write a full blog post following the project's style guide (see CLAUDE.md).

## Input

The argument may be:
- A topic name (e.g., "attention mechanisms") — research and write from scratch
- A path to a research brief in `_drafts/research/` — use it as your foundation

If a research brief exists, read it first and use it as your primary source material.

## Process

1. **Read the style guide** — Review CLAUDE.md for the blog's style guide, post template, and quality checklist.

2. **If no research brief exists**, conduct research:
   - Search for key papers, implementations, and explanations
   - Build a mental model of the concept hierarchy
   - Identify code examples and math that will be needed

3. **Plan the structure** — Create an outline with H2 sections. Follow the pattern:
   - Why this matters (motivation)
   - Background / prerequisites
   - Core technical content (2-4 sections)
   - Practical implications
   - Conclusion + references

4. **Write the post** — Follow these principles:
   - **Start strong**: Open with a compelling *why*. No "In this post, we will..."
   - **Build intuition first**: Explain the idea before the math. Use analogies.
   - **Show, don't just tell**: Include code that demonstrates concepts. Make it runnable.
   - **Use math carefully**: Every equation should be motivated and explained. No naked formulas.
   - **Be honest**: Mark speculation clearly. Acknowledge limitations.
   - **Write transitions**: Each section should flow naturally into the next.
   - **End with takeaways**: What should the reader remember and do?

5. **Add code examples** — Write Python code blocks that:
   - Are self-contained (include imports)
   - Have comments explaining non-obvious lines
   - Actually run and produce correct output
   - Demonstrate the concept, not just implement it

6. **Add figure placeholders** — Where a diagram would help, add:
   ```
   ![Description of what the figure shows]({{ site.baseurl }}/assets/images/post-slug/figure-name.png)
   *Figure N: Caption describing the figure.*
   ```

## Output

Save the post to `_drafts/YYYY-MM-DD-<topic-slug>.md` with complete Jekyll frontmatter.

After writing, run through the quality checklist from CLAUDE.md and fix any issues.

## Quality Standards

- Minimum 4,000 words for a standard topic
- At least 2 runnable code examples
- Math where appropriate (with intuition)
- Clear section transitions
- Blockquoted key takeaways
- Complete references section
- Proper frontmatter with categories, tags, and description
