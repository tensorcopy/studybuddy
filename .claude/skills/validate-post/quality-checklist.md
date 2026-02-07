# Quality Checklist

Full checklist for post validation. Each item should be checked.

## Frontmatter
- [ ] `title` is present and descriptive
- [ ] `date` is in YYYY-MM-DD format
- [ ] `categories` are present and reasonable
- [ ] `tags` are present (3-5 tags)
- [ ] `description` is present (1-2 sentences for SEO)
- [ ] `mathjax: true` if post contains math
- [ ] `toc: true` if post has 3+ sections

## Structure
- [ ] Opens with *why* this topic matters
- [ ] Does NOT start with "In this post, we will..."
- [ ] Flows from broad context to specific details
- [ ] Has clear H2 sections (4+ for a standard post)
- [ ] Each section is self-contained enough to skim
- [ ] Includes practical takeaways section
- [ ] Has a conclusion with forward-looking thoughts
- [ ] Has a references section with links

## Technical Content
- [ ] Math notation is correct LaTeX
- [ ] Inline math uses `$...$`, display uses `$$...$$`
- [ ] Every equation is explained (intuition before AND after)
- [ ] No "naked" formulas without context
- [ ] Claims are supported by references
- [ ] No obvious technical errors

## Code
- [ ] All code blocks have language tags
- [ ] Code is self-contained (includes imports)
- [ ] Code actually runs without errors
- [ ] Output matches what the text claims
- [ ] Comments explain non-obvious lines
- [ ] At least 2 runnable code examples total

## Style
- [ ] Tone is conversational authority (expert to peer)
- [ ] Speculation is marked ("I suspect...", "My intuition is...")
- [ ] Key terms are bolded on first use
- [ ] Blockquotes (`>`) highlight key insights (at least 2)
- [ ] Figures have captions in italics
- [ ] Numbered lists for sequences, bullets for unordered
- [ ] Word count is 4,000+ words

## Links and References
- [ ] No broken markdown links
- [ ] References section is complete (all citations listed)
- [ ] Image paths use `{{ site.baseurl }}/assets/images/...`
- [ ] External links have descriptive text (not "click here")
