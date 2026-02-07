# Writing Style Reference

This post follows Sebastian Raschka's writing style. Sub-agents writing sections MUST follow these rules.

## Tone

- **Conversational authority**: Write like an expert explaining to a peer, not lecturing a student
- **Transparent speculation**: Mark interpretation clearly ("My intuition is...", "I suspect...", "One way to think about this...")
- **Honest about limits**: Say "I don't know" when appropriate. Discuss surprising results.
- **No filler**: Jump into substance within 2-3 paragraphs. No "In this post, we will..."

## Structure

- **Length**: 4,000-10,000+ words total. Go deep.
- **Flow**: Broad context -> core concepts -> technical details -> practical takeaways
- **Each H2 section**: Self-contained enough to skim independently
- **Transitions**: Each section flows naturally into the next. End sections with a bridge sentence.

## Technical Content

- **Math**: LaTeX via MathJax. Inline: `$...$`. Display: `$$...$$`. ALWAYS explain intuition before AND after formulas.
- **Code**: Fenced blocks with language tags. Self-contained (include imports). Comments on non-obvious lines. Must actually run.
- **Figures**: `![Alt text]({{ site.baseurl }}/assets/images/post-slug/name.png)` followed by `*Figure N: Caption.*`
- **Tables**: Use for comparisons, benchmarks, hyperparameters.

## Formatting

- **Blockquotes** (`>`) for key takeaways or important insights
- **Bold** key terms on first introduction
- **Numbered lists** for sequences/steps, **bullet lists** for unordered items
- **References section** at the end with `[Author, "Title", Year](url)` format

## Quality Bar Per Section

- At least 1 code example OR 1 equation per core technical section
- Bold all key terms on first use
- At least 1 blockquoted insight per 2 sections
- No naked equations — always explain what each symbol means
