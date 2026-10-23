# Conversational book questions

Chat uses a conversational grounding mode: explain meanings from available facts and the recent dialogue, while keeping unknown exact facts unknown. Chapter summaries are not verbatim quotes. Machine citation identifiers remain in the API citations array rather than the assistant prose. Strict analysis retains its existing inline-source rules.

The three supplemental summaries were checked against Antoine de Saint-Exupéry, Le Petit Prince, chapter XXI: https://gutenberg.net.au/ebooks03/0300771h.html . They cover patient familiarity, the four-o-clock example, and the wheat-field memory. They are labeled summaries and do not replace, rebuild or expand the vector index. They are ranked only for relevant chat queries.

Acceptance requires real multi-turn replies: no internal IDs/source labels in prose, a substantive answer to why, preservation of follow-ups, a natural switch to everyday conversation, and no invented exact duration or acceptance of a false book premise. Model-output mocks validate transport and instructions only; they do not establish naturalness.
