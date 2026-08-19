# Ogham MCP — Extraction & Enrichment

`extraction.py` provides all NLP enrichment at store time. Everything is pure regex + dictionary lookup — no LLM calls, no NLP libraries beyond `parsedatetime` and `geotext`.

## Date Extraction (`extract_dates`)

Finds dates in content and returns sorted ISO strings.

**Detection layers:**
1. ISO dates: `2024-01-15`, `2024/01/15`
2. Natural dates: `January 15, 2024`, `15 January 2024`, abbreviated months
3. Relative dates (via parsedatetime): `last Tuesday`, `yesterday`, `3 weeks ago`

Relative dates are only attempted if no absolute dates are found.

## Entity Extraction (`extract_entities`)

Extracts named entities and returns prefixed tags (capped at 15):

| Prefix | Detection Method |
|--------|-----------------|
| `entity:CamelCase` | Regex: `[A-Z][a-z]+([A-Z][a-zA-Z]*)+` |
| `file:path/to/file.ext` | Regex: path-like patterns (max 5) |
| `error:SomeError` | Regex: words ending in `Error` or `Exception` |

Every class kept here has an unambiguous syntactic marker: interior
capitalisation, a path separator, an `Error`/`Exception` suffix.

**Removed in v0.18.0: `person:First Last`.** It matched two consecutive
capitalised words that were not stopwords, which cannot distinguish a name from
a product name -- to a shape rule, `Kevin Burns` and `Durable Objects` are
identical. Existing `person:` tags remain in your database and stay searchable;
no new ones are written. If you filter searches on a `person:` tag, that filter
will not match memories stored from v0.18.0 onward.

## Importance Scoring (`compute_importance`)

Returns 0.0–1.0 based on content signals:

| Signal | Score | Detection |
|--------|-------|-----------|
| Base | +0.2 | Always |
| Decision keywords | +0.3 | 16-language dictionary (decided, chose, migrated, etc.) |
| Error keywords | +0.2 | 16-language dictionary + `*Error`/`*Exception` regex |
| Architecture keywords | +0.2 | 16-language dictionary (design, pattern, refactor, etc.) |
| File paths | +0.1 | Path regex |
| Code (backticks) | +0.1 | Contains `` ` `` |
| Long content (>500 chars) | +0.1 | Length check |
| Rich tags (≥3) | +0.1 | Tag count |

Maximum score is capped at 1.0.

## Language Coverage

The keyword dictionaries cover **16 languages**: English, German, French, Italian, Spanish, Portuguese, Dutch, Polish, Turkish, Russian, Ukrainian, Irish, Arabic, Chinese, Japanese, Korean, Hindi.

The `stop-words` library is no longer a dependency. It backed only the removed
person-name classifier, where it contributed a 33-language stopword union of
12,705 entries applied to every memory regardless of the language it was
written in.

Day name recognition for recurrence extraction covers all 16 languages plus adverbial forms (e.g., German "montags" = "on Mondays").

## Recurrence Extraction (`extract_recurrence`)

Detects recurring day-of-week patterns. Returns sorted list of day indices (0=Sun..6=Sat) or None.

**Detection:**
1. Check for "every"/"each"/"weekly" keyword in 16 languages
2. If no "every" keyword, check for German adverbial day forms ("montags")
3. Scan for day names across all 16 languages (longest-first matching to avoid partial matches)
4. CJK/Arabic/Cyrillic: substring match; Alphabetic: word-boundary match

## Query Classification Functions

| Function | Pattern | Purpose |
|----------|---------|---------|
| `has_temporal_intent()` | Word intersection with temporal keywords | Gate for temporal re-ranking |
| `is_ordering_query()` | "what is the order of", "chronological order", "earliest to latest" | Route to strided + entity threading |
| `is_multi_hop_temporal()` | "how many months between", "which happened first", "across sessions" | Route to bridge retrieval |
| `is_broad_summary_query()` | "comprehensive summary", "summarize all", "how X has progressed" | Route to strided + MMR |

## Anchor Extraction (`extract_query_anchors`)

Extracts entity anchors from multi-hop temporal queries for bridge retrieval:

| Pattern | Example |
|---------|---------|
| `between X and Y` | "How many days between my trip and the conference?" |
| `X or Y` (after "first") | "Which happened first, the launch or the merger?" |
| `before X did Y` | "How long before the deadline did I start?" |
| `been X when Y` | "How long had I been coding when I got promoted?" |
| `since/after X` (fallback) | "What happened since the migration?" |

Anchors are cleaned of filler words (the, my, a, an, I).
