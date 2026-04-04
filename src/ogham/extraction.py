"""Date and entity extraction for memory enrichment.

Pure regex, no LLM calls. Runs at store_memory time to enrich
metadata (dates) and tags (entities) automatically.
"""

import logging
import re
from datetime import datetime

import parsedatetime
from stop_words import AVAILABLE_LANGUAGES, get_stop_words

from ogham.data.loader import (
    get_all_activity_words,
    get_all_architecture_words,
    get_all_day_names,
    get_all_decision_words,
    get_all_emotion_words,
    get_all_error_words,
    get_all_event_words,
    get_all_every_words,
    get_all_possessive_triggers,
    get_all_quantity_units,
    get_all_relationship_words,
    get_day_names,
    get_month_names,
    get_temporal_keywords,
    get_word_numbers,
)

logger = logging.getLogger(__name__)

# parsedatetime calendar for relative date parsing (zero deps, English)
_PDT_CAL = parsedatetime.Calendar()

# --- Stopwords (34 languages, loaded once) ---

_STOP_WORDS: set[str] = set()
for _lang in AVAILABLE_LANGUAGES:
    _STOP_WORDS.update(get_stop_words(_lang))

# --- Date extraction ---

_ISO_DATE = re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b")

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_NATURAL_DATE = re.compile(
    rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{4}}\b"
    rf"|\b\d{{1,2}}\s+{_MONTHS}\s+\d{{4}}\b",
    re.IGNORECASE,
)

# Patterns for relative date phrases to feed to parsedatetime
_RELATIVE_PHRASES = re.compile(
    r"\b((?:last|next|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|week|month|year)"
    r"|yesterday|today|tomorrow"
    r"|\d+\s+(?:days?|weeks?|months?|years?)\s+ago"
    r"|in\s+\d+\s+(?:days?|weeks?|months?|years?))\b",
    re.IGNORECASE,
)

# --- Multilingual day names → day index (0=Sun..6=Sat) ---
# Covers 16 languages matching the importance scoring dictionaries.
# Each entry maps a lowercase day name to its dow index.

_DAY_NAMES: dict[str, int] = get_all_day_names()

_EVERY_WORDS: set[str] = get_all_every_words()


def extract_recurrence(content: str) -> list[int] | None:
    """Extract recurring day-of-week patterns from content.

    Returns sorted list of day indices (0=Sun..6=Sat) or None.
    Supports 16 languages via static dictionary lookup.
    No LLM calls.
    """
    content_lower = content.lower()
    days_found: set[int] = set()

    # Check if content contains an "every" keyword
    has_every = any(word in content_lower for word in _EVERY_WORDS)

    if not has_every:
        # German adverbial forms ("montags") imply recurrence without "jeden"
        for name, idx in _DAY_NAMES.items():
            if name.endswith("s") and name in get_day_names("de"):
                if re.search(rf"\b{re.escape(name)}\b", content_lower):
                    days_found.add(idx)
        if not days_found:
            return None

    # Scan for day names in content
    if not days_found:
        # Longest names first to avoid partial matches (e.g. "dé luain" before "luan")
        for name, idx in sorted(_DAY_NAMES.items(), key=lambda x: -len(x[0])):
            if len(name) < 2:
                continue
            # CJK/Arabic/Cyrillic/Devanagari: substring match is fine (no spaces)
            if name[0].isascii() and name[0].isalpha():
                # Alphabetic: use word boundary to avoid "vendredi" matching "di"
                if re.search(rf"\b{re.escape(name)}\b", content_lower):
                    days_found.add(idx)
            elif name in content_lower:
                days_found.add(idx)

    return sorted(days_found) if days_found else None


TEMPORAL_KEYWORDS = frozenset(get_temporal_keywords("en"))


def extract_dates(content: str) -> list[str]:
    """Extract date strings from content. Returns sorted normalised ISO dates."""
    dates: set[str] = set()

    for match in _ISO_DATE.finditer(content):
        dates.add(match.group(1).replace("/", "-"))

    for match in _NATURAL_DATE.finditer(content):
        try:
            raw = re.sub(r"(st|nd|rd|th)", "", match.group(0))
            raw = raw.replace(",", "").strip()
            for fmt in ("%B %d %Y", "%d %B %Y", "%b %d %Y", "%d %b %Y"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    dates.add(dt.strftime("%Y-%m-%d"))
                    break
                except ValueError:
                    continue
        except Exception:
            logger.debug("Failed to parse natural date: %s", match.group(0))
            continue

    # Relative dates via parsedatetime ("last Tuesday", "yesterday", "two weeks ago")
    # Only attempt if no absolute dates found and content has temporal keywords
    if not dates:
        for phrase in _RELATIVE_PHRASES.findall(content):
            try:
                result, status = _PDT_CAL.parse(phrase)
                if status:
                    dt = datetime(*result[:6])
                    dates.add(dt.strftime("%Y-%m-%d"))
            except Exception:
                logger.debug("Failed to parse relative date: %s", phrase)
                continue

    return sorted(dates)


def has_temporal_intent(query: str) -> bool:
    """Check if a query has temporal intent (when, date, time, etc.)."""
    words = set(query.lower().split())
    return bool(words & TEMPORAL_KEYWORDS)


# --- Multi-hop temporal detection + entity extraction ---

_MULTI_HOP_PATTERNS = re.compile(
    r"how many (?:months?|weeks?|days?|years?) (?:between|since|before|after|passed)"
    r"|which (?:happened|came|occurred|did I do|event|did I|trip|device|gift|project) .{0,20}first"
    r"|what (?:is|was) the order"
    r"|how long (?:between|since|before|after|had passed|did it take)"
    r"|which .{0,30} (?:earlier|later|before|after)"
    r"|how (?:old|long) was I when"
    r"|how many .{0,40}across (?:my |our )?(?:sessions?|requests?|conversations?)"
    r"|across (?:all |my |our )?(?:sessions?|conversations?|requests?)",
    re.IGNORECASE,
)

# Pattern to extract entity anchors from "between X and Y" / "first, X or Y" queries
_BETWEEN_PATTERN = re.compile(
    r"between\s+(.+?)\s+and\s+(.+?)(?:\s*[?.!]|$)",
    re.IGNORECASE,
)
_OR_PATTERN = re.compile(
    r"(?:first|earlier|later),?\s+(.+?)\s+or\s+(?:the\s+)?(.+?)(?:\s*[?.!]|$)",
    re.IGNORECASE,
)
_SINCE_PATTERN = re.compile(
    r"(?:since|after|before)\s+(.+?)(?:\s+(?:did|was|had|have|how|when|$))",
    re.IGNORECASE,
)
# "how many days before X did Y" → extract X and Y
_BEFORE_DID_PATTERN = re.compile(
    r"(?:before|after)\s+(.+?)\s+did\s+(?:I\s+)?(.+?)(?:\s*[?.!]|$)",
    re.IGNORECASE,
)
# "how long had I been X when Y" → extract X and Y
_BEEN_WHEN_PATTERN = re.compile(
    r"been\s+(.+?)\s+when\s+(?:I\s+)?(.+?)(?:\s*[?.!]|$)",
    re.IGNORECASE,
)


_ORDERING_PATTERNS = re.compile(
    r"what is the order of"
    r"|from earliest to latest"
    r"|from latest to earliest"
    r"|in (?:chronological|reverse) order"
    r"|list the order"
    r"|walk me through the order"
    r"|in (?:what|which) order"
    r"|the sequence (?:of|in which)",
    re.IGNORECASE,
)


def is_ordering_query(query: str) -> bool:
    """Detect if a query asks for chronological ordering of multiple events."""
    return bool(_ORDERING_PATTERNS.search(query))


def is_multi_hop_temporal(query: str) -> bool:
    """Detect if a query requires multi-hop temporal reasoning (comparing two events)."""
    return bool(_MULTI_HOP_PATTERNS.search(query))


_SUMMARY_PATTERNS = re.compile(
    r"comprehensive summary"
    r"|summarize (?:all|everything|my)"
    r"|summary of (?:all|everything|how|my)"
    r"|overview of (?:all|everything|how|my)"
    r"|how .{0,30} has progressed"
    r"|give me a (?:full|complete|comprehensive)"
    r"|progress.{0,20}including"
    r"|across (?:all|my) (?:sessions?|conversations?)",
    re.IGNORECASE,
)


def is_broad_summary_query(query: str) -> bool:
    """Detect queries needing broad timeline coverage (summarization)."""
    return bool(_SUMMARY_PATTERNS.search(query))


def extract_query_anchors(query: str) -> list[str]:
    """Extract entity anchors from a multi-hop temporal query.

    Returns a list of 1-2 noun phrases representing the events being compared.
    Uses regex patterns -- no LLM or NLP library needed.
    """
    anchors = []

    # "between X and Y"
    match = _BETWEEN_PATTERN.search(query)
    if match:
        anchors = [match.group(1).strip(), match.group(2).strip()]

    # "X or Y" (from "which happened first, X or Y")
    if not anchors:
        match = _OR_PATTERN.search(query)
        if match:
            anchors = [match.group(1).strip(), match.group(2).strip()]

    # "before X did Y" / "after X did Y"
    if not anchors:
        match = _BEFORE_DID_PATTERN.search(query)
        if match:
            anchors = [match.group(1).strip(), match.group(2).strip()]

    # "been X when Y"
    if not anchors:
        match = _BEEN_WHEN_PATTERN.search(query)
        if match:
            anchors = [match.group(1).strip(), match.group(2).strip()]

    # "since/after X" (single anchor fallback)
    if not anchors:
        match = _SINCE_PATTERN.search(query)
        if match:
            anchors = [match.group(1).strip()]

    # Clean up common filler words from anchors
    cleaned = []
    for a in anchors:
        a = re.sub(r"^(?:the|my|a|an|I|i)\s+", "", a, flags=re.IGNORECASE)
        a = re.sub(r"\s+(?:the|my|a|an)\s+", " ", a, flags=re.IGNORECASE)
        a = a.strip().rstrip(",.")
        if len(a) > 2:
            cleaned.append(a)

    return cleaned


# --- Temporal query resolution ---

# Patterns for temporal range phrases ("in January", "last March", "four months ago")
_MONTH_NAMES = get_month_names("en")


def resolve_temporal_query(
    query: str,
    reference_date: datetime | None = None,
) -> tuple[str, str] | None:
    """Resolve a temporal query into a date range (start, end) as ISO strings.

    Uses parsedatetime first (free, handles ~80% of cases).
    Falls back to LLM (GPT-4o-mini via litellm) for complex expressions.
    Returns None if no temporal range can be resolved.
    """
    if reference_date is None:
        reference_date = datetime.now()

    # Try parsedatetime first
    result = _resolve_with_parsedatetime(query, reference_date)
    if result:
        return result

    # Try month name extraction ("in January", "last March")
    result = _resolve_month_reference(query, reference_date)
    if result:
        return result

    # LLM fallback (optional -- graceful if litellm not installed)
    if has_temporal_intent(query):
        result = _resolve_with_llm(query, reference_date)
        if result:
            return result

    return None


def _resolve_with_parsedatetime(query: str, reference_date: datetime) -> tuple[str, str] | None:
    """Try parsedatetime for relative date expressions."""
    # Look for range-like phrases
    range_patterns = [
        # "between X and Y"
        re.compile(r"between\s+(.+?)\s+and\s+(.+?)(?:\s*[?.!]|$)", re.IGNORECASE),
        # "from X to Y"
        re.compile(r"from\s+(.+?)\s+to\s+(.+?)(?:\s*[?.!]|$)", re.IGNORECASE),
    ]

    for pattern in range_patterns:
        match = pattern.search(query)
        if match:
            start_result, start_status = _PDT_CAL.parse(match.group(1), reference_date)
            end_result, end_status = _PDT_CAL.parse(match.group(2), reference_date)
            if start_status and end_status:
                start = datetime(*start_result[:6])
                end = datetime(*end_result[:6])
                return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    # Single point phrases → create a ±window
    for phrase in _RELATIVE_PHRASES.findall(query):
        result, status = _PDT_CAL.parse(phrase, reference_date)
        if status:
            dt = datetime(*result[:6])
            # Create a 7-day window around the resolved date
            from datetime import timedelta as _td

            start = dt - _td(days=3)
            end = dt + _td(days=4)
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    # "N months/weeks ago" → create a month/week window
    _WORD_NUMBERS = get_word_numbers("en")
    ago_match = re.search(
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an)"
        r"\s+(months?|weeks?|years?)\s+ago",
        query,
        re.IGNORECASE,
    )
    if ago_match:
        raw = ago_match.group(1).lower()
        num = _WORD_NUMBERS.get(raw, None)
        if num is None:
            num = int(raw)
        unit = ago_match.group(2).lower().rstrip("s")
        from datetime import timedelta

        if unit == "month":
            # Approximate: go back N months, create a 30-day window
            start = reference_date - timedelta(days=num * 30 + 15)
            end = (
                reference_date - timedelta(days=(num - 1) * 30 - 15) if num > 1 else reference_date
            )
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        elif unit == "week":
            start = reference_date - timedelta(weeks=num, days=3)
            end = reference_date - timedelta(weeks=num - 1) if num > 1 else reference_date
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        elif unit == "year":
            start = reference_date.replace(year=reference_date.year - num, month=1, day=1)
            end = reference_date.replace(year=reference_date.year - num, month=12, day=31)
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    return None


def _resolve_month_reference(query: str, reference_date: datetime) -> tuple[str, str] | None:
    """Resolve 'in January', 'last March', 'this April' to a month range."""
    query_lower = query.lower()

    for month_name, month_num in _MONTH_NAMES.items():
        if month_name not in query_lower:
            continue
        # Determine the year
        if "last" in query_lower or "previous" in query_lower:
            year = reference_date.year - 1
        elif "next" in query_lower:
            year = reference_date.year + 1
        else:
            # Default: most recent occurrence of that month
            year = reference_date.year
            if month_num > reference_date.month:
                year -= 1

        import calendar

        _, last_day = calendar.monthrange(year, month_num)
        start = f"{year}-{month_num:02d}-01"
        end = f"{year}-{month_num:02d}-{last_day:02d}"
        return (start, end)

    return None


def _resolve_with_llm(query: str, reference_date: datetime) -> tuple[str, str] | None:
    """LLM fallback for complex temporal expressions. Optional -- returns None if unavailable.

    Uses TEMPORAL_LLM_MODEL env var to determine the model. If empty, LLM
    fallback is disabled (parsedatetime only). Self-hosters can set this to
    an Ollama model (e.g. "ollama/llama3.2") or any litellm-compatible string.
    """
    from ogham.config import settings

    model = settings.temporal_llm_model
    if not model:
        logger.debug("TEMPORAL_LLM_MODEL not set, skipping LLM temporal resolution")
        return None

    try:
        import litellm
    except ImportError:
        logger.debug("litellm not installed, skipping LLM temporal resolution")
        return None

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Today is {reference_date.strftime('%Y-%m-%d')}. "
                        "Extract the date range from the user's query. "
                        'Return ONLY JSON: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} '
                        "or null if no date range can be determined."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=50,
            temperature=0,
        )
        import json

        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        if data and "start" in data and "end" in data:
            return (data["start"], data["end"])
    except Exception:
        logger.debug("LLM temporal resolution failed for: %s", query[:60])

    return None


# --- Shared regex patterns (used by both importance and entity extraction) ---

_CAMEL_CASE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-zA-Z]*)+\b")
_FILE_PATH = re.compile(r"(?:\.{0,2}/)?(?:[\w@.-]+/)+[\w@.-]+\.\w+")
_ERROR_TYPE = re.compile(r"\b\w*(?:Error|Exception)\b")

# --- Importance scoring (8 languages) ---

_DECISION_WORDS: set[str] = get_all_decision_words()
_ERROR_WORDS: set[str] = get_all_error_words()
_ARCHITECTURE_WORDS: set[str] = get_all_architecture_words()
_EVENT_WORDS: set[str] = get_all_event_words()
_ACTIVITY_WORDS: set[str] = get_all_activity_words()
_EMOTION_WORDS: set[str] = get_all_emotion_words()
_RELATIONSHIP_WORDS: set[str] = get_all_relationship_words()
_POSSESSIVE_TRIGGERS: set[str] = get_all_possessive_triggers()
_QUANTITY_UNITS: set[str] = get_all_quantity_units()

# GeoText for location extraction (pre-compiled city/country database from GeoNames)
try:
    from geotext import GeoText as _GeoText
except ImportError:
    _GeoText = None

# Pre-compile quantity pattern: number + unit, excluding years
_QUANTITY_PATTERN = re.compile(
    r"\b(?!(?:19|20)\d{2}\b)(\d+(?:\.\d+)?)\s+("
    + "|".join(re.escape(u) for u in sorted(_QUANTITY_UNITS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def _content_has_signal(content: str, word_set: set[str]) -> bool:
    """Check if content contains any word from the signal set."""
    content_lower = content.lower()
    return any(word in content_lower for word in word_set)


def compute_importance(content: str, tags: list[str] | None = None) -> float:
    """Score content importance based on signals. Returns 0.0-1.0.

    Checks 8 languages for decision, error, and architecture keywords.
    No LLM needed.
    """
    score = 0.2  # base score

    if _content_has_signal(content, _DECISION_WORDS):
        score += 0.3
    if _content_has_signal(content, _ERROR_WORDS) or _ERROR_TYPE.search(content):
        score += 0.2
    if _content_has_signal(content, _ARCHITECTURE_WORDS):
        score += 0.2
    if _FILE_PATH.search(content):
        score += 0.1
    if "```" in content or "`" in content:
        score += 0.1
    if len(content) > 500:
        score += 0.1
    if tags and len(tags) >= 3:
        score += 0.1

    return min(score, 1.0)


# --- Entity extraction ---
_PUNCT = str.maketrans("", "", ".,!?:;\"'()")


def extract_entities(content: str) -> list[str]:
    """Extract named entities from content for tagging.

    Returns sorted list of prefixed tags, capped at 20:
      person:FirstName LastName
      entity:CamelCaseName
      file:path/to/file.ext
      error:SomeError
      event:wedding
      activity:hiking
      emotion:frustrated
      relationship:sister
      quantity:3 tanks
    """
    entities: set[str] = set()

    # --- Technical entities (existing) ---
    for m in _CAMEL_CASE.finditer(content):
        entities.add(f"entity:{m.group(0)}")

    for i, m in enumerate(_FILE_PATH.finditer(content)):
        if i >= 5:
            break
        entities.add(f"file:{m.group(0)}")

    for m in _ERROR_TYPE.finditer(content):
        entities.add(f"error:{m.group(0)}")

    # --- Person names (existing): two consecutive capitalised words ---
    words = content.split()
    for i in range(len(words) - 1):
        w1 = words[i].translate(_PUNCT)
        w2 = words[i + 1].translate(_PUNCT)
        if (
            w1
            and w2
            and w1[0].isupper()
            and w2[0].isupper()
            and w1.isalpha()
            and w2.isalpha()
            and w1.lower() not in _STOP_WORDS
            and w2.lower() not in _STOP_WORDS
            and len(w1) > 1
            and len(w2) > 1
        ):
            entities.add(f"person:{w1} {w2}")

    # --- Enrichment entities (v1, multilingual) ---
    content_lower = content.lower()
    content_words = set(content_lower.split())

    def _match(word: str) -> bool:
        """Match word in content. Use substring for non-Latin/short CJK, word-set for Latin."""
        if len(word) < 2:
            return False
        # Non-ASCII (CJK, Arabic, Cyrillic, Devanagari, etc.) -- substring match
        if not word[0].isascii():
            return word in content_lower
        # Latin script -- word-set match to avoid partial matches
        return word in content_words

    # Events (cap 2)
    event_count = 0
    for word in _EVENT_WORDS:
        if event_count >= 2:
            break
        if _match(word):
            entities.add(f"event:{word}")
            event_count += 1

    # Activities (cap 2)
    activity_count = 0
    for word in _ACTIVITY_WORDS:
        if activity_count >= 2:
            break
        if _match(word):
            entities.add(f"activity:{word}")
            activity_count += 1

    # Emotions (cap 2)
    emotion_count = 0
    for word in _EMOTION_WORDS:
        if emotion_count >= 2:
            break
        if _match(word):
            entities.add(f"emotion:{word}")
            emotion_count += 1

    # Relationships: require social context (preposition or event/activity nearby)
    _SOCIAL_PREPS = {"with", "mit", "avec", "con", "com", "for", "für", "pour", "para"}
    rel_count = 0
    for i, w in enumerate(words):
        if rel_count >= 2:
            break
        w_lower = w.translate(_PUNCT).lower()
        # Pattern 1: preposition + relationship word ("with my sister")
        if w_lower in _SOCIAL_PREPS:
            for j in range(i + 1, min(i + 4, len(words))):
                next_w = words[j].translate(_PUNCT).lower()
                if next_w in _POSSESSIVE_TRIGGERS:
                    continue  # skip "with my" and check next
                if next_w in _RELATIONSHIP_WORDS:
                    entities.add(f"relationship:{next_w}")
                    rel_count += 1
                    break
        # Pattern 2: possessive + relationship word + event/activity nearby
        elif w_lower in _POSSESSIVE_TRIGGERS:
            for j in range(i + 1, min(i + 4, len(words))):
                next_w = words[j].translate(_PUNCT).lower()
                if next_w in _RELATIONSHIP_WORDS:
                    # Check if an event or activity word exists in the content
                    if content_words & _EVENT_WORDS or content_words & _ACTIVITY_WORDS:
                        entities.add(f"relationship:{next_w}")
                        rel_count += 1
                    break

    # Fallback: substring match for non-Latin relationship words (CJK, Arabic, Cyrillic)
    if rel_count < 2:
        for word in _RELATIONSHIP_WORDS:
            if rel_count >= 2:
                break
            if not word[0].isascii() and len(word) >= 2 and word in content_lower:
                entities.add(f"relationship:{word}")
                rel_count += 1

    # Quantities: number + unit noun (excluding years)
    qty_count = 0
    for m in _QUANTITY_PATTERN.finditer(content):
        if qty_count >= 3:
            break
        entities.add(f"quantity:{m.group(1)} {m.group(2).lower()}")
        qty_count += 1

    # Locations: GeoText city/country extraction (GeoNames database, no LLM)
    if _GeoText is not None:
        try:
            places = _GeoText(content)
            loc_count = 0
            for city in places.cities:
                if loc_count >= 2:
                    break
                entities.add(f"location:{city}")
                loc_count += 1
            for country in places.country_mentions:
                if loc_count >= 3:
                    break
                entities.add(f"location:{country}")
                loc_count += 1
        except Exception:
            logger.debug("GeoText extraction failed, skipping locations")

    return sorted(entities)[:20]
