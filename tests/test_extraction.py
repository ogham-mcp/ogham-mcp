"""Tests for date and entity extraction."""


def test_extract_iso_date():
    from ogham.extraction import extract_dates

    assert extract_dates("Meeting on 2023-05-07") == ["2023-05-07"]


def test_extract_iso_date_slashes():
    from ogham.extraction import extract_dates

    assert extract_dates("Due by 2023/05/07") == ["2023-05-07"]


def test_extract_natural_date_month_first():
    from ogham.extraction import extract_dates

    assert extract_dates("She went on May 7, 2023") == ["2023-05-07"]


def test_extract_natural_date_day_first():
    from ogham.extraction import extract_dates

    assert extract_dates("Event on 7 May 2023") == ["2023-05-07"]


def test_extract_natural_date_with_ordinal():
    from ogham.extraction import extract_dates

    result = extract_dates("On May 7th, 2023 something happened")
    assert "2023-05-07" in result


def test_extract_multiple_dates():
    from ogham.extraction import extract_dates

    text = "From 2023-01-15 to 2023-06-20"
    assert extract_dates(text) == ["2023-01-15", "2023-06-20"]


def test_extract_no_dates():
    from ogham.extraction import extract_dates

    assert extract_dates("No dates here at all") == []


def test_extract_relative_yesterday():
    from ogham.extraction import extract_dates

    result = extract_dates("We discussed this yesterday")
    assert len(result) == 1
    # Should be a valid ISO date (we can't assert exact value since it's relative)
    assert len(result[0]) == 10  # YYYY-MM-DD format


def test_extract_relative_last_tuesday():
    from ogham.extraction import extract_dates

    result = extract_dates("The meeting was last Tuesday")
    assert len(result) == 1
    assert len(result[0]) == 10


def test_extract_relative_weeks_ago():
    from ogham.extraction import extract_dates

    result = extract_dates("We decided this 2 weeks ago")
    assert len(result) == 1
    assert len(result[0]) == 10


def test_extract_absolute_preferred_over_relative():
    from ogham.extraction import extract_dates

    # When absolute dates exist, relative parsing is skipped
    result = extract_dates("On 2023-05-07 we had a meeting yesterday")
    assert "2023-05-07" in result


def test_temporal_intent_true():
    from ogham.extraction import has_temporal_intent

    assert has_temporal_intent("When did she go?") is True
    assert has_temporal_intent("What date was that?") is True
    assert has_temporal_intent("Last week we decided") is True


def test_temporal_intent_false():
    from ogham.extraction import has_temporal_intent

    assert has_temporal_intent("What did she say?") is False
    assert has_temporal_intent("Tell me about the project") is False


def test_extract_camelcase():
    from ogham.extraction import extract_entities

    entities = extract_entities("We use PostgreSQL and FastMCP for the server")
    assert "entity:PostgreSQL" in entities
    assert "entity:FastMCP" in entities


def test_extract_file_path():
    from ogham.extraction import extract_entities

    entities = extract_entities("Edit src/ogham/config.py for settings")
    assert "file:src/ogham/config.py" in entities


def test_extract_error_type():
    from ogham.extraction import extract_entities

    entities = extract_entities("Got a KeyError in the parser module")
    assert "error:KeyError" in entities


def test_extract_entities_cap():
    from ogham.extraction import extract_entities

    content = " ".join(f"FooBar{i} BazQux{i}" for i in range(20))
    entities = extract_entities(content)
    assert len(entities) <= 20


def test_compute_importance_decision():
    from ogham.extraction import compute_importance

    score = compute_importance("We decided to use PostgreSQL for the database")
    assert score >= 0.5  # decision keyword boosts


def test_compute_importance_error():
    from ogham.extraction import compute_importance

    score = compute_importance("Got a KeyError when accessing the config dict")
    assert score >= 0.4  # error keyword boosts


def test_compute_importance_plain():
    from ogham.extraction import compute_importance

    score = compute_importance("had a chat about the project")
    assert score <= 0.4  # no signals, low importance


def test_compute_importance_german_decision():
    from ogham.extraction import compute_importance

    score = compute_importance("Wir haben entschieden PostgreSQL zu verwenden")
    assert score >= 0.5  # German decision keyword


def test_compute_importance_french_error():
    from ogham.extraction import compute_importance

    score = compute_importance("Une erreur est survenue dans le module")
    assert score >= 0.4  # French error keyword


def test_compute_importance_chinese_architecture():
    from ogham.extraction import compute_importance

    score = compute_importance("我们需要重构这个模块的架构")
    assert score >= 0.4  # Chinese architecture keywords


def test_compute_importance_spanish_decision():
    from ogham.extraction import compute_importance

    score = compute_importance("Hemos decidido migrar a la nueva plataforma")
    assert score >= 0.5  # Spanish decision keyword


def test_compute_importance_arabic_error():
    from ogham.extraction import compute_importance

    score = compute_importance("حدث خطأ في النظام أثناء المعالجة")
    assert score >= 0.4  # Arabic error keyword


def test_compute_importance_turkish_decision():
    from ogham.extraction import compute_importance

    score = compute_importance("Yeni framework tercih ettik ve geçiş yaptık")
    assert score >= 0.5  # Turkish decision keyword


def test_extract_entities_empty():
    from ogham.extraction import extract_entities

    assert extract_entities("just some lowercase text here") == []


# --- Recurrence extraction tests ---


def test_recurrence_english_every_monday():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("Team standup every Monday at 9am") == [1]


def test_recurrence_english_multiple_days():
    from ogham.extraction import extract_recurrence

    result = extract_recurrence("Sync call every Tuesday and Thursday")
    assert result == [2, 4]


def test_recurrence_english_weekly():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("Weekly Friday retrospective") == [5]


def test_recurrence_german_jeden():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("Jeden Montag haben wir ein Standup") == [1]


def test_recurrence_german_adverbial():
    from ogham.extraction import extract_recurrence

    # "montags" implies recurrence without needing "jeden"
    assert extract_recurrence("Montags gibt es immer Meetings") == [1]


def test_recurrence_french_chaque():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("Chaque vendredi, réunion d'équipe") == [5]


def test_recurrence_spanish_cada():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("Cada lunes tenemos standup") == [1]


def test_recurrence_italian_ogni():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("Ogni mercoledì riunione di team") == [3]


def test_recurrence_dutch_elke():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("Elke dinsdag teamoverleg") == [2]


def test_recurrence_russian():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("Каждый понедельник стендап в 9 утра") == [1]


def test_recurrence_chinese():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("每周一开会讨论项目进度") == [1]


def test_recurrence_japanese():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("毎週金曜日にレビュー会議") == [5]


def test_recurrence_korean():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("매주 월요일 스탠드업 미팅") == [1]


def test_recurrence_irish():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("Gach Luan bíonn cruinniú againn") == [1]


def test_recurrence_arabic():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("كل الاثنين اجتماع الفريق") == [1]


def test_recurrence_no_every_keyword():
    from ogham.extraction import extract_recurrence

    # Day name without "every" should not match
    assert extract_recurrence("The meeting is on Monday") is None


def test_recurrence_no_day_name():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("We meet every week") is None


def test_recurrence_portuguese():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("Cada segunda-feira temos reunião") == [1]


def test_recurrence_turkish():
    from ogham.extraction import extract_recurrence

    assert extract_recurrence("Her pazartesi toplantı var") == [1]


# --- Temporal query resolution tests ---


def test_resolve_temporal_months_ago():
    from datetime import datetime

    from ogham.extraction import resolve_temporal_query

    ref = datetime(2026, 3, 21)
    result = resolve_temporal_query("What happened four months ago?", ref)
    assert result is not None
    start, end = result
    # 4 months ago from March 21 → around November 2025
    assert "2025-11" in start or "2025-10" in start


def test_resolve_temporal_last_january():
    from datetime import datetime

    from ogham.extraction import resolve_temporal_query

    ref = datetime(2026, 3, 21)
    result = resolve_temporal_query("What did I do last January?", ref)
    assert result is not None
    start, end = result
    assert start == "2025-01-01"
    assert end == "2025-01-31"


def test_resolve_temporal_in_march():
    from datetime import datetime

    from ogham.extraction import resolve_temporal_query

    ref = datetime(2026, 3, 21)
    result = resolve_temporal_query("What meetings did I have in March?", ref)
    assert result is not None
    start, end = result
    # Current month → this year
    assert "2026-03" in start


def test_resolve_temporal_last_week():
    from datetime import datetime

    from ogham.extraction import resolve_temporal_query

    ref = datetime(2026, 3, 21)
    result = resolve_temporal_query("What did we discuss last week?", ref)
    assert result is not None
    start, end = result
    assert "2026-03" in start


def test_resolve_temporal_no_temporal():
    from ogham.extraction import resolve_temporal_query

    result = resolve_temporal_query("Tell me about the project architecture")
    assert result is None


def test_resolve_temporal_two_years_ago():
    from datetime import datetime

    from ogham.extraction import resolve_temporal_query

    ref = datetime(2026, 3, 21)
    result = resolve_temporal_query("What was decided two years ago?", ref)
    assert result is not None
    start, end = result
    assert "2024" in start


def test_resolve_temporal_graceful_without_llm():
    """LLM fallback should not crash if litellm is not configured."""
    from ogham.extraction import resolve_temporal_query

    # Complex query that parsedatetime can't handle -- should return None, not crash
    result = resolve_temporal_query("the second Thursday of the quarter before last")
    # May return None (no LLM) or a result (if LLM is available) -- either is fine
    assert result is None or isinstance(result, tuple)


def test_person_entities_are_not_extracted():
    """Ogham does not extract person entities. Deleted 2026-08-18.

    Every entity class Ogham keeps has an unambiguous syntactic marker --
    CamelCase, a path separator, an Error suffix. ``person:`` had none: it
    inferred personhood from capitalisation, which does not distinguish
    "Kevin Burns" from "Durable Objects", and measured ~0% precision on
    technical prose (2,086 distinct tags over this repo's own docs, none of
    them people).

    Three gates were tried and none closed it -- a shape rule, an all-caps
    rejection, and a phrase denylist. The replacement that would have worked
    cost 70-120x the latency and 1,090x the package size.

    Independently: ogham-cli measured `person:` as the ONLY prefix on which
    the two implementations disagreed. Dropping it took extraction parity
    from 85.6% to 97/97.

    If this test fails, the classifier is back. Read the above first.
    """
    from ogham.extraction import extract_entities

    for content in (
        "We met Kevin Burns at the conference",
        "Reported by Jean McCarthy last week",
        "Caroline Smith went to the store",
        "Grant EXECUTE to SECURITY DEFINER functions",
        "We store state in Durable Objects for the coordinator",
        "Kevin Burns, Owen Fletcher and Luis Ramirez agreed.",
    ):
        assert not [e for e in extract_entities(content) if e.startswith("person:")], content


def test_the_classes_with_syntactic_markers_still_work():
    """The kept set, and why it is the kept set -- each has a real marker."""
    from ogham.extraction import extract_entities

    got = extract_entities("Edit src/ogham/config.py -- got a KeyError from PostgreSQL")
    assert "file:src/ogham/config.py" in got
    assert "error:KeyError" in got
    assert "entity:PostgreSQL" in got
