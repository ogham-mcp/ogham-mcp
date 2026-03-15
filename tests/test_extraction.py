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


def test_extract_person_name():
    from ogham.extraction import extract_entities

    entities = extract_entities("Caroline Smith went to the store")
    assert "person:Caroline Smith" in entities


def test_extract_person_not_stopwords():
    from ogham.extraction import extract_entities

    entities = extract_entities("The Quick brown fox")
    person_tags = [e for e in entities if e.startswith("person:")]
    assert len(person_tags) == 0


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
    assert len(entities) <= 15


def test_extract_entities_empty():
    from ogham.extraction import extract_entities

    assert extract_entities("just some lowercase text here") == []
