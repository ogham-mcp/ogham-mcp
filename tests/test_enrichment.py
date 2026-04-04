"""Tests for entity extraction enrichment and attention reordering."""

from datetime import datetime

from ogham.extraction import extract_entities
from ogham.service import _reorder_for_attention


class TestEntityExtraction:
    """Entity extraction produces correct tags across languages."""

    def test_english_wedding(self):
        tags = extract_entities("I went to Sarah's wedding with my sister and felt happy")
        assert "event:wedding" in tags
        assert "emotion:happy" in tags
        assert "relationship:sister" in tags

    def test_german_konzert(self):
        tags = extract_entities("Ich ging mit meiner Schwester zum Konzert in Berlin")
        assert "event:konzert" in tags
        assert "relationship:schwester" in tags

    def test_french_mariage(self):
        tags = extract_entities("Je suis allé au mariage avec ma soeur et j étais heureux")
        assert "event:mariage" in tags
        assert "emotion:heureux" in tags
        assert "relationship:soeur" in tags

    def test_spanish_boda(self):
        tags = extract_entities("Fui a la boda con mi hermana y estaba emocionada")
        assert "event:boda" in tags
        assert "relationship:hermana" in tags

    def test_chinese_wedding(self):
        tags = extract_entities("我和姐姐去参加婚礼，很开心")
        assert "event:婚礼" in tags
        assert "relationship:姐姐" in tags

    def test_japanese_wedding(self):
        tags = extract_entities("姉と結婚式に行って嬉しかった")
        assert "event:結婚式" in tags

    def test_korean_wedding(self):
        tags = extract_entities("언니와 결혼식에 갔는데 행복했어요")
        assert "event:결혼식" in tags

    def test_hindi_wedding(self):
        tags = extract_entities("मैं अपनी बहन के साथ शादी में गया और खुश था")
        assert "event:शादी" in tags
        assert "relationship:बहन" in tags

    def test_arabic_wedding(self):
        tags = extract_entities("ذهبت إلى حفل الزفاف مع أختي وكنت سعيداً")
        assert any(t.startswith("event:") for t in tags)

    def test_quantity_extraction(self):
        tags = extract_entities("We ran 5 miles and I was exhausted")
        assert "quantity:5 miles" in tags
        assert "emotion:exhausted" in tags

    def test_quantity_excludes_years(self):
        tags = extract_entities("In 2024 I bought 3 books")
        assert "quantity:3 books" in tags
        assert not any("2024" in t for t in tags if t.startswith("quantity:"))

    def test_no_boss_without_social_context(self):
        tags = extract_entities("My boss decided to use PostgreSQL")
        assert not any(t.startswith("relationship:") for t in tags)

    def test_location_extraction(self):
        tags = extract_entities("I went to Berlin for a conference")
        assert any("Berlin" in t for t in tags if t.startswith("location:"))

    def test_tag_cap(self):
        """Tags are capped at 20."""
        content = (
            "Sarah and John went to the wedding concert party marathon "
            "in Berlin Tokyo London and felt happy excited thrilled "
            "with my sister brother mother father cousin friend "
            "and bought 5 books 3 tickets 10 films"
        )
        tags = extract_entities(content)
        assert len(tags) <= 20

    def test_empty_content(self):
        tags = extract_entities("")
        assert tags == []

    def test_technical_entities_still_work(self):
        tags = extract_entities("Fixed the PostgreSQL connection error in /src/db.py")
        assert "entity:PostgreSQL" in tags
        assert any(t.startswith("file:") for t in tags)


class TestReorderForAttention:
    """Lost in the Middle reordering."""

    def test_short_list_unchanged(self):
        items = [{"id": i} for i in range(3)]
        assert _reorder_for_attention(items) == items

    def test_reorder_puts_top_first(self):
        items = [{"id": i, "score": 10 - i} for i in range(10)]
        reordered = _reorder_for_attention(items)
        # Top 30% (items 0, 1, 2) should still be at the front
        assert reordered[0]["id"] == 0
        assert reordered[1]["id"] == 1
        assert reordered[2]["id"] == 2

    def test_reorder_puts_bottom_last(self):
        items = [{"id": i, "score": 10 - i} for i in range(10)]
        reordered = _reorder_for_attention(items)
        # Bottom 20% (items 8, 9) should be at the end
        assert reordered[-1]["id"] == 9
        assert reordered[-2]["id"] == 8

    def test_reorder_preserves_all_items(self):
        items = [{"id": i} for i in range(20)]
        reordered = _reorder_for_attention(items)
        assert len(reordered) == 20
        assert set(r["id"] for r in reordered) == set(range(20))


class TestTimelineTable:
    """Timeline table builder."""

    def test_builds_table_from_dated_memories(self):
        from ogham.service import build_timeline_table

        results = [
            {"content": "Met Emma at coffee shop", "metadata": {"dates": ["2023-04-11"]}},
            {"content": "Started new job", "metadata": {"dates": ["2023-04-01"]}},
            {"content": "Went hiking in mountains", "metadata": {"dates": ["2023-04-15"]}},
        ]
        ref = datetime(2023, 4, 20)
        table = build_timeline_table(results, reference_date=ref)

        assert "CHRONOLOGICAL TIMELINE" in table
        assert "2023-04-01" in table
        assert "2023-04-11" in table
        assert "2023-04-15" in table
        assert "TODAY" in table
        assert "M1" in table or "M2" in table  # memory refs present

    def test_computes_days_ago(self):
        from ogham.service import build_timeline_table

        results = [
            {"content": "Event A", "metadata": {"dates": ["2023-04-10"]}},
            {"content": "Event B", "metadata": {"dates": ["2023-04-15"]}},
        ]
        ref = datetime(2023, 4, 20)
        table = build_timeline_table(results, reference_date=ref)

        assert "10 days" in table  # 2023-04-10 is 10 days before 2023-04-20
        assert "5 days" in table  # 2023-04-15 is 5 days before 2023-04-20

    def test_returns_empty_for_undated(self):
        from ogham.service import build_timeline_table

        results = [
            {"content": "No dates here", "metadata": {}},
            {"content": "Also no dates", "metadata": {}},
        ]
        assert build_timeline_table(results) == ""

    def test_returns_empty_for_single_date(self):
        from ogham.service import build_timeline_table

        results = [{"content": "Only one date", "metadata": {"dates": ["2023-04-11"]}}]
        assert build_timeline_table(results) == ""

    def test_merges_same_day_events(self):
        from ogham.service import build_timeline_table

        results = [
            {"content": "Morning event", "metadata": {"dates": ["2023-04-11"]}},
            {"content": "Afternoon event", "metadata": {"dates": ["2023-04-11"]}},
            {"content": "Next day event", "metadata": {"dates": ["2023-04-12"]}},
        ]
        ref = datetime(2023, 4, 20)
        table = build_timeline_table(results, reference_date=ref)

        # 2023-04-11 should appear once with both refs
        lines = [line for line in table.split("\n") if "2023-04-11" in line]
        assert len(lines) == 1
        assert "M1" in lines[0] and "M2" in lines[0]

    def test_defaults_to_now(self):
        from ogham.service import build_timeline_table

        results = [
            {"content": "Old event", "metadata": {"dates": ["2020-01-01"]}},
            {"content": "Recent event", "metadata": {"dates": ["2020-06-15"]}},
        ]
        table = build_timeline_table(results)
        assert "CHRONOLOGICAL TIMELINE" in table
        assert "2020-01-01" in table
