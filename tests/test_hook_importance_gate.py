"""The gate that decides what a PostToolUse hook is allowed to store (TBU-202).

Measured 2026-07-30 on the real `work` store: 33,703 of 36,025 memories --
93.6% -- came from `hook:post-tool`, and precision@10 over twelve realistic
queries was 25%. Three of every four results a user saw were hook exhaust.

Two causes:

1. The importance floor was 0.30. Sampling 800 stored rows, 47% score
   EXACTLY 0.30 -- they cleared it by a margin of zero -- and a further 35%
   score 0.20. Nothing scores between 0.30 and 0.40, so 0.35 is the whole
   gap and means "strictly above trivial".
2. The summariser applied a code-constant regex to non-code files, turning
   an edit to MEMORY.md into "MEMORY.md: changed TBU" -- an acronym lifted
   out of a Linear issue key.

Replaying 1,500 real stored rows through the gate as it now stands blocks
84% of current writes and keeps 100% of error captures.

A note on a road not taken, because the reasoning is not obvious. Gating
`type:code-change` was tried first and was wrong: `compute_importance`
keys on verb vocabulary rather than information, scoring the informative
"config.py: changed x from 1 to 2" identically to the contentless
"MEMORY.md: changed TBU" (both 0.20). Gating on that score discards the
good captures with the empty ones, which is why the fix belongs in the
summariser instead. Four existing tests in test_hooks.py caught it.

These tests pin the evidence with content taken verbatim from the store,
so lowering the floor or admitting a high-volume tag to the bypass fails
loudly rather than quietly restoring the flood.

Not covered here: `_recent_actions` is an in-process dict, but PostToolUse
spawns a fresh `ogham hooks inscribe` per tool call, so the dedup window
can never fire across invocations -- which is why 51% of hook rows are
byte-identical duplicates. Structural, tracked separately.
"""

import pytest

from ogham.extraction import compute_importance
from ogham.hooks import (
    _DEFAULT_IMPORTANCE_FLOOR,
    _HIGH_SIGNAL_TAGS,
    _get_importance_floor,
    _passes_importance_gate,
)

# Verbatim from the store. The first is the single most common shape in it.
TRIVIAL_CAPTURES = [
    "MEMORY.md: changed TBU [openbrain-sharedmemory]",
    "publish.yml: changed TBU [openbrain-sharedmemory]",
    "deploy/release command completed: make publish 2>&1 | tail -10 [openbrain-sharedmemory]",
    "git push: git push origin main",
    "Edit: /Users/x/Developer/Projects/MoveToCloud/graph-terragrunt/src/graph/nodes.py",
]


def test_floor_is_above_the_trivial_score():
    """0.30 was the modal score of trivial content, so it gated nothing."""
    assert _DEFAULT_IMPORTANCE_FLOOR > 0.30


def test_bypass_tags_are_rare_by_construction():
    """A bypass is only safe on tags that cannot flood.

    `type:action` sits on 33,867 rows. If a future tag is proposed for this
    set, the question is not "is it useful" but "can it flood".
    """
    assert "type:action" not in _HIGH_SIGNAL_TAGS


@pytest.mark.parametrize("content", TRIVIAL_CAPTURES)
def test_trivial_captures_are_rejected(content):
    """Each of these is currently in the store, most of them thousands of times."""
    assert not _passes_importance_gate(content, ["type:action"])


# --- the summary generator, which is where the empty rows came from --------


def test_uppercase_word_in_prose_is_not_treated_as_a_code_constant():
    """The defect behind the single most common row in the store.

    An ALL_CAPS regex meant to catch a module constant matched an acronym in
    Markdown, so editing MEMORY.md produced "MEMORY.md: changed TBU" -- TBU
    lifted out of a Linear issue key. Whatever the summary is now, it must
    not be that.
    """
    from ogham.hooks import _extract_edit_memory

    old = "- [x] TBU-100 done\n" + "filler line\n" * 5
    new = (
        "- [x] TBU-100 done\n"
        "- TBU-201 PyPI page is empty, no readme or urls declared in the project table\n"
        + "filler line\n"
        * 5
    )
    result = _extract_edit_memory(
        {"file_path": "/tmp/MEMORY.md", "old_string": old, "new_string": new}, "/tmp"
    )
    if result is not None:
        assert not result.content.endswith(": changed TBU")
        assert result.content != "MEMORY.md: changed TBU"


@pytest.mark.parametrize(
    "token",
    ["TBU", "NEW", "DONE", "POSTED", "SUPERSEDES", "B92DE9F9F3C4B7C0"],
)
def test_no_bare_acronym_summaries(token):
    """Every one of these shapes is in the store, thousands of times over."""
    from ogham.hooks import _extract_edit_memory

    old = "some existing prose line here\n" * 3
    new = f"some existing prose line here\nthe item is {token} and here is more prose text\n" * 3
    result = _extract_edit_memory(
        {"file_path": "/tmp/NOTES.md", "old_string": old, "new_string": new}, "/tmp"
    )
    if result is not None:
        assert result.content != f"NOTES.md: changed {token}"


def test_a_real_constant_definition_is_still_summarised():
    """The anchor must not break the case the regex was written for."""
    from ogham.hooks import _extract_edit_memory

    # The added text must clear `_diff_change_size` >= 20, or the summariser
    # declines before the regex is ever consulted.
    old = "import os\n\nOTHER = 1\n" + "pad = 0\n" * 5
    new = (
        "import os\n\nMAX_RETRIES = 5  # retry budget for the embedding provider\n"
        "OTHER = 1\n" + "pad = 0\n" * 5
    )
    result = _extract_edit_memory(
        {"file_path": "/tmp/settings.py", "old_string": old, "new_string": new}, "/tmp"
    )
    assert result is not None
    assert "MAX_RETRIES" in result.content


def test_substantive_error_output_is_kept():
    """Errors are the hook captures actually worth having.

    Sampled error captures clear the floor on content alone (median 0.40,
    100% above it), so this must pass WITHOUT relying on the tag bypass.
    """
    content = (
        "error: {'stdout': 'publish-check:\\n\\t@echo \"=== Publish-check: scanning "
        'dist/ for secrets ==="\\n\\t@if [ ! -d dist ] || [ -z "$$(ls dist/*.tar.gz '
        '2>/dev/null)" ]; then echo " No sdist in dist/ -- run make build first"; '
        "exit 1; fi'} [openbrain-sharedmemory]"
    )
    assert compute_importance(content) >= _DEFAULT_IMPORTANCE_FLOOR
    assert _passes_importance_gate(content, ["type:action"])


def test_terse_error_still_survives_via_the_bypass():
    """type:error keeps its bypass so a short error is not lost on length."""
    assert _passes_importance_gate("error: exit 1", ["type:action", "type:error"])


def test_decision_capture_survives():
    assert _passes_importance_gate("chose X over Y", ["type:action", "type:decision"])


# --- the config reader -----------------------------------------------------


def test_floor_defaults_when_config_has_no_key(monkeypatch):
    """The key is absent from the shipped shared file, so this is the live path."""
    monkeypatch.setattr("ogham.hooks._load_config", lambda: {"noise_commands": []})
    assert _get_importance_floor() == _DEFAULT_IMPORTANCE_FLOOR


def test_floor_is_read_from_config_when_present(monkeypatch):
    monkeypatch.setattr("ogham.hooks._load_config", lambda: {"importance_floor": 0.6})
    assert _get_importance_floor() == 0.6


def test_nonsense_floor_falls_back_rather_than_crashing_the_hook(monkeypatch):
    """A hook that raises loses the capture AND the tool result. Never crash."""
    monkeypatch.setattr("ogham.hooks._load_config", lambda: {"importance_floor": "high"})
    assert _get_importance_floor() == _DEFAULT_IMPORTANCE_FLOOR


def test_missing_config_falls_back(monkeypatch):
    monkeypatch.setattr("ogham.hooks._load_config", lambda: None)
    assert _get_importance_floor() == _DEFAULT_IMPORTANCE_FLOOR


# --- the negative case that matters ----------------------------------------


def test_gate_is_not_vacuous():
    """A gate that passes everything is indistinguishable from no gate.

    The 0.30 floor was effectively vacuous. Assert against the real corpus
    shape that the current setting actually rejects the bulk of it.
    """
    rejected = sum(1 for c in TRIVIAL_CAPTURES if not _passes_importance_gate(c, ["type:action"]))
    assert rejected == len(TRIVIAL_CAPTURES)
