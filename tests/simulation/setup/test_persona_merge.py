"""Tests for the ``deep_merge`` helper in ``simulation/setup/helpers.py``.

Verifies that ``deep_merge``:
1. Preserves base fields absent from the overlay
2. Overrides base fields present in the overlay
3. Handles ``None`` values in overlay without wiping base
4. Recursively merges nested dicts
5. Never mutates the base dict
"""

from __future__ import annotations

from corallab.simulation.setup.helpers import deep_merge

# ---------------------------------------------------------------------------
# Unit tests for deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    """Pure unit tests for the merge function."""

    def test_overlay_none_does_not_wipe_base(self):
        base = {"domain_state": {"compulsion_urge": 0.6}}
        overlay = {"domain_state": None}
        result = deep_merge(base, overlay)
        assert result["domain_state"] == {"compulsion_urge": 0.6}

    def test_overlay_value_overrides_base(self):
        base = {"severity": "low"}
        overlay = {"severity": "high"}
        result = deep_merge(base, overlay)
        assert result["severity"] == "high"

    def test_nested_dicts_merged_recursively(self):
        base = {"states": {"severity": "low", "domain_state": {"x": 1}}}
        overlay = {"states": {"severity": "high"}}
        result = deep_merge(base, overlay)
        assert result["states"]["severity"] == "high"
        assert result["states"]["domain_state"] == {"x": 1}  # preserved

    def test_overlay_adds_new_keys(self):
        base = {"a": 1}
        overlay = {"b": 2}
        result = deep_merge(base, overlay)
        assert result == {"a": 1, "b": 2}

    def test_base_not_mutated(self):
        base = {"states": {"x": 1}}
        overlay = {"states": {"y": 2}}
        deep_merge(base, overlay)
        assert base == {"states": {"x": 1}}

    def test_all_none_overlay_preserves_base(self):
        base = {"a": 1, "b": {"c": 3}}
        overlay = {"a": None, "b": None}
        result = deep_merge(base, overlay)
        assert result == {"a": 1, "b": {"c": 3}}

    def test_deeply_nested_merge(self):
        base = {"l1": {"l2": {"l3": {"keep": True, "override": "old"}}}}
        overlay = {"l1": {"l2": {"l3": {"override": "new"}}}}
        result = deep_merge(base, overlay)
        assert result["l1"]["l2"]["l3"]["keep"] is True
        assert result["l1"]["l2"]["l3"]["override"] == "new"
