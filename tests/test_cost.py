"""Tests for cost ceiling and budget management."""

from __future__ import annotations

import pytest

from consensus_council.cost import (
    BudgetExceededError,
    CostCeiling,
    CostTracker,
    estimate_cost,
    select_models_within_budget,
)


class TestEstimateCost:
    """Test cost estimation."""

    def test_returns_positive_number(self) -> None:
        """Even with fallback pricing, cost should be positive."""
        cost = estimate_cost("some-unknown-model", 1000, 500)
        assert cost > 0

    def test_more_tokens_more_cost(self) -> None:
        cost_small = estimate_cost("some-model", 100, 50)
        cost_large = estimate_cost("some-model", 10000, 5000)
        assert cost_large > cost_small

    def test_zero_tokens_zero_cost(self) -> None:
        cost = estimate_cost("some-model", 0, 0)
        assert cost == 0.0


class TestCostTracker:
    """Test cost accumulation and reporting."""

    def test_empty_tracker(self) -> None:
        tracker = CostTracker()
        assert tracker.total_cost == 0.0
        assert tracker.total_calls == 0

    def test_record_single_call(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", cost=0.05, prompt_tokens=100, completion_tokens=50)
        assert tracker.total_cost == pytest.approx(0.05)
        assert tracker.total_calls == 1
        assert tracker.cost_for_model("gpt-4o") == pytest.approx(0.05)

    def test_record_multiple_models(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", cost=0.05)
        tracker.record("claude", cost=0.03)
        assert tracker.total_cost == pytest.approx(0.08)
        assert tracker.total_calls == 2
        assert tracker.cost_for_model("gpt-4o") == pytest.approx(0.05)
        assert tracker.cost_for_model("claude") == pytest.approx(0.03)

    def test_record_same_model_multiple_times(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", cost=0.05, prompt_tokens=100, completion_tokens=50)
        tracker.record("gpt-4o", cost=0.03, prompt_tokens=80, completion_tokens=40)
        assert tracker.total_cost == pytest.approx(0.08)
        assert tracker.total_calls == 2
        assert tracker.cost_for_model("gpt-4o") == pytest.approx(0.08)

    def test_cost_for_unknown_model(self) -> None:
        tracker = CostTracker()
        assert tracker.cost_for_model("nonexistent") == 0.0

    def test_breakdown(self) -> None:
        tracker = CostTracker()
        tracker.record("a", cost=0.01, prompt_tokens=10, completion_tokens=5)
        tracker.record("b", cost=0.02, prompt_tokens=20, completion_tokens=10)
        breakdown = tracker.breakdown()
        assert "a" in breakdown
        assert "b" in breakdown
        assert breakdown["a"]["cost"] == pytest.approx(0.01)
        assert breakdown["a"]["calls"] == 1
        assert breakdown["a"]["prompt_tokens"] == 10
        assert breakdown["b"]["completion_tokens"] == 10

    def test_report_is_string(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", cost=0.05)
        report = tracker.report()
        assert isinstance(report, str)
        assert "gpt-4o" in report
        assert "$" in report


class TestCostCeiling:
    """Test budget enforcement."""

    def test_vote_under_budget(self) -> None:
        ceiling = CostCeiling(max_cost_per_vote=1.00)
        tracker = CostTracker()
        tracker.record("a", cost=0.30)
        # Should not raise
        ceiling.check_vote(tracker, estimated_next=0.30)

    def test_vote_over_budget(self) -> None:
        ceiling = CostCeiling(max_cost_per_vote=0.50)
        tracker = CostTracker()
        tracker.record("a", cost=0.30)
        with pytest.raises(BudgetExceededError) as exc_info:
            ceiling.check_vote(tracker, estimated_next=0.30)
        assert exc_info.value.limit == 0.50
        assert exc_info.value.current == pytest.approx(0.30)
        assert exc_info.value.requested == pytest.approx(0.30)

    def test_debate_under_budget(self) -> None:
        ceiling = CostCeiling(max_cost_per_debate=5.00)
        tracker = CostTracker()
        tracker.record("a", cost=2.00)
        ceiling.check_debate(tracker, estimated_next=1.00)

    def test_debate_over_budget(self) -> None:
        ceiling = CostCeiling(max_cost_per_debate=5.00)
        tracker = CostTracker()
        tracker.record("a", cost=4.50)
        with pytest.raises(BudgetExceededError):
            ceiling.check_debate(tracker, estimated_next=1.00)

    def test_budget_error_message(self) -> None:
        err = BudgetExceededError(limit=1.0, current=0.8, requested=0.5)
        assert "1.0" in str(err)
        assert "0.8" in str(err)


class TestSelectModelsWithinBudget:
    """Test greedy model selection within budget."""

    def test_all_fit(self) -> None:
        # With fallback pricing, small prompt should be cheap enough
        models = ["a", "b", "c"]
        selected = select_models_within_budget(models, "Short prompt", budget=10.0)
        assert selected == ["a", "b", "c"]

    def test_none_fit(self) -> None:
        models = ["a", "b"]
        selected = select_models_within_budget(
            models, "Test", budget=0.0, estimated_completion_tokens=500
        )
        assert selected == []

    def test_greedy_order_preserved(self) -> None:
        """First models in list are preferred."""
        models = ["cheap", "expensive"]
        # With a tiny budget, at least the first model should be selected
        selected = select_models_within_budget(
            models, "Test", budget=0.01, estimated_completion_tokens=10
        )
        # Either both or just the first
        if selected:
            assert selected[0] == "cheap"

    def test_empty_models(self) -> None:
        selected = select_models_within_budget([], "Test", budget=10.0)
        assert selected == []
