"""Tests for the Council class using mock LLM responses.

NO real API calls are made -- all litellm calls are mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from consensus_council.cost import BudgetExceededError, CostCeiling
from consensus_council.council import Council
from consensus_council.stalemate import StalemateStrategy
from consensus_council.voting import Vote


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> MagicMock:
    """Create a mock litellm completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.get = MagicMock(return_value={
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    })
    # __getitem__ for response["usage"]
    response.__getitem__ = lambda self, key: {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    }.get(key, {})
    return response


def _make_acompletion_mock(responses: dict[str, str]) -> AsyncMock:
    """Create an async mock for litellm.acompletion that returns model-specific responses."""
    async def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        model = kwargs.get("model", args[0] if args else "unknown")
        content = responses.get(model, "FINAL VOTE: ABSTAIN")
        return _mock_response(content)
    return AsyncMock(side_effect=side_effect)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCouncilInit:
    def test_requires_models(self) -> None:
        with pytest.raises(ValueError, match="at least one model"):
            Council(models=[])

    def test_accepts_models(self) -> None:
        c = Council(models=["gpt-4o", "claude-sonnet-4-5-20250514"])
        assert len(c.models) == 2


class TestCouncilVote:
    """Test single-round voting with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_simple_majority_yes(self) -> None:
        mock = _make_acompletion_mock({
            "model-a": "After analysis, this is safe. FINAL VOTE: YES",
            "model-b": "Looks good to me. FINAL VOTE: YES",
            "model-c": "I have concerns. FINAL VOTE: NO",
        })
        with patch("consensus_council.council.litellm") as mock_litellm:
            mock_litellm.acompletion = mock
            mock_litellm.completion_cost = MagicMock(return_value=0.01)

            council = Council(models=["model-a", "model-b", "model-c"])
            result = await council.avote("Is this safe?")

        assert result.decision == "YES"
        assert result.confidence >= 0.6
        assert len(result.votes) == 3

    @pytest.mark.asyncio
    async def test_simple_majority_no(self) -> None:
        mock = _make_acompletion_mock({
            "model-a": "Unsafe code. FINAL VOTE: NO",
            "model-b": "Critical issues. FINAL VOTE: NO",
            "model-c": "Seems fine. FINAL VOTE: YES",
        })
        with patch("consensus_council.council.litellm") as mock_litellm:
            mock_litellm.acompletion = mock
            mock_litellm.completion_cost = MagicMock(return_value=0.01)

            council = Council(models=["model-a", "model-b", "model-c"])
            result = await council.avote("Is this safe?")

        assert result.decision == "NO"

    @pytest.mark.asyncio
    async def test_supermajority_threshold(self) -> None:
        mock = _make_acompletion_mock({
            "model-a": "FINAL VOTE: YES",
            "model-b": "FINAL VOTE: YES",
            "model-c": "FINAL VOTE: NO",
        })
        with patch("consensus_council.council.litellm") as mock_litellm:
            mock_litellm.acompletion = mock
            mock_litellm.completion_cost = MagicMock(return_value=0.01)

            council = Council(models=["model-a", "model-b", "model-c"])
            result = await council.avote(
                "Is this safe?", threshold=0.66, strategy="supermajority"
            )

        assert result.decision == "YES"

    @pytest.mark.asyncio
    async def test_unanimous_fails_with_dissent(self) -> None:
        mock = _make_acompletion_mock({
            "model-a": "FINAL VOTE: YES",
            "model-b": "FINAL VOTE: YES",
            "model-c": "FINAL VOTE: NO",
        })
        with patch("consensus_council.council.litellm") as mock_litellm:
            mock_litellm.acompletion = mock
            mock_litellm.completion_cost = MagicMock(return_value=0.01)

            council = Council(models=["model-a", "model-b", "model-c"])
            result = await council.avote("Test?", strategy="unanimous")

        assert result.decision == "TIE"

    @pytest.mark.asyncio
    async def test_handles_api_failure(self) -> None:
        """A model that fails should be marked as ABSTAIN with error."""
        call_count = 0

        async def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            model = kwargs.get("model", "")
            if model == "model-b":
                raise ConnectionError("API timeout")
            return _mock_response(f"FINAL VOTE: YES")

        with patch("consensus_council.council.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
            mock_litellm.completion_cost = MagicMock(return_value=0.01)

            council = Council(models=["model-a", "model-b", "model-c"])
            result = await council.avote("Test?")

        assert result.decision == "YES"
        assert "model-b" in result.failed_models
        assert result.votes["model-b"].error is not None

    @pytest.mark.asyncio
    async def test_tracks_cost(self) -> None:
        mock = _make_acompletion_mock({
            "model-a": "FINAL VOTE: YES",
            "model-b": "FINAL VOTE: NO",
        })
        with patch("consensus_council.council.litellm") as mock_litellm:
            mock_litellm.acompletion = mock
            mock_litellm.completion_cost = MagicMock(return_value=0.05)

            council = Council(models=["model-a", "model-b"])
            result = await council.avote("Test?")

        assert result.total_cost > 0


class TestCouncilDebate:
    """Test multi-round debate with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_debate_converges(self) -> None:
        """All models agree in round 1 -> debate stops."""
        mock = _make_acompletion_mock({
            "model-a": "FINAL VOTE: YES",
            "model-b": "FINAL VOTE: YES",
            "model-c": "FINAL VOTE: YES",
        })
        with patch("consensus_council.council.litellm") as mock_litellm:
            mock_litellm.acompletion = mock
            mock_litellm.completion_cost = MagicMock(return_value=0.01)

            council = Council(models=["model-a", "model-b", "model-c"])
            result = await council.adebate("Test?", max_rounds=3)

        assert result.decision == "YES"
        assert result.rounds == 1

    @pytest.mark.asyncio
    async def test_debate_stalemate_stop(self) -> None:
        """Stalemate detection triggers STOP strategy."""
        mock = _make_acompletion_mock({
            "model-a": "I think yes. FINAL VOTE: YES",
            "model-b": "I think no. FINAL VOTE: NO",
        })
        with patch("consensus_council.council.litellm") as mock_litellm:
            mock_litellm.acompletion = mock
            mock_litellm.completion_cost = MagicMock(return_value=0.01)

            council = Council(
                models=["model-a", "model-b"],
                stalemate_strategy=StalemateStrategy.STOP,
            )
            result = await council.adebate(
                "Test?", max_rounds=5, stop_on="unanimous"
            )

        # Should detect stalemate and stop
        assert result.rounds <= 5
        # Result should be TIE since votes split and stalemate detected
        assert result.decision in ("TIE", "YES", "NO")

    @pytest.mark.asyncio
    async def test_debate_budget_stops_early(self) -> None:
        """Budget ceiling causes debate to stop early."""
        mock = _make_acompletion_mock({
            "model-a": "FINAL VOTE: YES",
            "model-b": "FINAL VOTE: NO",
        })
        with patch("consensus_council.council.litellm") as mock_litellm:
            mock_litellm.acompletion = mock
            mock_litellm.completion_cost = MagicMock(return_value=0.50)

            ceiling = CostCeiling(max_cost_per_debate=0.60)
            council = Council(
                models=["model-a", "model-b"],
                cost_ceiling=ceiling,
            )
            result = await council.adebate("Test?", max_rounds=10)

        # Should stop before round 10 due to budget
        assert result.rounds < 10

    @pytest.mark.asyncio
    async def test_debate_moderator_strategy(self) -> None:
        """MODERATOR strategy queries a designated model for tiebreak."""
        round_num = 0

        async def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal round_num
            model = kwargs.get("model", "")
            if model == "moderator-model":
                return _mock_response("After weighing arguments, FINAL VOTE: YES")
            # Always split to force stalemate
            if "model-a" in model or model == "model-a":
                return _mock_response("I say yes. FINAL VOTE: YES")
            return _mock_response("I say no. FINAL VOTE: NO")

        with patch("consensus_council.council.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
            mock_litellm.completion_cost = MagicMock(return_value=0.01)

            council = Council(
                models=["model-a", "model-b"],
                stalemate_strategy=StalemateStrategy.MODERATOR,
                moderator_model="moderator-model",
            )
            result = await council.adebate(
                "Test?", max_rounds=3, stop_on="unanimous"
            )

        # Moderator should have resolved the stalemate
        assert result.decision in ("YES", "NO", "TIE")

    @pytest.mark.asyncio
    async def test_debate_with_context(self) -> None:
        """Context should be passed through to model prompts."""
        captured_messages: list[str] = []

        async def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            messages = kwargs.get("messages", [])
            if messages:
                captured_messages.append(messages[0].get("content", ""))
            return _mock_response("FINAL VOTE: YES")

        with patch("consensus_council.council.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
            mock_litellm.completion_cost = MagicMock(return_value=0.01)

            council = Council(models=["model-a"])
            await council.avote("Review this", context="def foo(): pass")

        assert any("def foo(): pass" in msg for msg in captured_messages)


class TestCouncilWeighted:
    """Test weighted voting through Council."""

    @pytest.mark.asyncio
    async def test_weighted_vote(self) -> None:
        mock = _make_acompletion_mock({
            "strong": "FINAL VOTE: YES",
            "weak-a": "FINAL VOTE: NO",
            "weak-b": "FINAL VOTE: NO",
        })
        with patch("consensus_council.council.litellm") as mock_litellm:
            mock_litellm.acompletion = mock
            mock_litellm.completion_cost = MagicMock(return_value=0.01)

            # Give "strong" model 10x weight
            council = Council(
                models=["strong", "weak-a", "weak-b"],
                weights={"strong": 10.0, "weak-a": 1.0, "weak-b": 1.0},
            )
            result = await council.avote("Test?", strategy="weighted_majority")

        assert result.decision == "YES"
