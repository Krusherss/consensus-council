"""Tests for stalemate detection and resolution."""

from __future__ import annotations

from consensus_council.stalemate import (
    StalemateStrategy,
    build_moderator_prompt,
    detect_stalemate,
    resolve_stalemate,
)
from consensus_council.voting import Vote, VoteResult


def _make_vote(model: str, vote: Vote, reasoning: str = "Reasons.") -> VoteResult:
    return VoteResult(model=model, vote=vote, confidence=0.8, reasoning=reasoning)


class TestDetectStalemate:
    """Test stalemate detection logic."""

    def test_first_round_never_stalemate(self) -> None:
        current = {"a": Vote.YES, "b": Vote.NO}
        assert detect_stalemate(current, prev_votes=None) is False

    def test_same_votes_same_length_is_stalemate(self) -> None:
        votes = {"a": Vote.YES, "b": Vote.NO}
        responses = {"a": "I think yes." * 10, "b": "I think no." * 10}
        # Same responses for both rounds
        assert detect_stalemate(votes, votes, responses, responses) is True

    def test_changed_votes_not_stalemate(self) -> None:
        prev = {"a": Vote.YES, "b": Vote.NO}
        curr = {"a": Vote.NO, "b": Vote.NO}
        assert detect_stalemate(curr, prev) is False

    def test_same_votes_different_length_not_stalemate(self) -> None:
        votes = {"a": Vote.YES, "b": Vote.NO}
        prev_resp = {"a": "Short.", "b": "Short."}
        curr_resp = {"a": "Much longer response with new arguments." * 20, "b": "Short."}
        assert detect_stalemate(votes, votes, curr_resp, prev_resp) is False

    def test_same_votes_within_tolerance_is_stalemate(self) -> None:
        votes = {"a": Vote.YES, "b": Vote.NO}
        prev_resp = {"a": "x" * 100, "b": "y" * 100}
        curr_resp = {"a": "x" * 110, "b": "y" * 95}  # Within 20%
        assert detect_stalemate(votes, votes, curr_resp, prev_resp) is True

    def test_same_votes_exceeds_tolerance_not_stalemate(self) -> None:
        votes = {"a": Vote.YES, "b": Vote.NO}
        prev_resp = {"a": "x" * 100, "b": "y" * 100}
        curr_resp = {"a": "x" * 150, "b": "y" * 100}  # 50% change, > 20%
        assert detect_stalemate(votes, votes, curr_resp, prev_resp) is False

    def test_no_responses_same_votes_is_stalemate(self) -> None:
        """If no response text provided, same votes alone indicate stalemate."""
        votes = {"a": Vote.YES, "b": Vote.NO}
        assert detect_stalemate(votes, votes) is True

    def test_no_common_models(self) -> None:
        prev = {"a": Vote.YES}
        curr = {"b": Vote.NO}
        assert detect_stalemate(curr, prev) is False

    def test_custom_tolerance(self) -> None:
        votes = {"a": Vote.YES}
        prev_resp = {"a": "x" * 100}
        curr_resp = {"a": "x" * 115}  # 15% change
        # Default 20% tolerance -> stalemate
        assert detect_stalemate(votes, votes, curr_resp, prev_resp, length_tolerance=0.20) is True
        # Stricter 10% tolerance -> not stalemate
        assert detect_stalemate(votes, votes, curr_resp, prev_resp, length_tolerance=0.10) is False


class TestResolveStalemate:
    """Test stalemate resolution strategies."""

    def test_stop_returns_tie(self) -> None:
        votes = [_make_vote("a", Vote.YES), _make_vote("b", Vote.NO)]
        result = resolve_stalemate(votes, StalemateStrategy.STOP)
        assert result.decision == "TIE"
        assert "stalemate" in result.reasoning.lower()

    def test_random_tiebreak_picks_yes_or_no(self) -> None:
        votes = [_make_vote("a", Vote.YES), _make_vote("b", Vote.NO)]
        result = resolve_stalemate(votes, StalemateStrategy.RANDOM_TIEBREAK)
        assert result.decision in ("YES", "NO")
        assert result.confidence == 0.5
        assert "random" in result.reasoning.lower()

    def test_moderator_returns_pending(self) -> None:
        votes = [_make_vote("a", Vote.YES), _make_vote("b", Vote.NO)]
        result = resolve_stalemate(
            votes, StalemateStrategy.MODERATOR, moderator_model="gpt-4o"
        )
        assert result.decision == "PENDING_MODERATOR"
        assert "gpt-4o" in result.reasoning

    def test_escalate_lists_models(self) -> None:
        votes = [
            _make_vote("gpt-4o", Vote.YES),
            _make_vote("claude", Vote.NO),
        ]
        result = resolve_stalemate(votes, StalemateStrategy.ESCALATE_TO_HUMAN)
        assert result.decision == "ESCALATE"
        assert "gpt-4o" in result.reasoning
        assert "claude" in result.reasoning

    def test_all_vote_results_in_output(self) -> None:
        votes = [_make_vote("a", Vote.YES), _make_vote("b", Vote.NO)]
        result = resolve_stalemate(votes, StalemateStrategy.STOP)
        assert "a" in result.votes
        assert "b" in result.votes


class TestBuildModeratorPrompt:
    """Test moderator prompt construction."""

    def test_includes_original_question(self) -> None:
        votes = [_make_vote("a", Vote.YES, "Good code")]
        prompt = build_moderator_prompt("Is this safe?", votes)
        assert "Is this safe?" in prompt

    def test_includes_all_arguments(self) -> None:
        votes = [
            _make_vote("a", Vote.YES, "Argument A"),
            _make_vote("b", Vote.NO, "Argument B"),
        ]
        prompt = build_moderator_prompt("Question", votes)
        assert "Argument A" in prompt
        assert "Argument B" in prompt

    def test_includes_vote_values(self) -> None:
        votes = [
            _make_vote("a", Vote.YES),
            _make_vote("b", Vote.NO),
        ]
        prompt = build_moderator_prompt("Q", votes)
        assert "YES" in prompt
        assert "NO" in prompt

    def test_includes_context(self) -> None:
        votes = [_make_vote("a", Vote.YES)]
        prompt = build_moderator_prompt("Q", votes, context="def foo(): pass")
        assert "def foo(): pass" in prompt

    def test_asks_for_final_vote(self) -> None:
        votes = [_make_vote("a", Vote.YES)]
        prompt = build_moderator_prompt("Q", votes)
        assert "FINAL VOTE" in prompt
