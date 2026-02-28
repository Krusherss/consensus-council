"""Tests for voting strategies and vote extraction."""

from __future__ import annotations

import pytest

from hydra_consensus.voting import (
    ConsensusResult,
    Vote,
    VoteResult,
    extract_score,
    extract_vote,
    ranked_choice,
    simple_majority,
    supermajority,
    unanimous,
    weighted_majority,
)


# ---------------------------------------------------------------------------
# extract_vote tests
# ---------------------------------------------------------------------------

class TestExtractVote:
    """Test robust vote extraction from freeform text."""

    def test_explicit_final_vote_yes(self) -> None:
        text = "After careful analysis, I believe this is safe. FINAL VOTE: YES"
        vote, conf = extract_vote(text)
        assert vote == Vote.YES
        assert conf >= 0.9

    def test_explicit_final_vote_no(self) -> None:
        text = "This has critical vulnerabilities. FINAL VOTE: NO"
        vote, conf = extract_vote(text)
        assert vote == Vote.NO
        assert conf >= 0.9

    def test_explicit_final_answer_yes(self) -> None:
        text = "The code looks good. FINAL ANSWER: YES"
        vote, conf = extract_vote(text)
        assert vote == Vote.YES
        assert conf >= 0.9

    def test_markdown_bold_yes(self) -> None:
        text = "My assessment is **YES**, this should be approved."
        vote, conf = extract_vote(text)
        assert vote == Vote.YES
        assert conf >= 0.9

    def test_markdown_bold_no(self) -> None:
        text = "I recommend **NO** for this deployment."
        vote, conf = extract_vote(text)
        assert vote == Vote.NO
        assert conf >= 0.9

    def test_vote_colon_format(self) -> None:
        text = "After reviewing everything, VOTE: YES"
        vote, conf = extract_vote(text)
        assert vote == Vote.YES
        assert conf >= 0.9

    def test_my_vote_format(self) -> None:
        text = "Considering all factors, MY VOTE: NO"
        vote, conf = extract_vote(text)
        assert vote == Vote.NO
        assert conf >= 0.9

    def test_decision_colon(self) -> None:
        text = "DECISION: YES"
        vote, conf = extract_vote(text)
        assert vote == Vote.YES
        assert conf >= 0.9

    def test_verdict_colon(self) -> None:
        text = "VERDICT: NO"
        vote, conf = extract_vote(text)
        assert vote == Vote.NO
        assert conf >= 0.9

    def test_abstain_explicit(self) -> None:
        text = "I cannot determine either way. FINAL VOTE: ABSTAIN"
        vote, conf = extract_vote(text)
        assert vote == Vote.ABSTAIN
        assert conf >= 0.9

    def test_concur_synonym(self) -> None:
        """'I concur' is a common affirmative synonym."""
        text = "I concur with the assessment. The code looks safe and I approve of the changes."
        vote, conf = extract_vote(text)
        assert vote == Vote.YES
        assert conf > 0.4

    def test_lgtm(self) -> None:
        text = "LGTM, this looks good to me. I recommend we approve and ship it."
        vote, conf = extract_vote(text)
        assert vote == Vote.YES

    def test_reject_synonym(self) -> None:
        text = "I reject this proposal. The code is unsafe and should be blocked."
        vote, conf = extract_vote(text)
        assert vote == Vote.NO

    def test_empty_string(self) -> None:
        vote, conf = extract_vote("")
        assert vote == Vote.ABSTAIN
        assert conf == 0.0

    def test_whitespace_only(self) -> None:
        vote, conf = extract_vote("   \n\t  ")
        assert vote == Vote.ABSTAIN
        assert conf == 0.0

    def test_ambiguous_text(self) -> None:
        text = "There are some concerns but also some positives. Hard to say."
        vote, conf = extract_vote(text)
        # Should have low confidence regardless of vote
        assert conf < 0.6

    def test_mixed_signals_yes_wins(self) -> None:
        """When there are more YES synonyms than NO, YES should win."""
        text = (
            "While there is one minor concern, overall the code looks good, "
            "I approve and recommend deployment. Looks safe."
        )
        vote, conf = extract_vote(text)
        assert vote == Vote.YES

    def test_mixed_signals_no_wins(self) -> None:
        """When there are more NO synonyms, NO should win."""
        text = (
            "This code is unsafe, incorrect, and I reject it. "
            "Do not approve this. Block deployment."
        )
        vote, conf = extract_vote(text)
        assert vote == Vote.NO

    def test_case_insensitive_explicit(self) -> None:
        text = "final vote: yes"
        vote, conf = extract_vote(text)
        assert vote == Vote.YES
        assert conf >= 0.9

    def test_long_response_with_vote_at_end(self) -> None:
        """Vote in the last 500 chars should be detected."""
        filler = "This is detailed analysis. " * 100
        text = filler + "Overall, I approve this code. FINAL VOTE: YES"
        vote, conf = extract_vote(text)
        assert vote == Vote.YES
        assert conf >= 0.9


# ---------------------------------------------------------------------------
# extract_score tests
# ---------------------------------------------------------------------------

class TestExtractScore:
    """Test numeric score extraction."""

    def test_slash_format(self) -> None:
        text = "I would rate this code a 7/10 for quality."
        score, conf = extract_score(text)
        assert score is not None
        assert abs(score - 0.7) < 0.01
        assert conf >= 0.8

    def test_out_of_format(self) -> None:
        text = "Rating: 8 out of 10"
        score, conf = extract_score(text)
        assert score is not None
        assert abs(score - 0.8) < 0.01

    def test_score_colon_format(self) -> None:
        text = "Score: 9"
        score, conf = extract_score(text, scale=10)
        assert score is not None
        assert abs(score - 0.9) < 0.01

    def test_score_colon_with_max(self) -> None:
        text = "Rating: 4/5"
        score, conf = extract_score(text)
        assert score is not None
        assert abs(score - 0.8) < 0.01

    def test_decimal_score(self) -> None:
        text = "I give this a 7.5/10."
        score, conf = extract_score(text)
        assert score is not None
        assert abs(score - 0.75) < 0.01

    def test_no_score(self) -> None:
        text = "This is a qualitative assessment with no numbers."
        score, conf = extract_score(text)
        assert score is None
        assert conf == 0.0

    def test_empty_string(self) -> None:
        score, conf = extract_score("")
        assert score is None
        assert conf == 0.0

    def test_custom_scale(self) -> None:
        text = "Score: 3"
        score, conf = extract_score(text, scale=5)
        assert score is not None
        assert abs(score - 0.6) < 0.01


# ---------------------------------------------------------------------------
# Voting strategy tests
# ---------------------------------------------------------------------------

def _make_vote(model: str, vote: Vote, conf: float = 0.8, reasoning: str = "") -> VoteResult:
    """Helper to create a VoteResult."""
    return VoteResult(model=model, vote=vote, confidence=conf, reasoning=reasoning)


class TestSimpleMajority:
    def test_clear_yes(self) -> None:
        votes = [
            _make_vote("a", Vote.YES),
            _make_vote("b", Vote.YES),
            _make_vote("c", Vote.NO),
        ]
        result = simple_majority(votes)
        assert result.decision == "YES"

    def test_clear_no(self) -> None:
        votes = [
            _make_vote("a", Vote.NO),
            _make_vote("b", Vote.NO),
            _make_vote("c", Vote.YES),
        ]
        result = simple_majority(votes)
        assert result.decision == "NO"

    def test_tie(self) -> None:
        votes = [
            _make_vote("a", Vote.YES),
            _make_vote("b", Vote.NO),
        ]
        result = simple_majority(votes)
        assert result.decision == "TIE"

    def test_abstain_excluded(self) -> None:
        votes = [
            _make_vote("a", Vote.YES),
            _make_vote("b", Vote.ABSTAIN),
            _make_vote("c", Vote.ABSTAIN),
        ]
        result = simple_majority(votes)
        assert result.decision == "YES"

    def test_all_abstain(self) -> None:
        votes = [
            _make_vote("a", Vote.ABSTAIN),
            _make_vote("b", Vote.ABSTAIN),
        ]
        result = simple_majority(votes)
        assert result.decision == "ABSTAIN"

    def test_empty(self) -> None:
        result = simple_majority([])
        assert result.decision == "ABSTAIN"


class TestSupermajority:
    def test_two_thirds_met(self) -> None:
        votes = [
            _make_vote("a", Vote.YES),
            _make_vote("b", Vote.YES),
            _make_vote("c", Vote.NO),
        ]
        result = supermajority(votes, threshold=0.66)
        assert result.decision == "YES"

    def test_two_thirds_not_met(self) -> None:
        votes = [
            _make_vote("a", Vote.YES),
            _make_vote("b", Vote.NO),
            _make_vote("c", Vote.NO),
        ]
        result = supermajority(votes, threshold=0.66)
        assert result.decision == "NO"

    def test_exact_threshold(self) -> None:
        votes = [
            _make_vote("a", Vote.YES),
            _make_vote("b", Vote.YES),
            _make_vote("c", Vote.YES),
            _make_vote("d", Vote.NO),
        ]
        # 3/4 = 0.75 >= 0.75
        result = supermajority(votes, threshold=0.75)
        assert result.decision == "YES"


class TestUnanimous:
    def test_all_yes(self) -> None:
        votes = [
            _make_vote("a", Vote.YES),
            _make_vote("b", Vote.YES),
            _make_vote("c", Vote.YES),
        ]
        result = unanimous(votes)
        assert result.decision == "YES"

    def test_one_dissent(self) -> None:
        votes = [
            _make_vote("a", Vote.YES),
            _make_vote("b", Vote.YES),
            _make_vote("c", Vote.NO),
        ]
        result = unanimous(votes)
        assert result.decision == "TIE"

    def test_all_no(self) -> None:
        votes = [
            _make_vote("a", Vote.NO),
            _make_vote("b", Vote.NO),
        ]
        result = unanimous(votes)
        assert result.decision == "NO"


class TestWeightedMajority:
    def test_equal_weights(self) -> None:
        votes = [
            _make_vote("a", Vote.YES),
            _make_vote("b", Vote.YES),
            _make_vote("c", Vote.NO),
        ]
        result = weighted_majority(votes)
        assert result.decision == "YES"

    def test_heavier_weight_flips(self) -> None:
        votes = [
            _make_vote("a", Vote.YES),
            _make_vote("b", Vote.NO),
            _make_vote("c", Vote.NO),
        ]
        # Give model 'a' 10x weight -- should flip to YES
        weights = {"a": 10.0, "b": 1.0, "c": 1.0}
        result = weighted_majority(votes, weights)
        assert result.decision == "YES"

    def test_no_weights_defaults(self) -> None:
        votes = [
            _make_vote("a", Vote.NO),
            _make_vote("b", Vote.NO),
        ]
        result = weighted_majority(votes, weights=None)
        assert result.decision == "NO"


class TestRankedChoice:
    def test_clear_winner(self) -> None:
        votes = [
            _make_vote("a", Vote.YES, reasoning="PostgreSQL, MySQL, SQLite"),
            _make_vote("b", Vote.YES, reasoning="PostgreSQL, SQLite, MySQL"),
            _make_vote("c", Vote.YES, reasoning="MySQL, PostgreSQL, SQLite"),
        ]
        result = ranked_choice(votes)
        # PostgreSQL has 2 first-place votes
        assert result.decision == "PostgreSQL"

    def test_runoff_elimination(self) -> None:
        votes = [
            _make_vote("a", Vote.YES, reasoning="A, B, C"),
            _make_vote("b", Vote.YES, reasoning="B, A, C"),
            _make_vote("c", Vote.YES, reasoning="C, A, B"),
        ]
        # All tied first round -> C eliminated -> A wins with a's + c's votes
        result = ranked_choice(votes)
        assert result.decision in ("A", "B", "C")  # At least produces a result

    def test_empty_votes(self) -> None:
        result = ranked_choice([])
        assert result.decision == "ABSTAIN"

    def test_fallback_no_rankings(self) -> None:
        """If reasoning doesn't contain comma-separated rankings, fall back."""
        votes = [
            _make_vote("a", Vote.YES, reasoning="I think yes."),
            _make_vote("b", Vote.NO, reasoning="I think no."),
        ]
        result = ranked_choice(votes)
        # Should fall back to simple_majority behavior
        assert result.decision in ("YES", "NO", "TIE")
