"""Stalemate detection and resolution for consensus-council.

Detects when a debate is going in circles (same votes, no new arguments)
and applies a configurable resolution strategy.
"""

from __future__ import annotations

import random
from enum import Enum

from .voting import ConsensusResult, Vote, VoteResult


class StalemateStrategy(str, Enum):
    """How to resolve a stalemate."""
    STOP = "stop"                                       # Accept the tie, return TIE result
    RANDOM_TIEBREAK = "random_tiebreak"                 # Randomly pick YES or NO
    MODERATOR = "moderator"                             # Ask a designated model to break the tie
    ESCALATE_TO_HUMAN = "escalate"                      # Return a special result asking for human input
    STRUCTURED_DISAGREEMENT = "structured_disagreement" # Report where models agree/disagree


def detect_stalemate(
    current_votes: dict[str, Vote],
    prev_votes: dict[str, Vote] | None,
    current_responses: dict[str, str] | None = None,
    prev_responses: dict[str, str] | None = None,
    length_tolerance: float = 0.20,
) -> bool:
    """Detect whether the debate is in a stalemate.

    A stalemate is detected when:
    1. The votes haven't changed between rounds, AND
    2. The response lengths are within `length_tolerance` (20% by default),
       suggesting no substantially new arguments.

    Args:
        current_votes: Model -> Vote mapping for current round.
        prev_votes: Model -> Vote mapping for previous round (None = first round).
        current_responses: Model -> response text for current round.
        prev_responses: Model -> response text for previous round.
        length_tolerance: Fraction by which response lengths may differ (0.2 = 20%).

    Returns:
        True if the debate appears stalled.
    """
    # First round cannot be a stalemate
    if prev_votes is None:
        return False

    # Check if votes changed
    common_models = set(current_votes) & set(prev_votes)
    if not common_models:
        return False

    votes_changed = any(
        current_votes[m] != prev_votes[m] for m in common_models
    )
    if votes_changed:
        return False

    # Votes are the same -- check if arguments evolved
    if current_responses is None or prev_responses is None:
        # No response text to compare; votes alone are enough
        return True

    for model in common_models:
        curr_text = current_responses.get(model, "")
        prev_text = prev_responses.get(model, "")
        curr_len = len(curr_text)
        prev_len = len(prev_text)

        if prev_len == 0 and curr_len == 0:
            continue

        max_len = max(curr_len, prev_len)
        diff = abs(curr_len - prev_len)

        if max_len > 0 and diff / max_len > length_tolerance:
            # Significantly different length => new arguments being made
            return False

    # All models have same votes AND similar-length responses
    return True


def resolve_stalemate(
    votes: list[VoteResult],
    strategy: StalemateStrategy,
    moderator_model: str | None = None,
) -> ConsensusResult:
    """Apply a stalemate-resolution strategy and return a result.

    For StalemateStrategy.MODERATOR, the actual model call must be
    handled by the caller (Council). This function returns a placeholder
    result with instructions in `reasoning`.

    Args:
        votes: The final-round votes.
        strategy: Which resolution strategy to apply.
        moderator_model: Model to use for MODERATOR strategy.

    Returns:
        A ConsensusResult reflecting the chosen strategy.
    """
    vote_map = {v.model: v for v in votes}

    if strategy == StalemateStrategy.STOP:
        return ConsensusResult(
            decision="TIE",
            confidence=0.0,
            votes=vote_map,
            reasoning="Stalemate detected. Debate ended without resolution.",
        )

    if strategy == StalemateStrategy.RANDOM_TIEBREAK:
        pick = random.choice([Vote.YES, Vote.NO])
        return ConsensusResult(
            decision=pick.value,
            confidence=0.5,
            votes=vote_map,
            reasoning=f"Stalemate resolved by random tiebreak: {pick.value}",
        )

    if strategy == StalemateStrategy.MODERATOR:
        # The actual LLM call is performed by Council; we return a marker
        model_label = moderator_model or "unspecified"
        return ConsensusResult(
            decision="PENDING_MODERATOR",
            confidence=0.0,
            votes=vote_map,
            reasoning=(
                f"Stalemate detected. Escalating to moderator model '{model_label}' "
                "for tiebreak."
            ),
        )

    if strategy == StalemateStrategy.ESCALATE_TO_HUMAN:
        yes_models = [v.model for v in votes if v.vote == Vote.YES]
        no_models = [v.model for v in votes if v.vote == Vote.NO]
        return ConsensusResult(
            decision="ESCALATE",
            confidence=0.0,
            votes=vote_map,
            reasoning=(
                f"Stalemate detected. Escalating to human.\n"
                f"  YES: {', '.join(yes_models) or 'none'}\n"
                f"  NO:  {', '.join(no_models) or 'none'}\n"
                f"Please make the final call."
            ),
        )

    if strategy == StalemateStrategy.STRUCTURED_DISAGREEMENT:
        summary = build_disagreement_summary(votes)
        lines = ["Stalemate: models could not reach agreement.", "Vote breakdown:"]
        for vote_val, models in summary["vote_breakdown"].items():
            lines.append(f"  {vote_val}: {', '.join(models)}")
        return ConsensusResult(
            decision="NO_CONSENSUS",
            confidence=0.0,
            votes=vote_map,
            reasoning="\n".join(lines),
        )

    # Fallback (should not happen with enum)
    return ConsensusResult(
        decision="TIE",
        confidence=0.0,
        votes=vote_map,
        reasoning=f"Unknown stalemate strategy: {strategy}",
    )


def build_disagreement_summary(votes: list[VoteResult]) -> dict:
    """Build a structured summary of where models agree and disagree.

    Used when a stalemate cannot be resolved — instead of returning a bare
    TIE, callers can inspect exactly which models voted which way and why.

    Args:
        votes: The final-round votes.

    Returns:
        Dict with vote_breakdown, majority_vote, majority_models, is_split,
        positions (per-model vote + confidence + reasoning summary), and
        optionally minority_vote / minority_models.
    """
    vote_groups: dict[str, list[str]] = {}
    for v in votes:
        key = v.vote.value
        if key not in vote_groups:
            vote_groups[key] = []
        vote_groups[key].append(v.model)

    sorted_groups = sorted(vote_groups.items(), key=lambda x: -len(x[1]))
    majority_vote = sorted_groups[0][0] if sorted_groups else "ABSTAIN"
    majority_models = sorted_groups[0][1] if sorted_groups else []

    result: dict = {
        "vote_breakdown": vote_groups,
        "majority_vote": majority_vote,
        "majority_models": majority_models,
        "is_split": len(vote_groups) > 1,
        "positions": {
            v.model: {
                "vote": v.vote.value,
                "confidence": v.confidence,
                "reasoning_summary": (
                    v.reasoning[:500] + "..." if len(v.reasoning) > 500 else v.reasoning
                ),
            }
            for v in votes
        },
    }

    if len(sorted_groups) > 1:
        result["minority_vote"] = sorted_groups[1][0]
        result["minority_models"] = sorted_groups[1][1]

    return result


def build_moderator_prompt(
    original_prompt: str,
    votes: list[VoteResult],
    context: str = "",
) -> str:
    """Build a prompt for the moderator model to break a tie.

    The moderator sees all arguments and votes and is asked to make
    a final decision.
    """
    parts = [
        "You are a MODERATOR in a multi-model debate. The models have reached a stalemate.",
        "Your job is to review ALL arguments and make the FINAL decision.",
        "",
    ]

    if context:
        parts.append("=== CONTEXT ===")
        parts.append(context)
        parts.append("=== END CONTEXT ===")
        parts.append("")

    parts.append("=== ORIGINAL QUESTION ===")
    parts.append(original_prompt)
    parts.append("=== END QUESTION ===")
    parts.append("")

    parts.append("=== MODEL ARGUMENTS ===")
    for v in votes:
        parts.append(f"--- {v.model} (voted {v.vote.value}, confidence {v.confidence:.2f}) ---")
        parts.append(v.reasoning)
        parts.append("")
    parts.append("=== END ARGUMENTS ===")
    parts.append("")

    parts.append("Based on the QUALITY OF ARGUMENTS (not vote count), make your decision.")
    parts.append("State: FINAL VOTE: YES or FINAL VOTE: NO")

    return "\n".join(parts)
