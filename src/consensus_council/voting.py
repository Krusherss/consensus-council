"""Voting strategies and vote extraction for consensus-council.

Provides robust extraction of votes (YES/NO/ABSTAIN) and numeric scores
from freeform LLM text, plus configurable voting strategies from simple
majority to ranked choice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Vote(str, Enum):
    """Possible vote values."""
    YES = "YES"
    NO = "NO"
    ABSTAIN = "ABSTAIN"


@dataclass
class VoteResult:
    """A single model's vote with metadata."""
    model: str
    vote: Vote
    confidence: float  # 0.0 - 1.0
    reasoning: str
    raw_response: str = ""
    error: str | None = None


@dataclass
class ConsensusResult:
    """Aggregated result from a voting round or debate."""
    decision: str  # "YES", "NO", "ABSTAIN", "TIE", or free-text for ranked-choice
    confidence: float  # 0.0 - 1.0
    votes: dict[str, VoteResult] = field(default_factory=dict)
    reasoning: str = ""
    rounds: int = 1
    total_cost: float = 0.0
    failed_models: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Vote extraction
# ---------------------------------------------------------------------------

# Explicit markers that models often use
_EXPLICIT_PATTERNS: list[tuple[re.Pattern[str], Vote]] = [
    # FINAL VOTE: YES / FINAL ANSWER: NO  etc.
    (re.compile(r"FINAL\s+(?:VOTE|ANSWER|DECISION)\s*:\s*(YES|NO|ABSTAIN)", re.I), None),  # type: ignore[arg-type]
    # **YES** or **NO** (markdown bold)
    (re.compile(r"\*\*(YES|NO|ABSTAIN)\*\*", re.I), None),  # type: ignore[arg-type]
    # VOTE: YES
    (re.compile(r"(?:MY\s+)?VOTE\s*:\s*(YES|NO|ABSTAIN)", re.I), None),  # type: ignore[arg-type]
    # DECISION: YES
    (re.compile(r"DECISION\s*:\s*(YES|NO|ABSTAIN)", re.I), None),  # type: ignore[arg-type]
    # VERDICT: YES
    (re.compile(r"VERDICT\s*:\s*(YES|NO|ABSTAIN)", re.I), None),  # type: ignore[arg-type]
    # ANSWER: YES
    (re.compile(r"ANSWER\s*:\s*(YES|NO|ABSTAIN)", re.I), None),  # type: ignore[arg-type]
]

# Affirmative / negative synonyms for fallback counting
_YES_WORDS = {
    "yes", "approve", "approved", "accept", "accepted", "agree", "agreed",
    "safe", "correct", "true", "affirmative", "pass", "passed", "good",
    "recommend", "recommended", "concur", "support", "supported", "lgtm",
    "ship it", "looks good",
}

_NO_WORDS = {
    "no", "reject", "rejected", "deny", "denied", "disagree", "disagreed",
    "unsafe", "incorrect", "false", "negative", "fail", "failed", "bad",
    "do not recommend", "block", "blocked", "oppose", "opposed", "nack",
    "not safe", "do not approve", "not recommended",
}


def extract_vote(response_text: str) -> tuple[Vote, float]:
    """Extract a YES/NO/ABSTAIN vote from freeform model response.

    Returns (vote, confidence) where confidence is 0.0-1.0 indicating
    how certain we are about the extraction.

    Strategy:
    1. Look for explicit vote markers (FINAL VOTE: YES, **NO**, etc.)
    2. Fall back to counting YES/NO synonyms in the last 500 chars
    3. If still ambiguous, return ABSTAIN
    """
    if not response_text or not response_text.strip():
        return Vote.ABSTAIN, 0.0

    text = response_text.strip()

    # --- Phase 1: Explicit markers (high confidence) ---
    for pattern, _ in _EXPLICIT_PATTERNS:
        match = pattern.search(text)
        if match:
            raw = match.group(1).upper()
            return Vote(raw), 0.95

    # --- Phase 2: Synonym counting in the tail (medium confidence) ---
    # Models tend to state their conclusion at the end
    tail = text[-500:].lower()

    yes_count = sum(1 for w in _YES_WORDS if w in tail)
    no_count = sum(1 for w in _NO_WORDS if w in tail)

    total = yes_count + no_count
    if total == 0:
        return Vote.ABSTAIN, 0.1

    if yes_count > no_count:
        confidence = min(0.85, 0.5 + (yes_count - no_count) / total * 0.4)
        return Vote.YES, confidence
    elif no_count > yes_count:
        confidence = min(0.85, 0.5 + (no_count - yes_count) / total * 0.4)
        return Vote.NO, confidence
    else:
        # Tie in word counts
        return Vote.ABSTAIN, 0.2


def extract_score(response_text: str, scale: int = 10) -> tuple[float | None, float]:
    """Extract a numeric score from freeform model response.

    Looks for patterns like '7/10', '8 out of 10', 'score: 7',
    'rating: 8/10', etc.

    Returns (normalized_score, confidence) where normalized_score is
    0.0-1.0 (mapped to the given scale) or None if not found.
    """
    if not response_text or not response_text.strip():
        return None, 0.0

    text = response_text.strip()

    patterns = [
        # "7/10", "8/10"
        re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+)", re.I),
        # "8 out of 10"
        re.compile(r"(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+)", re.I),
        # "score: 7" or "score: 7/10"
        re.compile(r"(?:score|rating|grade)\s*:\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+))?", re.I),
    ]

    for pattern in patterns:
        match = pattern.search(text)
        if match:
            value = float(match.group(1))
            max_val = float(match.group(2)) if match.group(2) else float(scale)
            if max_val > 0:
                normalized = min(1.0, max(0.0, value / max_val))
                return normalized, 0.9
            return None, 0.0

    # Try bare number at end of response (low confidence)
    tail_match = re.search(r"\b(\d+(?:\.\d+)?)\b", text[-100:])
    if tail_match:
        value = float(tail_match.group(1))
        if 0 <= value <= scale:
            return value / scale, 0.4

    return None, 0.0


# ---------------------------------------------------------------------------
# Voting strategies
# ---------------------------------------------------------------------------

def simple_majority(votes: Sequence[VoteResult]) -> ConsensusResult:
    """Simple majority: > 50% agreement wins."""
    return _threshold_vote(votes, threshold=0.5, label="simple_majority")


def supermajority(votes: Sequence[VoteResult], threshold: float = 0.66) -> ConsensusResult:
    """Supermajority: configurable threshold (default 2/3)."""
    return _threshold_vote(votes, threshold=threshold, label="supermajority")


def unanimous(votes: Sequence[VoteResult]) -> ConsensusResult:
    """All non-abstaining models must agree."""
    return _threshold_vote(votes, threshold=1.0, label="unanimous")


def weighted_majority(
    votes: Sequence[VoteResult],
    weights: dict[str, float] | None = None,
) -> ConsensusResult:
    """Per-model reliability weights. Defaults to equal weight."""
    if not votes:
        return ConsensusResult(decision="ABSTAIN", confidence=0.0)

    effective = [v for v in votes if v.vote != Vote.ABSTAIN]
    if not effective:
        return ConsensusResult(
            decision="ABSTAIN",
            confidence=0.0,
            votes={v.model: v for v in votes},
        )

    default_weight = 1.0
    yes_weight = 0.0
    no_weight = 0.0
    total_weight = 0.0

    for v in effective:
        w = (weights or {}).get(v.model, default_weight)
        total_weight += w
        if v.vote == Vote.YES:
            yes_weight += w
        elif v.vote == Vote.NO:
            no_weight += w

    if total_weight == 0:
        return ConsensusResult(
            decision="ABSTAIN",
            confidence=0.0,
            votes={v.model: v for v in votes},
        )

    yes_ratio = yes_weight / total_weight
    no_ratio = no_weight / total_weight

    if yes_ratio > no_ratio:
        decision = "YES"
        confidence = yes_ratio
    elif no_ratio > yes_ratio:
        decision = "NO"
        confidence = no_ratio
    else:
        decision = "TIE"
        confidence = 0.5

    reasoning_parts = [f"{v.model}: {v.vote.value} ({v.reasoning})" for v in votes]
    return ConsensusResult(
        decision=decision,
        confidence=confidence,
        votes={v.model: v for v in votes},
        reasoning="\n".join(reasoning_parts),
    )


def ranked_choice(votes: Sequence[VoteResult]) -> ConsensusResult:
    """Ranked-choice voting for multi-option questions.

    Each VoteResult's `reasoning` field should contain comma-separated
    ranked preferences (e.g. "PostgreSQL, MySQL, SQLite").
    Falls back to first-choice counting if parsing fails.
    """
    if not votes:
        return ConsensusResult(decision="ABSTAIN", confidence=0.0)

    # Parse ranked preferences from reasoning
    ballots: list[list[str]] = []
    for v in votes:
        choices = [c.strip() for c in v.reasoning.split(",") if c.strip()]
        if choices:
            ballots.append(choices)

    if not ballots:
        # Fallback: use vote value as the single choice
        return simple_majority(votes)

    # If no ballot contains more than one choice, there are no real
    # rankings -- fall back to simple majority on the vote values.
    if all(len(b) <= 1 for b in ballots):
        return simple_majority(votes)

    # Collect all candidates
    all_candidates: set[str] = set()
    for ballot in ballots:
        all_candidates.update(ballot)

    remaining = set(all_candidates)

    # Instant-runoff rounds
    for _ in range(len(all_candidates)):
        if not remaining:
            break

        # Count first-place votes among remaining candidates
        counts: dict[str, int] = {c: 0 for c in remaining}
        for ballot in ballots:
            for choice in ballot:
                if choice in remaining:
                    counts[choice] += 1
                    break

        total = sum(counts.values())
        if total == 0:
            break

        # Check for majority
        for candidate, count in sorted(counts.items(), key=lambda x: -x[1]):
            if count / total > 0.5:
                return ConsensusResult(
                    decision=candidate,
                    confidence=count / total,
                    votes={v.model: v for v in votes},
                    reasoning=f"Won ranked-choice with {count}/{total} votes",
                )

        # Eliminate lowest
        min_count = min(counts.values())
        eliminated = {c for c, n in counts.items() if n == min_count}
        remaining -= eliminated

    # No majority found -- pick plurality winner
    counts_final: dict[str, int] = {}
    for ballot in ballots:
        if ballot:
            first = ballot[0]
            counts_final[first] = counts_final.get(first, 0) + 1

    if counts_final:
        winner = max(counts_final, key=lambda k: counts_final[k])
        return ConsensusResult(
            decision=winner,
            confidence=counts_final[winner] / len(ballots),
            votes={v.model: v for v in votes},
            reasoning=f"Plurality winner with {counts_final[winner]}/{len(ballots)} first-choice votes",
        )

    return ConsensusResult(
        decision="TIE",
        confidence=0.0,
        votes={v.model: v for v in votes},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _threshold_vote(
    votes: Sequence[VoteResult],
    threshold: float,
    label: str,
) -> ConsensusResult:
    """Generic threshold-based voting."""
    if not votes:
        return ConsensusResult(decision="ABSTAIN", confidence=0.0)

    effective = [v for v in votes if v.vote != Vote.ABSTAIN]
    if not effective:
        return ConsensusResult(
            decision="ABSTAIN",
            confidence=0.0,
            votes={v.model: v for v in votes},
        )

    yes_count = sum(1 for v in effective if v.vote == Vote.YES)
    no_count = sum(1 for v in effective if v.vote == Vote.NO)
    total = len(effective)

    yes_ratio = yes_count / total
    no_ratio = no_count / total

    # A tie is when neither side exceeds the threshold by a strict margin,
    # or when both sides have equal votes.  For threshold-based strategies
    # (simple_majority uses 0.5, supermajority uses 0.66, unanimous uses 1.0),
    # we require the winning side to *strictly exceed* the losing side AND
    # meet the threshold.
    if yes_ratio > no_ratio and yes_ratio >= threshold:
        decision = "YES"
        confidence = yes_ratio
    elif no_ratio > yes_ratio and no_ratio >= threshold:
        decision = "NO"
        confidence = no_ratio
    else:
        decision = "TIE"
        confidence = max(yes_ratio, no_ratio)

    reasoning_parts = [f"{v.model}: {v.vote.value} ({v.reasoning})" for v in votes]
    return ConsensusResult(
        decision=decision,
        confidence=confidence,
        votes={v.model: v for v in votes},
        reasoning="\n".join(reasoning_parts),
    )
