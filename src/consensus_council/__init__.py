"""Multi-model voting and three-stage deliberation with blind peer review."""

from .council import Council
from .deliberation import DeliberationResult, PeerReview
from .cost import CostTracker, CostCeiling, BudgetExceededError
from .stalemate import StalemateStrategy, build_disagreement_summary
from .voting import ConsensusResult, Vote, VoteResult, extract_score, extract_vote
from .web_search import search as web_search, has_search_tags, resolve_searches, SEARCH_INSTRUCTION

__all__ = [
    # Core
    "Council",
    "DeliberationResult",
    "PeerReview",
    "ConsensusResult",
    "Vote",
    "VoteResult",
    # Cost
    "CostTracker",
    "CostCeiling",
    "BudgetExceededError",
    # Stalemate
    "StalemateStrategy",
    "build_disagreement_summary",
    # Voting helpers
    "extract_score",
    "extract_vote",
    # Web search
    "web_search",
    "has_search_tags",
    "resolve_searches",
    "SEARCH_INSTRUCTION",
]

__version__ = "0.3.0"
