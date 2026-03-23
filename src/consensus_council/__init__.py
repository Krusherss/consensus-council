"""consensus-council: Drop-in multi-model voting with anti-sycophancy and stalemate resolution."""

from .council import Council
from .cost import CostTracker, CostCeiling, BudgetExceededError
from .stalemate import StalemateStrategy, build_disagreement_summary
from .voting import ConsensusResult, Vote, VoteResult, extract_score, extract_vote
from .web_search import search as web_search, has_search_tags, resolve_searches, SEARCH_INSTRUCTION

__all__ = [
    # Core
    "Council",
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

__version__ = "0.2.0"
