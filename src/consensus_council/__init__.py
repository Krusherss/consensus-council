"""consensus-council: Drop-in multi-model voting with anti-sycophancy and stalemate resolution."""

from .council import Council
from .voting import ConsensusResult, Vote, VoteResult, extract_score, extract_vote

__all__ = [
    "Council",
    "ConsensusResult",
    "Vote",
    "VoteResult",
    "extract_score",
    "extract_vote",
]

__version__ = "0.1.0"
