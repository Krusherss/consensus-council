"""Anti-sycophancy and anti-anchoring measures for consensus-council.

Prevents models from copying each other by:
- Randomising query order to avoid anchoring bias
- Blind voting mode where models see only the original prompt
- Directive injection that instructs models to disagree when warranted
- Debate prompts that frame other responses as arguments to evaluate
"""

from __future__ import annotations

import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Anti-sycophancy directive
# ---------------------------------------------------------------------------

_ANTI_SYCOPHANCY_DIRECTIVE = """\
IMPORTANT INSTRUCTIONS FOR YOUR RESPONSE:
- You are an independent evaluator. Your job is to give your HONEST assessment.
- Do NOT change your vote just because other models disagree with you.
- Only change your position if the LOGIC is undeniable and you genuinely find it compelling.
- Do NOT soften your position to avoid conflict or to be polite.
- Genuine disagreement is MORE VALUABLE than false consensus.
- If you think something is wrong, SAY it is wrong, even if every other model says otherwise.
- State your vote clearly at the end as: FINAL VOTE: YES or FINAL VOTE: NO
"""


def build_anti_sycophancy_directive() -> str:
    """Return the standard anti-sycophancy prompt prefix.

    This should be prepended to prompts sent to models so they resist
    the urge to agree just to be agreeable.
    """
    return _ANTI_SYCOPHANCY_DIRECTIVE


# ---------------------------------------------------------------------------
# Rotation ordering
# ---------------------------------------------------------------------------

@dataclass
class RotationOrder:
    """Generate different query orderings per round to prevent anchoring bias.

    The model that responds first in a debate tends to anchor the others.
    By rotating the order each round, no single model consistently leads.
    """

    n_models: int
    _seed: int | None = None

    def order_for_round(self, round_num: int) -> list[int]:
        """Return a permutation of model indices for the given round.

        Round 0 returns the natural order, subsequent rounds are shuffled
        with a deterministic seed so results are reproducible.
        """
        indices = list(range(self.n_models))
        if round_num == 0:
            return indices
        rng = random.Random(
            (self._seed if self._seed is not None else 42) + round_num
        )
        rng.shuffle(indices)
        return indices

    def all_orders(self, n_rounds: int) -> list[list[int]]:
        """Return orderings for multiple rounds at once."""
        return [self.order_for_round(r) for r in range(n_rounds)]


# ---------------------------------------------------------------------------
# Blind voting
# ---------------------------------------------------------------------------

@dataclass
class BlindVoting:
    """Ensure each model sees ONLY the original prompt, not other models' responses.

    Used for simple vote rounds where cross-contamination is undesirable.
    """

    prompt: str
    context: str = ""

    def build_prompt(self, model_name: str) -> str:
        """Build the complete prompt for a model in blind-voting mode.

        Includes the anti-sycophancy directive, the original prompt, and
        any context -- but explicitly excludes other models' responses.
        """
        parts = [
            build_anti_sycophancy_directive(),
            "",
            f"You are evaluating the following as model '{model_name}'.",
            "You have NOT seen any other model's response. Give your independent assessment.",
            "",
        ]
        if self.context:
            parts.append("=== CONTEXT ===")
            parts.append(self.context)
            parts.append("=== END CONTEXT ===")
            parts.append("")
        parts.append("=== QUESTION ===")
        parts.append(self.prompt)
        parts.append("=== END QUESTION ===")
        parts.append("")
        parts.append(
            "Give your reasoning, then state your final vote as: FINAL VOTE: YES or FINAL VOTE: NO"
        )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Debate / cross-talk prompt
# ---------------------------------------------------------------------------

def build_crosstalk_prompt(
    model_name: str,
    round_num: int,
    prev_responses: dict[str, str],
    prev_votes: dict[str, str],
    original_prompt: str,
    context: str = "",
) -> str:
    """Build a debate prompt where a model can see others' prior arguments.

    The prompt frames previous responses as arguments to EVALUATE, not
    authority to defer to. Anti-sycophancy language is embedded.

    Args:
        model_name: The model being prompted.
        round_num: Current debate round (1-indexed for display).
        prev_responses: Mapping of model name -> previous response text.
        prev_votes: Mapping of model name -> previous vote ("YES"/"NO").
        original_prompt: The original user question.
        context: Optional additional context.

    Returns:
        The complete prompt string for this debate round.
    """
    parts: list[str] = [
        build_anti_sycophancy_directive(),
        "",
        f"=== DEBATE ROUND {round_num} ===",
        f"You are '{model_name}'. This is round {round_num} of a multi-round debate.",
        "",
    ]

    if context:
        parts.append("=== ORIGINAL CONTEXT ===")
        parts.append(context)
        parts.append("=== END CONTEXT ===")
        parts.append("")

    parts.append("=== ORIGINAL QUESTION ===")
    parts.append(original_prompt)
    parts.append("=== END QUESTION ===")
    parts.append("")

    if prev_responses:
        parts.append("=== OTHER MODELS' ARGUMENTS (evaluate on merit, do NOT defer) ===")
        for other_model, response in prev_responses.items():
            if other_model == model_name:
                continue
            vote_str = prev_votes.get(other_model, "UNKNOWN")
            parts.append(f"--- {other_model} (voted {vote_str}) ---")
            # Truncate very long responses to keep context manageable
            if len(response) > 2000:
                parts.append(response[:2000] + "\n[... truncated ...]")
            else:
                parts.append(response)
            parts.append("")
        parts.append("=== END OTHER ARGUMENTS ===")
        parts.append("")

    parts.append("INSTRUCTIONS FOR THIS ROUND:")
    parts.append("1. Consider the arguments above ON THEIR MERITS.")
    parts.append("2. If a new argument is logically compelling, you MAY update your position.")
    parts.append("3. If not, HOLD YOUR GROUND. Do not flip just to reach consensus.")
    parts.append("4. Explain your reasoning, then state: FINAL VOTE: YES or FINAL VOTE: NO")

    return "\n".join(parts)
