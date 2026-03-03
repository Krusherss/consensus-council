"""Cost ceiling and budget management for consensus-council.

Tracks per-model and total spend, estimates costs before calls,
and enforces hard budget limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import litellm


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate the cost (USD) for a model call.

    Uses litellm's cost-per-token data when available, falling back to
    a conservative default.

    Args:
        model: LiteLLM model string (e.g. "gpt-4o").
        prompt_tokens: Estimated input token count.
        completion_tokens: Estimated output token count.

    Returns:
        Estimated cost in USD.
    """
    try:
        cost = litellm.completion_cost(
            model=model,
            prompt=str(prompt_tokens),
            completion=str(completion_tokens),
        )
        if cost is not None and cost > 0:
            return float(cost)
    except Exception:
        pass

    # Fallback: try to get model info from litellm
    try:
        info = litellm.get_model_info(model)
        input_cost = info.get("input_cost_per_token", 0.0)
        output_cost = info.get("output_cost_per_token", 0.0)
        if input_cost > 0 or output_cost > 0:
            return prompt_tokens * input_cost + completion_tokens * output_cost
    except Exception:
        pass

    # Hard fallback: assume expensive model pricing ($10/1M input, $30/1M output)
    return prompt_tokens * 10.0 / 1_000_000 + completion_tokens * 30.0 / 1_000_000


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------

@dataclass
class _ModelCostEntry:
    """Internal tracking entry for a single model."""
    total_cost: float = 0.0
    call_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


@dataclass
class CostTracker:
    """Accumulates costs across a vote or debate session.

    Records per-model and total spend, and can report a breakdown.
    """

    _models: dict[str, _ModelCostEntry] = field(default_factory=dict)
    _start_time: float = field(default_factory=time.time)

    def record(
        self,
        model: str,
        cost: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record the cost of a single model call."""
        if model not in self._models:
            self._models[model] = _ModelCostEntry()
        entry = self._models[model]
        entry.total_cost += cost
        entry.call_count += 1
        entry.total_prompt_tokens += prompt_tokens
        entry.total_completion_tokens += completion_tokens

    @property
    def total_cost(self) -> float:
        """Total cost across all models."""
        return sum(e.total_cost for e in self._models.values())

    @property
    def total_calls(self) -> int:
        """Total number of model calls."""
        return sum(e.call_count for e in self._models.values())

    def cost_for_model(self, model: str) -> float:
        """Cost for a specific model."""
        entry = self._models.get(model)
        return entry.total_cost if entry else 0.0

    def breakdown(self) -> dict[str, dict[str, float | int]]:
        """Return a per-model cost breakdown.

        Returns:
            Dict of model -> {cost, calls, prompt_tokens, completion_tokens}.
        """
        result: dict[str, dict[str, float | int]] = {}
        for model, entry in self._models.items():
            result[model] = {
                "cost": entry.total_cost,
                "calls": entry.call_count,
                "prompt_tokens": entry.total_prompt_tokens,
                "completion_tokens": entry.total_completion_tokens,
            }
        return result

    def report(self) -> str:
        """Human-readable cost report."""
        lines = [f"Total cost: ${self.total_cost:.6f} ({self.total_calls} calls)"]
        for model, entry in sorted(self._models.items()):
            lines.append(
                f"  {model}: ${entry.total_cost:.6f} "
                f"({entry.call_count} calls, "
                f"{entry.total_prompt_tokens} prompt + "
                f"{entry.total_completion_tokens} completion tokens)"
            )
        elapsed = time.time() - self._start_time
        lines.append(f"Elapsed: {elapsed:.1f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cost ceiling
# ---------------------------------------------------------------------------

class BudgetExceededError(Exception):
    """Raised when a cost ceiling is breached."""

    def __init__(self, limit: float, current: float, requested: float) -> None:
        self.limit = limit
        self.current = current
        self.requested = requested
        super().__init__(
            f"Budget exceeded: limit=${limit:.4f}, "
            f"spent=${current:.4f}, requested=${requested:.4f}"
        )


@dataclass
class CostCeiling:
    """Enforce hard budget limits on votes and debates.

    Args:
        max_cost_per_vote: Maximum USD for a single vote() call.
        max_cost_per_debate: Maximum USD for a full debate() call.
    """

    max_cost_per_vote: float = 0.50
    max_cost_per_debate: float = 5.00

    def check_vote(self, tracker: CostTracker, estimated_next: float = 0.0) -> None:
        """Raise BudgetExceededError if the next call would exceed the vote budget."""
        if tracker.total_cost + estimated_next > self.max_cost_per_vote:
            raise BudgetExceededError(
                limit=self.max_cost_per_vote,
                current=tracker.total_cost,
                requested=estimated_next,
            )

    def check_debate(self, tracker: CostTracker, estimated_next: float = 0.0) -> None:
        """Raise BudgetExceededError if the next call would exceed the debate budget."""
        if tracker.total_cost + estimated_next > self.max_cost_per_debate:
            raise BudgetExceededError(
                limit=self.max_cost_per_debate,
                current=tracker.total_cost,
                requested=estimated_next,
            )


# ---------------------------------------------------------------------------
# Model selection within budget
# ---------------------------------------------------------------------------

def select_models_within_budget(
    models: list[str],
    prompt: str,
    budget: float,
    estimated_completion_tokens: int = 500,
) -> list[str]:
    """Filter models that can fit within the given budget.

    Estimates the cost of calling each model with the given prompt and
    returns only those whose combined cost stays within budget.

    Models are added in the order provided, greedily. The first models
    in the list are preferred.

    Args:
        models: Candidate model strings.
        prompt: The prompt text (used to estimate input tokens).
        budget: Maximum total cost in USD.
        estimated_completion_tokens: Assumed output length per model.

    Returns:
        List of models that fit within budget (may be empty).
    """
    # Rough token estimate: 1 token ~ 4 chars
    estimated_prompt_tokens = max(1, len(prompt) // 4)

    selected: list[str] = []
    running_cost = 0.0

    for model in models:
        cost = estimate_cost(model, estimated_prompt_tokens, estimated_completion_tokens)
        if running_cost + cost <= budget:
            selected.append(model)
            running_cost += cost

    return selected
