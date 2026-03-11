"""Main Council class for consensus-council.

The Council orchestrates multi-model voting and debate, integrating
anti-sycophancy measures, stalemate detection, and cost control.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import anyio
import litellm

from .anti_sycophancy import (
    BlindVoting,
    RotationOrder,
    build_crosstalk_prompt,
)
from .cost import (
    BudgetExceededError,
    CostCeiling,
    CostTracker,
    estimate_cost,
)
from .stalemate import (
    StalemateStrategy,
    build_moderator_prompt,
    detect_stalemate,
    resolve_stalemate,
)
from .voting import (
    ConsensusResult,
    Vote,
    VoteResult,
    extract_vote,
    simple_majority,
    supermajority,
    unanimous,
    weighted_majority,
)

logger = logging.getLogger(__name__)


class Council:
    """Multi-model voting council.

    Args:
        models: List of LiteLLM model strings (e.g. ["gpt-4o", "claude-sonnet-4-5-20250514"]).
        cost_ceiling: Optional CostCeiling to enforce budget limits.
        weights: Optional per-model reliability weights for weighted voting.
        stalemate_strategy: How to handle debate stalemates.
        moderator_model: Model for MODERATOR stalemate strategy.
        max_tokens: Maximum tokens per model response.
        temperature: Sampling temperature for model calls.
    """

    def __init__(
        self,
        models: list[str],
        cost_ceiling: CostCeiling | None = None,
        weights: dict[str, float] | None = None,
        stalemate_strategy: StalemateStrategy = StalemateStrategy.STOP,
        moderator_model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> None:
        if not models:
            raise ValueError("Council requires at least one model.")
        self.models = list(models)
        self.cost_ceiling = cost_ceiling
        self.weights = weights
        self.stalemate_strategy = stalemate_strategy
        self.moderator_model = moderator_model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def vote(
        self,
        prompt: str,
        context: str = "",
        threshold: float = 0.5,
        strategy: str = "simple_majority",
    ) -> ConsensusResult:
        """Synchronous voting. Queries all models and aggregates votes.

        Args:
            prompt: The question to vote on.
            context: Optional context (code diff, document, etc.).
            threshold: Agreement threshold (used by supermajority).
            strategy: One of "simple_majority", "supermajority", "unanimous",
                      "weighted_majority", "ranked_choice".

        Returns:
            ConsensusResult with the aggregated decision.
        """
        return anyio.from_thread.run_sync(
            lambda: anyio.run(self.avote, prompt, context, threshold, strategy)
        ) if _in_async_context() else anyio.run(
            self.avote, prompt, context, threshold, strategy
        )

    def debate(
        self,
        prompt: str,
        context: str = "",
        max_rounds: int = 3,
        stop_on: str = "majority",
        threshold: float = 0.66,
    ) -> ConsensusResult:
        """Synchronous multi-round debate.

        Args:
            prompt: The question to debate.
            context: Optional context.
            max_rounds: Maximum debate rounds.
            stop_on: Stop condition -- "unanimous", "majority", or "supermajority".
            threshold: Threshold for supermajority stop condition.

        Returns:
            ConsensusResult from the final round.
        """
        return anyio.from_thread.run_sync(
            lambda: anyio.run(self.adebate, prompt, context, max_rounds, stop_on, threshold)
        ) if _in_async_context() else anyio.run(
            self.adebate, prompt, context, max_rounds, stop_on, threshold
        )

    async def avote(
        self,
        prompt: str,
        context: str = "",
        threshold: float = 0.5,
        strategy: str = "simple_majority",
    ) -> ConsensusResult:
        """Async voting -- queries all models concurrently."""
        tracker = CostTracker()
        blind = BlindVoting(prompt=prompt, context=context)

        # Check budget before starting
        if self.cost_ceiling:
            est = sum(
                estimate_cost(m, max(1, len(prompt) // 4), self.max_tokens)
                for m in self.models
            )
            self.cost_ceiling.check_vote(tracker, est)

        # Query all models concurrently
        vote_results = await self._query_all_blind(blind, tracker)

        # Apply voting strategy
        result = _apply_strategy(vote_results, strategy, threshold, self.weights)
        result.total_cost = tracker.total_cost
        return result

    async def adebate(
        self,
        prompt: str,
        context: str = "",
        max_rounds: int = 3,
        stop_on: str = "majority",
        threshold: float = 0.66,
    ) -> ConsensusResult:
        """Async multi-round debate with anti-sycophancy and stalemate detection."""
        tracker = CostTracker()
        rotation = RotationOrder(len(self.models))

        prev_votes: dict[str, Vote] | None = None
        prev_responses: dict[str, str] | None = None
        last_result: ConsensusResult | None = None
        all_vote_results: list[VoteResult] = []

        for round_num in range(max_rounds):
            # Budget check
            if self.cost_ceiling:
                est = sum(
                    estimate_cost(m, max(1, len(prompt) // 4), self.max_tokens)
                    for m in self.models
                )
                try:
                    self.cost_ceiling.check_debate(tracker, est)
                except BudgetExceededError:
                    logger.warning("Budget exceeded at round %d, stopping debate.", round_num)
                    break

            # Determine query order for this round
            order = rotation.order_for_round(round_num)
            ordered_models = [self.models[i] for i in order]

            if round_num == 0:
                # First round: blind voting
                blind = BlindVoting(prompt=prompt, context=context)
                vote_results = await self._query_all_blind(blind, tracker)
            else:
                # Subsequent rounds: cross-talk debate
                vote_results = await self._query_all_debate(
                    ordered_models=ordered_models,
                    round_num=round_num + 1,
                    prev_responses=prev_responses or {},
                    prev_votes={m: v.value for m, v in (prev_votes or {}).items()},
                    original_prompt=prompt,
                    context=context,
                    tracker=tracker,
                )

            all_vote_results = vote_results

            # Build current state
            current_votes = {v.model: v.vote for v in vote_results}
            current_responses = {v.model: v.reasoning for v in vote_results}

            # Check stop condition
            result = _apply_strategy(
                vote_results, _stop_to_strategy(stop_on), threshold, self.weights
            )
            result.rounds = round_num + 1
            result.total_cost = tracker.total_cost
            last_result = result

            if result.decision in ("YES", "NO") and result.decision != "TIE":
                # Consensus reached
                return result

            # Check stalemate
            if detect_stalemate(current_votes, prev_votes, current_responses, prev_responses):
                logger.info("Stalemate detected at round %d.", round_num + 1)
                stalemate_result = resolve_stalemate(
                    vote_results, self.stalemate_strategy, self.moderator_model
                )

                if stalemate_result.decision == "PENDING_MODERATOR" and self.moderator_model:
                    # Run moderator
                    mod_result = await self._query_moderator(
                        prompt, vote_results, context, tracker
                    )
                    mod_result.rounds = round_num + 1
                    mod_result.total_cost = tracker.total_cost
                    return mod_result

                stalemate_result.rounds = round_num + 1
                stalemate_result.total_cost = tracker.total_cost
                return stalemate_result

            prev_votes = current_votes
            prev_responses = current_responses

        # Max rounds exhausted
        if last_result is not None:
            last_result.reasoning += "\n[Max debate rounds reached without full consensus]"
            return last_result

        return ConsensusResult(
            decision="TIE",
            confidence=0.0,
            reasoning="Debate ended without result.",
        )

    def route(self, prompt: str, route_model: str | None = None) -> str:
        """Classify a question as 'vote' or 'debate' using a lightweight model.

        STRUCTURED questions (clear YES/NO, defined options) → 'vote'.
        OPEN_ENDED questions (design, strategy, analysis) → 'debate'.

        Args:
            prompt: The question to classify.
            route_model: LiteLLM model for classification. Defaults to the first
                         model in the council's list.

        Returns:
            'vote' or 'debate'.
        """
        classifier = route_model or self.models[0]
        classification_prompt = (
            "Classify this question as either STRUCTURED or OPEN_ENDED.\n\n"
            "STRUCTURED: Has clear proposals to vote on, asks for YES/NO decisions, "
            "evaluates specific options with defined criteria.\n"
            "OPEN_ENDED: Asks for design, strategy, architecture, or complex "
            "qualitative analysis with no single right answer.\n\n"
            "Respond with exactly one word: STRUCTURED or OPEN_ENDED\n\n"
            f"Question: {prompt[:1000]}"
        )
        result = anyio.from_thread.run_sync(
            lambda: anyio.run(self._classify, classifier, classification_prompt)
        ) if _in_async_context() else anyio.run(
            self._classify, classifier, classification_prompt
        )
        return "vote" if "STRUCTURED" in result.upper() else "debate"

    def decide(
        self,
        prompt: str,
        context: str = "",
        route_model: str | None = None,
        **kwargs: Any,
    ) -> ConsensusResult:
        """Auto-route to vote() or debate() based on question type.

        Uses :meth:`route` to classify the question, then dispatches to the
        appropriate method. Extra kwargs are forwarded as appropriate.

        Args:
            prompt: The question.
            context: Optional context (code diff, document, etc.).
            route_model: Model for routing classification.
            **kwargs: Forwarded to vote() (threshold, strategy) or
                      debate() (max_rounds, stop_on, threshold) as applicable.

        Returns:
            ConsensusResult from the selected method.
        """
        mode = self.route(prompt, route_model=route_model)
        if mode == "vote":
            vote_keys = {"threshold", "strategy"}
            return self.vote(
                prompt,
                context=context,
                **{k: v for k, v in kwargs.items() if k in vote_keys},
            )
        debate_keys = {"max_rounds", "stop_on", "threshold"}
        return self.debate(
            prompt,
            context=context,
            **{k: v for k, v in kwargs.items() if k in debate_keys},
        )

    async def _classify(self, model: str, prompt: str) -> str:
        """Run a single low-cost classification call."""
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
                drop_params=True,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return "OPEN_ENDED"  # Safe default — prefer debate over missing it

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _query_model(
        self,
        model: str,
        prompt_text: str,
        tracker: CostTracker,
    ) -> VoteResult:
        """Query a single model and return a VoteResult."""
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                drop_params=True,
            )

            content = response.choices[0].message.content or ""
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Track cost
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = estimate_cost(model, prompt_tokens, completion_tokens)

            tracker.record(
                model=model,
                cost=cost,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            vote, confidence = extract_vote(content)
            return VoteResult(
                model=model,
                vote=vote,
                confidence=confidence,
                reasoning=content,
                raw_response=content,
            )
        except Exception as exc:
            logger.error("Model %s failed: %s", model, exc)
            return VoteResult(
                model=model,
                vote=Vote.ABSTAIN,
                confidence=0.0,
                reasoning="",
                raw_response="",
                error=str(exc),
            )

    async def _query_all_blind(
        self,
        blind: BlindVoting,
        tracker: CostTracker,
    ) -> list[VoteResult]:
        """Query all models in blind-voting mode concurrently."""
        results: list[VoteResult] = []

        async with anyio.create_task_group() as tg:
            result_holder: dict[str, VoteResult] = {}

            async def _run(model: str) -> None:
                prompt_text = blind.build_prompt(model)
                result_holder[model] = await self._query_model(model, prompt_text, tracker)

            for model in self.models:
                tg.start_soon(_run, model)

        # Preserve model order
        for model in self.models:
            if model in result_holder:
                results.append(result_holder[model])

        return results

    async def _query_all_debate(
        self,
        ordered_models: list[str],
        round_num: int,
        prev_responses: dict[str, str],
        prev_votes: dict[str, str],
        original_prompt: str,
        context: str,
        tracker: CostTracker,
    ) -> list[VoteResult]:
        """Query all models in debate mode concurrently."""
        results: list[VoteResult] = []

        async with anyio.create_task_group() as tg:
            result_holder: dict[str, VoteResult] = {}

            async def _run(model: str) -> None:
                prompt_text = build_crosstalk_prompt(
                    model_name=model,
                    round_num=round_num,
                    prev_responses=prev_responses,
                    prev_votes=prev_votes,
                    original_prompt=original_prompt,
                    context=context,
                )
                result_holder[model] = await self._query_model(model, prompt_text, tracker)

            for model in ordered_models:
                tg.start_soon(_run, model)

        for model in ordered_models:
            if model in result_holder:
                results.append(result_holder[model])

        return results

    async def _query_moderator(
        self,
        original_prompt: str,
        votes: list[VoteResult],
        context: str,
        tracker: CostTracker,
    ) -> ConsensusResult:
        """Query the moderator model to break a tie."""
        if not self.moderator_model:
            return ConsensusResult(
                decision="TIE",
                confidence=0.0,
                reasoning="No moderator model configured.",
            )

        prompt_text = build_moderator_prompt(original_prompt, votes, context)
        mod_vote = await self._query_model(self.moderator_model, prompt_text, tracker)

        return ConsensusResult(
            decision=mod_vote.vote.value,
            confidence=mod_vote.confidence,
            votes={v.model: v for v in votes} | {mod_vote.model: mod_vote},
            reasoning=f"Moderator ({self.moderator_model}) decided: {mod_vote.vote.value}\n{mod_vote.reasoning}",
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _in_async_context() -> bool:
    """Check if we're already inside an async event loop."""
    try:
        import sniffio
        sniffio.current_async_library()
        return True
    except (ImportError, sniffio.AsyncLibraryNotFoundError):
        return False


def _stop_to_strategy(stop_on: str) -> str:
    """Map debate stop_on value to a voting strategy name."""
    mapping = {
        "unanimous": "unanimous",
        "majority": "simple_majority",
        "supermajority": "supermajority",
    }
    return mapping.get(stop_on, "simple_majority")


def _apply_strategy(
    votes: Sequence[VoteResult],
    strategy: str,
    threshold: float,
    weights: dict[str, float] | None,
) -> ConsensusResult:
    """Apply the named voting strategy to a list of votes."""
    strategies = {
        "simple_majority": lambda v: simple_majority(v),
        "supermajority": lambda v: supermajority(v, threshold),
        "unanimous": lambda v: unanimous(v),
        "weighted_majority": lambda v: weighted_majority(v, weights),
    }
    fn = strategies.get(strategy, strategies["simple_majority"])
    result = fn(votes)

    # Attach failed models
    result.failed_models = [v.model for v in votes if v.error]
    return result
