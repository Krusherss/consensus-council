"""Main Council class for consensus-council.

The Council orchestrates multi-model voting and debate, integrating
anti-sycophancy measures, stalemate detection, and cost control.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import anyio
import litellm

from .web_search import has_search_tags, resolve_searches, SEARCH_INSTRUCTION

from .deliberation import (
    DeliberationResult,
    PeerReview,
    build_chair_prompt,
    build_deliberation_crosstalk_prompt,
    build_independent_prompt,
    build_peer_review_prompt,
    rotated_label_map,
    write_deliberation_artifact,
)

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
        models: List of LiteLLM model strings (e.g. ["openai/o3", "xai/grok-4"]).
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
        enable_search: bool = False,
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
        return (
            anyio.from_thread.run_sync(
                lambda: anyio.run(
                    self.avote, prompt, context, threshold, strategy, enable_search
                )
            )
            if _in_async_context()
            else anyio.run(
                self.avote, prompt, context, threshold, strategy, enable_search
            )
        )

    def debate(
        self,
        prompt: str,
        context: str = "",
        max_rounds: int = 3,
        stop_on: str = "majority",
        threshold: float = 0.66,
        enable_search: bool = False,
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
        return (
            anyio.from_thread.run_sync(
                lambda: anyio.run(
                    self.adebate,
                    prompt,
                    context,
                    max_rounds,
                    stop_on,
                    threshold,
                    enable_search,
                )
            )
            if _in_async_context()
            else anyio.run(
                self.adebate,
                prompt,
                context,
                max_rounds,
                stop_on,
                threshold,
                enable_search,
            )
        )

    def deliberate(
        self,
        prompt: str,
        chair_model: str,
        context: str = "",
        mode: str = "auto",
        debate_rounds: int = 2,
        route_model: str | None = None,
        enable_search: bool = False,
        output_dir: str | None = None,
    ) -> DeliberationResult:
        """Run the three-stage council and return the chairman's synthesis.

        vote and debate remain the right APIs for binary decisions.
        Use this method for open-ended research, design, and review questions.

        Args:
            prompt: The question for the council.
            chair_model: LiteLLM model used only for final synthesis.
            context: Optional code, document, or other supporting context.
            mode: auto, simple, or debate.
            debate_rounds: Total panel rounds in debate mode, including the
                initial independent round.
            route_model: Optional low-cost model used when mode is auto.
            enable_search: Allow panelists to request DuckDuckGo searches.
            output_dir: Optional directory for partial and final Markdown
                artifacts. No files are written when omitted.
        """
        args = (
            prompt,
            chair_model,
            context,
            mode,
            debate_rounds,
            route_model,
            enable_search,
            output_dir,
        )
        return (
            anyio.from_thread.run_sync(lambda: anyio.run(self.adeliberate, *args))
            if _in_async_context()
            else anyio.run(self.adeliberate, *args)
        )

    async def adeliberate(
        self,
        prompt: str,
        chair_model: str,
        context: str = "",
        mode: str = "auto",
        debate_rounds: int = 2,
        route_model: str | None = None,
        enable_search: bool = False,
        output_dir: str | None = None,
    ) -> DeliberationResult:
        """Async three-stage deliberation with blind review and synthesis."""
        if not chair_model:
            raise ValueError("A chair_model is required for synthesis.")
        if mode not in {"auto", "simple", "debate"}:
            raise ValueError("mode must be 'auto', 'simple', or 'debate'.")
        if debate_rounds < 1:
            raise ValueError("debate_rounds must be at least 1.")

        selected_mode = mode
        if mode == "auto":
            classification_prompt = (
                "Classify this question as STRUCTURED or OPEN_ENDED. "
                "STRUCTURED means a narrow factual or defined decision question. "
                "OPEN_ENDED means design, strategy, architecture, research, or complex "
                "qualitative analysis. Respond with exactly one label.\n\n"
                f"Question: {prompt[:1000]}"
            )
            classification = await self._classify(
                route_model or self.models[0], classification_prompt
            )
            selected_mode = (
                "simple" if "STRUCTURED" in classification.upper() else "debate"
            )

        tracker = CostTracker()

        def _check_budget(prompts: dict[str, str]) -> None:
            if not self.cost_ceiling:
                return
            estimated = sum(
                estimate_cost(
                    model,
                    max(1, len(prompt_text) // 4),
                    self.max_tokens,
                )
                for model, prompt_text in prompts.items()
            )
            self.cost_ceiling.check_debate(tracker, estimated)

        search_instruction = SEARCH_INSTRUCTION if enable_search else ""
        independent_prompt = build_independent_prompt(
            prompt, context=context, search_instruction=search_instruction
        )
        stage_prompts = {model: independent_prompt for model in self.models}
        _check_budget(stage_prompts)
        responses, stage_errors = await self._query_text_all(
            stage_prompts, tracker, enable_search=enable_search
        )
        failed_models = list(stage_errors)
        if not responses:
            raise RuntimeError("All panel models failed during Stage 1.")

        completed_rounds = 1
        if selected_mode == "debate":
            for round_num in range(2, debate_rounds + 1):
                debate_prompts = {
                    model: build_deliberation_crosstalk_prompt(
                        prompt,
                        model,
                        responses,
                        round_num,
                        context=context,
                        search_instruction=search_instruction,
                    )
                    for model in responses
                }
                _check_budget(debate_prompts)
                updated, round_errors = await self._query_text_all(
                    debate_prompts, tracker, enable_search=enable_search
                )
                responses.update(updated)
                failed_models.extend(round_errors)
                completed_rounds = round_num

        review_question = prompt
        if context:
            review_question = f"{prompt}\n\nSUPPLIED CONTEXT:\n{context}"

        review_holder: dict[str, PeerReview] = {}
        panel_models = list(responses)
        review_label_maps = {
            reviewer: rotated_label_map(panel_models, index)
            for index, reviewer in enumerate(panel_models)
        }
        review_prompts = {
            reviewer: build_peer_review_prompt(
                review_question, responses, review_label_maps[reviewer]
            )
            for reviewer in panel_models
        }
        _check_budget(review_prompts)

        async with anyio.create_task_group() as tg:

            async def _review(reviewer: str) -> None:
                review, error = await self._query_text_model(
                    reviewer, review_prompts[reviewer], tracker
                )
                review_holder[reviewer] = PeerReview(
                    reviewer=reviewer,
                    review=review,
                    label_map=review_label_maps[reviewer],
                    error=error,
                )

            for reviewer in panel_models:
                tg.start_soon(_review, reviewer)

        reviews = [review_holder[model] for model in panel_models]
        failed_models.extend(item.reviewer for item in reviews if item.error)

        partial_result = DeliberationResult(
            question=prompt,
            synthesis="",
            responses=responses,
            reviews=reviews,
            chair_model=chair_model,
            mode=selected_mode,
            rounds=completed_rounds,
            total_cost=tracker.total_cost,
            failed_models=list(dict.fromkeys(failed_models)),
        )
        if output_dir:
            write_deliberation_artifact(output_dir, partial_result, partial=True)

        chair_prompt = build_chair_prompt(review_question, responses, reviews)
        _check_budget({chair_model: chair_prompt})
        synthesis, chair_error = await self._query_text_model(
            chair_model, chair_prompt, tracker
        )
        if chair_error:
            failed_models.append(chair_model)
            synthesis = "Chairman synthesis failed; inspect the saved panel checkpoint."

        result = DeliberationResult(
            question=prompt,
            synthesis=synthesis,
            responses=responses,
            reviews=reviews,
            chair_model=chair_model,
            mode=selected_mode,
            rounds=completed_rounds,
            total_cost=tracker.total_cost,
            failed_models=list(dict.fromkeys(failed_models)),
        )
        if output_dir:
            artifact = write_deliberation_artifact(output_dir, result)
            result.artifact_path = str(artifact)
        return result

    async def avote(
        self,
        prompt: str,
        context: str = "",
        threshold: float = 0.5,
        strategy: str = "simple_majority",
        enable_search: bool = False,
    ) -> ConsensusResult:
        """Async voting -- queries all models concurrently."""
        tracker = CostTracker()
        full_prompt = f"{SEARCH_INSTRUCTION}\n\n{prompt}" if enable_search else prompt
        blind = BlindVoting(prompt=full_prompt, context=context)

        # Check budget before starting
        if self.cost_ceiling:
            est = sum(
                estimate_cost(m, max(1, len(full_prompt) // 4), self.max_tokens)
                for m in self.models
            )
            self.cost_ceiling.check_vote(tracker, est)

        # Query all models concurrently
        vote_results = await self._query_all_blind(blind, tracker)

        # Resolve web search tags if enabled
        if enable_search:
            for r in vote_results:
                if has_search_tags(r.reasoning):
                    resolved, _ = resolve_searches(r.reasoning)
                    r.reasoning = resolved
                    r.raw_response = resolved
                    r.vote, r.confidence = extract_vote(resolved)

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
        enable_search: bool = False,
    ) -> ConsensusResult:
        """Async multi-round debate with anti-sycophancy and stalemate detection."""
        tracker = CostTracker()
        prompt = f"{SEARCH_INSTRUCTION}\n\n{prompt}" if enable_search else prompt
        rotation = RotationOrder(len(self.models))

        prev_votes: dict[str, Vote] | None = None
        prev_responses: dict[str, str] | None = None
        last_result: ConsensusResult | None = None

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
                    logger.warning(
                        "Budget exceeded at round %d, stopping debate.", round_num
                    )
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

            # Resolve web search tags if enabled
            if enable_search:
                for r in vote_results:
                    if has_search_tags(r.reasoning):
                        resolved, _ = resolve_searches(r.reasoning)
                        r.reasoning = resolved
                        r.raw_response = resolved
                        r.vote, r.confidence = extract_vote(resolved)

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
            if detect_stalemate(
                current_votes, prev_votes, current_responses, prev_responses
            ):
                logger.info("Stalemate detected at round %d.", round_num + 1)
                stalemate_result = resolve_stalemate(
                    vote_results, self.stalemate_strategy, self.moderator_model
                )

                if (
                    stalemate_result.decision == "PENDING_MODERATOR"
                    and self.moderator_model
                ):
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
            last_result.reasoning += (
                "\n[Max debate rounds reached without full consensus]"
            )
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
        result = (
            anyio.from_thread.run_sync(
                lambda: anyio.run(self._classify, classifier, classification_prompt)
            )
            if _in_async_context()
            else anyio.run(self._classify, classifier, classification_prompt)
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
            vote_keys = {"threshold", "strategy", "enable_search"}
            return self.vote(
                prompt,
                context=context,
                **{k: v for k, v in kwargs.items() if k in vote_keys},
            )
        debate_keys = {"max_rounds", "stop_on", "threshold", "enable_search"}
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

    async def _query_text_model(
        self,
        model: str,
        prompt_text: str,
        tracker: CostTracker,
    ) -> tuple[str, str | None]:
        """Query one model, recording cost while redacting provider errors."""
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
            return content, None
        except Exception as exc:
            error_name = type(exc).__name__
            logger.error("Model %s failed (%s).", model, error_name)
            return "", f"{error_name}: provider call failed"

    async def _query_text_with_search(
        self,
        model: str,
        prompt_text: str,
        tracker: CostTracker,
        enable_search: bool,
    ) -> tuple[str, str | None]:
        """Query a model and, when requested, return a source-grounded revision."""
        content, error = await self._query_text_model(model, prompt_text, tracker)
        if error or not enable_search or not has_search_tags(content):
            return content, error

        resolved, search_log = await anyio.to_thread.run_sync(resolve_searches, content)
        if not search_log:
            return resolved, None

        followup_prompt = (
            "You requested live web searches. Below is your draft with the actual "
            "DuckDuckGo/Trafilatura source material inserted. Rewrite the answer using "
            "only supported claims, cite the supplied URLs inline, and never invent a "
            "citation.\n\n"
            f"{resolved}"
        )
        revised, followup_error = await self._query_text_model(
            model, followup_prompt, tracker
        )
        if followup_error:
            return resolved, followup_error
        return revised, None

    async def _query_text_all(
        self,
        prompts: dict[str, str],
        tracker: CostTracker,
        *,
        enable_search: bool = False,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Query model-specific prompts concurrently, preserving input order."""
        result_holder: dict[str, str] = {}
        error_holder: dict[str, str] = {}

        async with anyio.create_task_group() as tg:

            async def _run(model: str, prompt_text: str) -> None:
                text, error = await self._query_text_with_search(
                    model, prompt_text, tracker, enable_search
                )
                if text:
                    result_holder[model] = text
                if error:
                    error_holder[model] = error

            for model, prompt_text in prompts.items():
                tg.start_soon(_run, model, prompt_text)

        results = {
            model: result_holder[model] for model in prompts if model in result_holder
        }
        errors = {
            model: error_holder[model] for model in prompts if model in error_holder
        }
        return results, errors

    async def _query_model(
        self,
        model: str,
        prompt_text: str,
        tracker: CostTracker,
    ) -> VoteResult:
        """Query a single model and return a VoteResult."""
        content, error = await self._query_text_model(model, prompt_text, tracker)
        if error:
            return VoteResult(
                model=model,
                vote=Vote.ABSTAIN,
                confidence=0.0,
                reasoning="",
                raw_response="",
                error=error,
            )
        vote, confidence = extract_vote(content)
        return VoteResult(
            model=model,
            vote=vote,
            confidence=confidence,
            reasoning=content,
            raw_response=content,
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
                result_holder[model] = await self._query_model(
                    model, prompt_text, tracker
                )

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
                result_holder[model] = await self._query_model(
                    model, prompt_text, tracker
                )

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
