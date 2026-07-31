"""Tests for three-stage deliberation. No provider calls are made."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from consensus_council.cost import CostTracker
from consensus_council.council import Council
from consensus_council.deliberation import (
    PeerReview,
    build_chair_prompt,
    build_peer_review_prompt,
    rotated_label_map,
)


def _response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.get = MagicMock(
        return_value={"prompt_tokens": 100, "completion_tokens": 50}
    )
    return response


def test_rotated_labels_change_for_each_reviewer() -> None:
    models = ["model-a", "model-b", "model-c"]

    first = rotated_label_map(models, 0)
    second = rotated_label_map(models, 1)

    assert set(first.values()) == set(models)
    assert set(second.values()) == set(models)
    assert first["Response A"] == "model-a"
    assert second["Response A"] == "model-b"


def test_peer_review_prompt_hides_model_ids() -> None:
    responses = {"model-a": "First answer", "model-b": "Second answer"}
    labels = {"Response A": "model-b", "Response B": "model-a"}

    prompt = build_peer_review_prompt("Question?", responses, labels)

    assert "model-a" not in prompt
    assert "model-b" not in prompt
    assert "Response A:\nSecond answer" in prompt
    assert "Response B:\nFirst answer" in prompt


def test_chair_prompt_deanonymizes_reviews() -> None:
    review = PeerReview(
        reviewer="reviewer-model",
        review="FINAL RANKING:\n1. Response A",
        label_map={"Response A": "panel-model"},
    )

    prompt = build_chair_prompt("Question?", {"panel-model": "Answer"}, [review])

    assert "Reviewer: reviewer-model" in prompt
    assert "Response A = panel-model" in prompt
    assert "never invent a source" in prompt


@pytest.mark.asyncio
async def test_simple_deliberation_runs_all_three_stages(tmp_path: Any) -> None:
    captured: list[tuple[str, str]] = []

    async def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        model = kwargs["model"]
        prompt = kwargs["messages"][0]["content"]
        captured.append((model, prompt))
        if model == "chair-model":
            return _response("Final synthesis")
        if "You are reviewing candidate answers" in prompt:
            return _response("Review\nFINAL RANKING:\n1. Response A")
        return _response(f"Independent answer from {model}")

    with patch("consensus_council.council.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
        mock_litellm.completion_cost = MagicMock(return_value=0.01)
        council = Council(models=["model-a", "model-b"], max_tokens=256)
        result = await council.adeliberate(
            "Design a cache",
            chair_model="chair-model",
            mode="simple",
            output_dir=str(tmp_path),
        )

    assert result.synthesis == "Final synthesis"
    assert result.rounds == 1
    assert list(result.responses) == ["model-a", "model-b"]
    assert len(result.reviews) == 2
    assert result.failed_models == []
    assert len(captured) == 5
    assert result.artifact_path is not None
    assert len(list(tmp_path.glob("*.md"))) == 2


@pytest.mark.asyncio
async def test_search_results_are_sent_back_for_cited_revision() -> None:
    calls = 0

    async def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response("Needs evidence [SEARCH: example query]")
        return _response("Revised claim [source: https://example.test/source]")

    search_record = [{"query": "example query", "status": "ok"}]
    with (
        patch("consensus_council.council.litellm") as mock_litellm,
        patch(
            "consensus_council.council.resolve_searches",
            return_value=("Resolved source material", search_record),
        ),
    ):
        mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
        mock_litellm.completion_cost = MagicMock(return_value=0.01)
        council = Council(models=["model-a"])
        text, error = await council._query_text_with_search(
            "model-a",
            "Question",
            CostTracker(verbose=False),
            enable_search=True,
        )

    assert error is None
    assert "https://example.test/source" in text
    assert calls == 2


@pytest.mark.asyncio
async def test_provider_errors_do_not_echo_sensitive_details() -> None:
    sensitive_detail = "private" + "-credential-value"

    with patch("consensus_council.council.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(side_effect=RuntimeError(sensitive_detail))
        council = Council(models=["model-a"])
        text, error = await council._query_text_model(
            "model-a", "Question", CostTracker(verbose=False)
        )

    assert text == ""
    assert error is not None
    assert sensitive_detail not in error
    assert error == "RuntimeError: provider call failed"
