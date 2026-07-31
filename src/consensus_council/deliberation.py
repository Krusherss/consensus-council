"""Three-stage deliberation primitives.

The deliberation path complements yes/no voting:

1. independent answers (or optional debate rounds),
2. anonymous peer review with per-reviewer label rotation,
3. a named chairman synthesising the final answer.

Prompt construction lives in this module so the anonymity boundary can be
tested without making provider calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .anti_sycophancy import build_anti_sycophancy_directive


@dataclass
class PeerReview:
    """One anonymous review and the private label mapping used to create it."""

    reviewer: str
    review: str
    label_map: dict[str, str]
    error: str | None = None


@dataclass
class DeliberationResult:
    """Complete, auditable result from the three-stage council."""

    question: str
    synthesis: str
    responses: dict[str, str]
    reviews: list[PeerReview]
    chair_model: str
    mode: str = "simple"
    rounds: int = 1
    total_cost: float = 0.0
    failed_models: list[str] = field(default_factory=list)
    artifact_path: str | None = None


def build_independent_prompt(
    question: str,
    context: str = "",
    search_instruction: str = "",
) -> str:
    """Build a Stage 1 prompt that contains no other model's answer."""
    parts = [build_anti_sycophancy_directive()]
    if search_instruction:
        parts.extend([search_instruction, ""])
    if context:
        parts.extend(["=== CONTEXT ===", context, "=== END CONTEXT ===", ""])
    parts.extend(
        [
            "=== QUESTION ===",
            question,
            "=== END QUESTION ===",
            "",
            "Give a complete independent answer. Identify uncertainty and do not "
            "assume that another model will correct you.",
        ]
    )
    return "\n".join(parts)


def rotated_label_map(models: list[str], reviewer_index: int) -> dict[str, str]:
    """Map anonymous labels to models, rotated independently per reviewer."""
    if not models:
        return {}
    offset = reviewer_index % len(models)
    rotated = models[offset:] + models[:offset]
    return {f"Response {chr(65 + i)}": model for i, model in enumerate(rotated)}


def build_peer_review_prompt(
    question: str,
    responses: dict[str, str],
    label_map: dict[str, str],
) -> str:
    """Build a Stage 2 prompt containing anonymous labels, not model IDs."""
    response_block = "\n\n".join(
        f"{label}:\n{responses[model]}" for label, model in label_map.items()
    )
    return (
        "You are reviewing candidate answers to the question below. The answers "
        "are anonymized; judge them only on correctness, evidence, reasoning, and "
        "usefulness. Do not guess authorship.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CANDIDATE ANSWERS:\n{response_block}\n\n"
        "Evaluate every response, flag unsupported or conflicting claims, and end "
        "with an exact best-to-worst ranking in this format:\n"
        "FINAL RANKING:\n1. Response B\n2. Response A"
    )


def build_deliberation_crosstalk_prompt(
    question: str,
    model: str,
    responses: dict[str, str],
    round_num: int,
    context: str = "",
    search_instruction: str = "",
) -> str:
    """Build a debate-round prompt without forcing a binary vote."""
    other_answers = "\n\n".join(
        f"Candidate {index}:\n{text}"
        for index, (other_model, text) in enumerate(responses.items(), start=1)
        if other_model != model
    )
    parts = [build_anti_sycophancy_directive()]
    if search_instruction:
        parts.extend([search_instruction, ""])
    if context:
        parts.extend(["=== CONTEXT ===", context, "=== END CONTEXT ===", ""])
    parts.extend(
        [
            f"=== DELIBERATION ROUND {round_num} ===",
            f"Original question: {question}",
            "",
            "Other independently produced answers (anonymized):",
            other_answers,
            "",
            "Give your updated answer. Challenge unsupported claims and adopt another "
            "answer's point only when its evidence or reasoning is stronger. You are not "
            "required to reach consensus.",
        ]
    )
    return "\n".join(parts)


def build_chair_prompt(
    question: str,
    responses: dict[str, str],
    reviews: list[PeerReview],
) -> str:
    """Build the de-anonymised Stage 3 chairman prompt."""
    response_block = "\n\n".join(
        f"Model: {model}\nResponse:\n{text}" for model, text in responses.items()
    )
    review_parts: list[str] = []
    for item in reviews:
        mapping = ", ".join(
            f"{label} = {model}" for label, model in item.label_map.items()
        )
        review_parts.append(
            f"Reviewer: {item.reviewer}\nLabel mapping: {mapping}\nReview:\n{item.review}"
        )
    review_block = "\n\n".join(review_parts)
    return (
        "You are the chairman of an LLM council. Produce the final answer to the "
        "user; do not merely summarize the meeting. Reconcile disagreements, prefer "
        "claims supported by supplied evidence, preserve material uncertainty, and "
        "never invent a source or imply that model agreement proves correctness.\n\n"
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"STAGE 1 - INDEPENDENT RESPONSES:\n{response_block}\n\n"
        f"STAGE 2 - ANONYMOUS PEER REVIEWS (now de-anonymized):\n{review_block}\n\n"
        "Return one clear, well-reasoned final answer. Where sources appear in the "
        "record, keep citations attached to the claims they support."
    )


def write_deliberation_artifact(
    output_dir: str | Path,
    result: DeliberationResult,
    *,
    partial: bool = False,
) -> Path:
    """Write a Markdown checkpoint or completed deliberation artifact."""
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    slug = (
        re.sub(r"[^a-z0-9]+", "-", result.question.lower()).strip("-")[:48]
        or "question"
    )
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    suffix = "-partial" if partial else ""
    path = destination / f"{stamp}-{slug}{suffix}.md"

    lines = [
        f"# Council: {result.question[:100]}",
        "",
        f"*Mode: {result.mode} | rounds: {result.rounds} | chair: {result.chair_model}*",
        "",
        f"**Question:** {result.question}",
        "",
        "## Stage 1 - Independent responses",
        "",
    ]
    for model, response in result.responses.items():
        lines.extend([f"### {model}", "", response, ""])
    lines.extend(["## Stage 2 - Anonymous peer reviews", ""])
    for review in result.reviews:
        lines.extend([f"### {review.reviewer}", "", review.review, ""])
    if not partial:
        lines.extend(["## Stage 3 - Chairman synthesis", "", result.synthesis, ""])
    lines.extend(
        [
            "## Run metadata",
            "",
            f"- Estimated/provider-reported cost: ${result.total_cost:.6f}",
            f"- Failed models: {', '.join(result.failed_models) or 'none'}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
