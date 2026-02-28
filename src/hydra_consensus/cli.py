"""CLI interface for hydra-consensus.

Provides `hydra vote` and `hydra debate` commands with Rich-formatted output.
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table

from .cost import CostCeiling
from .council import Council
from .stalemate import StalemateStrategy

console = Console()


@click.group()
@click.version_option(package_name="hydra-consensus")
def main() -> None:
    """Hydra Consensus -- multi-model voting with anti-sycophancy."""
    pass


@main.command()
@click.argument("prompt")
@click.option(
    "--model", "-m",
    multiple=True,
    required=True,
    help="Model to include in the council (repeat for multiple).",
)
@click.option(
    "--threshold", "-t",
    default=0.5,
    type=float,
    help="Agreement threshold (0.0-1.0). Default: 0.5.",
)
@click.option(
    "--strategy", "-s",
    default="simple_majority",
    type=click.Choice([
        "simple_majority", "supermajority", "unanimous", "weighted_majority",
    ]),
    help="Voting strategy. Default: simple_majority.",
)
@click.option(
    "--context-file", "-c",
    type=click.Path(exists=True),
    help="File to include as context.",
)
@click.option(
    "--max-cost",
    default=0.50,
    type=float,
    help="Maximum cost in USD for this vote. Default: 0.50.",
)
@click.option(
    "--max-tokens",
    default=1024,
    type=int,
    help="Maximum tokens per model response. Default: 1024.",
)
def vote(
    prompt: str,
    model: tuple[str, ...],
    threshold: float,
    strategy: str,
    context_file: str | None,
    max_cost: float,
    max_tokens: int,
) -> None:
    """Run a single-round vote across models.

    Example:
        hydra vote "Is this code safe?" -m gpt-4o -m claude-sonnet-4-5-20250514 -t 0.66
    """
    context = ""
    if context_file:
        with open(context_file) as f:
            context = f.read()

    ceiling = CostCeiling(max_cost_per_vote=max_cost)
    council = Council(
        models=list(model),
        cost_ceiling=ceiling,
        max_tokens=max_tokens,
    )

    with console.status("[bold green]Querying models..."):
        try:
            result = council.vote(
                prompt=prompt,
                context=context,
                threshold=threshold,
                strategy=strategy,
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            sys.exit(1)

    _display_result(result)


@main.command()
@click.argument("prompt")
@click.option(
    "--model", "-m",
    multiple=True,
    required=True,
    help="Model to include in the council (repeat for multiple).",
)
@click.option(
    "--rounds", "-r",
    default=3,
    type=int,
    help="Maximum debate rounds. Default: 3.",
)
@click.option(
    "--stop-on",
    default="majority",
    type=click.Choice(["unanimous", "majority", "supermajority"]),
    help="Stop condition. Default: majority.",
)
@click.option(
    "--threshold", "-t",
    default=0.66,
    type=float,
    help="Threshold for supermajority. Default: 0.66.",
)
@click.option(
    "--context-file", "-c",
    type=click.Path(exists=True),
    help="File to include as context.",
)
@click.option(
    "--max-cost",
    default=5.00,
    type=float,
    help="Maximum cost in USD for this debate. Default: 5.00.",
)
@click.option(
    "--stalemate",
    default="stop",
    type=click.Choice(["stop", "random_tiebreak", "moderator", "escalate"]),
    help="Stalemate resolution strategy. Default: stop.",
)
@click.option(
    "--moderator",
    default=None,
    type=str,
    help="Moderator model for tiebreaking (used with --stalemate moderator).",
)
@click.option(
    "--max-tokens",
    default=1024,
    type=int,
    help="Maximum tokens per model response. Default: 1024.",
)
def debate(
    prompt: str,
    model: tuple[str, ...],
    rounds: int,
    stop_on: str,
    threshold: float,
    context_file: str | None,
    max_cost: float,
    stalemate: str,
    moderator: str | None,
    max_tokens: int,
) -> None:
    """Run a multi-round debate across models.

    Example:
        hydra debate "Best database?" -m gpt-4o -m claude-sonnet-4-5-20250514 -r 3
    """
    context = ""
    if context_file:
        with open(context_file) as f:
            context = f.read()

    ceiling = CostCeiling(max_cost_per_debate=max_cost)
    council = Council(
        models=list(model),
        cost_ceiling=ceiling,
        stalemate_strategy=StalemateStrategy(stalemate),
        moderator_model=moderator,
        max_tokens=max_tokens,
    )

    with console.status("[bold green]Running debate..."):
        try:
            result = council.debate(
                prompt=prompt,
                context=context,
                max_rounds=rounds,
                stop_on=stop_on,
                threshold=threshold,
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            sys.exit(1)

    _display_result(result)


def _display_result(result: "ConsensusResult") -> None:
    """Display a ConsensusResult using Rich tables."""
    from .voting import ConsensusResult  # noqa: F811 (for type reference)

    # Header
    decision_color = {
        "YES": "green",
        "NO": "red",
        "TIE": "yellow",
        "ABSTAIN": "dim",
        "ESCALATE": "magenta",
    }.get(result.decision, "white")

    console.print()
    console.print(
        f"[bold {decision_color}]Decision: {result.decision}[/]"
        f"  (confidence: {result.confidence:.0%})"
    )
    if result.rounds > 1:
        console.print(f"Rounds: {result.rounds}")
    console.print(f"Cost: ${result.total_cost:.4f}")
    console.print()

    # Vote table
    if result.votes:
        table = Table(title="Model Votes")
        table.add_column("Model", style="cyan")
        table.add_column("Vote", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Reasoning", max_width=60)

        for model, vr in result.votes.items():
            vote_style = {"YES": "green", "NO": "red", "ABSTAIN": "dim"}.get(
                vr.vote.value, "white"
            )
            reasoning_preview = (
                vr.reasoning[:100] + "..." if len(vr.reasoning) > 100 else vr.reasoning
            )
            error_note = f" [red](ERROR: {vr.error})[/]" if vr.error else ""
            table.add_row(
                model,
                f"[{vote_style}]{vr.vote.value}[/]",
                f"{vr.confidence:.0%}",
                reasoning_preview + error_note,
            )

        console.print(table)

    # Failed models
    if result.failed_models:
        console.print(
            f"\n[yellow]Failed models:[/] {', '.join(result.failed_models)}"
        )

    console.print()
