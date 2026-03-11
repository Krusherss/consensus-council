# consensus-council

> Drop-in multi-model voting with anti-sycophancy and stalemate resolution.

Ask multiple LLMs the same question and get a reliable, consensus-driven answer. Consensus Council prevents models from copying each other, detects when debates stall, and keeps your costs under control.

**Background:** Battle-tested across 73 books in a multi-model annotation pipeline.

## Installation

```bash
pip install consensus-council
```

## Quick Start

```python
from consensus_council import Council

council = Council(models=["gpt-4o", "claude-sonnet-4-5-20250514", "gemini-2.0-flash"])
result = council.vote("Is this code safe to deploy?", context=code_diff, threshold=0.66)
print(result.decision, result.confidence)
```

## Features

### Voting Strategies

Consensus Council supports five voting strategies for different reliability requirements:

| Strategy | Description | Use When |
|----------|-------------|----------|
| `simple_majority` | >50% agreement wins | Quick checks, low-stakes |
| `supermajority` | Configurable threshold (default 2/3) | Production deployments, security reviews |
| `unanimous` | All models must agree | Safety-critical decisions |
| `weighted_majority` | Per-model reliability weights | You trust some models more than others |
| `ranked_choice` | Instant-runoff for multi-option questions | "Which database should we use?" |

```python
# Simple majority (default)
result = council.vote("Approve this PR?")

# Supermajority -- 2/3 must agree
result = council.vote("Safe to deploy?", threshold=0.66, strategy="supermajority")

# Unanimous -- all must agree
result = council.vote("Delete production data?", strategy="unanimous")

# Weighted -- trust GPT-4o more
council = Council(
    models=["gpt-4o", "claude-sonnet-4-5-20250514", "gemini-2.0-flash"],
    weights={"gpt-4o": 2.0, "claude-sonnet-4-5-20250514": 1.5, "gemini-2.0-flash": 1.0},
)
result = council.vote("Is this correct?", strategy="weighted_majority")
```

### Anti-Sycophancy

LLMs tend to agree with each other (sycophancy) and anchor on the first response they see (anchoring bias). Consensus Council fights both:

- **Blind voting:** In simple votes, each model sees only the original prompt -- never other models' responses.
- **Rotation ordering:** In debates, the query order is shuffled every round so no model consistently leads.
- **Anti-sycophancy directive:** Every prompt includes explicit instructions:
  - "Do NOT change your vote just because other models disagree"
  - "Only change if the LOGIC is undeniable"
  - "Do NOT soften your position to avoid conflict"
  - "Genuine disagreement is MORE VALUABLE than false consensus"
- **Merit-based cross-talk:** In debate rounds, models see others' arguments but are instructed to evaluate them on logical merit, not defer to authority.

### Multi-Round Debate

For complex questions, run a multi-round debate where models can see and respond to each other's arguments:

```python
result = council.debate(
    prompt="What's the best database for this use case?",
    context=requirements,
    max_rounds=3,
    stop_on="supermajority",  # or "unanimous", "majority"
    threshold=0.66,
)
print(f"Decision: {result.decision} after {result.rounds} rounds")
```

**Stop conditions:**
- `"majority"` -- stop when >50% agree
- `"supermajority"` -- stop when threshold is met (default 2/3)
- `"unanimous"` -- stop only when all agree

### Stalemate Resolution

When a debate goes in circles (same votes, no new arguments), Consensus Council detects the stalemate and applies your chosen strategy:

```python
from consensus_council.stalemate import StalemateStrategy

council = Council(
    models=["gpt-4o", "claude-sonnet-4-5-20250514"],
    stalemate_strategy=StalemateStrategy.MODERATOR,
    moderator_model="gpt-4o",
)
```

| Strategy | Behavior |
|----------|----------|
| `STOP` | Accept the tie, return `TIE` result |
| `RANDOM_TIEBREAK` | Randomly pick YES or NO |
| `MODERATOR` | Query a designated model with all arguments to break the tie |
| `ESCALATE_TO_HUMAN` | Return an `ESCALATE` result for human review |

### Cost Control

Set hard budget limits to prevent runaway costs:

```python
from consensus_council.cost import CostCeiling

council = Council(
    models=["gpt-4o", "claude-sonnet-4-5-20250514", "gemini-2.0-flash"],
    cost_ceiling=CostCeiling(
        max_cost_per_vote=0.50,    # USD per vote() call
        max_cost_per_debate=5.00,  # USD per debate() call
    ),
)
```

Every result includes the total cost:

```python
result = council.vote("Is this safe?")
print(f"Cost: ${result.total_cost:.4f}")
```

You can also filter models that fit within a budget before creating the council:

```python
from consensus_council.cost import select_models_within_budget

affordable = select_models_within_budget(
    models=["gpt-4o", "claude-sonnet-4-5-20250514", "gemini-2.0-flash"],
    prompt="My question here",
    budget=0.10,
)
council = Council(models=affordable)
```

### Vote Extraction

Consensus Council robustly extracts YES/NO votes from freeform model responses. It handles:

- Explicit markers: `FINAL VOTE: YES`, `**NO**`, `DECISION: YES`
- Synonyms: "I concur", "LGTM", "reject", "block", "unsafe"
- Ambiguous text: falls back to counting affirmative/negative words in the tail
- Numeric scores: `7/10`, `8 out of 10`, `score: 9`

```python
from consensus_council import extract_vote, extract_score

vote, confidence = extract_vote("After careful review, I approve. FINAL VOTE: YES")
# vote=Vote.YES, confidence=0.95

score, confidence = extract_score("I would rate this a 7/10.")
# score=0.7, confidence=0.9
```

## CLI

Consensus Council includes a command-line interface for quick experiments:

```bash
# Simple vote
consensus-council vote "Is this approach correct?" \
    -m gpt-4o \
    -m claude-sonnet-4-5-20250514 \
    -t 0.66

# Vote with context from a file
consensus-council vote "Is this code safe?" \
    -m gpt-4o \
    -m gemini-2.0-flash \
    --context-file code.py

# Multi-round debate
consensus-council debate "Best approach for caching?" \
    -m gpt-4o \
    -m claude-sonnet-4-5-20250514 \
    -r 3 \
    --stop-on supermajority

# With stalemate handling
consensus-council debate "Should we migrate to Rust?" \
    -m gpt-4o \
    -m claude-sonnet-4-5-20250514 \
    -r 5 \
    --stalemate moderator \
    --moderator gpt-4o
```

## Result Object

Every `vote()` and `debate()` call returns a `ConsensusResult`:

```python
result.decision      # "YES", "NO", "TIE", "ABSTAIN", or "ESCALATE"
result.confidence    # 0.0 - 1.0
result.votes         # {model_name: VoteResult}
result.reasoning     # Merged reasoning from all models
result.rounds        # Number of debate rounds (1 for simple vote)
result.total_cost    # Total USD spent
result.failed_models # Models that errored out
```

Each `VoteResult` contains:

```python
vote.model      # Model name
vote.vote       # Vote.YES, Vote.NO, or Vote.ABSTAIN
vote.confidence # 0.0 - 1.0
vote.reasoning  # Full model response
vote.error      # Error message if the model failed
```

## Async Support

All operations support async via anyio:

```python
import anyio
from consensus_council import Council

async def main():
    council = Council(models=["gpt-4o", "claude-sonnet-4-5-20250514"])
    result = await council.avote("Is this safe?")
    print(result.decision)

anyio.run(main)
```

## Model Support

Consensus Council uses [LiteLLM](https://docs.litellm.ai/) under the hood, so it supports any model LiteLLM supports:

- OpenAI: `gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini`
- Anthropic: `claude-sonnet-4-5-20250514`, `claude-opus-4-20250514`, `claude-haiku-3-5-20241022`
- Google: `gemini-2.0-flash`, `gemini-2.5-pro`
- AWS Bedrock: `bedrock/anthropic.claude-3-sonnet`
- Azure: `azure/gpt-4o`
- Local: `ollama/llama3`, `vllm/...`

Set the appropriate API keys as environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc.).

## Disclaimer

Consensus Council is a tool for aggregating LLM opinions. It does **not** guarantee correctness, safety, or fitness for any particular purpose. Model outputs can be wrong, biased, or inconsistent.

**Do not rely on this library for medical, legal, financial, or safety-critical decisions without independent human review.** The authors accept no liability for decisions made based on model outputs, whether or not they reached consensus.

Use at your own risk.

## License

MIT -- see [LICENSE](LICENSE).
