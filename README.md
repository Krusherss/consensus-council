# consensus-council

> Multi-model voting and three-stage deliberation with blind review, optional live sourcing, and cost controls.

Ask multiple LLMs independently, rotate anonymous peer-review labels, and let a separate chairman synthesize the evidence and disagreements. The original voting and debate APIs remain available for structured decisions.

The library is provider-configurable through LiteLLM. A featured profile below mirrors the author's current Claude Code council while keeping every model replaceable.

## Three-stage council

The new `deliberate()` path mirrors the current Claude Code workflow:

1. **Independent panel:** every model answers without seeing the others.
2. **Blind peer review:** each reviewer sees anonymous answers with a different label rotation, reducing position and identity bias.
3. **Chairman synthesis:** a separate model receives the de-anonymized answers and reviews, reconciles disagreements, and returns the final response.

```python
from consensus_council import Council

council = Council(
    models=[
        "openai/o3",
        "gemini/gemini-2.5-pro",
        "xai/grok-4",
    ],
    max_tokens=4096,
)

result = council.deliberate(
    "Review this architecture and identify the safest design.",
    chair_model="anthropic/claude-opus-4-6",
    mode="auto",           # routes to simple or debate
    debate_rounds=2,
    enable_search=True,     # optional DuckDuckGo + Trafilatura sourcing
    output_dir=None,        # no transcripts are written by default
)

print(result.synthesis)
```

Model aliases change over time and vary by provider account. Treat the names above as a profile, not a lock-in requirement; any LiteLLM-supported panel and chairman can be supplied.

### What web sourcing does

When `enable_search=True`, panelists can emit `[SEARCH: query]`. The search helper:

- searches DuckDuckGo without an additional search API key;
- extracts readable source-page content with Trafilatura;
- limits each response to five searches and runs them concurrently;
- sends the retrieved material back to the requesting model for a cited revision;
- tells the chairman to retain claim-level citations and never invent sources.

Search provenance improves auditability; it does not prove that a source or model conclusion is correct.

### CLI

```bash
consensus-council deliberate "Review this design" \
  -m openai/o3 \
  -m gemini/gemini-2.5-pro \
  -m xai/grok-4 \
  --chair anthropic/claude-opus-4-6 \
  --mode auto \
  --search
```

Add `--output-dir ./council-runs` only when you intentionally want Markdown checkpoints and the final audit artifact.

### Secret and data handling

- API credentials are read from normal provider environment variables; no credentials are bundled.
- `.env`, virtual environments, build products, and standalone session outputs are Git-ignored.
- Provider errors are reduced to exception classes instead of echoing potentially sensitive response details.
- Search and output artifacts are opt-in. Review generated artifacts before committing them because prompts and model responses may contain private input.


**Standalone compatibility:** The root-level `council_consensus.py` is retained for existing v0.2 users. The packaged `consensus_council.Council` API is the source of truth for the three-stage workflow and featured Grok profile.

## Installation

```bash
pip install consensus-council

# With live web search support
pip install consensus-council[search]
```

## Quick Start

```python
from consensus_council import Council

council = Council(models=["openai/o3", "xai/grok-4", "gemini/gemini-2.5-pro"])
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
    models=["openai/o3", "xai/grok-4", "gemini/gemini-2.5-pro"],
    weights={"openai/o3": 2.0, "xai/grok-4": 1.5, "gemini/gemini-2.5-pro": 1.0},
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
    models=["openai/o3", "xai/grok-4"],
    stalemate_strategy=StalemateStrategy.MODERATOR,
    moderator_model="openai/o3",
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
    models=["openai/o3", "xai/grok-4", "gemini/gemini-2.5-pro"],
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
    models=["openai/o3", "xai/grok-4", "gemini/gemini-2.5-pro"],
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

### Live Web Search

Enable real-time web search so models can fetch evidence mid-debate. Models emit `[SEARCH: query]` tags — the council resolves them via DuckDuckGo (free, no API key) and feeds results back before the final answer.

```python
# Any model can search the web during debate
result = council.debate(
    "What are the latest clinical trial results for KRAS G12C inhibitors?",
    enable_search=True,
    max_rounds=3,
)

# Works with vote() and decide() too
result = council.vote("Is sotorasib approved for second-line NSCLC?", enable_search=True)
result = council.decide("Best approach for this architecture?", enable_search=True)
```

Models write searches inline:
```
Based on recent data [SEARCH: sotorasib CodeBreaK 200 trial 2025 results]
the evidence suggests...
```

The council fetches full page content (via trafilatura), strips boilerplate, and injects it back into the model's response before voting. Up to 5 searches per model per round, run in parallel.

```python
# Use web search standalone
from consensus_council import web_search, resolve_searches

results = web_search("KRAS G12C resistance mechanisms 2025")
print(results)  # Formatted string ready to inject into any prompt

# Resolve [SEARCH: ...] tags in any text
resolved_text, search_log = resolve_searches(model_response)
```

## CLI

Consensus Council includes a command-line interface for quick experiments:

```bash
# Simple vote
consensus-council vote "Is this approach correct?" \
    -m openai/o3 \
    -m xai/grok-4 \
    -t 0.66

# Vote with context from a file
consensus-council vote "Is this code safe?" \
    -m openai/o3 \
    -m gemini/gemini-2.5-pro \
    --context-file code.py

# Multi-round debate
consensus-council debate "Best approach for caching?" \
    -m openai/o3 \
    -m xai/grok-4 \
    -r 3 \
    --stop-on supermajority

# With stalemate handling
consensus-council debate "Should we migrate to Rust?" \
    -m openai/o3 \
    -m xai/grok-4 \
    -r 5 \
    --stalemate moderator \
    --moderator openai/o3
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
    council = Council(models=["openai/o3", "xai/grok-4"])
    result = await council.avote("Is this safe?")
    print(result.decision)

anyio.run(main)
```

## Model Support

Consensus Council uses [LiteLLM](https://docs.litellm.ai/) under the hood, so it supports any model LiteLLM supports:

- OpenAI: `openai/o3`
- Anthropic: `anthropic/claude-opus-4-6`
- Google: `gemini/gemini-2.5-pro`
- xAI: `xai/grok-4`
- Local: `ollama/llama3`, `vllm/...`

Set the appropriate API keys as environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, etc.). Model aliases may change; verify them against your installed LiteLLM version and provider account.

## Disclaimer

Consensus Council is a tool for aggregating LLM opinions. It does **not** guarantee correctness, safety, or fitness for any particular purpose. Model outputs can be wrong, biased, or inconsistent.

**Do not rely on this library for medical, legal, financial, or safety-critical decisions without independent human review.** The authors accept no liability for decisions made based on model outputs, whether or not they reached consensus.

Use at your own risk.

## Acknowledgments

The three-stage answer → peer-review → chairman pattern is inspired by [Andrej Karpathy's `llm-council`](https://github.com/karpathy/llm-council). This project adds configurable voting, rotating blind labels, debate rounds, cost ceilings, keyless source retrieval, checkpoint artifacts, and a reusable Python/CLI interface.

## License

MIT -- see [LICENSE](LICENSE).
