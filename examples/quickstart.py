"""Quickstart example for hydra-consensus.

Demonstrates basic voting and debate across multiple LLM models.
Requires API keys set as environment variables (e.g. OPENAI_API_KEY,
ANTHROPIC_API_KEY, GEMINI_API_KEY).
"""

from hydra_consensus import Council

# --- Simple vote ---
council = Council(models=["gpt-4o", "claude-sonnet-4-5-20250514", "gemini-2.0-flash"])

result = council.vote(
    prompt="Is Python a good choice for building a REST API?",
    threshold=0.66,
    strategy="supermajority",
)

print(f"Decision: {result.decision}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Cost: ${result.total_cost:.4f}")
print()

for model, vote in result.votes.items():
    print(f"  {model}: {vote.vote.value} (confidence: {vote.confidence:.0%})")
    if vote.error:
        print(f"    ERROR: {vote.error}")

print()

# --- Multi-round debate ---
result = council.debate(
    prompt="Should we use a microservices or monolith architecture for a new SaaS product?",
    max_rounds=3,
    stop_on="supermajority",
    threshold=0.66,
)

print(f"Debate result: {result.decision}")
print(f"Rounds: {result.rounds}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Cost: ${result.total_cost:.4f}")
