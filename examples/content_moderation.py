"""Content moderation example using hydra-consensus.

Use multiple models to decide whether user-submitted content violates
community guidelines, with debate for borderline cases.
"""

from hydra_consensus import Council
from hydra_consensus.cost import CostCeiling
from hydra_consensus.stalemate import StalemateStrategy

# Example user-submitted content
user_content = """
Hey everyone! I found this amazing trick to get free premium accounts.
Just visit this link and enter your existing credentials to verify your
identity, and you'll get lifetime premium access. Works 100% of the time!
Link: http://totally-legit-site.example.com/free-premium
"""

# Set up council with stalemate resolution
council = Council(
    models=["gpt-4o", "claude-sonnet-4-5-20250514", "gemini-2.0-flash"],
    cost_ceiling=CostCeiling(max_cost_per_debate=2.00),
    stalemate_strategy=StalemateStrategy.MODERATOR,
    moderator_model="gpt-4o",
)

# First: quick vote
print("=== Quick Vote ===")
result = council.vote(
    prompt=(
        "Does this user-submitted content violate community guidelines? "
        "Consider: phishing attempts, scam promotion, credential theft, "
        "and misleading claims. Vote YES if it violates guidelines, NO if acceptable."
    ),
    context=user_content,
    threshold=0.66,
    strategy="supermajority",
)

print(f"Violates guidelines? {result.decision}")
print(f"Confidence: {result.confidence:.0%}")
print()

# If borderline, escalate to debate
if result.decision == "TIE" or result.confidence < 0.8:
    print("=== Escalating to Debate ===")
    debate_result = council.debate(
        prompt=(
            "Debate whether this content should be removed. "
            "Consider: is this a phishing attempt? Is the user being "
            "malicious or just naive? What's the harm potential?"
        ),
        context=user_content,
        max_rounds=3,
        stop_on="supermajority",
        threshold=0.66,
    )

    print(f"Final decision: {debate_result.decision}")
    print(f"Rounds needed: {debate_result.rounds}")
    print(f"Total cost: ${debate_result.total_cost:.4f}")
else:
    print(f"Clear consensus reached. Total cost: ${result.total_cost:.4f}")
