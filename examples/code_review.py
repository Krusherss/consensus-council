"""Code review example using hydra-consensus.

Ask multiple models to review a code diff and reach consensus on
whether it is safe to deploy.
"""

from hydra_consensus import Council
from hydra_consensus.cost import CostCeiling

# Example code diff to review
code_diff = '''
--- a/auth.py
+++ b/auth.py
@@ -42,6 +42,12 @@ def authenticate(request):
     token = request.headers.get("Authorization")
     if not token:
         return None
+
+    # New: also check query parameter for API clients
+    if not token:
+        token = request.args.get("token")
+    if not token:
+        return None

     try:
-        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
+        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256", "none"])
         return User.query.get(payload["user_id"])
     except jwt.InvalidTokenError:
         return None
'''

# Set up a council with cost control
council = Council(
    models=["gpt-4o", "claude-sonnet-4-5-20250514", "gemini-2.0-flash"],
    cost_ceiling=CostCeiling(max_cost_per_vote=0.50),
)

# Run the review
result = council.vote(
    prompt=(
        "Review this code diff for security vulnerabilities. "
        "Is this change safe to deploy? "
        "Pay special attention to authentication bypasses, injection, "
        "and algorithm confusion attacks."
    ),
    context=code_diff,
    threshold=0.66,
    strategy="supermajority",
)

print(f"Safe to deploy? {result.decision}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Cost: ${result.total_cost:.4f}")
print()

for model, vote in result.votes.items():
    print(f"--- {model}: {vote.vote.value} ---")
    print(vote.reasoning[:500])
    print()
