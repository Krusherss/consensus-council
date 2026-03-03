"""Tests for anti-sycophancy and anti-anchoring measures."""

from __future__ import annotations

from consensus_council.anti_sycophancy import (
    BlindVoting,
    RotationOrder,
    build_anti_sycophancy_directive,
    build_crosstalk_prompt,
)


class TestAntiSycophancyDirective:
    """Test the anti-sycophancy prompt directive."""

    def test_directive_not_empty(self) -> None:
        directive = build_anti_sycophancy_directive()
        assert len(directive) > 50

    def test_directive_contains_key_phrases(self) -> None:
        directive = build_anti_sycophancy_directive()
        assert "Do NOT change your vote" in directive
        assert "LOGIC" in directive
        assert "Do NOT soften your position" in directive
        assert "Genuine disagreement" in directive

    def test_directive_mentions_conflict(self) -> None:
        directive = build_anti_sycophancy_directive()
        assert "conflict" in directive.lower()

    def test_directive_mentions_honest(self) -> None:
        directive = build_anti_sycophancy_directive()
        assert "HONEST" in directive


class TestRotationOrder:
    """Test that rotation ordering prevents anchoring bias."""

    def test_round_zero_is_natural_order(self) -> None:
        rotation = RotationOrder(n_models=4)
        order = rotation.order_for_round(0)
        assert order == [0, 1, 2, 3]

    def test_subsequent_rounds_differ(self) -> None:
        rotation = RotationOrder(n_models=4)
        orders = rotation.all_orders(5)
        # Round 0 is natural order; at least one later round should differ
        natural = orders[0]
        assert any(o != natural for o in orders[1:])

    def test_all_indices_present(self) -> None:
        """Every round must contain all model indices exactly once."""
        rotation = RotationOrder(n_models=5)
        for round_num in range(10):
            order = rotation.order_for_round(round_num)
            assert sorted(order) == [0, 1, 2, 3, 4]

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same orderings."""
        r1 = RotationOrder(n_models=4, _seed=123)
        r2 = RotationOrder(n_models=4, _seed=123)
        for rnd in range(5):
            assert r1.order_for_round(rnd) == r2.order_for_round(rnd)

    def test_different_seeds_differ(self) -> None:
        r1 = RotationOrder(n_models=4, _seed=100)
        r2 = RotationOrder(n_models=4, _seed=200)
        # At least one round should differ
        differ = any(
            r1.order_for_round(r) != r2.order_for_round(r) for r in range(1, 10)
        )
        assert differ

    def test_single_model(self) -> None:
        rotation = RotationOrder(n_models=1)
        assert rotation.order_for_round(0) == [0]
        assert rotation.order_for_round(5) == [0]


class TestBlindVoting:
    """Test blind voting prompt construction."""

    def test_prompt_includes_original_question(self) -> None:
        blind = BlindVoting(prompt="Is this safe?")
        prompt = blind.build_prompt("gpt-4o")
        assert "Is this safe?" in prompt

    def test_prompt_includes_model_name(self) -> None:
        blind = BlindVoting(prompt="Review this code")
        prompt = blind.build_prompt("claude-sonnet-4-5-20250514")
        assert "claude-sonnet-4-5-20250514" in prompt

    def test_prompt_excludes_other_responses(self) -> None:
        blind = BlindVoting(prompt="Test question")
        prompt = blind.build_prompt("model-a")
        assert "other model" not in prompt.lower() or "NOT seen any other model" in prompt

    def test_prompt_includes_context(self) -> None:
        blind = BlindVoting(prompt="Review this", context="def foo(): pass")
        prompt = blind.build_prompt("model-a")
        assert "def foo(): pass" in prompt

    def test_prompt_without_context(self) -> None:
        blind = BlindVoting(prompt="Simple question")
        prompt = blind.build_prompt("model-a")
        assert "CONTEXT" not in prompt or "END CONTEXT" not in prompt

    def test_prompt_includes_anti_sycophancy(self) -> None:
        blind = BlindVoting(prompt="Test")
        prompt = blind.build_prompt("model-a")
        assert "Do NOT change your vote" in prompt

    def test_prompt_asks_for_final_vote(self) -> None:
        blind = BlindVoting(prompt="Test")
        prompt = blind.build_prompt("model-a")
        assert "FINAL VOTE" in prompt


class TestCrosstalkPrompt:
    """Test debate cross-talk prompt construction."""

    def test_includes_original_prompt(self) -> None:
        prompt = build_crosstalk_prompt(
            model_name="model-a",
            round_num=2,
            prev_responses={"model-b": "I think yes."},
            prev_votes={"model-b": "YES"},
            original_prompt="Should we deploy?",
        )
        assert "Should we deploy?" in prompt

    def test_includes_other_models_responses(self) -> None:
        prompt = build_crosstalk_prompt(
            model_name="model-a",
            round_num=2,
            prev_responses={"model-b": "Deploy is fine."},
            prev_votes={"model-b": "YES"},
            original_prompt="Deploy?",
        )
        assert "Deploy is fine." in prompt
        assert "model-b" in prompt

    def test_excludes_own_response(self) -> None:
        """A model should not see its own previous response as 'another model'."""
        prompt = build_crosstalk_prompt(
            model_name="model-a",
            round_num=2,
            prev_responses={
                "model-a": "My old response",
                "model-b": "Other response",
            },
            prev_votes={"model-a": "YES", "model-b": "NO"},
            original_prompt="Test?",
        )
        # model-a's response should not appear under OTHER MODELS section
        # (it may appear elsewhere, but the section heading should precede model-b only)
        lines = prompt.split("\n")
        in_others_section = False
        found_own = False
        for line in lines:
            if "OTHER MODELS" in line:
                in_others_section = True
            if "END OTHER ARGUMENTS" in line:
                in_others_section = False
            if in_others_section and "model-a" in line and "---" in line:
                found_own = True
        assert not found_own

    def test_includes_anti_sycophancy(self) -> None:
        prompt = build_crosstalk_prompt(
            model_name="a",
            round_num=2,
            prev_responses={"b": "Response"},
            prev_votes={"b": "YES"},
            original_prompt="Test",
        )
        assert "Do NOT change your vote" in prompt

    def test_includes_context(self) -> None:
        prompt = build_crosstalk_prompt(
            model_name="a",
            round_num=2,
            prev_responses={},
            prev_votes={},
            original_prompt="Test",
            context="Some code here",
        )
        assert "Some code here" in prompt

    def test_truncates_long_responses(self) -> None:
        long_response = "x" * 5000
        prompt = build_crosstalk_prompt(
            model_name="a",
            round_num=2,
            prev_responses={"b": long_response},
            prev_votes={"b": "YES"},
            original_prompt="Test",
        )
        assert "truncated" in prompt.lower()

    def test_round_number_displayed(self) -> None:
        prompt = build_crosstalk_prompt(
            model_name="a",
            round_num=3,
            prev_responses={},
            prev_votes={},
            original_prompt="Test",
        )
        assert "ROUND 3" in prompt
