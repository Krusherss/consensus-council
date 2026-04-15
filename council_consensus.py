"""Council Consensus — Multi-model consensus engine.

Queries Gemini, Claude, and GPT in parallel with blind voting,
anti-sycophancy measures, stalemate detection, and cost tracking.

Modes:
  - independent: Single round, simple majority vote.
  - debate: Up to 3 rounds with cross-talk, supermajority consensus,
            stalemate detection, and weighted majority fallback.
  - auto: Haiku classifies the question and picks the best mode.
"""

import sys
import io
import os
import re
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# Fix Windows encoding (guard against double-wrapping on re-import)
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from council_utils import (  # type: ignore[import]
    query_gemini,
    query_claude,
    query_gpt_pro,
    query_haiku,
    with_citation_instruction,
    CostTracker,
)
from web_search import (  # type: ignore[import]
    SEARCH_INSTRUCTION,
    has_search_tags,
    resolve_searches,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = ['gemini', 'claude', 'gpt']

MODEL_FUNCTIONS = {
    'gemini': query_gemini,
    'claude': query_claude,
    'gpt':    query_gpt_pro,
}

MODEL_LABELS = {
    'gemini': 'Model A',
    'claude': 'Model B',
    'gpt':    'Model C',
}

MODEL_TIMEOUT = 300   # seconds per model call
MAX_DEBATE_ROUNDS = 3

# Anti-sycophancy directive prepended to all prompts
_ANTI_SYCOPHANCY = """\
IMPORTANT: You are an independent evaluator. Give your HONEST assessment.
- Do NOT change your position just because others disagree.
- Only update if the LOGIC is undeniable and genuinely compelling.
- Genuine disagreement is MORE VALUABLE than false consensus.
- If you think something is wrong, SAY so, even if every other model disagrees.
"""


# ---------------------------------------------------------------------------
# Blind-voting label rotation (anti-position-bias)
# ---------------------------------------------------------------------------

def _get_rotation(round_num: int) -> dict:
    """Return {model_name: blind_label} for the given round.

    Labels rotate so no model occupies the same position twice:
      Round 0: gemini=A, claude=B, gpt=C
      Round 1: gemini=B, claude=C, gpt=A
      Round 2: gemini=C, claude=A, gpt=B
    """
    labels = list(MODEL_LABELS.values())  # ['Model A', 'Model B', 'Model C']
    return {
        MODELS[i]: labels[(i + round_num) % 3]
        for i in range(3)
    }


# ---------------------------------------------------------------------------
# Vote / confidence extraction
# ---------------------------------------------------------------------------

_VOTE_RE = re.compile(
    r'(?:FINAL\s+)?(?:VOTE|DECISION|VERDICT|ANSWER)\s*:\s*(YES|NO|ABSTAIN|MODIFY)',
    re.IGNORECASE,
)

_CONFIDENCE_RE = re.compile(
    r'CONFIDENCE\s*:\s*(\d{1,3})\s*%',
    re.IGNORECASE,
)


def _extract_vote(text: str) -> str:
    """Extract a vote from freeform model text.

    Scans for patterns like VOTE: YES, FINAL VOTE: NO, DECISION: MODIFY.
    Returns the uppercase vote string, or 'ABSTAIN' if nothing found.
    """
    match = _VOTE_RE.search(text)
    if match:
        return match.group(1).upper()
    return 'ABSTAIN'


def _extract_confidence(text: str) -> float:
    """Extract a confidence percentage from model text.

    Looks for CONFIDENCE: XX% and returns a float 0.0-1.0.
    Defaults to 0.5 if not found.
    """
    match = _CONFIDENCE_RE.search(text)
    if match:
        pct = int(match.group(1))
        return max(0.0, min(1.0, pct / 100.0))
    return 0.5


# ---------------------------------------------------------------------------
# Stalemate detection
# ---------------------------------------------------------------------------

def _detect_stalemate(votes_r1: dict, votes_r2: dict) -> bool:
    """Return True if all votes are identical between two rounds (no movement).

    Args:
        votes_r1: {model_name: vote_string} from round N-1.
        votes_r2: {model_name: vote_string} from round N.
    """
    if not votes_r1 or not votes_r2:
        return False
    for model in MODELS:
        if votes_r1.get(model) != votes_r2.get(model):
            return False
    return True


# ---------------------------------------------------------------------------
# Voting strategies
# ---------------------------------------------------------------------------

def _apply_voting_strategy(votes: dict, strategy: str) -> dict:
    """Apply a voting strategy to a {model: vote_string} dict.

    Args:
        votes: {model_name: {'vote': str, 'confidence': float, ...}}
        strategy: 'simple_majority', 'supermajority', or 'weighted_majority'.

    Returns:
        {'decision': str, 'count': int, 'total': int, ...}
    """
    vote_values = {m: v['vote'] for m, v in votes.items()}
    total = len(vote_values)

    if strategy == 'simple_majority':
        # 2+ of 3 agree → consensus
        counts: dict = {}
        for v in vote_values.values():
            counts[v] = counts.get(v, 0) + 1
        winner = max(counts, key=lambda k: counts[k])
        if counts[winner] >= 2:
            return {'decision': winner, 'count': counts[winner], 'total': total}
        return {'decision': 'TIE', 'count': 0, 'total': total}

    elif strategy == 'supermajority':
        # All 3 must agree (unanimous for a 3-model council)
        vals = list(vote_values.values())
        if vals[0] == vals[1] == vals[2] and vals[0] != 'ABSTAIN':
            return {'decision': vals[0], 'count': 3, 'total': total}
        return {'decision': 'TIE', 'count': 0, 'total': total}

    elif strategy == 'weighted_majority':
        # Weight by confidence; highest weighted side wins if >50% of total
        weight_by_vote: dict = {}
        total_weight = 0.0
        for _m, data in votes.items():
            v = data['vote']
            c = data['confidence']
            weight_by_vote[v] = weight_by_vote.get(v, 0.0) + c
            total_weight += c
        if total_weight == 0:
            return {'decision': 'TIE', 'count': 0, 'total': total}
        winner = max(weight_by_vote, key=lambda k: weight_by_vote[k])
        if weight_by_vote[winner] / total_weight > 0.5:
            count = sum(1 for v in vote_values.values() if v == winner)
            return {
                'decision': winner,
                'count': count,
                'total': total,
                'weight': weight_by_vote[winner],
                'total_weight': total_weight,
            }
        return {'decision': 'TIE', 'count': 0, 'total': total}

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Mode router (Haiku)
# ---------------------------------------------------------------------------

def _route_mode(prompt: str) -> str:
    """Use Haiku to classify the question as independent or debate.

    STRUCTURED questions (clear YES/NO proposals) → independent.
    OPEN_ENDED questions (design, strategy, analysis) → debate.
    """
    classification_prompt = (
        "Classify this question as either STRUCTURED or OPEN_ENDED.\n\n"
        "STRUCTURED: Has clear proposals to vote on, asks for YES/NO decisions, "
        "evaluates specific options.\n"
        "OPEN_ENDED: Asks for design recommendations, strategy, architecture, "
        "or complex analysis.\n\n"
        "Respond with exactly one word: STRUCTURED or OPEN_ENDED\n\n"
        f"Question: {prompt}"
    )
    result = query_haiku(classification_prompt).strip().upper()
    if 'STRUCTURED' in result:
        return 'independent'
    return 'debate'


# ---------------------------------------------------------------------------
# Parallel model querying with blind labels
# ---------------------------------------------------------------------------

def _query_all_parallel(prompt: str, round_num: int = 0, blind: bool = True,
                        enable_search: bool = False) -> dict:
    """Query all 3 models in parallel with optional blind labeling.

    Args:
        prompt: The base prompt to send.
        round_num: Current round (affects label rotation).
        blind: If True, wrap prompt with blind identity instructions.
        enable_search: If True, prepend Tavily search instructions and resolve
                       any [SEARCH: query] tags models emit before final answer.

    Returns:
        {model: {'response': str, 'elapsed': float, 'vote': str, 'confidence': float}}
    """
    rotation = _get_rotation(round_num)
    results = {}

    # Prepend search instruction when enabled
    search_prefix = f"{SEARCH_INSTRUCTION}\n\n" if enable_search else ""

    def _call(model_name: str):
        fn = MODEL_FUNCTIONS[model_name]
        label = rotation[model_name]

        if blind:
            full_prompt = (
                f"{_ANTI_SYCOPHANCY}\n"
                f"You are {label}. Other models will also answer but you cannot "
                f"see their responses yet.\n\n"
                f"{search_prefix}{prompt}\n\n"
                f"End your response with VOTE: YES/NO/ABSTAIN and CONFIDENCE: XX%."
            )
        else:
            full_prompt = f"{search_prefix}{prompt}"

        start = time.monotonic()
        response = fn(full_prompt)

        # Resolve any [SEARCH: query] tags the model emitted
        if enable_search and has_search_tags(response):
            print(f"  [{model_name}] Web searches detected — fetching results...")
            resolved, search_log = resolve_searches(response)
            if search_log:
                queries_done = [s["query"] for s in search_log]
                print(f"  [{model_name}] Searches completed: {queries_done}")
                followup = (
                    f"You requested web searches. Here are the live results:\n\n"
                    f"{resolved}\n\n"
                    f"Now provide your complete final response incorporating these results.\n"
                    f"End with VOTE: YES/NO/ABSTAIN and CONFIDENCE: XX%."
                )
                response = fn(followup)

        elapsed = time.monotonic() - start
        return model_name, response, elapsed, full_prompt

    futures = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for model in MODELS:
            future = executor.submit(_call, model)
            futures[future] = model

        for future in as_completed(futures, timeout=MODEL_TIMEOUT + 10):
            model_name = futures[future]
            try:
                _, response, elapsed, full_prompt = future.result(timeout=0)
            except TimeoutError:
                response = f"ERROR: timed out after {MODEL_TIMEOUT}s"
                elapsed = MODEL_TIMEOUT
                full_prompt = prompt
            except Exception as e:
                response = f"ERROR: {e}"
                elapsed = 0.0
                full_prompt = prompt

            results[model_name] = {
                'response': response,
                'elapsed': elapsed,
                'vote': _extract_vote(response),
                'confidence': _extract_confidence(response),
                'label': rotation[model_name],
                '_full_prompt': full_prompt,
            }
            print(f"  {model_name.capitalize()} ({rotation[model_name]}): "
                  f"received ({elapsed:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Session directory management
# ---------------------------------------------------------------------------

def _session_dir() -> str:
    """Create and return a timestamped session directory."""
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions', stamp)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _build_round_md(title: str, timestamp: str, responses: dict, round_label: str = "") -> str:
    """Build a markdown summary of one round's responses."""
    label = f" -- {round_label}" if round_label else ""
    lines = [f"# {title}{label}", f"**Timestamp:** {timestamp}", ""]
    for model, data in responses.items():
        vote = data['vote']
        conf = data['confidence']
        elapsed = data['elapsed']
        blind_label = data.get('label', model)
        lines += [
            f"## {model.capitalize()} [{blind_label}] ({elapsed:.1f}s) "
            f"| Vote: {vote} | Confidence: {conf:.0%}",
            "",
            data['response'],
            "",
        ]
    return "\n".join(lines)


def _print_responses(responses: dict, limit: int = 0):
    """Print model responses, optionally truncated."""
    for model, data in responses.items():
        label = data.get('label', model)
        elapsed = data['elapsed']
        vote = data['vote']
        conf = data['confidence']
        resp = data['response']
        print(f"\n--- {model.capitalize()} [{label}] ({elapsed:.1f}s) "
              f"| Vote: {vote} | Confidence: {conf:.0%} ---")
        if limit and len(resp) > limit:
            print(resp[:limit] + f"\n... [{len(resp) - limit} chars truncated]")
        else:
            print(resp)


def _print_vote_tally(responses: dict):
    """Print a compact vote tally."""
    print("\n[Vote Tally]")
    for model, data in responses.items():
        print(f"  {model.capitalize()}: {data['vote']} "
              f"(confidence: {data['confidence']:.0%})")


def _build_disagreement_output(responses: dict) -> dict:
    """Build structured output showing where models agree and disagree.

    Returns a dict with vote_breakdown, majority/minority info, and
    per-model position summaries for the session JSON.
    """
    vote_groups: dict[str, list[str]] = {}
    for model, data in responses.items():
        v = data['vote']
        if v not in vote_groups:
            vote_groups[v] = []
        vote_groups[v].append(model)

    sorted_groups = sorted(vote_groups.items(), key=lambda x: -len(x[1]))
    majority_vote = sorted_groups[0][0] if sorted_groups else 'ABSTAIN'
    majority_models = sorted_groups[0][1] if sorted_groups else []

    result: dict = {
        'vote_breakdown': {v: models for v, models in vote_groups.items()},
        'majority_vote': majority_vote,
        'majority_models': majority_models,
        'is_split': len(vote_groups) > 1,
        'positions': {
            model: {
                'vote': data['vote'],
                'confidence': data['confidence'],
                'summary': data['response'][:500] + '...' if len(data['response']) > 500 else data['response'],
            }
            for model, data in responses.items()
        },
    }

    if len(sorted_groups) > 1:
        result['minority_vote'] = sorted_groups[1][0]
        result['minority_models'] = sorted_groups[1][1]

    return result


# ---------------------------------------------------------------------------
# Main council function
# ---------------------------------------------------------------------------

def ask_council(prompt: str, mode: str = "auto", enable_search: bool = False) -> dict:
    """Run the Council Consensus on a question.

    Args:
        prompt: The user's question.
        mode: 'auto' (Haiku routes), 'independent', or 'debate'.
        enable_search: If True, models can use [SEARCH: query] to run live
                       Tavily web searches before giving their final answer.

    Returns:
        Dict with mode, responses, votes, decision, cost, session path.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    session_path = _session_dir()
    tracker = CostTracker(log_dir=session_path)

    # ------------------------------------------------------------------
    # Step 1: Mode routing
    # ------------------------------------------------------------------
    if mode == 'auto':
        print("\n[Mode Router] Classifying question with Haiku...")
        mode = _route_mode(prompt)
        print(f"[Mode Router] Selected: {mode}")
    elif mode not in ('independent', 'debate'):
        print(f"[Warning] Unknown mode '{mode}', defaulting to independent.")
        mode = 'independent'

    title = f"Council Consensus -- {mode.capitalize()} Mode"

    # ------------------------------------------------------------------
    # Step 2: Add citation instruction
    # ------------------------------------------------------------------
    prompt = with_citation_instruction(prompt)

    # ------------------------------------------------------------------
    # Step 3a: INDEPENDENT mode
    # ------------------------------------------------------------------
    if mode == 'independent':
        print(f"\n[Independent] Querying all models in parallel...")
        responses = _query_all_parallel(prompt, round_num=0, blind=True,
                                        enable_search=enable_search)

        # Vote tally
        _print_responses(responses, limit=800)
        _print_vote_tally(responses)

        strategy_result = _apply_voting_strategy(responses, 'simple_majority')
        decision = strategy_result['decision']
        print(f"\n[Decision] {decision} "
              f"({strategy_result['count']}/{strategy_result['total']} agree)")

        # Cost
        for model, data in responses.items():
            tracker.log_call(model, data.get('_full_prompt', prompt), data['response'])
        tracker.print_summary()
        total_cost = tracker.session_total

        # Save result.md
        result_md = _build_round_md(title, timestamp, responses)
        result_md += f"\n---\n**Decision:** {decision}\n"
        with open(os.path.join(session_path, 'result.md'), 'w',
                  encoding='utf-8', errors='replace') as f:
            f.write(result_md)

        # Save result.json
        result_data = {
            'mode': 'independent',
            'timestamp': timestamp,
            'prompt': prompt,
            'responses': {
                m: {
                    'response': d['response'],
                    'elapsed': d['elapsed'],
                    'vote': d['vote'],
                    'confidence': d['confidence'],
                    'label': d['label'],
                }
                for m, d in responses.items()
            },
            'votes': {m: d['vote'] for m, d in responses.items()},
            'decision': decision,
            'strategy': 'simple_majority',
            'strategy_result': strategy_result,
            'cost_total': total_cost,
        }
        with open(os.path.join(session_path, 'result.json'), 'w',
                  encoding='utf-8', errors='replace') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"\nSaved to: {session_path}")
        return result_data

    # ------------------------------------------------------------------
    # Step 3b: DEBATE mode
    # ------------------------------------------------------------------
    all_rounds = []       # list of response dicts per round
    all_votes = []        # list of {model: vote_str} per round
    decision = None

    for round_num in range(MAX_DEBATE_ROUNDS):
        round_display = round_num + 1
        print(f"\n{'='*60}")
        print(f"[Round {round_display}/{MAX_DEBATE_ROUNDS}]")
        print(f"{'='*60}")

        if round_num == 0:
            # First round: blind independent query
            print("Querying all models independently (blind)...")
            responses = _query_all_parallel(prompt, round_num=0, blind=True,
                                            enable_search=enable_search)
        else:
            # Subsequent rounds: cross-talk with other models' responses
            print("Building cross-talk prompts...")
            prev = all_rounds[-1]
            rotation = _get_rotation(round_num)

            # Query each model with the others' prior responses
            def _build_crosstalk(model_name: str) -> str:
                label = rotation[model_name]
                other_parts = []
                for other_model, other_data in prev.items():
                    if other_model == model_name:
                        continue
                    other_label = other_data.get('label', other_model)
                    other_resp = other_data['response']
                    if len(other_resp) > 2000:
                        other_resp = other_resp[:2000] + "\n[... truncated ...]"
                    other_parts.append(
                        f"- {other_label}: {other_resp}"
                    )

                search_prefix = f"{SEARCH_INSTRUCTION}\n\n" if enable_search else ""
                return (
                    f"{_ANTI_SYCOPHANCY}\n"
                    f"You are {label}. This is round {round_display} of a debate.\n\n"
                    f"{search_prefix}"
                    f"Original question: {prompt}\n\n"
                    f"Round {round_display - 1} responses from other models:\n"
                    f"{''.join(other_parts)}\n\n"
                    f"Challenge any unsupported claims from other models. "
                    f"Back your own assertions with sources. "
                    f"State whether you agree or disagree and why. "
                    f"End with VOTE: YES/NO/ABSTAIN and CONFIDENCE: XX%."
                )

            # Query in parallel with custom per-model prompts
            responses = {}
            futures = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                for model in MODELS:
                    ct_prompt = _build_crosstalk(model)

                    def _call(m=model, p=ct_prompt):
                        fn = MODEL_FUNCTIONS[m]
                        start = time.monotonic()
                        resp = fn(p)
                        # Resolve search tags in cross-talk rounds too
                        if enable_search and has_search_tags(resp):
                            print(f"  [{m}] Web searches detected in debate round — fetching...")
                            resolved, search_log = resolve_searches(resp)
                            if search_log:
                                queries_done = [s["query"] for s in search_log]
                                print(f"  [{m}] Searches completed: {queries_done}")
                                followup = (
                                    f"You requested web searches. Here are the live results:\n\n"
                                    f"{resolved}\n\n"
                                    f"Now provide your complete updated response with citations. "
                                    f"Challenge unsupported claims from other models. "
                                    f"End with VOTE: YES/NO/ABSTAIN and CONFIDENCE: XX%."
                                )
                                resp = fn(followup)
                        elapsed = time.monotonic() - start
                        return m, resp, elapsed, p

                    futures[executor.submit(_call)] = model

                for future in as_completed(futures, timeout=MODEL_TIMEOUT + 10):
                    model_name = futures[future]
                    try:
                        _, response, elapsed, full_prompt = future.result(timeout=0)
                    except Exception as e:
                        response = f"ERROR: {e}"
                        elapsed = 0.0
                        full_prompt = prompt

                    rot = _get_rotation(round_num)
                    responses[model_name] = {
                        'response': response,
                        'elapsed': elapsed,
                        'vote': _extract_vote(response),
                        'confidence': _extract_confidence(response),
                        'label': rot[model_name],
                        '_full_prompt': full_prompt,
                    }
                    print(f"  {model_name.capitalize()} ({rot[model_name]}): "
                          f"received ({elapsed:.1f}s)")

        # Record this round
        all_rounds.append(responses)
        round_votes = {m: d['vote'] for m, d in responses.items()}
        all_votes.append(round_votes)

        # Log costs for this round
        for model, data in responses.items():
            tracker.log_call(model, data.get('_full_prompt', prompt), data['response'])

        # Print results
        _print_responses(responses, limit=800)
        _print_vote_tally(responses)

        # Save round markdown
        round_md = _build_round_md(title, timestamp, responses,
                                    round_label=f"Round {round_display}")
        with open(os.path.join(session_path, f'round{round_display}.md'), 'w',
                  encoding='utf-8', errors='replace') as f:
            f.write(round_md)

        # Check supermajority (all 3 agree)
        strategy_result = _apply_voting_strategy(responses, 'supermajority')
        if strategy_result['decision'] != 'TIE':
            decision = strategy_result['decision']
            print(f"\n[Consensus] Unanimous agreement: {decision} "
                  f"(round {round_display})")
            break

        # Check stalemate (round 2+)
        if round_num > 0 and _detect_stalemate(all_votes[-2], all_votes[-1]):
            print(f"\n[Stalemate] Positions locked between rounds "
                  f"{round_display - 1} and {round_display}.")
            wm_result = _apply_voting_strategy(responses, 'weighted_majority')
            if wm_result['decision'] != 'TIE':
                decision = wm_result['decision']
                print(f"[Weighted Majority] Decision: {decision}")
            else:
                disagree = _build_disagreement_output(responses)
                print("[No Consensus] Models could not reach agreement. Vote breakdown:")
                for vote, models in disagree['vote_breakdown'].items():
                    print(f"  {vote}: {', '.join(m.capitalize() for m in models)}")
                decision = 'NO_CONSENSUS'
            break

    # If we exhausted all rounds without consensus
    if decision is None:
        print(f"\n[No Consensus] Max rounds ({MAX_DEBATE_ROUNDS}) reached.")
        last_responses = all_rounds[-1]
        fallback = _apply_voting_strategy(last_responses, 'simple_majority')
        if fallback['decision'] != 'TIE':
            decision = fallback['decision']
            print(f"[Fallback] Simple majority: {decision} "
                  f"({fallback['count']}/{fallback['total']})")
        else:
            wm_fallback = _apply_voting_strategy(last_responses, 'weighted_majority')
            if wm_fallback['decision'] != 'TIE':
                decision = wm_fallback['decision']
                print(f"[Fallback] Weighted majority: {decision}")
            else:
                disagree = _build_disagreement_output(last_responses)
                print("[No Consensus] Models could not agree after all rounds. Vote breakdown:")
                for vote, models in disagree['vote_breakdown'].items():
                    print(f"  {vote}: {', '.join(m.capitalize() for m in models)}")
                decision = 'NO_CONSENSUS'

    # Cost summary
    tracker.print_summary()
    total_cost = tracker.session_total

    # Save result.json
    result_data = {
        'mode': 'debate',
        'timestamp': timestamp,
        'prompt': prompt,
        'rounds': len(all_rounds),
        'all_responses': {
            f'round{i+1}': {
                m: {
                    'response': d['response'],
                    'elapsed': d['elapsed'],
                    'vote': d['vote'],
                    'confidence': d['confidence'],
                    'label': d['label'],
                }
                for m, d in round_data.items()
            }
            for i, round_data in enumerate(all_rounds)
        },
        'votes': all_votes,
        'decision': decision,
        'cost_total': total_cost,
    }
    with open(os.path.join(session_path, 'result.json'), 'w',
              encoding='utf-8', errors='replace') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {session_path}")
    return result_data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== Council Consensus ===")
    user_prompt = input("Enter your question: ").strip()
    if not user_prompt:
        print("No question provided. Exiting.")
        sys.exit(0)

    mode_input = input(
        "Mode? [a]uto / [i]ndependent / [d]ebate (default: auto): "
    ).strip().lower()

    if mode_input.startswith('i'):
        chosen_mode = 'independent'
    elif mode_input.startswith('d'):
        chosen_mode = 'debate'
    else:
        chosen_mode = 'auto'

    ask_council(user_prompt, mode=chosen_mode)
