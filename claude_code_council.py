#!/usr/bin/env python3
"""
LLM Council — Karpathy 3-stage + Hydra enhancements (CLI)
Primary source: https://github.com/karpathy/llm-council

Stage 1: Parallel independent query — anti-sycophancy, optional web search
Stage 2: Anonymous peer review + FINAL RANKING — blind label rotation
Stage 3: Chairman synthesis (de-anonymized)

Modes:
  auto    — Haiku classifies question and picks simple or debate (default)
  simple  — Karpathy 3-stage only
  debate  — 2 rounds of cross-talk before Stage 2/3

Usage:
    python council.py "Your question here"
    python council.py --mode debate "Your question here"
    python council.py --file question.md
    cat diff.patch | python council.py "Review this diff:"

Required env vars: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY
Panel: o3 (OpenAI) · Gemini 2.5 Pro (Google) · Grok 4 (xAI) | Chairman: Claude Opus 4.6

Output: ~/council/YYYY-MM-DD-HHMM-<slug>.md  (full session)
        ~/council/chairman-log.md             (rolling synthesis log, recall-indexed)
"""

import asyncio
import io
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from textwrap import dedent

from typing import cast

import anthropic
import openai
import requests as _http
from anthropic.types import TextBlock

# Fix Windows encoding
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Optional web search -- works out of the box if web_search.py is present
try:
    from web_search import (  # type: ignore[import]
        SEARCH_INSTRUCTION,
        SEARCH_TAG_RE,
        resolve_searches,
    )
    _WEB_SEARCH = True
except ImportError:
    _WEB_SEARCH = False
    SEARCH_INSTRUCTION = ""
    SEARCH_TAG_RE = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE)
    def resolve_searches(text: str) -> tuple:  # type: ignore[misc]
        return text, []

# ---- CONFIG ------------------------------------------------------------------

COUNCIL_MODELS = [
    {"id": "o3",               "name": "o3",               "provider": "openai"},
    {"id": "gemini-2.5-pro",   "name": "Gemini 2.5 Pro",   "provider": "gemini"},
    {"id": "grok-4",           "name": "Grok 4",           "provider": "xai"},
]

CHAIRMAN_MODEL = {"id": "claude-opus-4-6",          "name": "Claude Opus 4.6",  "provider": "anthropic"}
HAIKU_MODEL    = {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "provider": "anthropic"}

OUTPUT_DIR   = Path.home() / "council"
CHAIRMAN_LOG = Path.home() / "council" / "chairman-log.md"
MAX_TOKENS   = 4096   # visible output tokens per model call
MODEL_TIMEOUT = 120   # seconds per model call

# o3 bundles reasoning + output tokens in one budget; give it extra headroom
# so reasoning doesn't consume the entire limit, leaving nothing for the response.
O3_MAX_COMPLETION_TOKENS = 16384
GEMINI_MAX_OUTPUT_TOKENS = 8192   # 2.5 Pro is a thinking model; 4096 triggers MAX_TOKENS cutoff

COST_PER_1M = {
    "openai":    {"input": 10.00,  "output": 40.00},   # o3
    "gemini":    {"input":  1.25,  "output": 10.00},   # gemini-2.5-pro <200K ctx
    "anthropic": {"input": 15.00,  "output": 75.00},   # claude-opus-4-6 (chairman)
    "xai":       {"input":  5.00,  "output": 15.00},   # grok-4
}

_ANTI_SYCOPHANCY = (
    "IMPORTANT: You are an independent evaluator. Give your HONEST assessment.\n"
    "- Do NOT change your position just because others disagree.\n"
    "- Only update if the LOGIC is undeniable and genuinely compelling.\n"
    "- Genuine disagreement is MORE VALUABLE than false consensus.\n"
    "- If you think something is wrong, SAY so, even if every other model disagrees.\n\n"
)

_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
}

# ---- API CALLERS -------------------------------------------------------------

_SAFE_LOCAL_ERRORS = {
    "OPENAI_API_KEY not set",
    "GEMINI_API_KEY not set",
    "ANTHROPIC_API_KEY not set",
    "XAI_API_KEY not set",
}


def _safe_error(exc: BaseException) -> str:
    """Return an actionable error label without persisting provider details."""
    message = str(exc)
    return message if message in _SAFE_LOCAL_ERRORS else type(exc).__name__


def _call_openai(model_id: str, text: str) -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=key)
    # o3 and o-series models use max_completion_tokens; gpt-* use max_tokens.
    # O3_MAX_COMPLETION_TOKENS is larger than MAX_TOKENS because o3 bundles
    # reasoning tokens + visible output in the same budget.
    _is_o_series = model_id.startswith("o") and not model_id.startswith("gpt")
    kwargs: dict = (
        {"max_completion_tokens": O3_MAX_COMPLETION_TOKENS}
        if _is_o_series
        else {"max_tokens": MAX_TOKENS}
    )
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": text}],  # type: ignore[arg-type]
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def _call_gemini(model_id: str, text: str) -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    resp = _http.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
        headers={"x-goog-api-key": key},
        json={
            "contents": [{"parts": [{"text": text}], "role": "user"}],
            "generationConfig": {"maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS},
        },
        timeout=MODEL_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        candidate = data["candidates"][0]
        # Gemini 2.5 Pro may return a candidate with no 'content' on MAX_TOKENS finish
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            finish = candidate.get("finishReason", "UNKNOWN")
            raise RuntimeError(f"Gemini returned no content parts (finishReason={finish})")
        return parts[0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError("Unexpected Gemini response shape") from exc


def _call_anthropic(model_id: str, text: str, max_tokens: int = MAX_TOKENS) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": text}],  # type: ignore[arg-type]
    )
    block = resp.content[0]
    if isinstance(block, TextBlock):
        return block.text
    raise RuntimeError(f"Unexpected content type: {type(block)}")


def _call_xai(model_id: str, text: str) -> str:
    key = os.environ.get("XAI_API_KEY", "")
    if not key:
        raise RuntimeError("XAI_API_KEY not set")
    client = openai.OpenAI(api_key=key, base_url="https://api.x.ai/v1")
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": text}],  # type: ignore[arg-type]
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""


def call_model(model: dict, text: str) -> str:
    provider = model["provider"]
    if provider == "openai":
        return _call_openai(model["id"], text)
    if provider == "gemini":
        return _call_gemini(model["id"], text)
    if provider == "anthropic":
        return _call_anthropic(model["id"], text)
    if provider == "xai":
        return _call_xai(model["id"], text)
    raise ValueError(f"Unknown provider: {provider}")


def _active_models() -> list[dict]:
    """Return COUNCIL_MODELS filtered to those with API keys present."""
    active = [m for m in COUNCIL_MODELS if os.environ.get(_KEY_VARS.get(m["provider"], ""), "")]
    skipped = [m["name"] for m in COUNCIL_MODELS if m not in active]
    if skipped:
        print(f"[Info] Skipping (no key): {', '.join(skipped)}")
    if len(active) < 2:
        print("[Error] Need at least 2 models. Set OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY.")
        sys.exit(1)
    return active


# ---- COST TRACKING -----------------------------------------------------------

def _estimate_cost(provider: str, prompt: str, response: str) -> float:
    rates = COST_PER_1M.get(provider, COST_PER_1M["anthropic"])
    return (len(prompt) // 4 * rates["input"] + len(response) // 4 * rates["output"]) / 1_000_000


def _print_costs(costs: list[tuple[str, float]]) -> float:
    total = sum(c for _, c in costs)
    print("\n[Cost Estimate]")
    for name, cost in costs:
        print(f"  {name}: ${cost:.5f}")
    print(f"  TOTAL: ${total:.5f}")
    return total


# ---- WEB SEARCH RESOLUTION ---------------------------------------------------

def _resolve_web_searches(model_name: str, response: str, model: dict) -> str:
    """Resolve [SEARCH: ...] tags in response if web_search.py is available."""
    if not _WEB_SEARCH or not SEARCH_TAG_RE.search(response):
        return response
    print(f"  [{model_name}] Resolving web searches...")
    enriched, _ = resolve_searches(response)
    follow_up = (
        f"{response}\n\n=== SEARCH RESULTS ===\n{enriched}\n\n"
        "Now provide your final answer, citing the sources found above."
    )
    return call_model(model, follow_up)


# ---- MODE ROUTING ------------------------------------------------------------

def route_mode(query: str) -> str:
    """Use Haiku to classify query: 'simple' for structured, 'debate' for open-ended."""
    print("\n[Mode Router] Classifying with Haiku...")
    prompt = (
        "Classify this question as STRUCTURED or OPEN_ENDED.\n\n"
        "STRUCTURED: Has clear proposals to evaluate, asks YES/NO, compares specific options.\n"
        "OPEN_ENDED: Asks for design, strategy, architecture, or complex analysis.\n\n"
        "Respond with EXACTLY one word: STRUCTURED or OPEN_ENDED\n\n"
        f"Question: {query}"
    )
    try:
        result = _call_anthropic(HAIKU_MODEL["id"], prompt, max_tokens=10).strip().upper()
        mode = "simple" if "STRUCTURED" in result else "debate"
        print(f"[Mode Router] Selected: {mode}")
        return mode
    except Exception as e:
        print(f"[Mode Router] Error ({_safe_error(e)}), defaulting to 'simple'")
        return "simple"


# ---- STAGE 0 -- Debate pre-rounds (cross-talk) -------------------------------

async def stage0_debate(query: str, models: list[dict], rounds: int = 2) -> list[dict]:
    """Cross-talk debate rounds. Final round responses feed directly into Stage 2.

    Round 1: blind independent query with anti-sycophancy.
    Round 2+: each model sees others' prior responses and may update its position.
    Anti-sycophancy directive prevents capitulation under social pressure.
    """
    base_query = f"{SEARCH_INSTRUCTION}\n\n{query}" if _WEB_SEARCH else query
    prev_results: list[dict] = []

    for round_num in range(rounds):
        print(f"\n[Debate Round {round_num + 1}/{rounds}] Querying in parallel...")
        loop = asyncio.get_event_loop()

        def _call_one(args: tuple) -> dict:
            model, idx = args
            t0 = time.monotonic()
            if round_num == 0 or not prev_results:
                prompt = f"{_ANTI_SYCOPHANCY}{base_query}"
            else:
                others = "\n\n".join(
                    f"Model {chr(65 + j)} argued:\n{r['response']}"
                    for j, r in enumerate(prev_results)
                    if j != idx
                )
                prompt = (
                    f"{_ANTI_SYCOPHANCY}"
                    f"Original question: {query}\n\n"
                    f"Other models' Round {round_num} positions:\n{others}\n\n"
                    "Give your updated response. State where you agree or disagree and why. "
                    "You are NOT required to reach consensus."
                )
            try:
                response = call_model(model, prompt)
                response = _resolve_web_searches(model["name"], response, model)
            except Exception as e:
                response = f"[ERROR: {_safe_error(e)}]"
            return {"model": model, "response": response, "_prompt": prompt, "_elapsed": time.monotonic() - t0}

        tasks = [loop.run_in_executor(None, _call_one, (m, i)) for i, m in enumerate(models)]
        prev_results = list(await asyncio.gather(*tasks))
        for r in prev_results:
            print(f"  [OK] {r['model']['name']} ({r['_elapsed']:.1f}s)")

    return prev_results


# ---- STAGE 1 -- Parallel independent query -----------------------------------

async def stage1_query(query: str, models: list[dict]) -> list[dict]:
    """Independent parallel query with anti-sycophancy. No system prompt (Karpathy)."""
    print("\n[Stage 1] Querying council members in parallel...")
    prefix = f"{SEARCH_INSTRUCTION}\n\n{_ANTI_SYCOPHANCY}" if _WEB_SEARCH else _ANTI_SYCOPHANCY
    loop = asyncio.get_event_loop()

    def _call_one(model: dict) -> dict:
        t0 = time.monotonic()
        prompt = f"{prefix}{query}"
        try:
            response = call_model(model, prompt)
            response = _resolve_web_searches(model["name"], response, model)
        except Exception as e:
            response = f"[ERROR: {_safe_error(e)}]"
        return {"model": model, "response": response, "_prompt": prompt, "_elapsed": time.monotonic() - t0}

    raw = await asyncio.gather(*[loop.run_in_executor(None, _call_one, m) for m in models],
                               return_exceptions=True)
    results: list[dict] = []
    for model, item in zip(models, raw):
        if isinstance(item, Exception):
            error = _safe_error(item)
            print(f"  [ERROR] {model['name']}: {error}")
            results.append({"model": model, "response": f"[ERROR: {error}]", "_prompt": "", "_elapsed": 0.0})
        else:
            d = cast(dict, item)
            print(f"  [OK]    {d['model']['name']} ({d['_elapsed']:.1f}s, {len(d['response'])} chars)")
            results.append(d)
    return results


# ---- STAGE 2 -- Anonymous peer review with label rotation --------------------

async def stage2_peer_review(query: str, stage1_results: list[dict]) -> list[dict]:
    """Karpathy peer review with blind label rotation to prevent position bias.

    Each reviewer sees responses in a rotated order so 'Response A' is a
    different model's work for each reviewer -- eliminating the tendency to
    consistently favor the first-listed response.
    """
    print("\n[Stage 2] Peer review (anonymous ranking, label rotation)...")
    n = len(stage1_results)

    def _build_prompt(reviewer_idx: int) -> tuple[str, dict[str, str]]:
        rotated = [stage1_results[(reviewer_idx + i) % n] for i in range(n)]
        labels = [chr(65 + i) for i in range(n)]
        label_map = {
            f"Response {label}": r["model"]["name"]
            for label, r in zip(labels, rotated)
        }
        responses_block = "\n\n".join(
            f"Response {label}:\n{r['response']}" for label, r in zip(labels, rotated)
        )
        prompt = dedent(f"""
            You are evaluating different responses to the following question:

            Question: {query}

            Here are the responses from different models (anonymized):

            {responses_block}

            Your task:
            1. Evaluate each response individually -- what it does well, what it misses.
            2. At the very end, provide a final ranking.

            IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
            - Start with the line "FINAL RANKING:" (all caps, with colon)
            - List responses from best to worst as a numbered list
            - Each line: number, period, space, ONLY the label (e.g., "1. Response A")
            - No other text in the ranking section

            Example:
            Response A covers X well but misses Y...
            Response B is accurate but shallow on Z...

            FINAL RANKING:
            1. Response C
            2. Response A
            3. Response B

            Now provide your evaluation and ranking:
        """).strip()
        return prompt, label_map

    loop = asyncio.get_event_loop()

    def _call_one(args: tuple) -> dict:
        model, idx = args
        t0 = time.monotonic()
        prompt, label_map = _build_prompt(idx)
        try:
            review = call_model(model, prompt)
        except Exception as e:
            review = f"[ERROR: {_safe_error(e)}]"
        return {
            "model": model,
            "review": review,
            "label_map": label_map,
            "_elapsed": time.monotonic() - t0,
        }

    raw = await asyncio.gather(
        *[loop.run_in_executor(None, _call_one, (r["model"], i)) for i, r in enumerate(stage1_results)],
        return_exceptions=True,
    )
    results: list[dict] = []
    for r, item in zip(stage1_results, raw):
        model = r["model"]
        if isinstance(item, Exception):
            error = _safe_error(item)
            print(f"  [ERROR] {model['name']}: {error}")
            results.append({"model": model, "review": f"[ERROR: {error}]", "label_map": {}})
        else:
            d = cast(dict, item)
            print(f"  [OK]    {d['model']['name']} reviewed ({d['_elapsed']:.1f}s)")
            results.append(d)
    return results


# ---- STAGE 3 -- Chairman synthesis -------------------------------------------

def stage3_chairman(query: str, stage1: list[dict], stage2: list[dict]) -> str:
    """Chairman sees all responses + rankings de-anonymized and synthesizes."""
    print(f"\n[Stage 3] Chairman ({CHAIRMAN_MODEL['name']}) synthesizing...")
    stage1_block = "\n\n".join(
        f"Model: {r['model']['name']}\nResponse:\n{r['response']}" for r in stage1
    )
    stage2_block = "\n\n".join(
        f"Model: {r['model']['name']}\n"
        f"Label map: {', '.join(f'{label}={name}' for label, name in r.get('label_map', {}).items())}\n"
        f"Ranking/Review:\n{r['review']}"
        for r in stage2
    )
    chairman_prompt = dedent(f"""
        You are the Chairman of an LLM Council. Multiple AI models provided responses
        to a user's question, then ranked each other's responses anonymously.

        Original Question: {query}

        STAGE 1 - Individual Responses:
        {stage1_block}

        STAGE 2 - Peer Rankings:
        {stage2_block}

        Your task as Chairman is to synthesize this into a single, comprehensive, accurate
        answer. Consider:
        - The individual responses and their insights
        - The peer rankings and what they reveal about response quality
        - Patterns of agreement and disagreement
        - Where models disagreed, explain why and which position is better supported

        Provide a clear, well-reasoned final answer that represents the council's collective wisdom:
    """).strip()

    t0 = time.monotonic()
    result = call_model(CHAIRMAN_MODEL, chairman_prompt)
    print(f"  [OK]    Chairman ({len(result)} chars, {time.monotonic() - t0:.1f}s)")
    return result


# ---- OUTPUT ------------------------------------------------------------------

def save_output(
    query: str,
    mode: str,
    stage1: list[dict],
    stage2: list[dict],
    stage3: str,
    costs: list[tuple[str, float]],
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H%M")
    slug = "".join(c if c.isalnum() else "-" for c in query[:50].lower())[:40]
    out_path = OUTPUT_DIR / f"{ts}-{slug}.md"

    sections = [
        f"# Council: {query[:80]}\n",
        f"*{datetime.now().strftime('%Y-%m-%d %H:%M')} | mode: {mode}*\n",
        f"**Question:** {query}\n",
        "---\n## Stage 1 -- Individual Responses\n",
    ]
    for r in stage1:
        sections.append(f"### {r['model']['name']}\n\n{r['response']}\n")
    sections.append("---\n## Stage 2 -- Peer Reviews\n")
    for r in stage2:
        sections.append(f"### {r['model']['name']}\n\n{r['review']}\n")
    sections.append("---\n## Stage 3 -- Chairman Synthesis\n\n")
    sections.append(stage3)
    if costs:
        total = sum(c for _, c in costs)
        cost_lines = "\n".join(f"  - {n}: ${c:.5f}" for n, c in costs)
        sections.append(f"\n\n---\n## Cost\n\n{cost_lines}\n  - **TOTAL: ${total:.5f}**\n")

    out_path.write_text("\n".join(sections), encoding="utf-8")

    ts_full = datetime.now().strftime("%Y-%m-%d %H:%M")
    with CHAIRMAN_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n---\n## {ts_full} -- {query[:80]}\n\n{stage3}\n")

    return out_path


# ---- CHECKPOINT SAVE ---------------------------------------------------------

def _save_partial(query: str, mode: str, s1: list[dict], s2: list[dict]) -> None:
    """Write Stage 1+2 results to a checkpoint file before the chairman call."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H%M")
    slug = "".join(c if c.isalnum() else "-" for c in query[:50].lower())[:40]
    partial_path = OUTPUT_DIR / f"{ts}-{slug}-partial.md"
    lines = [
        f"# Council (partial): {query[:80]}\n",
        f"*{datetime.now().strftime('%Y-%m-%d %H:%M')} | mode: {mode} | stage: 1+2 saved*\n",
        "---\n## Stage 1 -- Individual Responses\n",
    ]
    for r in s1:
        lines.append(f"### {r['model']['name']}\n\n{r['response']}\n")
    lines.append("---\n## Stage 2 -- Peer Reviews\n")
    for r in s2:
        lines.append(f"### {r['model']['name']}\n\n{r['review']}\n")
    partial_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [Checkpoint] Stage 1+2 saved → {partial_path.name}")


# ---- RUNNER ------------------------------------------------------------------

async def run_council(query: str, mode: str = "auto") -> str:
    models = _active_models()

    if mode == "auto":
        mode = route_mode(query)

    # Stage 0 (debate cross-talk) or Stage 1 (independent)
    if mode == "debate":
        s1 = await stage0_debate(query, models, rounds=2)
    else:
        s1 = await stage1_query(query, models)

    s2 = await stage2_peer_review(query, s1)

    # Checkpoint: save Stage 1+2 before calling chairman so a credit/network
    # failure at Stage 3 doesn't lose the panelist work.
    _save_partial(query, mode, s1, s2)

    s3 = stage3_chairman(query, s1, s2)

    costs = [
        (r["model"]["name"], _estimate_cost(r["model"]["provider"], r.get("_prompt", ""), r["response"]))
        for r in s1
    ]
    costs.append((CHAIRMAN_MODEL["name"], _estimate_cost(CHAIRMAN_MODEL["provider"], query, s3)))
    _print_costs(costs)

    out_path = save_output(query, mode, s1, s2, s3, costs)
    print(f"\n[Saved] {out_path}")
    print("\n" + "=" * 60)
    print("CHAIRMAN SYNTHESIS")
    print("=" * 60)
    print(s3)
    return s3


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    mode = "auto"
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 >= len(args):
            print("Error: --mode requires a value (auto, simple, debate)")
            sys.exit(1)
        mode = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if args and args[0] == "--file" and len(args) >= 2:
        query = Path(args[1]).read_text(encoding="utf-8").strip()
    else:
        query = " ".join(args)
        if not sys.stdin.isatty():
            extra = sys.stdin.read().strip()
            if extra:
                query = f"{query}\n\n{extra}"

    if not query:
        print("Error: empty query")
        sys.exit(1)

    asyncio.run(run_council(query, mode=mode))


if __name__ == "__main__":
    main()
