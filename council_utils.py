import sys
import io
import os
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

# Fix Windows encoding (guard against double-wrapping on re-import)
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value

_load_env()


# ---------------------------------------------------------------------------
# Citation instruction
# ---------------------------------------------------------------------------

CITATION_INSTRUCTION = (
    "When making factual claims, cite your sources with author, year, and "
    "publication where possible. If you are drawing on general knowledge "
    "without a specific source, state that explicitly."
)


def with_citation_instruction(prompt: str) -> str:
    """Prepend the citation instruction to any prompt."""
    return f"{CITATION_INSTRUCTION}\n\n{prompt}"


# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens)
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    'gemini': {'input': 1.25,  'output': 10.00},  # gemini-2.5-pro
    'claude': {'input': 15.00, 'output': 75.00},  # claude-opus-4-6
    'gpt':    {'input': 30.00, 'output': 120.00}, # gpt-5.4-pro (estimate)
    'haiku':  {'input': 0.80,  'output': 4.00},   # claude-haiku-4-5
}

MODEL_IDS: dict[str, str] = {
    'gemini': 'gemini-2.5-pro',
    'claude': 'claude-opus-4-6',
    'gpt':    'gpt-5.4-pro',
    'haiku':  'claude-haiku-4-5',
}


def estimate_cost(model: str, prompt: str, response: str) -> float:
    """Estimate the USD cost for a prompt/response pair (4 chars/token heuristic)."""
    rates = PRICING.get(model, PRICING['claude'])
    input_tokens = max(1, len(prompt) // 4)
    output_tokens = max(1, len(response) // 4)
    return (input_tokens * rates['input'] + output_tokens * rates['output']) / 1_000_000


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Thread-safe cost tracker with per-call console logging and JSONL persistence.

    Adapted from MHA Factory Revamped cost_tracker.py.
    """

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._calls: list[dict] = []
        self._session_start = datetime.now()
        self._log_path: Optional[Path] = None
        self._log_dir: Optional[Path] = Path(log_dir) if log_dir else None

    def log_call(self, model: str, prompt_text: str, response_text: str) -> float:
        """Log one API call, print a one-liner, persist to JSONL. Returns cost."""
        rates = PRICING.get(model, PRICING['claude'])
        input_tokens = max(1, len(prompt_text) // 4)
        output_tokens = max(1, len(response_text) // 4)
        cost = (input_tokens * rates['input'] + output_tokens * rates['output']) / 1_000_000

        record = {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'model_id': MODEL_IDS.get(model, model),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost_usd': round(cost, 6),
        }

        with self._lock:
            self._calls.append(record)
            session_total = sum(c['cost_usd'] for c in self._calls)

        model_id = MODEL_IDS.get(model, model)
        print(f"  [COST] {model} ({model_id}) | "
              f"{input_tokens:,} in / {output_tokens:,} out | "
              f"${cost:.4f} | Session: ${session_total:.4f}")

        # JSONL log
        if self._log_dir:
            if self._log_path is None:
                self._log_dir.mkdir(parents=True, exist_ok=True)
                ts = self._session_start.strftime('%Y-%m-%d_%H%M%S')
                self._log_path = self._log_dir / f'costs_{ts}.jsonl'
            try:
                with open(self._log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + '\n')
            except Exception:
                pass

        return cost

    @property
    def session_total(self) -> float:
        with self._lock:
            return sum(c['cost_usd'] for c in self._calls)

    def print_summary(self) -> None:
        """Print a formatted cost breakdown table."""
        with self._lock:
            calls = list(self._calls)

        if not calls:
            return

        by_model: dict[str, dict] = {}
        grand_total = 0.0
        for c in calls:
            m = c['model']
            if m not in by_model:
                by_model[m] = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0.0}
            by_model[m]['calls'] += 1
            by_model[m]['input_tokens'] += c['input_tokens']
            by_model[m]['output_tokens'] += c['output_tokens']
            by_model[m]['cost_usd'] += c['cost_usd']
            grand_total += c['cost_usd']

        print("\n" + "=" * 65)
        print("  COST SUMMARY")
        print(f"  Session started: {self._session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 65)
        print(f"\n  {'Model':<10} {'Calls':>6} {'Input Tok':>12} {'Output Tok':>12} {'Cost (USD)':>12}")
        print("  " + "-" * 54)
        for model, d in sorted(by_model.items(), key=lambda x: -x[1]['cost_usd']):
            print(f"  {model:<10} {d['calls']:>6} {d['input_tokens']:>12,} "
                  f"{d['output_tokens']:>12,} ${d['cost_usd']:>11.4f}")
        print("\n" + "=" * 65)
        print(f"  TOTAL CALLS:  {len(calls):>10,}")
        print(f"  GRAND TOTAL:  ${grand_total:>10.4f}")
        print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Model query functions
# ---------------------------------------------------------------------------

def query_gemini(prompt: str) -> str:
    try:
        import google.genai as genai
        client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY', ''))
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=genai.types.GenerateContentConfig(temperature=0.3),
        )
        return response.text or ""
    except Exception as e:
        return f"ERROR: {e}"


def query_claude(prompt: str) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY', ''))
        message = client.messages.create(
            model='claude-opus-4-20250514',
            max_tokens=2000,
            messages=[{'role': 'user', 'content': prompt}],
        )
        for block in message.content:
            text = getattr(block, 'text', None)
            if text is not None:
                return str(text)
        return ""
    except Exception as e:
        return f"ERROR: {e}"


def query_gpt_pro(prompt: str) -> str:
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))
        models_to_try = ['gpt-5.4-pro', 'gpt-5.4']
        last_error = None
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                )
                return response.choices[0].message.content or ""
            except openai.NotFoundError:
                last_error = f"model {model!r} not found"
                continue
            except Exception as e:
                raise
        return f"ERROR: {last_error}"
    except Exception as e:
        return f"ERROR: {e}"


def query_haiku(prompt: str) -> str:
    """Query Claude Haiku for lightweight routing/moderation tasks."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY', ''))
        message = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=500,
            messages=[{'role': 'user', 'content': prompt}],
        )
        for block in message.content:
            text = getattr(block, 'text', None)
            if text is not None:
                return str(text)
        return ""
    except Exception as e:
        return f"ERROR: {e}"
