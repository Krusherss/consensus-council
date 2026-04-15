"""
Tavily Web Search module for Hydra Council.

Models can trigger real-time web searches by including [SEARCH: query] in their responses.
The council intercepts these tags, runs Tavily, injects results, then re-queries the model
for a final grounded response.

Usage in prompts:
    If you need current information, write [SEARCH: your query] and results will be provided.

Example:
    [SEARCH: 2026 NCAA tournament bracket first round matchups]
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

TAVILY_TIMEOUT = 30
SEARCH_TAG_RE = re.compile(r'\[SEARCH:\s*(.+?)\]', re.IGNORECASE)
MAX_SEARCHES_PER_RESPONSE = 5  # cap to avoid runaway costs


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    """Load Tavily API key from env or local .env file."""
    key = os.environ.get("TAVILY_API_KEY", "")
    if key:
        return key
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("TAVILY_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return ""


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def search(query: str, max_results: int = 5) -> str:
    """
    Run a Tavily web search and return formatted results as a plain string
    ready to inject into a model prompt.

    Returns an error string (never raises) so the council can continue even
    if search fails.
    """
    api_key = _load_api_key()
    if not api_key:
        return "[SEARCH FAILED: No TAVILY_API_KEY found in environment or .env]"

    try:
        from tavily import TavilyClient
    except ImportError:
        return "[SEARCH FAILED: tavily-python not installed — run: pip install tavily-python]"

    client = TavilyClient(api_key=api_key)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client.search,
            query=query,
            search_depth="basic",
            max_results=max_results,
        )
        try:
            response = future.result(timeout=TAVILY_TIMEOUT)
        except FuturesTimeout:
            return f"[SEARCH FAILED: Tavily timed out after {TAVILY_TIMEOUT}s for: {query!r}]"
        except Exception as e:
            return f"[SEARCH FAILED: {e}]"

    results = response.get("results", [])
    if not results:
        return f"[SEARCH: No results found for: {query!r}]"

    lines = [f"=== WEB SEARCH RESULTS: {query!r} ==="]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = (r.get("content", "") or "")[:400].strip()
        lines.append(f"\n[{i}] {title}\n    URL: {url}\n    {content}")
    lines.append("=== END SEARCH RESULTS ===")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tag extraction and resolution
# ---------------------------------------------------------------------------

def extract_search_queries(text: str) -> list[str]:
    """Return all unique [SEARCH: query] values found in text, preserving order."""
    seen: set[str] = set()
    queries: list[str] = []
    for q in SEARCH_TAG_RE.findall(text):
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
    return queries[:MAX_SEARCHES_PER_RESPONSE]


def has_search_tags(text: str) -> bool:
    """Return True if the text contains any [SEARCH: ...] tags."""
    return bool(SEARCH_TAG_RE.search(text))


def resolve_searches(text: str) -> tuple[str, list[dict]]:
    """
    Find all [SEARCH: query] tags in text, execute Tavily searches in parallel,
    replace each tag with its results inline.

    Returns:
        (modified_text, search_log)
        where search_log is a list of {query, results} dicts for session logging.
    """
    queries = extract_search_queries(text)
    if not queries:
        return text, []

    # Run all searches in parallel
    search_results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(len(queries), 3)) as executor:
        futures = {executor.submit(search, q): q for q in queries}
        for future, query in futures.items():
            try:
                search_results[query] = future.result(timeout=TAVILY_TIMEOUT + 5)
            except Exception as e:
                search_results[query] = f"[SEARCH FAILED: {e}]"

    search_log = []
    modified = text
    for query in queries:
        result_text = search_results.get(query, "[SEARCH: no result]")
        search_log.append({"query": query, "results": result_text})
        # Replace [SEARCH: query] tag with the actual results
        modified = SEARCH_TAG_RE.sub(
            lambda m, q=query, r=result_text: r if m.group(1).strip() == q else m.group(0),
            modified,
        )

    return modified, search_log


# ---------------------------------------------------------------------------
# Prompt instruction (prepended to all council prompts when search is enabled)
# ---------------------------------------------------------------------------

SEARCH_INSTRUCTION = """\
REAL-TIME WEB SEARCH AVAILABLE:
You have access to live web search. To search the internet, include this tag anywhere in your response:
    [SEARCH: your search query here]

You can use multiple searches. Results will be fetched and returned to you before your final answer.
Use this for: current events, recent data, live statistics, up-to-date information.
Example: [SEARCH: 2026 NCAA tournament bracket full results]
"""
