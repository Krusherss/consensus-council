"""
web_search.py — Free, unlimited web search for Hydra Council (and any project).

No API keys. No accounts. No rate limits for normal use.

Stack:
    duckduckgo-search  — fetch search result URLs + snippets
    trafilatura        — fetch full page content, strip boilerplate

Drop-in replacement for tavily_search.py.
Copy this file to any project and it works immediately.

Usage:
    from web_search import search, has_search_tags, resolve_searches, SEARCH_INSTRUCTION

Models trigger searches by writing [SEARCH: query] in their response.
The council resolves all tags before the model gives its final answer.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FETCH_TIMEOUT = 15        # seconds per page fetch
SEARCH_TIMEOUT = 20       # seconds for DDG query
MAX_SEARCHES_PER_RESPONSE = 5
MAX_RESULTS_PER_QUERY = 5
MAX_CONTENT_CHARS = 1200  # per page — enough context without flooding the prompt

SEARCH_TAG_RE = re.compile(r'\[SEARCH:\s*(.+?)\]', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Core search + fetch
# ---------------------------------------------------------------------------

def _ddg_search(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[dict]:
    """
    Query DuckDuckGo and return list of {title, url, snippet} dicts.
    Returns [] on any failure.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": (r.get("body", "") or "")[:400],
            }
            for r in results
        ]
    except Exception as e:
        print(f"  [web_search] DDG error: {e}")
        return []


def _fetch_page(url: str) -> Optional[str]:
    """
    Fetch a URL and extract clean main content using trafilatura.
    Returns None if fetch fails or content is empty.
    """
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return text
    except Exception:
        return None


def _fetch_pages_parallel(urls: list[str], timeout: int = FETCH_TIMEOUT) -> dict[str, Optional[str]]:
    """Fetch multiple pages in parallel. Returns {url: content_or_None}."""
    results: dict[str, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=min(len(urls), 3)) as executor:
        futures = {executor.submit(_fetch_page, url): url for url in urls}
        for future, url in futures.items():
            try:
                results[url] = future.result(timeout=timeout)
            except FuturesTimeout:
                results[url] = None
            except Exception:
                results[url] = None
    return results


# ---------------------------------------------------------------------------
# Public search function
# ---------------------------------------------------------------------------

def search(query: str, max_results: int = MAX_RESULTS_PER_QUERY,
           fetch_content: bool = True) -> str:
    """
    Search the web for query and return formatted results as a plain string
    ready to inject into a model prompt.

    Args:
        query:         Search query string
        max_results:   Number of results to retrieve (default 5)
        fetch_content: If True, fetch full page content via trafilatura.
                       If False, return DDG snippets only (faster).

    Returns:
        Formatted string with results. Never raises — returns error string on failure.
    """
    print(f"  [web_search] Searching: {query!r}")
    t0 = time.monotonic()

    # Step 1: DDG search
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_ddg_search, query, max_results)
            ddg_results = future.result(timeout=SEARCH_TIMEOUT)
    except FuturesTimeout:
        return f"[SEARCH FAILED: DuckDuckGo timed out after {SEARCH_TIMEOUT}s for: {query!r}]"
    except Exception as e:
        return f"[SEARCH FAILED: {e}]"

    if not ddg_results:
        return f"[SEARCH: No results found for: {query!r}]"

    # Step 2: Fetch full page content (parallel)
    if fetch_content:
        urls = [r["url"] for r in ddg_results if r["url"]]
        page_contents = _fetch_pages_parallel(urls)
    else:
        page_contents = {}

    elapsed = time.monotonic() - t0
    print(f"  [web_search] {len(ddg_results)} results in {elapsed:.1f}s")

    # Step 3: Format output
    lines = [f"=== WEB SEARCH RESULTS: {query!r} ==="]
    for i, r in enumerate(ddg_results, 1):
        title = r["title"]
        url = r["url"]
        snippet = r["snippet"]

        # Use full page content if available, else fall back to snippet
        full_text = page_contents.get(url) if fetch_content else None
        if full_text:
            content = full_text[:MAX_CONTENT_CHARS].strip()
            source_label = "full text"
        else:
            content = snippet
            source_label = "snippet"

        lines.append(f"\n[{i}] {title} ({source_label})")
        lines.append(f"    URL: {url}")
        if content:
            lines.append(f"    {content}")

    lines.append("=== END SEARCH RESULTS ===")
    raw = "\n".join(lines)
    # Strip characters that cause Windows cp1252 encoding errors
    return raw.encode("ascii", errors="replace").decode("ascii")


# ---------------------------------------------------------------------------
# Tag parsing and resolution
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
    Find all [SEARCH: query] tags in text, run searches in parallel,
    replace each tag with its results inline.

    Returns:
        (modified_text, search_log)
        search_log is a list of {query, results} dicts for session saving.
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
                search_results[query] = future.result(timeout=SEARCH_TIMEOUT + FETCH_TIMEOUT + 10)
            except Exception as e:
                search_results[query] = f"[SEARCH FAILED: {e}]"

    search_log = []
    modified = text
    for query in queries:
        result_text = search_results.get(query, "[SEARCH: no result]")
        search_log.append({"query": query, "results": result_text})
        # Replace the [SEARCH: query] tag with the actual results
        pattern = re.compile(r'\[SEARCH:\s*' + re.escape(query) + r'\]', re.IGNORECASE)
        modified = pattern.sub(f"\n{result_text}\n", modified, count=1)

    return modified, search_log


# ---------------------------------------------------------------------------
# Prompt instruction injected into council prompts when search is enabled
# ---------------------------------------------------------------------------

SEARCH_INSTRUCTION = """\
REAL-TIME WEB SEARCH AVAILABLE — CITATIONS REQUIRED:
You have access to live web search via DuckDuckGo (full page content extraction).
To search the internet, include this tag anywhere in your response:
    [SEARCH: your search query here]

You may use up to 5 searches. Results will be fetched and returned before your final answer.

MANDATORY CITATION RULES:
- Every factual claim MUST be backed by a source URL or a [SEARCH: query] to find one.
- Format citations inline: "Duke finished 32-2 [source: apnews.com/...]"
- If you cannot find a source for a claim, explicitly state: "[UNSUPPORTED - lower confidence]"
- Assertions without sources will be treated as less trustworthy by the other models.
- Do NOT make up citations. Only cite URLs that appeared in actual search results.

Examples:
    [SEARCH: 2026 NCAA tournament bracket full results]
    [SEARCH: Duke basketball 2026 season record injuries]
    [SEARCH: KRAS G12C inhibitor resistance latest clinical trials 2026]
"""


# ---------------------------------------------------------------------------
# Quick test (run directly: python web_search.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== web_search.py self-test ===\n")
    result = search("2026 NCAA March Madness tournament bracket Duke", max_results=3)
    print(result)
    print(f"\nTag detection: {has_search_tags('Check [SEARCH: test query] here')}")
    print(f"Query extract: {extract_search_queries('Check [SEARCH: test query] here')}")
