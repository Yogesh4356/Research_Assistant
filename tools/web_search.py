from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    DuckDuckGo web search.
    Returns list of dicts: title, url, snippet
    """
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")
            })

    return results


def format_search_results(results: list[dict]) -> str:
    """
    Convert search results into a readable string.
    Provide context for the LLM.
    """
    if not results:
        return "No results found."

    formatted = ""
    for i, r in enumerate(results, 1):
        formatted += f"""
Result {i}:
Title: {r['title']}
URL: {r['url']}
Summary: {r['snippet']}
---"""

    return formatted.strip()


def search_and_format(query: str, max_results: int = 5) -> tuple[str, list[dict]]:
    """
    Search and format in a single call.
    Returns: formatted string + raw results
    """
    results = web_search(query, max_results=max_results)
    formatted = format_search_results(results)
    return formatted, results