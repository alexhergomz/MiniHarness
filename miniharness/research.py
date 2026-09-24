"""Research: a bounded sub-loop that writes a markdown file.

Promethean's rabbit-hole mode was ~1,600 lines: a workspace store, a
sub-question tree, BM25-ranked synthesis, contradiction detection, a live event
feed, and a sandboxed background agent with slot save/restore.

This is the same capability at a tenth the size. The data model is a markdown
file. The file *is* the workspace, *is* the resume state, and *is* the report —
a document of claims with citations is what the BM25 reconstruction was trying
to produce anyway. You can open it in an editor while the run is still going.

What was lost: contradiction detection, ranked retrieval across findings, the
live event feed. For a research feature driving a 9B model, that trade is
obviously right.

Not a top-level tool: it runs as ``/research`` so its schema never costs tokens
on ordinary coding turns, and so WebSearch stays out of the main tool set where
it would only tempt the model to search instead of reading the code.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
from pathlib import Path

from . import net as requests

from .config import HOME
from .provider import AssistantTurn, TextChunk, stream

WORKSPACES = HOME / "research"

SYSTEM = """You are a research agent. Answer the question by searching the web \
and reading sources.

Method: break the question into sub-questions. For each one, search, fetch the \
most promising sources, and record what you found with save_finding. Every \
finding needs a URL. When you have covered the question, call finish.

Record findings as you go — do not batch them at the end. Prefer primary \
sources (papers, docs, source code) over summaries. If sources disagree, save \
both and say so in the claim."""


def _ddg_search(query: str, max_results: int = 8) -> str:
    """Search via DuckDuckGo's HTML endpoint. No API key, no dependency."""
    try:
        r = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; miniharness/0.1)"},
            timeout=20,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        return f"Search failed: {e}"

    out = []
    for m in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r.text, re.S
    ):
        url, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2))
        # DDG wraps results in a redirect; unwrap to the real URL.
        if "uddg=" in url:
            url = urllib.parse.unquote(url.split("uddg=")[1].split("&")[0])
        out.append(f"{html.unescape(title).strip()}\n  {url}")
        if len(out) >= max_results:
            break
    return "\n".join(out) if out else "No results."


class Workspace:
    """The markdown file. That's the whole data model."""

    def __init__(self, question: str, slug: str):
        WORKSPACES.mkdir(parents=True, exist_ok=True)
        self.path = WORKSPACES / f"{slug}.md"
        self.question = question
        self.n_findings = 0
        if not self.path.exists():
            self.path.write_text(f"# {question}\n\n", encoding="utf-8")

    def append(self, text: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text)

    def read(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def save_finding(self, claim: str, url: str, sub_question: str = "") -> str:
        body = self.read()
        heading = f"## {sub_question}\n" if sub_question else "## Findings\n"
        if heading not in body:
            self.append("\n" + heading)
        self.append(f"- {claim.strip()} — [source]({url})\n")
        self.n_findings += 1
        return "Saved."


SCHEMAS = [
    {
        "name": "web_search",
        "description": "Search the web. Returns titles and URLs.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "WebFetch",
        "description": "Fetch a URL and return its text with HTML stripped.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "save_finding",
        "description": "Record one claim with the URL that supports it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "url": {"type": "string"},
                "sub_question": {
                    "type": "string",
                    "description": "Which part of the question this answers",
                },
            },
            "required": ["claim", "url"],
        },
    },
    {
        "name": "finish",
        "description": "Call when the question is covered.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]


def slugify(question: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")
    return s[:60] or "research"


def run(question: str, config: dict, on_event=None, resume: bool = False):
    """Run the sub-loop until it finishes or hits the call cap.

    Yields nothing; calls ``on_event(kind, text)`` for UI. Returns the path.
    """
    from .tools import _webfetch

    ws = Workspace(question, slugify(question))
    emit = on_event or (lambda kind, text: None)

    prompt = question
    if resume and (prior := ws.read()).strip():
        prompt = (f"{question}\n\nYou already gathered the following. "
                  f"Continue from here; do not repeat findings.\n\n{prior[:8000]}")

    messages = [{"role": "user", "content": prompt}]
    cap = int(config.get("research_max_calls", 40))
    empty_retries = 0
    n_searches = 0

    for _ in range(cap):
        turn = None
        for event in stream(config["model"], SYSTEM, messages, SCHEMAS, config):
            if isinstance(event, TextChunk):
                emit("text", event.text)
            elif isinstance(event, AssistantTurn):
                # Only the turn itself. Reasoning and tool-draft progress also
                # arrive here, and were saved as "the turn" until the real one
                # overwrote them.
                turn = event
        if turn is None:
            break

        messages.append(turn.to_message())
        calls = [tc for tc in turn.tool_calls if "_raw" not in tc.get("input", {})]
        if not calls:
            # Same early-stopping failure as the main loop (DESIGN.md §5.3c):
            # an empty turn silently ended the run with an empty report. A
            # research run is not finished until it says finish.
            if empty_retries < 2:
                empty_retries += 1
                messages[-1]["content"] = messages[-1].get("content") or "[no output]"
                messages.append({"role": "user", "content": (
                    "You produced no tool call. Continue: search, fetch a source, "
                    "call save_finding for what you have learned, or call finish.")})
                emit("tool", "(empty response — prompting to continue)")
                continue
            break

        done = False
        for tc in calls:
            name, p = tc["name"], tc["input"]
            emit("tool", f"{name} {json.dumps(p)[:120]}")
            if name == "web_search":
                result = _ddg_search(p.get("query", ""))
                n_searches += 1
                # Observed: the model re-queries five times in a row, refining
                # wording, never extracting anything — and the run ends with an
                # empty report. Escalate at the point of failure rather than
                # hoping the system prompt covers it.
                if ws.n_findings == 0 and n_searches >= 2:
                    result += (
                        f"\n\n[you have searched {n_searches} times and recorded "
                        f"nothing. Stop searching. Either call save_finding with "
                        f"what you already know from these results, or open one of "
                        f"these URLs with WebFetch and record what it says.]")
            elif name == "WebFetch":
                result = _webfetch({"url": p.get("url", "")}, config, None)[:8000]
                # Models reliably fetch and then move on without recording
                # anything, leaving an empty report. Remind them at the exact
                # moment the information is in front of them.
                if ws.n_findings == 0:
                    result += ("\n\n[reminder: call save_finding with each useful "
                               "claim and its URL before fetching anything else]")
            elif name == "save_finding":
                result = ws.save_finding(p.get("claim", ""), p.get("url", ""),
                                         p.get("sub_question", ""))
            elif name == "finish":
                ws.append(f"\n## Summary\n{p.get('summary', '')}\n")
                result, done = "Finished.", True
            else:
                result = f"Unknown tool {name}"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        if done:
            break
    else:
        ws.append(f"\n_[stopped at the {cap}-call cap; resume to continue]_\n")

    if ws.n_findings == 0:
        # A report containing only its own title reads like a finished
        # document. Say plainly that nothing was recorded.
        ws.append("\n_[no findings were recorded — the model searched but never "
                  "called save_finding. Re-run or resume to retry.]_\n")
        emit("tool", "(finished with 0 findings)")

    return ws.path


def list_workspaces() -> list[Path]:
    if not WORKSPACES.exists():
        return []
    return sorted(WORKSPACES.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
