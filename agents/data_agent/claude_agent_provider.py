"""Claude Agent SDK bridge for the "Generate & Test" handbook pipeline.

Mirrors ``WebUI_revamped/providers/claude_agent.py``'s async pattern
(``query()`` + ``ClaudeAgentOptions``), but exposes a synchronous, blocking
API: ``handbook_generator.py`` and ``DataRetrieverAgent.py`` run inside
Flask's sync request/worker threads, not an async framework, so callers here
just get a plain function that returns once the agent session finishes.

The SDK shells out to the ``claude`` CLI (a separate Node-based binary, not
just the pip package) -- ``available()``/``unavailable_reason()`` let callers
check both halves before committing to a Claude-backed run.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil


def package_available() -> bool:
    return importlib.util.find_spec("claude_agent_sdk") is not None


def cli_available() -> bool:
    """The Agent SDK shells out to the ``claude`` CLI -- a missing CLI fails
    inside the SDK's own subprocess launch with a much less actionable error,
    so check for it up front."""
    return shutil.which("claude") is not None


def available() -> bool:
    return package_available() and cli_available()


def unavailable_reason() -> str:
    """Empty string when the Claude Agent SDK is fully usable; otherwise a
    user-facing explanation of what's missing."""
    if not package_available():
        return "The claude-agent-sdk Python package is not installed on the server."
    if not cli_available():
        return ("The Claude Code CLI ('claude') is not on the server's PATH -- "
                "the Agent SDK shells out to it. Install it with "
                "'npm install -g @anthropic-ai/claude-code'.")
    return ""


def load_anthropic_key():
    """Return the caller's Anthropic API key.

    Inside a Flask request the key comes from the UI (``g.anthropic_api_key``,
    set from the ``X-Anthropic-Key`` header). Outside of a request
    (background worker threads, notebooks) it falls back to the
    ``ANTHROPIC_API_KEY`` env var / ``.env`` -- mirrors
    ``utils.agm_helper.load_OpenAI_key``.
    """
    try:
        from flask import g
        key = getattr(g, "anthropic_api_key", None)
        if key:
            return key
    except RuntimeError:
        pass  # Not in a Flask request context

    from pathlib import Path
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass
    return os.getenv("ANTHROPIC_API_KEY")


class AgentSessionResult:
    """Outcome of one ``run_agent`` call.

    ``text`` is the agent's final answer (its last assistant text, or the
    SDK's own summarized ``ResultMessage.result`` when that's richer).
    ``tool_events`` is a best-effort log of tool calls the agent made (each
    updated in place with its ``result``/``is_error`` once the matching
    ``ToolResultBlock`` arrives), for callers that want more than the
    formatted ``on_activity`` lines.
    """

    __slots__ = ("text", "success", "error", "tool_events", "usage")

    def __init__(self):
        self.text = ""
        self.success = True
        self.error = ""
        self.tool_events = []  # [{"id","name","input","result","is_error"}, ...]
        self.usage = {}


# input-dict keys, in priority order, worth surfacing per tool -- everything
# else in the input is still available via AgentSessionResult.tool_events for
# callers that want it, this is just what's worth a human-readable log line.
_TOOL_INPUT_HINT_KEYS = {
    "Bash": ("command",), "Write": ("file_path",), "Edit": ("file_path",),
    "Read": ("file_path",), "Glob": ("pattern",), "Grep": ("pattern",),
    "WebSearch": ("query",), "WebFetch": ("url",),
}
_ACTIVITY_PREVIEW_LIMIT = 400


def _describe_tool_use(name, tool_input):
    """One human-readable line for a tool call, e.g. 'Running: python x.py'
    or 'Writing retrieve.py'. Falls back to a generic 'Using <tool>' for
    tools/inputs this doesn't specifically know how to phrase."""
    verbs = {"Bash": "Running:", "Write": "Writing", "Edit": "Editing",
             "Read": "Reading", "Glob": "Searching for", "Grep": "Searching for",
             "WebSearch": "Searching the web:", "WebFetch": "Fetching"}
    hint_keys = _TOOL_INPUT_HINT_KEYS.get(name, ())
    detail = next((str(tool_input.get(k)) for k in hint_keys
                   if tool_input.get(k)), "")
    if detail:
        if len(detail) > 200:
            detail = detail[:200] + "…"
        return f"{verbs.get(name, f'Using {name}:')} {detail}"
    return f"Using tool: {name}"


def _tool_result_text(content):
    """``ToolResultBlock.content`` is ``str | list[dict] | None`` -- flatten
    it to plain text for logging (only the ``type: "text"`` items in the
    list form carry human-readable content; others, e.g. images, are noted
    but not rendered here)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _describe_tool_result(text, is_error):
    text = text.strip()
    if not text:
        return None
    if len(text) > _ACTIVITY_PREVIEW_LIMIT:
        text = text[:_ACTIVITY_PREVIEW_LIMIT] + "…"
    prefix = "  -> error: " if is_error else "  -> "
    return prefix + text.replace("\n", " ")


def run_agent(prompt, *, system_prompt=None, tools=None, cwd=None,
              max_turns=8, model=None, api_key=None, extra_env=None,
              on_activity=None):
    """Run one Claude Agent SDK session to completion and return an
    ``AgentSessionResult``.

    Blocking -- runs its own event loop via ``asyncio.run``, so it's safe to
    call from a plain Flask request thread or a background worker thread.
    Never call it from inside code that's already running an asyncio loop.

    ``tools`` is the Agent SDK's built-in tool-name list (e.g. ``["Read",
    "Write", "Bash"]``, ``["WebSearch", "WebFetch"]``, or ``[]`` for a
    tool-free single-turn completion) -- both ``tools`` and ``allowed_tools``
    are set to it, so nothing outside this explicit list is ever available.
    ``on_activity`` is an optional callback taking one formatted line of
    live progress per event -- assistant text verbatim, one line per tool
    call ("Running: python download.py"), and one indented preview line per
    tool result (truncated, "  -> " or "  -> error: " prefixed) -- so the
    UI can show what the agent is actually doing turn by turn instead of
    going silent until the whole session finishes.
    """
    if not package_available():
        raise RuntimeError(
            "Claude Agent SDK is not installed on the server. "
            "Install with: pip install claude-agent-sdk")
    if not cli_available():
        raise RuntimeError(
            "The Claude Code CLI ('claude') is not on the server's PATH -- "
            "the Agent SDK shells out to it. Install it with "
            "'npm install -g @anthropic-ai/claude-code'.")
    key = api_key or load_anthropic_key()
    if not key:
        raise RuntimeError(
            "Generating with Claude requires an Anthropic API key. Enter "
            "one in Settings, or set ANTHROPIC_API_KEY on the server.")

    from claude_agent_sdk import (
        AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock,
        ToolResultBlock, ToolUseBlock, UserMessage, query,
    )

    tool_list = list(tools or [])
    # The SDK forwards this environment only to the CLI subprocess it spawns
    # for this one call -- the Flask process's own environment (and any
    # concurrent request's key) is never touched. Base it on a copy of the
    # current environment (PATH, etc. that the CLI/Bash tool need to function)
    # rather than replacing it outright, then layer in any caller-supplied
    # credentials (e.g. a data source's API key for a Bash-executed download)
    # and finally our own key so it always wins.
    session_env = dict(os.environ)
    session_env.update({k: v for k, v in (extra_env or {}).items() if v})
    session_env["ANTHROPIC_API_KEY"] = key
    options_kwargs = {
        "max_turns": max_turns,
        "tools": tool_list,
        "allowed_tools": tool_list,
        "env": session_env,
        # Without this, Bash/Write/Edit calls block waiting for an
        # interactive permission decision that never arrives -- there is no
        # terminal or human attached to this session, it's a headless Flask
        # backend. Every tool this module ever grants is already scoped to
        # an explicit allowlist (``tools``) and, for anything that writes or
        # executes, a caller-chosen sandbox directory (``cwd``) -- the same
        # trust boundary this app already applies to LLM-generated code run
        # via subprocess/exec elsewhere, just enforced by the caller's tool
        # list instead of a per-call prompt.
        "permission_mode": "bypassPermissions",
    }
    if system_prompt:
        options_kwargs["system_prompt"] = system_prompt
    if cwd:
        options_kwargs["cwd"] = str(cwd)
    if model:
        options_kwargs["model"] = model
    options = ClaudeAgentOptions(**options_kwargs)

    result = AgentSessionResult()
    tool_events_by_id = {}  # ToolUseBlock.id -> the dict already in tool_events

    async def _consume():
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        result.text += block.text
                        if on_activity:
                            on_activity(block.text)
                    elif isinstance(block, ToolUseBlock):
                        entry = {"id": block.id, "name": str(block.name),
                                 "input": block.input, "result": "",
                                 "is_error": False}
                        result.tool_events.append(entry)
                        tool_events_by_id[block.id] = entry
                        if on_activity:
                            on_activity(_describe_tool_use(entry["name"], block.input))
            elif isinstance(message, UserMessage):
                content = message.content
                blocks = content if isinstance(content, list) else []
                for block in blocks:
                    if not isinstance(block, ToolResultBlock):
                        continue
                    text = _tool_result_text(block.content)
                    entry = tool_events_by_id.get(block.tool_use_id)
                    if entry is not None:
                        entry["result"] = text
                        entry["is_error"] = bool(block.is_error)
                    if on_activity:
                        line = _describe_tool_result(text, bool(block.is_error))
                        if line:
                            on_activity(line)
            elif isinstance(message, ResultMessage):
                if getattr(message, "result", None):
                    text = str(message.result)
                    if text and text not in result.text:
                        result.text = text
                # The same result message carries token counts alongside the
                # cost figure. Without them a Claude-provider run recorded a
                # call with no tokens at all, which the ledger then reports as
                # an unmeasured call -- so the run looked instrumented while
                # its token totals stayed at zero.
                sdk_usage = getattr(message, "usage", None) or {}
                if not isinstance(sdk_usage, dict):
                    sdk_usage = {name: getattr(sdk_usage, name, None) for name in (
                        "input_tokens", "output_tokens", "total_tokens")}
                result.usage = {
                    "cost_usd": getattr(message, "total_cost_usd", None),
                    "duration_ms": getattr(message, "duration_ms", None),
                    "turns": getattr(message, "num_turns", None),
                    "input_tokens": sdk_usage.get("input_tokens"),
                    "output_tokens": sdk_usage.get("output_tokens"),
                    "total_tokens": sdk_usage.get("total_tokens"),
                }
                if getattr(message, "is_error", False):
                    result.success = False
                    result.error = (
                        result.text or "Claude Agent SDK returned an error.")

    asyncio.run(_consume())
    if not result.text.strip() and result.success:
        result.success = False
        result.error = "Claude Agent SDK returned no text output."
    return result
