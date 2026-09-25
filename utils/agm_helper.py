"""LLM routing for handbook generation and retrieval: GIBD proxy,
OpenAI, or a self-hosted OpenAI-compatible server, chosen by key."""
import configparser
import json as _json
import os
import sys

import requests as _requests


# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the directory to sys.path
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

def load_config():
    config = configparser.ConfigParser()

    config_path = os.path.join(current_script_dir, 'config.ini')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            config.write(f)

    # print(config_path)
    config.read(config_path)
    return config


def load_OpenAI_key():
    """Return the GIBD API key (``USER_KEY``).

    Inside a Flask request the key comes from the UI (``g.api_key``).
    Outside of a request it falls back to the ``OPENAI_API_KEY`` env var / ``.env``.
    """
    # When called inside a Flask request, use the key the user provided via the UI
    try:
        from flask import g
        key = getattr(g, 'api_key', None)
        if key:
            return key
    except RuntimeError:
        pass  # Not in a Flask request context (e.g. notebooks, standalone scripts)

    # Outside of a web request, fall back to the environment / .env
    from pathlib import Path
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass
    return os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# GIBD-only enforcement
# ---------------------------------------------------------------------------
# Easy toggle: set ALLOW_OPENAI = True to re-enable plain OpenAI keys,
# or False (default) to accept GIBD keys only. The OpenAI routing code below
# is kept intact either way — just flip this one flag when you need OpenAI.
# (Mirrors the frontend ALLOW_OPENAI_KEY flag in script.js.)
ALLOW_OPENAI = True


def _ensure_gibd_only(key):
    """When ALLOW_OPENAI is False, reject a present non-GIBD key instead of
    silently falling back to OpenAI. GIBD keys and the local 'no-api' sentinel
    are always allowed."""
    if ALLOW_OPENAI:
        return
    if key and not str(key).startswith("gibd-") and key != "no-api":
        raise RuntimeError(
            "This app is configured for GIBD keys only — the provided key is not "
            "a GIBD key (it should start with 'gibd-'). Enter your GIBD key in "
            "Settings. (To re-enable OpenAI, set ALLOW_OPENAI = True in "
            "utils/agm_helper.py.)"
        )

# ---------------------------------------------------------------------------
# UI-selected model & reasoning effort (set by agm_workflow.py on each request)
# ---------------------------------------------------------------------------

_ui_model = None
_ui_reasoning_effort = None

def set_ui_model(model, reasoning_effort=None):
    """Called by agm_workflow.py to propagate UI selections to all agents."""
    global _ui_model, _ui_reasoning_effort
    _ui_model = model
    _ui_reasoning_effort = reasoning_effort

def get_ui_model():
    """Return the model selected in the UI, or 'gpt-5.2' as fallback."""
    return _ui_model or "gpt-5.2"

def get_ui_reasoning_effort():
    """Return the reasoning effort selected in the UI, or None."""
    return _ui_reasoning_effort

# ---------------------------------------------------------------------------
# GIBD API helpers
# ---------------------------------------------------------------------------

GIBD_API_URL = "https://www.gibd.online"
# GIBD_SERVICE = "Autonomous Geographic Modelling Agent"
GIBD_SERVICE = "GIS Co-Scientist"
LOCAL_BASE_URL = "http://128.118.54.16:11434/v1"
LOCAL_MODEL_NAME = "deepseek-v3:latest"  # Override applied for every "no-api" call


LOCAL_EXTRA_BODY = {"enable_thinking": False}  # Disable Gemma's thinking phase


def _usable_openai_fallback_key():
    """Return an API key that is safe to pass to the public OpenAI API.

    A GIBD user key (``gibd-...``) is NOT a valid OpenAI key — sending it to
    api.openai.com only produces a confusing 401 and leaks the key into
    OpenAI's error logs. So the fallback only kicks in if the env-configured
    key clearly looks like a real OpenAI key (``sk-...`` / ``sk-proj-...``).
    Returns ``None`` if no usable key is configured.
    """
    from pathlib import Path
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass
    api_key = os.getenv("OPENAI_API_KEY") or ""
    api_key = api_key.strip()
    if not api_key:
        return None
    if api_key.startswith("gibd-") or api_key.startswith("gibd_"):
        return None
    if not api_key.startswith("sk-"):
        # Not a recognizable OpenAI key shape — refuse to try it so we don't
        # mask the real GIBD error with a generic 401.
        return None
    return api_key


class _DotDict:
    """Wrap a dict so that nested keys can be accessed with dot notation.

    This lets existing code like ``response.choices[0].message.content``
    keep working when the underlying data comes from a raw JSON dict
    instead of the OpenAI Python SDK.
    """

    def __init__(self, d):
        for k, v in (d if isinstance(d, dict) else {}).items():
            if isinstance(v, dict):
                setattr(self, k, _DotDict(v))
            elif isinstance(v, list):
                setattr(self, k, [_DotDict(i) if isinstance(i, dict) else i for i in v])
            else:
                setattr(self, k, v)

    def __getattr__(self, name):
        # Return None for missing attributes, matching OpenAI SDK behaviour
        # (e.g. chunk.choices[0].delta.content is None when key absent).
        return None

    def get(self, key, default=None):
        val = getattr(self, key, default)
        return default if val is None else val


# ── HTTP resilience for GIBD calls ───────────────────────────────────────────
# The GIBD proxy occasionally times out on connect or returns transient 5xx
# responses.  Without timeouts and retries a single blip kills a long-running
# manuscript pipeline (8 sequential LLM calls).  These constants apply to
# every request to gibd.online.

# (connect_timeout, read_timeout) — read is generous for slow LLM responses
GIBD_HTTP_TIMEOUT = (15, 600)
GIBD_MAX_RETRIES = 4
GIBD_BACKOFF_BASE = 2.0  # seconds — 2, 4, 8, 16


def _gibd_post_with_retry(url, *, json=None, stream=False, label="GIBD"):
    """
    POST to a GIBD endpoint with timeouts and exponential backoff on
    transient failures (connection errors, read timeouts, 5xx, 429).

    Raises RuntimeError on permanent failure (4xx other than 429, or
    after all retries exhausted).
    """
    import time as _time

    last_err = None
    for attempt in range(GIBD_MAX_RETRIES):
        try:
            resp = _requests.post(
                url, json=json, stream=stream, timeout=GIBD_HTTP_TIMEOUT
            )
        except (_requests.ConnectionError, _requests.Timeout) as e:
            last_err = e
            if attempt < GIBD_MAX_RETRIES - 1:
                wait = GIBD_BACKOFF_BASE * (2 ** attempt)
                print(f"[{label}] {type(e).__name__} on attempt {attempt + 1}/{GIBD_MAX_RETRIES} — retrying in {wait}s...")
                _time.sleep(wait)
                continue
            raise RuntimeError(
                f"{label} unreachable after {GIBD_MAX_RETRIES} attempts: {e}"
            ) from e

        # Retry transient server errors and rate limits
        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if attempt < GIBD_MAX_RETRIES - 1:
                wait = GIBD_BACKOFF_BASE * (2 ** attempt)
                print(f"[{label}] HTTP {resp.status_code} on attempt {attempt + 1}/{GIBD_MAX_RETRIES} — retrying in {wait}s...")
                _time.sleep(wait)
                continue
            raise RuntimeError(
                f"{label} failed after {GIBD_MAX_RETRIES} attempts: HTTP {resp.status_code}"
            )

        return resp

    # Defensive — loop should always either return or raise
    raise RuntimeError(f"{label} failed: {last_err}")


def _gibd_handshake(user_key):
    """Step 1 of GIBD flow – obtain a ``question_id``."""
    resp = _gibd_post_with_retry(
        f"{GIBD_API_URL}/api/request-question-id",
        json={"user_api_key": user_key, "service_name": GIBD_SERVICE},
        label="GIBD handshake",
    )
    if resp.status_code != 201:
        raise RuntimeError(f"GIBD handshake failed ({resp.status_code}): {resp.text}")
    return resp.json()["question_id"]


def gibd_chat_completion(user_key, model, messages, **kwargs):
    """Non-streaming chat completion through the GIBD proxy.

    Extra OpenAI parameters (``temperature``, ``response_format``, etc.)
    can be passed as keyword arguments and will be forwarded in the payload.

    Returns a ``_DotDict`` that supports the same attribute access as an
    OpenAI ``ChatCompletion`` object, e.g.
    ``response.choices[0].message.content``.
    """

    q_id = _gibd_handshake(user_key)
    payload = {
        "question_id": q_id,
        "service_name": GIBD_SERVICE,
        "model": model,
        "messages": messages,
        "stream": False,
        **kwargs,
    }
    resp = _gibd_post_with_retry(
        f"{GIBD_API_URL}/api/openai/{user_key}",
        json=payload,
        label="GIBD inference",
    )
    if resp.status_code != 200:
        raise RuntimeError(f"GIBD inference failed ({resp.status_code}): {resp.text}")
    return _DotDict(resp.json())


def gibd_chat_completion_stream(user_key, model, messages, **kwargs):
    """Streaming chat completion through the GIBD proxy.

    Extra OpenAI parameters (``temperature``, ``response_format``, etc.)
    can be passed as keyword arguments and will be forwarded in the payload.

    Yields ``_DotDict`` chunks that support the same attribute access as
    OpenAI streaming chunks, e.g. ``chunk.choices[0].delta.content``.
    """

    q_id = _gibd_handshake(user_key)
    payload = {
        "question_id": q_id,
        "service_name": GIBD_SERVICE,
        "model": model,
        "messages": messages,
        "stream": True,
        **kwargs,
    }
    resp = _gibd_post_with_retry(
        f"{GIBD_API_URL}/api/openai/{user_key}",
        json=payload,
        stream=True,
        label="GIBD stream",
    )
    if resp.status_code != 200:
        raise RuntimeError(f"GIBD stream failed ({resp.status_code}): {resp.text}")

    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").strip()
        # Strip the SSE "data:" prefix (with or without trailing space)
        if decoded.startswith("data:"):
            decoded = decoded[5:].lstrip()
        if not decoded or decoded == "[DONE]":
            break
        try:
            chunk = _json.loads(decoded)
            if "error" in chunk:
                raise RuntimeError(f"LLM API error: {chunk['error']}")
            yield _DotDict(chunk)
        except _json.JSONDecodeError:
            continue


def openai_chat_completion(user_key, model, messages, **kwargs):
    """Non-streaming chat completion via the OpenAI API directly.

      Extra OpenAI parameters (``temperature``, ``response_format``, etc.)
      can be passed as keyword arguments.

      Returns the native OpenAI ``ChatCompletion`` object, so callers can use
      ``response.choices[0].message.content`` as usual.
    """
    from openai import OpenAI
    client = OpenAI(api_key=user_key)
    return client.chat.completions.create(
        model=model, messages=messages, stream=False, **kwargs
    )


def local_chat_completion(model, messages, base_url=None, api_key=None, **kwargs):
    """Non Streaming chat completion from a local/self-hosted OpenAI-compatible
    server. ``base_url``/``model`` default to the shared LOCAL_* constants but
    can be overridden per-call (e.g. from an Experiment 2 study's controls)."""
    from openai import OpenAI
    client = OpenAI(
        base_url=base_url or LOCAL_BASE_URL,
        api_key=api_key or "no-api",
    )
    return client.chat.completions.create(
        model=model or LOCAL_MODEL_NAME,
        messages=messages, stream=False,
        # extra_body=LOCAL_EXTRA_BODY,
        **kwargs
    )


def openai_chat_completion_stream(user_key, model, messages, **kwargs):
    """Streaming chat completion via the OpenAI API directly.

    Extra OpenAI parameters (``temperature``, ``response_format``, etc.)
    can be passed as keyword arguments.

    Yields native OpenAI streaming chunks, e.g.
    ``chunk.choices[0].delta.content``.
    """
    from openai import OpenAI
    client = OpenAI(api_key=user_key)
    response = client.chat.completions.create(
        model=model, messages=messages, stream=True, **kwargs
    )
    for chunk in response:
        yield chunk

def local_chat_completion_stream(model, messages, base_url=None, api_key=None, **kwargs):
    """Streaming chat completion from a local/self-hosted OpenAI-compatible
    server. ``base_url``/``model`` default to the shared LOCAL_* constants but
    can be overridden per-call (e.g. from an Experiment 2 study's controls)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url or LOCAL_BASE_URL, api_key=api_key or "no-api")

    response = client.chat.completions.create(
        model=model or LOCAL_MODEL_NAME, messages=messages, stream=True,
        # extra_body=LOCAL_EXTRA_BODY,
        **kwargs
    )
    for chunk in response:
        yield chunk


def _route_chat_completion(model, messages, user_key=None, base_url=None, **kwargs):
    """Route a non-streaming chat completion to GIBD, a custom/local server,
    or OpenAI based on ``base_url``/the key.

    A caller-supplied ``base_url`` (e.g. an Experiment 2 study's open-source
    model settings) always wins and routes to that OpenAI-compatible server.
    Otherwise, if ``user_key`` starts with ``"gibd-"`` the call is forwarded
    to :func:`gibd_chat_completion`; a ``"no-api"`` key routes to the default
    local server; anything else goes directly to :func:`openai_chat_completion`.

    Returns an object that supports ``response.choices[0].message.content``
    in every path (``_DotDict`` from GIBD, native ``ChatCompletion`` otherwise).
    """
    key = user_key or load_OpenAI_key()
    _ensure_gibd_only(key)
    if base_url:
        print(f"Using custom base_url ({base_url}) without streaming")
        return local_chat_completion(
            model, messages, base_url=base_url, api_key=key, **kwargs)
    if key and key.startswith("gibd-"):
        # print(f"[DEBUG PRINT]: key = {key}")
        print("Using GIBD-Key without streaming")
        return gibd_chat_completion(key, model, messages, **kwargs)
    elif key and key == "no-api":
        print("Using Local model without streaming")
        return local_chat_completion(model, messages, **kwargs)
    else:
        # print(f"[DEBUG PRINT]: key = {key}")
        print("Using openai API key and without streaming")
        return openai_chat_completion(key, model, messages, **kwargs)


def _route_chat_completion_stream(model, messages, user_key=None, base_url=None, **kwargs):
    """Route a streaming chat completion to GIBD, a custom/local server, or
    OpenAI based on ``base_url``/the key. See :func:`client_chat_completion`
    for the routing rules.

    Yields streaming chunks with the same ``chunk.choices[0].delta.content``
    shape in every path.
    """
    key = user_key or load_OpenAI_key()
    _ensure_gibd_only(key)
    if base_url:
        print(f"Using custom base_url ({base_url}) and streaming")
        yield from local_chat_completion_stream(
            model, messages, base_url=base_url, api_key=key, **kwargs)
    elif key and key.startswith("gibd-"):
        # print(f"[DEBUG PRINT]: key = {key}")
        print("Using GIBD-key and streaming")
        yield from gibd_chat_completion_stream(key, model, messages, **kwargs)
    elif key and key=="no-api":
        print("Using Local model and streaming")
        yield from local_chat_completion_stream(model, messages, **kwargs)
    else:
        # print(f"[DEBUG PRINT]: key = {key}")
        print("Using openai API key and streaming")
        yield from openai_chat_completion_stream(key, model, messages, **kwargs)


def _usage_from_chunk(chunk):
    """The usage block on a streamed chunk, whatever object carries it.

    OpenAI SDK chunks expose ``.usage``; the GIBD proxy's chunks arrive as
    ``_DotDict``, which returns None for absent keys. Both are handled, and a
    chunk without usage yields None so callers can keep the last one seen.
    """
    usage = getattr(chunk, "usage", None)
    if usage is None and isinstance(chunk, dict):
        usage = chunk.get("usage")
    return usage or None


def client_chat_completion(model, messages, user_key=None, base_url=None,
                           record_usage=True, **kwargs):
    """Non-streaming chat completion, recorded against the active usage ledger.

    This is the second of the two LLM families in this codebase (the other is
    :class:`GIBDChat`). The Research, Spatial Analysis and Data Retrieval modes
    all reach it -- the task manager, the spatial analysis executor and the RQ
    understanding agent call it directly -- so it is where their token cost has
    to be captured.

    Pass ``record_usage=False`` for a caller that records the same call itself;
    see ``DataRetrieverAgent._stream_with_usage``, which does its own
    per-chunk accounting and would otherwise be counted twice.
    """
    response = _route_chat_completion(
        model, messages, user_key=user_key, base_url=base_url, **kwargs)
    if record_usage:
        _record_llm_usage(model, _usage_from_chunk(response) or {}, source="chat")
    return response


def client_chat_completion_stream(model, messages, user_key=None, base_url=None,
                                  record_usage=True, **kwargs):
    """Streaming chat completion, recorded against the active usage ledger.

    OpenAI-style backends send the usage figures only on a final chunk, and
    only when ``stream_options`` asks for them. Not every backend this routes
    to accepts that argument, and because the router is a GENERATOR a rejection
    does not surface until the first chunk is pulled -- so a try/except around
    the call alone would never fire. The first chunk is therefore pulled here:
    if that fails, the whole stream is retried without ``stream_options``
    before any content has reached the caller. Losing the token count is
    acceptable; losing the stream is not.
    """
    if not record_usage or "stream_options" in kwargs:
        # The caller is doing its own accounting, or has already specified how
        # it wants usage reported. Either way, stay out of the way.
        yield from _route_chat_completion_stream(
            model, messages, user_key=user_key, base_url=base_url, **kwargs)
        return

    first = None
    iterator = None
    try:
        iterator = iter(_route_chat_completion_stream(
            model, messages, user_key=user_key, base_url=base_url,
            stream_options={"include_usage": True}, **kwargs))
        first = next(iterator)
    except StopIteration:
        iterator = None
    except Exception:
        print("[client_chat_completion_stream] backend rejected stream_options; "
              "retrying without usage capture")
        yield from _route_chat_completion_stream(
            model, messages, user_key=user_key, base_url=base_url, **kwargs)
        return

    usage = None
    if first is not None:
        usage = _usage_from_chunk(first) or usage
        yield first
        for chunk in iterator:
            usage = _usage_from_chunk(chunk) or usage
            yield chunk

    # Recorded once, after the stream is exhausted -- one call, not one per
    # chunk. A backend that accepted stream_options but sent no usage leaves
    # this empty, and the ledger counts the call and flags it as missing rather
    # than pretending it was free.
    _record_llm_usage(model, usage or {}, source="chat-stream")



# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------
#
# Every LLM call the Research, Spatial Analysis and Data Retrieval modes make
# goes through GIBDChat, so this is the one place where usage can be captured
# for all three. It has to happen at call time: once a run finishes, per-request
# usage is unrecoverable -- the GIBD proxy and OpenAI expose no way to look up
# historical usage keyed by anything this application knows.
#
# Recording is a no-op when no ledger is active (notebooks, CLI, tests), so
# instrumenting a call site never changes behaviour outside a collection scope.

def _usage_ledger():
    """Import lazily so a partially-installed environment still loads helper."""
    try:
        from agents.data_agent import usage_ledger
        return usage_ledger
    except Exception:
        return None


def _record_llm_usage(model, usage, source=""):
    """Record one call against the active ledger; returns the entry or None."""
    ledger = _usage_ledger()
    if ledger is None:
        return None
    try:
        return ledger.record(model, usage, source=source)
    except Exception:
        # Accounting must never take down a working LLM call.
        return None

