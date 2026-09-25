import ast
import concurrent.futures
import contextlib
import datetime
import json
import re
import unicodedata
import sys
import time
import traceback
from urllib.parse import urlsplit
import geopandas as gpd
from IPython.display import display, HTML, Code

from click import prompt
import tomli
from glob import glob
import logging
import os
import configparser

from openai import OpenAI
from dotenv import load_dotenv
import requests

import utils.agm_helper as helper
from agents.data_agent import claude_agent_provider
from agents.data_agent import usage_ledger


def _stream_with_usage(**kwargs):
    """helper.client_chat_completion_stream, asking for a final usage chunk.

    Token cost can only be captured while the call happens, and OpenAI-style
    backends only send the usage chunk when stream_options requests it. Not
    every backend this helper routes to (GIBD, a local server) accepts that
    argument, and because the helper is a GENERATOR the rejection does not
    surface until the first chunk is pulled -- a try/except around the call
    itself would never fire. So the first chunk is pulled here: if that fails,
    the whole stream is retried without stream_options, before any content has
    been yielded to the caller.
    """
    try:
        # record_usage=False: this wrapper's callers record each call
        # themselves (with their own phase and source labels), so letting the
        # helper record as well would count every retrieval call twice.
        stream = helper.client_chat_completion_stream(
            stream_options={"include_usage": True}, record_usage=False, **kwargs)
        iterator = iter(stream)
        first = next(iterator)
    except StopIteration:
        return
    except Exception:
        logging.info("Backend rejected stream_options; retrying without "
                     "usage capture (token cost will be unavailable).")
        yield from helper.client_chat_completion_stream(record_usage=False, **kwargs)
        return
    yield first
    yield from iterator


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load environment variables from .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Use this in .py files
# BASE_DIR = os.getcwd()  # Notebook fallback

EVALUATION = False

# Retry/timeout budget for a single data request's execute-and-debug loop in
# LLM_Find. Worst case used to be 10 trials x 300s = 50 minutes per outer
# attempt (and data_download retries the whole request up to 3 times on top).
# In practice a repair that hasn't converged by trial 5 almost never converges
# by trial 10 — the same-error early-stop usually fires first, but a debugger
# that keeps producing NEW errors could still burn the full budget. Both knobs
# are env-overridable so evaluation runs can restore the old, patient budget
# (AGSI_DOWNLOAD_TRIALS=10 AGSI_DOWNLOAD_ATTEMPT_TIMEOUT=300) without a code
# change.
def _int_env(name, default):
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default

DOWNLOAD_TRIALS = max(1, _int_env("AGSI_DOWNLOAD_TRIALS", 5))
DOWNLOAD_ATTEMPT_TIMEOUT = max(30, _int_env("AGSI_DOWNLOAD_ATTEMPT_TIMEOUT", 180))

# Evidence-grounded handbook revisions to attempt before falling back to
# _re_research_skill, once the code-repair loop above is exhausted. Deliberately
# far below Handbook Studio's 3: Studio is refining ONE handbook while the user
# watches that and nothing else, whereas a research run fans out over every data
# request the RQ breakdown produced, and each refinement here costs an LLM call
# plus a full re-download attempt that the rest of the workflow waits on. One
# revision buys the common case (a single wrong parameter or endpoint detail);
# a second rarely converges before the source itself turns out to be wrong,
# which is what re-research is for. Env-overridable for evaluation runs, and
# settable to 0 to restore the previous debug-then-re-research ladder exactly.
HANDBOOK_REFINE_ATTEMPTS = max(0, _int_env("AGSI_HANDBOOK_REFINE_ATTEMPTS", 1))

# After which failed execute trial the loop asks the failure classifier, once,
# whether the error is fixable at all (see execute_complete_program). Trial 1
# is too early — a single failure is usually a plain bug the first fix solves;
# waiting until the budget is spent is what this exists to avoid. 0 disables
# the early check and restores the debug-until-exhausted behaviour.
EARLY_TRIAGE_AFTER_TRIAL = max(0, _int_env("AGSI_EARLY_TRIAGE_AFTER_TRIAL", 2))

# Model used for data retrieval — source selection, download-code generation,
# and every debug/repair attempt — when the caller doesn't specify one (the
# production Spatial Analysis / Research path does not).
#
# Matched to handbook_generator.HANDBOOK_GEN_MODEL on purpose. These were
# previously mismatched (handbooks authored by gpt-5.2, the code consuming
# them written by gpt-4o), and the weaker consumer repeatedly failed to follow
# a correct handbook — inventing endpoints and leaving literal placeholder
# credentials in the generated code. Callers that DO pass a model explicitly
# (Experiment 2's harness, Handbook Studio's picker) are unaffected.
DATA_RETRIEVAL_MODEL = (
    os.environ.get("AGSI_DATA_RETRIEVAL_MODEL", "").strip() or "gpt-5.2")



# Control-arm prompt surgery. The treatment prompts name a technical handbook in
# three places; the control has no handbook, so saying any of it would tell the
# model something untrue about its own inputs. Each pair below is applied as a
# replacement AGAINST the treatment text, which is what keeps the treatment
# prompt byte-for-byte unchanged -- and which also means rewording one of these
# sentences upstream would turn the swap into a silent no-op.
# tests/test_control_prompt.py asserts the control prompt never says "handbook";
# that is what catches such a drift.
_HANDBOOK_ROLE_CLAUSE = (
    "When downloading geo-spatial data, the technical handbook for a "
    "particular data source is provided; you can follow it, and write "
    "Python code carefully to download the data.")
_NO_HANDBOOK_ROLE_CLAUSE = (
    "When downloading geo-spatial data, write Python code carefully to "
    "download the data.")
# Deliberately excludes the leading "match it"/"match" so the one pair covers
# both the generation requirement and the debug requirement, whose wording
# differs by exactly that word.
_HANDBOOK_PATTERN_CLAUSE = "the exact naming pattern documented in the handbook"
_NO_HANDBOOK_PATTERN_CLAUSE = "the exact naming pattern the source documents"

# Feedback-driven revision (get_feedback_revision_prompt) says what to do when
# the reviewer contradicts the handbook. The control arm has no handbook, so
# that requirement would point at nothing -- it gets the second wording, which
# says the same thing about the code's own assumptions instead.
_FEEDBACK_GUIDELINES_CLAUSE = (
    "If the feedback contradicts the technical guidelines, follow the feedback "
    "and say so in your explanation: the reviewer can see things the "
    "guidelines do not record, such as a deprecated endpoint or a wrong study "
    "area.")
_FEEDBACK_NO_GUIDELINES_CLAUSE = (
    "If the feedback contradicts what the code currently assumes, follow the "
    "feedback and say so in your explanation: the reviewer can see things the "
    "code does not record, such as a deprecated endpoint or a wrong study "
    "area.")


def _claude_agent_instructions(script_name, try_count, control):
    """The write/run/fix instructions appended after the download prompt on the
    Claude Agent SDK path.

    The credentials sentence used to name the handbook as where the
    environment-variable names are documented. In the control arm no handbook
    was given, so that sentence pointed at nothing -- the same dangling
    reference the generation and debug prompts carried, on the one path that
    builds its prompt here rather than in create_download_prompt. The
    instruction that matters (read them from the environment, never invent one)
    is identical in both arms; only the provenance clause differs.
    """
    where = ("under their exact names" if control
             else "under the exact names documented in the handbook")
    return (
        "You are running in an empty, sandboxed working directory. "
        f"Write the complete program to a file named {script_name}, "
        f"then run it with `python {script_name}` via the Bash tool. "
        "If it fails, diagnose the real cause from the error (never "
        f"guess), fix {script_name} in place, and run it again. Repeat "
        f"until it finishes successfully or you have made {try_count} "
        "run attempts, whichever comes first — do not exceed that many "
        "runs. Required credentials, if any, are already present in "
        f"the environment {where} — read them via os.environ; never invent or "
        "hard-code a value.")


class CredentialError(Exception):
    """Raised when a data download fails because the API key / credentials are
    invalid, unauthorized — or simply absent. Unlike a code bug, none of these
    can be fixed by retrying or LLM-debugging the download code, so it
    short-circuits the execute/retry loop and is surfaced to the user, who is
    the only one who can supply or correct a key.

    ``missing_keys`` names credentials the code needed and could not find
    (as opposed to ones the provider rejected). The interactive re-prompt
    uses it to ask for exactly those, even when the handbook never declared
    them."""

    def __init__(self, message, missing_keys=None):
        super().__init__(message)
        self.missing_keys = [str(n) for n in (missing_keys or []) if n]


# Wording that says a credential is ABSENT rather than wrong. Paired with a
# known key name in the same message so an unrelated "required parameter"
# error can never be mistaken for a missing key.
_MISSING_CREDENTIAL_CONTEXT = re.compile(
    r"missing|not set|unset|not found|not provided|not configured|required|"
    r"environment variable|env var|no value|empty", re.I)


def _missing_credential_names(err, candidate_names):
    """Which of ``candidate_names`` the failure says are absent.

    A ``KeyError('FIRMS_MAP_KEY')`` from ``os.environ[...]``, or any message
    that names the credential alongside "missing"/"not set"/"environment
    variable" wording, counts. Walks the exception chain like the other
    detectors. Returns the names in the order they were listed, or [].
    """
    names = [str(n).strip() for n in (candidate_names or []) if str(n).strip()]
    if not names:
        return []
    found, seen = [], set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        if isinstance(err, KeyError) and err.args:
            wanted = str(err.args[0]).strip().lower()
            for n in names:
                if n.lower() == wanted and n not in found:
                    found.append(n)
        text = f"{type(err).__name__}: {err}"
        if _MISSING_CREDENTIAL_CONTEXT.search(text):
            for n in names:
                if n not in found and re.search(
                        r"(?<![A-Za-z0-9_])" + re.escape(n) + r"(?![A-Za-z0-9_])",
                        text, re.I):
                    found.append(n)
        err = err.__cause__ or err.__context__
    return found


_CREDENTIAL_ERROR_SIGNS = (
    'unauthorized', 'forbidden', 'access denied', 'permission denied',
    '401 client error', '403 client error',
    'http error 401', 'http error 403',
    'error 401', 'error 403',
    'invalid api key', 'invalid api_key', 'invalid apikey', 'invalid_api_key',
    'invalid key', 'invalid token', 'invalid map_key', 'invalid map key',
    'api key is invalid', 'apikey is invalid', 'wrong api key', 'bad api key',
    'authentication failed', 'authentication error', 'invalid authentication',
    'bad credentials', 'invalid credentials', 'missing api key', 'api key required',
    
    'key are invalid', 'key is invalid', 'email and/or key', 'email or key',
    'not a registered', 'no longer valid', 'expired key', 'expired token',
)



_CREDENTIAL_HTTP_STATUS = (401, 403)

# Hosts the generated code may call as an internal HELPER (e.g. resolving a
# place name to coordinates) that are never themselves the data source a
# handbook declares credentials for. A 401/403 from one of these means the
# helper is rate-limiting/blocking the request (e.g. Nominatim requiring a
# proper User-Agent per its usage policy) — a fixable code bug, not evidence
# that THIS source's API key is wrong. See create_download_prompt's guidance
# on using osmnx/a descriptive User-Agent for Nominatim specifically.
_NON_SOURCE_HELPER_HOSTS = (
    "nominatim.openstreetmap.org",
)


def _http_status_of(err):
    """Return the HTTP status code carried by an exception, or None.

    Works across HTTP libraries without importing them, by duck-typing:
    - ``requests``' HTTPError exposes ``err.response.status_code``
    - ``urllib``'s HTTPError exposes ``err.code``
    """
    resp = getattr(err, "response", None)
    status = getattr(resp, "status_code", None)
    if isinstance(status, int):
        return status
    code = getattr(err, "code", None)
    if isinstance(code, int):
        return code
    return None


def _http_host_of(err):
    """Return the hostname the failing HTTP request was made to, or None.

    Works across HTTP libraries without importing them: ``requests``'
    HTTPError exposes ``err.response.url``; ``urllib``'s HTTPError exposes
    ``err.filename``/``err.url``.
    """
    resp = getattr(err, "response", None)
    url = (getattr(resp, "url", None) or getattr(err, "url", None)
           or getattr(err, "filename", None))
    if not isinstance(url, str) or not url:
        return None
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return None



_ENV_READ_PATTERNS = (
    re.compile(r'os\.environ\s*\[\s*[\'"]([A-Za-z_][A-Za-z0-9_]*)[\'"]\s*\]'),
    re.compile(r'os\.environ\.get\s*\(\s*[\'"]([A-Za-z_][A-Za-z0-9_]*)[\'"]'),
    re.compile(r'os\.getenv\s*\(\s*[\'"]([A-Za-z_][A-Za-z0-9_]*)[\'"]'),
)

# Reading these says nothing about the data source needing a credential.
_ENV_READS_TO_IGNORE = frozenset({
    "PATH", "HOME", "TMPDIR", "TEMP", "TMP", "PWD", "USER", "USERNAME",
    "PYTHONPATH", "PYTHONUNBUFFERED", "LANG", "LC_ALL", "SHELL",
    "REEXEC_HTTP_LOG", "RERUN_HTTP_LOG", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GDAL_DATA", "PROJ_LIB", "AWS_REGION", "NO_PROXY", "HTTP_PROXY",
    "HTTPS_PROXY",
})


def credential_names_in_code(code):
    """Environment variables the generated code expects to read.

    Without a handbook there is no declared list of required credentials, so
    the only honest source of truth is the code the model actually wrote: if it
    reads FIRMS_MAP_KEY, that is the credential this run needs, under the exact
    name the code will look for. Names that are plainly environment plumbing
    rather than source credentials are ignored, as is the app's own LLM key --
    prompting for those would be noise.
    """
    found, seen = [], set()
    for pattern in _ENV_READ_PATTERNS:
        for name in pattern.findall(code or ""):
            upper = name.upper()
            if upper in _ENV_READS_TO_IGNORE or name in seen:
                continue
            seen.add(name)
            found.append(name)
    return found


def _looks_like_credential_error(err):
    """Decide whether ``err`` is an invalid/unauthorized credential failure
    (so we can stop retrying and warn the user) rather than a code bug.

    Walks the exception chain (``__cause__``/``__context__``) because the
    download code often wraps the original HTTP error in a later one. For each
    link we first check the HTTP status code structurally (immune to error
    wording), then fall back to matching known credential phrases in the text.
    A link whose request went to a known non-source helper host (e.g.
    Nominatim, used purely for geocoding) is skipped entirely — a 401/403
    there says nothing about the actual data source's credentials.
    """
    seen = set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))

        if _http_host_of(err) in _NON_SOURCE_HELPER_HOSTS:
            err = err.__cause__ or err.__context__
            continue

        # 1. Structural: an HTTP 401/403 is unambiguous, whatever the wording.
        if _http_status_of(err) in _CREDENTIAL_HTTP_STATUS:
            return True

        # 2. Text: the exception string AND the response body (when present).
        #    Providers like EPA AQS reject a bad key with HTTP 200/400 and put
        #    the real reason ("Email and/or key are invalid.") only in the body,
        #    so raise_for_status() yields a generic "400 Client Error" — we must
        #    look past it at the body itself.
        text = f"{type(err).__name__}: {err}".lower()
        resp = getattr(err, "response", None)
        body = getattr(resp, "text", None)
        if isinstance(body, str):
            text += " " + body.lower()
        if any(sign in text for sign in _CREDENTIAL_ERROR_SIGNS):
            return True

        err = err.__cause__ or err.__context__
    return False


class DataUnavailableError(Exception):
    """Raised when the source answered but has nothing for what was asked:
    the requested period, area, variable, or product lies outside what it
    covers. Like a credential error this is not a code bug — no rewrite of
    the download code can produce data the provider does not hold — so it
    short-circuits the execute/debug loop, the handbook-repair ladder, AND
    the outer per-request retry in ``data_download``. The alternative the
    debugger otherwise drifts into (shrinking the date range, swapping the
    county, dropping the variable until a call finally returns 200) is
    worse than failing: the analysis then runs on data nobody asked for."""
    pass


# The marker the download/debug prompts tell the model to raise (or reply
# with) when it recognises that the provider has no data for the request,
# so the detector below never has to guess from wording alone.
DATA_UNAVAILABLE_MARKER = "DATA_UNAVAILABLE"

# Phrases a provider (or the generated code's own coverage check) uses to
# say "there is nothing here for what you asked" — as opposed to "your
# request was malformed". Deliberately specific: a generic "not found" or
# "does not exist" also describes a missing local directory, a typo'd
# endpoint, or a wrong column name, all of which the debugger CAN fix.
_DATA_UNAVAILABLE_ERROR_SIGNS = (
    'no data available', 'no data is available', 'no data are available',
    'no data found', 'no data were found', 'no data was found',
    'no data returned', 'returned no data', 'no data exists', 'no data exist',
    'no data for the', 'no data for this', 'no data in the requested',
    'data not available', 'data is not available', 'data are not available',
    'data is unavailable', 'data are unavailable', 'data unavailable',
    'no records found', 'no records were found', 'no records match',
    'no matching records', 'no matching data', 'no results found',
    'no results were found', 'no features found', 'no observations found',
    'no observations were found', 'zero records returned',
    'out of range', 'outside the available', 'outside of the available',
    'outside the coverage', 'outside the data coverage',
    'outside the temporal', 'outside the spatial', 'outside the valid',
    'not within the available', 'not within the coverage',
    'outside the period', 'outside the range', 'outside of the range',
    'no coverage for', 'not covered by this', 'is not covered', 'are not covered',
    'exceeds the available', 'beyond the available', 'beyond the end of',
    'before the earliest', 'after the latest', 'earlier than the earliest',
    'later than the latest', 'later than the last available',
    'invalid date range', 'date range is invalid', 'date range not available',
    'date range is not available', 'requested period is not',
    'requested date range', 'requested time range', 'requested time period',
    'not yet available', 'is not available for', 'are not available for',
    'not available for the requested', 'unavailable for the requested',
    'not available for this', 'unavailable for this',
    'not yet released', 'has not been released', 'have not been released',
    'not yet published', 'has not been published',
    'no longer available', 'is no longer served', 'has been retired',
    'no such dataset', 'no such product', 'dataset not available',
    'product not available', 'variable not available', 'not a valid variable',
    'no data at the requested', 'no data at this location',
)


# A DATA_UNAVAILABLE marker whose reason is about the dataset's SHAPE rather
# than its coverage. The generated code sometimes probes a row, fails to see
# a column named the way the request is phrased ("no year field, so it
# cannot be filtered to 2023"), and raises the marker on its own inference —
# no provider ever said the data was missing. That is a schema-reading
# problem the debugger can fix (the period is implicit in a release-year
# dataset; the variable is a wide-format column), so it must not end the run.
_SCHEMA_GUESS_RE = re.compile(
    r"\b(field|fields|column|columns|schema|attribute|attributes|property|"
    r"properties|key|keys|header|headers)\b|\$select|no recognizable|"
    r"does not expose|doesn't expose|exposes no|not in the (row|response)",
    re.IGNORECASE)


def _marker_reason_is_schema_guess(text):
    """True when the text after a DATA_UNAVAILABLE marker blames a missing
    field/column/schema element instead of quoting a provider's coverage
    statement."""
    if DATA_UNAVAILABLE_MARKER not in text:
        return False
    reason = text.split(DATA_UNAVAILABLE_MARKER, 1)[1]
    return bool(_SCHEMA_GUESS_RE.search(reason))


def _looks_like_data_unavailable_error(err):
    """Decide whether ``err`` says the source has no data for this request
    (so we can stop debugging and tell the user) rather than that the code
    is wrong.

    Same shape as ``_looks_like_credential_error``: walk the exception
    chain, skip helper hosts (a geocoder saying "no results" is a place-name
    problem the debugger can fix), and match on the exception text plus the
    response body. The generated code's own ``DATA_UNAVAILABLE:`` marker is
    the strongest signal and is checked first.
    """
    seen = set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))

        if _http_host_of(err) in _NON_SOURCE_HELPER_HOSTS:
            err = err.__cause__ or err.__context__
            continue

        text = f"{type(err).__name__}: {err}"
        if DATA_UNAVAILABLE_MARKER in text:
            if not _marker_reason_is_schema_guess(text):
                return True
            # The code raised the marker on its own reading of the schema,
            # not on anything the provider said. Ignore the marker itself;
            # the phrase check below and the rest of the chain still run,
            # so a genuine provider statement further down is honoured.
            logging.warning(
                "DATA_UNAVAILABLE marker not honoured — its reason is about "
                f"the schema, not coverage; treating as a code bug: {err}")
        text = text.lower()
        resp = getattr(err, "response", None)
        body = getattr(resp, "text", None)
        if isinstance(body, str):
            text += " " + body.lower()
        if any(sign in text for sign in _DATA_UNAVAILABLE_ERROR_SIGNS):
            return True

        err = err.__cause__ or err.__context__
    return False


def _data_unavailable_reason(err):
    """The provider's own words for WHY there is no data, for the user-facing
    message. In order of preference: the text after the code's own marker;
    the response body or exception message (whichever carried the coverage
    phrase, walking the chain outermost-in); else the innermost exception's
    message."""
    def _snip(text):
        return " ".join(str(text).split())[:400]

    seen = set()
    innermost = err
    matched = None
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        text = str(err)
        if DATA_UNAVAILABLE_MARKER in text:
            reason = text.split(DATA_UNAVAILABLE_MARKER, 1)[1].lstrip(" :-—")
            if reason.strip():
                return _snip(reason)
        if matched is None and _http_host_of(err) not in _NON_SOURCE_HELPER_HOSTS:
            body = getattr(getattr(err, "response", None), "text", None)
            if isinstance(body, str) and any(
                    sign in body.lower() for sign in _DATA_UNAVAILABLE_ERROR_SIGNS):
                matched = body
            elif any(sign in text.lower() for sign in _DATA_UNAVAILABLE_ERROR_SIGNS):
                matched = text
        innermost = err
        err = err.__cause__ or err.__context__
    if matched:
        return _snip(matched)
    reason = str(innermost).strip() or type(innermost).__name__
    return _snip(reason.splitlines()[0])


_DATA_UNAVAILABLE_VERDICT_RE = re.compile(
    r"^\s*(?:\*\*)?" + DATA_UNAVAILABLE_MARKER + r"(?:\*\*)?\s*[:\-—]\s*(.+)",
    re.MULTILINE)


def _data_unavailable_verdict(debug_reply):
    """The reason the debugger gave when it declined to "fix" the code
    because the data does not exist for this request, or None.

    The debug prompt tells the model to answer with a line starting
    ``DATA_UNAVAILABLE: <reason>`` and NO code block in that case. A reply
    that carries both the marker and a code block is treated as a fix, not a
    verdict: the model hedged, and running its code is the cheaper way to
    find out which half it meant."""
    reply = str(debug_reply or "")
    if "```" in reply:
        return None
    m = _DATA_UNAVAILABLE_VERDICT_RE.search(reply)
    if not m:
        return None
    reason = m.group(1).strip()[:400]
    if reason and _marker_reason_is_schema_guess(f"{DATA_UNAVAILABLE_MARKER}: {reason}"):
        # Same demotion as _looks_like_data_unavailable_error: a verdict
        # that blames a missing field/column is the debugger guessing the
        # schema, not the provider refusing the request.
        logging.warning(
            "Debugger's DATA_UNAVAILABLE verdict not honoured — its reason "
            f"is about the schema, not coverage: {reason}")
        return None
    return reason or None


_RATE_LIMIT_ERROR_SIGNS = (
    'too many requests', 'rate limit', 'rate-limited', 'rate limited',
    '429 client error', 'http error 429', 'error 429',
    'quota exceeded', 'throttled', 'throttling',
)

_RATE_LIMIT_HTTP_STATUS = (429,)


def _looks_like_rate_limit_error(err):
    """Decide whether ``err`` is a transient throttling failure rather than a
    code bug. Unlike a credential error, this is not fatal and unlike a
    generic bug it does not need an LLM to "fix" the code — the code was
    never wrong, so the right move is to wait and retry the SAME code
    instead of spending an LLM debug round-trip on it."""
    seen = set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))

        if _http_status_of(err) in _RATE_LIMIT_HTTP_STATUS:
            return True

        text = f"{type(err).__name__}: {err}".lower()
        resp = getattr(err, "response", None)
        body = getattr(resp, "text", None)
        if isinstance(body, str):
            text += " " + body.lower()
        if any(sign in text for sign in _RATE_LIMIT_ERROR_SIGNS):
            return True

        err = err.__cause__ or err.__context__
    return False


def _text_has_credential_error_signs(text):
    """Text-only counterpart of ``_looks_like_credential_error`` for the
    Claude Agent SDK trial path, where failures surface as the agent's own
    prose report rather than a Python exception object -- there is no
    exception chain or ``.response`` body to walk, only the phrases
    themselves."""
    return any(sign in str(text or "").lower() for sign in _CREDENTIAL_ERROR_SIGNS)


def _install_response_json_body_patch():
    """Make ``requests.Response.json()`` include the response body in its error
    when the body isn't valid JSON.

    Why: some providers (e.g. the US Census API) reject a bad key with HTTP 200
    and an HTML "Invalid Key" page instead of JSON. ``raise_for_status()`` then
    passes, and ``response.json()`` raises a bare ``JSONDecodeError`` ("Expecting
    value: line 1 ...") that has no ``.response`` attached — so the real reason
    is lost before the credential detector or the debugger ever sees it. By
    appending the (truncated) body to the error message, the credential signs
    ("invalid key", etc.) become visible to ``_looks_like_credential_error``.

    The body is only appended when parsing already failed, so working downloads
    are unaffected. Guarded so re-imports don't stack the wrapper."""
    if getattr(requests.models.Response.json, "_agm_body_patch", False):
        return
    _orig_json = requests.models.Response.json

    def _json_with_body(self, **kwargs):
        try:
            return _orig_json(self, **kwargs)
        except ValueError as e:  # JSONDecodeError subclasses ValueError
            snippet = (self.text or "")[:500]
            raise ValueError(
                f"{e} | Response body was not valid JSON "
                f"(HTTP {self.status_code}): {snippet}"
            ) from e

    _json_with_body._agm_body_patch = True
    requests.models.Response.json = _json_with_body


_install_response_json_body_patch()


def parse_llm_object(reply):
    """Parse an LLM reply that is supposed to be a single JSON/dict object.

    LLMs intermittently ignore the "no ```json fences" instruction, and a
    bare ``ast.literal_eval`` chokes on the fence (SyntaxError) as well as on
    JSON ``true``/``false``/``null`` tokens. This normalizes the common
    failure shapes before parsing:

      1. Strip ```json / ``` markdown fences.
      2. Slice from the first ``{`` to the last ``}`` so any prose around the
         object is discarded.
      3. Try ``json.loads`` first (handles true/false/null), then fall back to
         ``ast.literal_eval`` (handles the single-quoted Python dict the
         prompt example shows).

    Returns the parsed object, or raises ValueError if nothing parses.
    """
    if not isinstance(reply, str):
        raise ValueError(f"expected a string reply, got {type(reply).__name__}")

    cleaned = reply.strip()
    # Strip markdown code fences, e.g. ```json\n{...}\n```
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    # Discard any prose surrounding the object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    return ast.literal_eval(cleaned)


def format_data_request_for_display(data_request):
    """Render a ``rq_breakdown['data_requests']`` entry for display in a
    workflow card or log line.

    Entries are sometimes a plain sentence and sometimes a structured dict
    (``{name, source, temporal_scope, geographic_scope, ...}``) depending on
    how the task-breakdown model phrased them. ``str()``-ing the dict form
    produces Python's single-quoted repr (e.g. ``"{'name': '...', 'source':
    '...'}"``), which reads as a leaked internal object rather than a
    description a person wrote. This keeps plain strings untouched and turns
    the dict form back into a sentence.
    """
    if not isinstance(data_request, dict):
        return str(data_request)
    name = str(data_request.get("name") or "").strip()
    if name:
        source = str(data_request.get("source") or "").strip()
        if source and source.lower() not in name.lower():
            return f"{name} (source: {source})"
        return name
    # No 'name' field — fall back to a readable "Label: value" list rather
    # than Python's dict repr.
    parts = [
        f"{str(k).replace('_', ' ').strip().capitalize()}: {v}"
        for k, v in data_request.items() if v not in (None, "")
    ]
    return "; ".join(parts) if parts else str(data_request)


# Formats worth opening to count records / read a CRS. Anything else is
# reported by name and size only.
_TABULAR_SUMMARY_EXTS = (".csv", ".xlsx", ".xls")
_VECTOR_SUMMARY_EXTS = (".geojson", ".json", ".gpkg", ".shp", ".zip", ".kml", ".gml")
_RASTER_SUMMARY_EXTS = (".tif", ".tiff", ".nc", ".img", ".vrt")


# Which captured writes actually constitute the delivered data.
#
# FileWriteTracker patches open()/to_file()/to_csv(), so it captures EVERY file
# a generated script writes -- including its own scratch. A Penn State boundary
# download was observed writing twelve Overpass tile dumps and a one-row index
# CSV, failing to build a single polygon, and still being reported as
# "success" with thirteen "downloaded files", because the only test applied was
# "did any file appear on disk".
#
# A run has delivered something only when at least one captured file is a
# dataset. Scratch stays visible as a supporting file -- it is useful evidence
# when diagnosing a failure -- but it can no longer stand in for the data.

# Extensions that are a dataset by their nature. Deliberately generous: a .zip
# or .nc that this process cannot open is still what the user asked for, and
# calling a real download a failure is worse than the reverse.
_DELIVERED_DATASET_EXTS = (
    ".geojson", ".geojsonl", ".gpkg", ".shp", ".fgb", ".gpx", ".kml", ".gml",
    ".dxf", ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".jsonl", ".ndjson",
    ".tif", ".tiff", ".nc", ".img", ".vrt", ".asc", ".dem", ".bil",
    ".grib", ".grib2", ".h5", ".hdf", ".hdf5", ".las", ".laz",
    ".sqlite", ".mbtiles", ".zip", ".gz", ".bz2", ".tar", ".7z",
)

# ".json" is genuinely ambiguous -- GeoJSON is routinely saved under it, and so
# is a raw API dump. Decided by content rather than by name.
_AMBIGUOUS_DATASET_EXTS = (".json",)


def _json_looks_like_data(path):
    """True when a .json file is GeoJSON or a non-trivial record collection."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    if '"FeatureCollection"' in head or '"Feature"' in head:
        return True
    # A bare list of records is data; an object keyed by API-specific fields
    # (Overpass "elements", a status envelope) is not, on its own.
    return head.lstrip()[:1] == "["


def classify_delivered_files(paths):
    """Split captured writes into (datasets, supporting).

    ``datasets`` is what the request actually delivered; ``supporting`` is
    everything else the script happened to write. Both are reported, but only
    a non-empty ``datasets`` means the download succeeded.
    """
    datasets, supporting = [], []
    for path in sorted(paths or []):
        ext = os.path.splitext(path)[1].lower()
        if ext in _DELIVERED_DATASET_EXTS:
            datasets.append(path)
        elif ext in _AMBIGUOUS_DATASET_EXTS and _json_looks_like_data(path):
            datasets.append(path)
        else:
            supporting.append(path)
    # An archive the script downloaded and then extracted is the wrapper, not
    # a second copy of the result: tl_2023_us_county.zip next to
    # tl_2023_us_county.shp is one dataset, and listing both put an 80 MB
    # zip in the Data Layers panel. Demote a zip whose stem matches another
    # delivered file.
    stems = {os.path.splitext(os.path.basename(p))[0].lower()
             for p in datasets if not p.lower().endswith(".zip")}
    wrappers = [p for p in datasets if p.lower().endswith(".zip")
                and os.path.splitext(os.path.basename(p))[0].lower() in stems]
    if wrappers:
        datasets = [p for p in datasets if p not in wrappers]
        supporting = sorted(supporting + wrappers)
    return datasets, supporting


def no_dataset_error(supporting):
    """The message shown when a run wrote only scratch."""
    names = ", ".join(os.path.basename(p) for p in supporting[:4])
    more = f" (+{len(supporting) - 4} more)" if len(supporting) > 4 else ""
    return (
        "The code ran but produced no dataset — only intermediate files "
        f"({names}{more}). This usually means the source returned records the "
        "script could not turn into the requested output (for example an "
        "Overpass relation whose geometry sits under members[].geometry, not "
        "at the top level). Re-run with feedback, or edit the code."
    )


def summarize_downloaded_files(paths):
    """Describe what a data download actually returned, for human review.

    Returns ``{"files": [...], "total_records": int, "all_empty": bool,
    "warnings": [str]}``. Each file entry carries whatever could be read
    cheaply: record count, column names, CRS, bounding box, size.

    This exists because "the download succeeded" is not the same as "the
    download returned the data you asked for". A header-only CSV writes a
    file and exits 0 — the workflow then computes a confident answer from
    nothing (observed live: an empty hospitals CSV produced "0 earthquakes
    near hospitals"). Surfacing counts/extent lets a human catch that before
    the analysis runs.

    Never raises: any file that cannot be inspected is reported with a note
    rather than failing the review.
    """
    summary = {"files": [], "total_records": 0, "all_empty": False,
               "warnings": []}
    real_paths = [p for p in (paths or [])
                  if p and os.path.exists(p) and os.path.isfile(p)]
    if not real_paths:
        summary["warnings"].append("No files were produced by this download.")
        summary["all_empty"] = True
        return summary

    countable = 0          # files whose records we could actually count
    countable_empty = 0
    for path in real_paths:
        ext = os.path.splitext(path)[1].lower()
        entry = {
            "filename": os.path.basename(path),
            "path": path,
            "format": ext.lstrip(".") or "unknown",
            "size_bytes": 0,
            "records": None,     # None = not countable for this format
            "columns": [],
            "crs": "",
            "bbox": None,
            "note": "",
        }
        try:
            entry["size_bytes"] = os.path.getsize(path)
        except OSError:
            pass

        try:
            if ext in _TABULAR_SUMMARY_EXTS:
                import pandas as pd
                df = (pd.read_csv(path, low_memory=False) if ext == ".csv"
                      else pd.read_excel(path))
                entry["records"] = int(len(df))
                entry["columns"] = [str(c) for c in df.columns][:40]
            elif ext in _VECTOR_SUMMARY_EXTS:
                import geopandas as _gpd
                gdf = _gpd.read_file(path)
                entry["records"] = int(len(gdf))
                entry["columns"] = [str(c) for c in gdf.columns
                                    if c != gdf.geometry.name][:40]
                if gdf.crs is not None:
                    epsg = gdf.crs.to_epsg()
                    entry["crs"] = f"EPSG:{epsg}" if epsg else str(gdf.crs)
                if len(gdf) and not gdf.geometry.is_empty.all():
                    b = gdf.total_bounds  # minx, miny, maxx, maxy
                    entry["bbox"] = [round(float(v), 6) for v in b]
            elif ext in _RASTER_SUMMARY_EXTS:
                import rasterio
                with rasterio.open(path) as src:
                    entry["crs"] = str(src.crs) if src.crs else ""
                    entry["note"] = (f"raster {src.width}x{src.height}, "
                                     f"{src.count} band(s)")
                    b = src.bounds
                    entry["bbox"] = [round(float(v), 6) for v in
                                     (b.left, b.bottom, b.right, b.top)]
            else:
                entry["note"] = "Format not inspected."
        except Exception as e:
            entry["note"] = f"Could not inspect: {e}"

        if entry["records"] is not None:
            countable += 1
            summary["total_records"] += entry["records"]
            if entry["records"] == 0:
                countable_empty += 1
                summary["warnings"].append(
                    f"{entry['filename']} contains 0 records "
                    f"(the file was written but holds no data).")
        summary["files"].append(entry)

    # "all empty" only when we could actually count something and all of it
    # was empty — an uninspectable format must never be reported as empty.
    summary["all_empty"] = bool(countable) and countable_empty == countable
    return summary


class FileWriteTracker:
    """Context manager that intercepts common file-write operations during
    code execution so we can tell exactly which files were written — even if
    they overwrite existing files or land outside our expected output dir.

    Wraps ``builtins.open`` (write modes), ``GeoDataFrame.to_file``, common
    pandas save methods, ``xarray`` ``to_netcdf``/``to_zarr``, and
    ``rasterio.open`` (write mode). On exit all originals are restored.
    """

    _WRITE_MODE_CHARS = ("w", "a", "x", "+")

    def __init__(self):
        self.captured_paths = set()
        self._patches = []  # list of (owner, attr, original)

    def _record(self, path):
        if path is None:
            return
        try:
            abspath = os.path.abspath(os.fspath(path))
        except Exception:
            return
        self.captured_paths.add(abspath)

    def reset(self):
        """Discard paths captured so far. Called between execute/debug retries
        so a failed attempt's partial/orphaned writes aren't reported as the
        final downloaded files once a later attempt succeeds (or the loop
        gives up) — without this, ``captured_paths`` accumulates across every
        retry inside the same ``with FileWriteTracker()`` block."""
        self.captured_paths = set()

    def _patch(self, owner, attr, wrapper_factory):
        try:
            original = getattr(owner, attr)
        except AttributeError:
            return
        try:
            setattr(owner, attr, wrapper_factory(original))
            self._patches.append((owner, attr, original))
        except Exception:
            pass

    def __enter__(self):
        tracker = self

        # 1. builtins.open — catches raw file writes incl. urllib/requests streams
        import builtins
        def _open_wrapper(_orig):
            def tracking_open(file, mode="r", *args, **kwargs):
                if isinstance(mode, str) and any(c in mode for c in tracker._WRITE_MODE_CHARS):
                    tracker._record(file)
                return _orig(file, mode, *args, **kwargs)
            return tracking_open
        self._patch(builtins, "open", _open_wrapper)

        # 2. geopandas.GeoDataFrame.to_file
        try:
            import geopandas as _gpd
            def _gdf_wrapper(_orig):
                def tracking_to_file(self_gdf, filename, *args, **kwargs):
                    tracker._record(filename)
                    return _orig(self_gdf, filename, *args, **kwargs)
                return tracking_to_file
            self._patch(_gpd.GeoDataFrame, "to_file", _gdf_wrapper)
        except Exception:
            pass

        # 3. pandas DataFrame save methods (first positional arg is the path)
        try:
            import pandas as _pd
            for _method in ("to_csv", "to_parquet", "to_excel", "to_json", "to_feather", "to_pickle", "to_hdf"):
                def _df_wrapper(_orig):
                    def tracking_save(self_df, path_or_buf=None, *args, **kwargs):
                        if isinstance(path_or_buf, (str, os.PathLike)):
                            tracker._record(path_or_buf)
                        return _orig(self_df, path_or_buf, *args, **kwargs)
                    return tracking_save
                self._patch(_pd.DataFrame, _method, _df_wrapper)
        except Exception:
            pass

        # 4. xarray Dataset/DataArray save methods
        try:
            import xarray as _xr
            for _cls_name in ("Dataset", "DataArray"):
                _cls = getattr(_xr, _cls_name, None)
                if _cls is None:
                    continue
                for _method in ("to_netcdf", "to_zarr"):
                    def _xr_wrapper(_orig):
                        def tracking_save(self_obj, path=None, *args, **kwargs):
                            if isinstance(path, (str, os.PathLike)):
                                tracker._record(path)
                            return _orig(self_obj, path, *args, **kwargs)
                        return tracking_save
                    self._patch(_cls, _method, _xr_wrapper)
        except Exception:
            pass

        # 5. rasterio.open (write/append modes)
        try:
            import rasterio as _rio
            def _rio_wrapper(_orig):
                def tracking_rio_open(fp, mode="r", *args, **kwargs):
                    if isinstance(mode, str) and any(c in mode for c in ("w", "a")):
                        tracker._record(fp)
                    return _orig(fp, mode, *args, **kwargs)
                return tracking_rio_open
            self._patch(_rio, "open", _rio_wrapper)
        except Exception:
            pass

        # 6. shutil file movers — grabs paths even when code downloads via a
        #    temp file and then moves/copies it into its final location.
        import shutil as _shutil
        for _method in ("copy", "copy2", "copyfile", "move"):
            def _shutil_wrapper(_orig):
                def tracking_shutil(src, dst, *args, **kwargs):
                    tracker._record(dst)
                    return _orig(src, dst, *args, **kwargs)
                return tracking_shutil
            self._patch(_shutil, _method, _shutil_wrapper)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore all patches in reverse order
        for owner, attr, original in reversed(self._patches):
            try:
                setattr(owner, attr, original)
            except Exception:
                pass
        self._patches = []
        return False


class HttpRequestTracker:
    """Record every HTTP request the generated download code makes (evaluation
    mode only). Because the code is exec'd in-process, patching the client
    libraries here sees all of its traffic — no proxy needed.

    Wraps ``requests.sessions.Session.request`` (which every requests call —
    get/post/Session — funnels through, covering osmnx, census wrappers, etc.)
    and ``urllib.request.OpenerDirector.open`` (urlopen fallback). Each record
    is ``{attempt, method, url, time}`` where ``attempt`` is the execute/debug
    trial that issued it — evaluation levels L1/L2 are scored on the requests
    of the final attempt. Same patch/restore pattern as FileWriteTracker."""

    def __init__(self):
        self.captured_requests = []
        self.attempt = 0
        self._patches = []  # list of (owner, attr, original)

    def set_attempt(self, n):
        """Tag subsequent requests with the execute-trial number ``n``."""
        self.attempt = n

    def _record(self, method, url):
        try:
            url = str(url)
        except Exception:
            return
        self.captured_requests.append({
            "attempt": self.attempt,
            "method": str(method or "GET").upper(),
            "url": url,
            "time": time.time(),
        })

    def _patch(self, owner, attr, wrapper_factory):
        try:
            original = getattr(owner, attr)
        except AttributeError:
            return
        try:
            setattr(owner, attr, wrapper_factory(original))
            self._patches.append((owner, attr, original))
        except Exception:
            pass

    def __enter__(self):
        tracker = self

        # 1. requests — Session.request is the single funnel for get/post/etc.
        try:
            import requests as _requests
            def _req_wrapper(_orig):
                def tracking_request(self_session, method, url, *args, **kwargs):
                    tracker._record(method, url)
                    return _orig(self_session, method, url, *args, **kwargs)
                return tracking_request
            self._patch(_requests.sessions.Session, "request", _req_wrapper)
        except Exception:
            pass

        # 2. urllib — OpenerDirector.open backs urllib.request.urlopen.
        try:
            import urllib.request as _urllib_request
            def _url_wrapper(_orig):
                def tracking_open(self_opener, fullurl, *args, **kwargs):
                    # fullurl is a str or a urllib.request.Request.
                    url = getattr(fullurl, "full_url", fullurl)
                    method = getattr(fullurl, "method", None) or "GET"
                    tracker._record(method, url)
                    return _orig(self_opener, fullurl, *args, **kwargs)
                return tracking_open
            self._patch(_urllib_request.OpenerDirector, "open", _url_wrapper)
        except Exception:
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for owner, attr, original in reversed(self._patches):
            try:
                setattr(owner, attr, original)
            except Exception:
                pass
        self._patches.clear()
        return False


class DataRetrieverAgent:
    def __init__(self, api_key=None, extra_handbook_dirs=None, evaluation=False,
                 base_url=None, model=None, reasoning_effort=None,
                 provider="openai", anthropic_api_key=None):
        self.provider = provider or "openai"
        if self.provider == "claude":
            self.anthropic_api_key = (
                anthropic_api_key or claude_agent_provider.load_anthropic_key())
            if not self.anthropic_api_key:
                raise ValueError(
                    "Anthropic API key must be provided or set as an "
                    "environment variable.")
            # Not required on this path, but kept populated (best-effort)
            # for any code that still reads self.api_key unconditionally.
            self.api_key = api_key or helper.load_OpenAI_key()
        else:
            self.anthropic_api_key = None
            self.api_key = api_key or helper.load_OpenAI_key()
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided or set as an environment variable.")
        # When set, routes every LLM call below to this OpenAI-compatible
        # server instead of GIBD/OpenAI (used by Experiment 2's open-source
        # model controls).
        self.base_url = base_url
        # Evaluation mode — see the EVALUATION switch at the top of this file.
        # ON if any of: the module-level EVALUATION constant, this constructor
        # arg, or the AGSI_EVALUATION env var.
        self.evaluation = EVALUATION or bool(evaluation) or (
            str(os.environ.get("AGSI_EVALUATION", "")).strip().lower()
            in ("1", "true", "yes"))
        self.eval_record = None
        # Report of the most recent execute_complete_program call:
        # {attempts_used, success, timed_out, traceback}. Always maintained
        # (cheap), primarily consumed in evaluation mode (L3 + repair count).
        self.last_execution_report = None
        # Caller-selected data-retrieval model (drives code generation and
        # debugging for the actual retrieval trial) -- defaults to the
        # platform's long-standing default so existing callers that don't
        # pass one see no behavior change.
        self.model = model or (
            "claude-sonnet-5" if self.provider == "claude"
            else DATA_RETRIEVAL_MODEL)
        # Only meaningful for models that support it (gpt-5.x); None/"" is
        # correctly a no-op everywhere it's used below. "max" is a real
        # level for some models (e.g. gpt-5.6-sol) but only on the
        # Responses API -- every LLM call in this class goes through Chat
        # Completions (helper.client_chat_completion_stream), which 400s on
        # "max", so downgrade once here to the next-highest tier it does
        # support rather than fail at every call site.
        self.reasoning_effort = "xhigh" if reasoning_effort == "max" else reasoning_effort
        self.handbook_dir = os.path.join(BASE_DIR, "DataRetriever_Handbooks", "Handbooks")
        self.keys_dir = os.path.join(BASE_DIR, "DataRetriever_Handbooks", "Keys")
        # Additional handbook directories searched alongside the curated global
        # catalog — used to expose a user's private, contributed sources to
        # their own tasks only. The global catalog always takes precedence on a
        # name collision, so a user source can never shadow a curated one.
        self.extra_handbook_dirs = [
            d for d in (extra_handbook_dirs or []) if d and os.path.isdir(d)
        ]
        # self.output_dir = os.path.join(BASE_DIR, "..", "..", "outputs", "DataRetrieverOutput")
        self.output_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "outputs", "DataRetrieverOutput"))
        os.makedirs(self.output_dir, exist_ok=True)
        # self.data_request= ctx["artifacts"]["rq_breakdown"]["data_request_description"]
        # self.client = OpenAI(api_key=self.api_key)
        # self.data_request = data_request # Temporary use


    def LLM_Find(self, data_request, stream_callback=None, interactive_callback=None, user_keys=None, synthesis_stream_callback=None):
        # synthesis_stream_callback: optional separate stream for need-driven
        # skill synthesis ("Generate & Use New Skill") — the Web UI passes one
        # tagged with its own step id so the synthesis progress renders in its
        # OWN workflow card instead of the data-request card. Falls back to
        # stream_callback when not provided.
        # Effective data-source keys: start from any keys the user has already
        # provided (from the Web UI), and we may add more if the confirmation
        # prompt collects missing ones.
        effective_user_keys = dict(user_keys or {})

        # Evaluation mode: everything the benchmark scorer needs, filled in as
        # the run progresses and left on self.eval_record — populated on every
        # exit path, so the harness can read it however this call returns.
        eval_rec = None
        if self.evaluation:
            eval_rec = {
                "data_request": str(data_request),
                "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "llm_selected_source": None,   # the LLM's finalized pick
                "llm_ideal_source": None,      # its unconstrained suggestion
                "confidence": None,
                "final_source": None,          # what the run actually used (L1)
                "http_requests": [],           # every request, tagged by attempt (L1/L2)
                "execution": {},               # attempts/success/traceback (L3)
                "downloaded_files": [],        # written + surviving files (L4 input)
            }
            self.eval_record = eval_rec

        def _stream(text):
            if stream_callback:
                stream_callback(text)

        def _warn(msg):
            logging.warning(msg)
        #     _stream(f"\n⚠️ **Warning:** {msg}\n")

        # _stream("\n\n### Selecting data source...\n\n")
        logging.info("Starting data retrieval process, AI is selecting data source...")
        source_select_prompt_str = self.create_select_prompt(task=data_request)
        select_source_reply = self.select_source(select_prompt_str=source_select_prompt_str, stream_callback=stream_callback)
        logging.info(f"LLM's reply for data source selection: {select_source_reply} ")

        # Guard against upstream LLM failure — select_source returns None when
        # the GIBD proxy (or its OpenAI fallback) errors out. Without this
        # check the next line would crash with an opaque
        # `ValueError: malformed node or string` from ast.literal_eval(None).
        if not select_source_reply:
            err_msg = (
                "Data source selection failed: the LLM returned no response. "
                "This usually means the GIBD API key is invalid/expired or "
                "the GIBD proxy is unreachable. Check the server logs for the "
                "underlying error."
            )
            logging.error(err_msg)
            _warn(err_msg)
            raise RuntimeError(err_msg)

        try:
            select_source = parse_llm_object(select_source_reply)
        except (ValueError, SyntaxError) as parse_err:
            err_msg = (
                f"Data source selection failed: LLM reply could not be parsed "
                f"as a JSON/Python object ({parse_err}). Raw reply: "
                f"{select_source_reply[:300]!r}"
            )
            logging.error(err_msg)
            _warn(err_msg)
            raise RuntimeError(err_msg) from parse_err

        if not isinstance(select_source, dict) or "Selected data source" not in select_source:
            err_msg = (
                f"Data source selection failed: LLM reply did not contain the "
                f"expected 'Selected data source' key. Parsed reply: {select_source!r}"
            )
            logging.error(err_msg)
            _warn(err_msg)
            raise RuntimeError(err_msg)

        selected_data_source= select_source['Selected data source']
        # New structured fields requested from the prompt — tolerate missing keys
        data_to_download = select_source.get('Data to download', '') or ''
        # Two-step reasoning fields: ideal source (what the LLM thinks is best
        # in a perfect world) vs. selected source (what it settled on from the
        # available handbooks).
        ideal_source = (select_source.get('Ideal source') or '').strip()
        ideal_source_reasoning = (
            select_source.get('Ideal source reasoning') or ''
        ).strip()
        try:
            confidence = int(select_source.get('Confidence', 0) or 0)
        except (TypeError, ValueError):
            confidence = 0
        logging.info(
            f"LLM selected data source: {selected_data_source} "
            f"(ideal={ideal_source!r}, confidence={confidence}, "
            f"data='{data_to_download}')"
        )
        if eval_rec is not None:
            eval_rec["llm_selected_source"] = selected_data_source
            eval_rec["llm_ideal_source"] = ideal_source
            eval_rec["confidence"] = confidence
        if ideal_source:
            _stream(
                f"\n\n**Ideal source (LLM suggestion):** {ideal_source}\n"
            )
        _stream(
            f"\n**Selected data source (finalized):** {selected_data_source} "
            f"(confidence: {confidence}/10)\n"
            f"**Data to download:** {data_to_download}\n"
        )


        # Retrieve the Handbook file for the selected data source
        handbook_files = self.collect_handbook_files(source_dir=self.handbook_dir)
        descriptions_str, data_source_dict = self.assemble_handbook_description(handbook_files)
        # Tolerant lookup: an exact dict hit first, then a normalized match.
        # The model very often echoes the catalog's own name with a trailing
        # full stop, which an exact lookup misses — the source then shows as
        # "not recognized" even though its handbook is right there, and the
        # UI hides Proceed because that button is gated on recognition.
        selected_data_source, selected_data_source_ID = self.resolve_source_name(
            selected_data_source, data_source_dict)
        is_recognized = selected_data_source_ID != "Unknown"
        if not is_recognized:
            _warn(f"Selected data source '{selected_data_source}' is not recognized. No handbook file found.")

        # ── Interactive confirmation (before generating download code) ──
        # Always ask the user to confirm the LLM's choice, not only on Unknown.
        # The user can Proceed, change the source, Retry, or Abort. They can
        # also click "Always Proceed" to skip this prompt for remaining
        # requests in this run.
        # Which API keys does the chosen source still need? (handbook
        # placeholders not satisfied by the bundled .keys files nor by keys the
        # user already provided.) These are surfaced in the confirmation prompt
        # so the user can supply them just-in-time.
        def _compute_missing_keys(source_id):
            if not source_id or source_id == "Unknown":
                return []
            required = self.get_required_key_names(source_id, keys_dir=self.keys_dir)
            file_keys = {}
            kf = os.path.join(self.keys_dir, f"{source_id}.keys")
            if os.path.exists(kf):
                try:
                    file_keys = self.load_keys(source_id, self.keys_dir)
                except Exception:
                    file_keys = {}
            missing = []
            for name in required:
                if self.key_value_is_set(effective_user_keys.get(name)):
                    continue
                if self.key_value_is_set(file_keys.get(name)):
                    continue
                missing.append(name)
            return missing

        # Resolve the "apply for a key" URL for a given key name from the
        # source's optional [Links] section (per-key override → website → blank).
        def _key_url(source_id, key_name, _links_cache={}):
            if source_id not in _links_cache:
                _links_cache[source_id] = self.load_key_links(source_id, self.keys_dir)
            links = _links_cache[source_id]
            return (links.get(key_name) or links.get('website')
                    or links.get('signup_url') or '')

        if interactive_callback is not None:
            available_sources = [k for k in data_source_dict.keys() if k != "Unknown"]
            missing_keys = _compute_missing_keys(selected_data_source_ID)
            # _stream(
            #     "\n\n⏸ **Waiting for user confirmation of the selected data "
            #     "source...**\n"
            # )
            try:
                user_response = interactive_callback({
                    "data_request": format_data_request_for_display(data_request),
                    "llm_choice": selected_data_source,
                    "llm_explanation": select_source.get("Explanation", ""),
                    "ideal_source": ideal_source,
                    "ideal_source_reasoning": ideal_source_reasoning,
                    "data_to_download": data_to_download,
                    "confidence": confidence,
                    "is_recognized": is_recognized,
                    "available_sources": available_sources,
                    "handbook_descriptions": descriptions_str,
                    "required_keys": [
                        {"name": n, "url": _key_url(selected_data_source_ID, n)}
                        for n in missing_keys
                    ],
                    "caveats": self.load_source_caveats(selected_data_source_ID),
                })
            except Exception as cb_err:
                _warn(f"Interactive source-select callback failed: {cb_err}")
                user_response = None

            # Merge any data-source keys the user entered in the prompt so they
            # are applied when the handbook is assembled below.
            if isinstance(user_response, dict):
                entered_keys = user_response.get("data_source_keys") or {}
                if isinstance(entered_keys, dict):
                    for k, v in entered_keys.items():
                        if v:
                            effective_user_keys[k] = v

            # None/empty response → proceed if recognized, else fail
            if not user_response or not isinstance(user_response, dict):
                if not is_recognized:
                    return None
                # fall through with LLM's original choice
            else:
                action = (user_response.get("action") or "").lower()

                if action == "abort":
                    _stream("\n\n❌ **User aborted this data request.**\n")
                    return {
                        "status": "failed",
                        "error": "User aborted the data request.",
                    }

                if action == "retry":
                    _stream("\n\n🔄 **Retrying with LLM source selection...**\n")
                    return None  # outer loop will retry LLM_Find

                if action == "use":
                    chosen = (user_response.get("source") or "").strip()
                    if chosen and chosen in data_source_dict and chosen != "Unknown":
                        selected_data_source = chosen
                        selected_data_source_ID = data_source_dict[chosen]["ID"]
                        is_recognized = True
                        _stream(
                            f"\n\n✓ **Using user-selected data source:** "
                            f"{selected_data_source}\n"
                        )
                    else:
                        _warn(
                            f"User-selected source '{chosen}' is not in the "
                            f"handbook. Aborting this data request."
                        )
                        return {
                            "status": "failed",
                            "error": f"Unknown user-selected source: {chosen}",
                        }
                elif action in ("proceed", "always_proceed"):
                    if not is_recognized:
                        # Autonomous mode returns "proceed" without a human at
                        # the dropdown, so an unrecognized source would dead-end
                        # here. Instead, auto-substitute the best AVAILABLE
                        # handbook source — the same choice a person makes from
                        # the picker — so the run can continue.
                        substitute = self.select_available_substitute(
                            data_request=data_request,
                            available_sources=available_sources,
                            descriptions_str=descriptions_str,
                        )
                        # Same tolerant lookup — the substitute is another LLM
                        # echo of a catalog name and picks up the same stray
                        # punctuation.
                        sub_name, sub_id = self.resolve_source_name(
                            substitute, data_source_dict)
                        if substitute and sub_id != "Unknown":
                            selected_data_source = sub_name
                            selected_data_source_ID = sub_id
                            is_recognized = True
                            _stream(
                                f"\n\n↩ **Auto-substituted unavailable source "
                                f"with closest available handbook:** "
                                f"{selected_data_source}\n"
                            )
                            # The substituted source may need its own API key(s)
                            # the original (Unknown) source didn't. Warn — we
                            # can't re-prompt here in an autonomous run.
                            sub_missing = _compute_missing_keys(selected_data_source_ID)
                            if sub_missing:
                                _warn(
                                    f"Substituted source '{selected_data_source}' "
                                    f"needs API key(s) not provided: "
                                    f"{', '.join(sub_missing)}. Download may fail "
                                    f"without them."
                                )
                        else:
                            _warn(
                                "Source not recognized and no available "
                                "substitute could be selected. Aborting this "
                                "request."
                            )
                            return {
                                "status": "failed",
                                "error": "Cannot proceed: source not recognized.",
                            }
                    else:
                        _stream(
                            f"\n\n✓ **Proceeding with selected (finalized) data "
                            f"source:** {selected_data_source}\n"
                        )
                    # 'always_proceed' is handled inside the workflow callback,
                    # which sets a closure flag to skip future prompts.
                elif action == "generate":
                    # ── Need-driven autonomous synthesis ──────────────────
                    # No suitable handbook exists for the data this request
                    # needs, so author one on the spot: generate + self-verify a
                    # new data-source skill and save it to the user's private
                    # library. Progress streams to its OWN card (the synthesis
                    # callback), and the user must CONFIRM before the download
                    # continues with the new skill. Looped so "generate again"
                    # from the confirmation card starts another round.
                    synth_stream = synthesis_stream_callback or stream_callback
                    gen_query = ((user_response.get("generate_query") or "").strip()
                                 or ideal_source or selected_data_source)
                    gen_website = (user_response.get("generate_website") or "").strip()
                    while True:
                        result = self.synthesize_source_skill(
                            query=gen_query, website=gen_website,
                            stream_callback=synth_stream,
                        )
                        if not result:
                            return {
                                "status": "failed",
                                "error": "Need-driven knowledge generation failed; "
                                         "review the logs and try again.",
                            }
                        # Re-discover handbooks so the freshly written skill is
                        # selectable, then switch this run to it.
                        handbook_files = self.collect_handbook_files(source_dir=self.handbook_dir)
                        descriptions_str, data_source_dict = self.assemble_handbook_description(handbook_files)
                        available_sources = [k for k in data_source_dict.keys()
                                             if k != "Unknown"]
                        selected_data_source = result["name"]
                        selected_data_source_ID = result["id"]
                        is_recognized = True

                        # ── Pause: ask the user to confirm before the download
                        # continues with the freshly generated skill (it may
                        # also need an API key they haven't provided yet).
                        verification = result.get("verification") or {}
                        confirm_missing = _compute_missing_keys(selected_data_source_ID)
                        _stream(
                            f"\n\n✓ **New data-source knowledge ready:** "
                            f"{selected_data_source}\n"
                            "\n⏸ **Waiting for user confirmation to proceed "
                            "with the new knowledge...**\n"
                        )
                        try:
                            confirm = interactive_callback({
                                "data_request": format_data_request_for_display(data_request),
                                "llm_choice": selected_data_source,
                                "llm_explanation": "",
                                "data_to_download": data_to_download,
                                "confidence": confidence,
                                "is_recognized": True,
                                "available_sources": available_sources,
                                "handbook_descriptions": descriptions_str,
                                "required_keys": [
                                    {"name": n,
                                     "url": _key_url(selected_data_source_ID, n)}
                                    for n in confirm_missing
                                ],
                                "caveats": self.load_source_caveats(selected_data_source_ID),
                                "synthesized": True,
                                "synthesized_status": verification.get("status", ""),
                                "synthesized_note": verification.get("note", ""),
                            })
                        except Exception as cb_err:
                            _warn(f"Knowledge-confirmation callback failed: {cb_err}")
                            confirm = None

                        if not confirm or not isinstance(confirm, dict):
                            break  # no answer possible → proceed with the skill
                        entered = confirm.get("data_source_keys") or {}
                        if isinstance(entered, dict):
                            for k, v in entered.items():
                                if v:
                                    effective_user_keys[k] = v
                        c_action = (confirm.get("action") or "").lower()
                        if c_action == "abort":
                            _stream("\n\n❌ **User aborted this data request.**\n")
                            return {"status": "failed",
                                    "error": "User aborted the data request."}
                        if c_action == "retry":
                            _stream("\n\n🔄 **Retrying with LLM source selection...**\n")
                            return None  # outer loop will retry LLM_Find
                        if c_action == "use":
                            chosen = (confirm.get("source") or "").strip()
                            if (chosen and chosen in data_source_dict
                                    and chosen != "Unknown"):
                                selected_data_source = chosen
                                selected_data_source_ID = data_source_dict[chosen]["ID"]
                                _stream(
                                    f"\n\n✓ **Using user-selected data source:** "
                                    f"{selected_data_source}\n"
                                )
                            else:
                                _warn(
                                    f"User-selected source '{chosen}' is not in "
                                    f"the handbook. Continuing with the "
                                    f"generated knowledge instead."
                                )
                            break
                        if c_action == "generate":
                            # Another synthesis round with the (possibly
                            # edited) description.
                            gen_query = ((confirm.get("generate_query") or "").strip()
                                         or gen_query)
                            gen_website = ((confirm.get("generate_website") or "").strip()
                                           or gen_website)
                            continue
                        # proceed / always_proceed / anything else → continue.
                        # If verification was skipped for lack of a key and the
                        # user just supplied one, verify the skill with it FIRST
                        # (auto-fixing on failure) instead of downloading blind.
                        if entered and verification.get("status") != "passed":
                            self._verify_skill_with_keys(
                                selected_data_source_ID, effective_user_keys,
                                _stream, _warn)
                        _stream(
                            f"\n\n✓ **Proceeding with newly generated data "
                            f"source:** {selected_data_source}\n"
                        )
                        break
                else:
                    _warn(f"Unknown action from user: '{action}'. Proceeding with caution.")
                    if not is_recognized:
                        return None

        elif not is_recognized:
            # No interactive callback and source not recognized → legacy behaviour
            return None

        # The source is final here (after any user overrides / substitution /
        # synthesis) — this is what evaluation L1 compares against the request
        # log's queried domains.
        if eval_rec is not None:
            eval_rec["final_source"] = {"name": selected_data_source,
                                        "id": selected_data_source_ID,
                                        "is_recognized": is_recognized}

        handbook_str = self.collect_a_handbook(source_ID=selected_data_source_ID, source_dir=self.handbook_dir, keys_dir=self.keys_dir, user_keys=effective_user_keys)
        # Check if handbook was successfully loaded
        if handbook_str is None:
            _warn(f"Could not load handbook for data source '{selected_data_source}'. No handbook file found or error in loading.")
            handbook_str = ""  # Set to empty string to continue the flow
        else:
            logging.info(f"Successfully loaded handbook for data source '{selected_data_source}'.")


        # GENERATE & EXECUTE — with an interactive credential-retry loop.
        # If the download fails because the data-source API key is invalid, we
        # re-show the confirmation card as a "check your API key" warning, let
        # the user correct or confirm the key, then regenerate + re-execute with
        # it. Capped so confirming the same wrong key can't loop forever. When
        # there's no UI to prompt (autonomous run), we surface the failure as
        # before.
        _MAX_CRED_RETRIES = 3
        cred_attempts = 0
        cred_error = None
        tracker = None
        research_attempted = False
        refine_attempts = 0
        # A handbook revision that has not yet earned persistence. While one is
        # pending it is the handbook in force for this request, but it stays in
        # memory until a retry proves it (see _refine_skill_from_failure).
        pending_refined_book = None

        def _handbook_in_force():
            """The handbook text and worked example the next attempt should
            use: a pending in-memory revision when one is under test, else
            whatever is saved on disk.

            Every path that rebuilds the download prompt goes through here. A
            path that re-read the .toml directly would quietly run the OLD
            handbook while a revision was pending, and the revision would then
            be credited or blamed for a run it took no part in.
            """
            if pending_refined_book is not None:
                keys = self.effective_keys(selected_data_source_ID,
                                           self.keys_dir, effective_user_keys)
                return (self.render_handbook_text(pending_refined_book, keys),
                        self.render_code_example(pending_refined_book, keys))
            return (self.collect_a_handbook(
                        source_ID=selected_data_source_ID,
                        source_dir=self.handbook_dir, keys_dir=self.keys_dir,
                        user_keys=effective_user_keys) or "",
                    self.collect_code_example(
                        selected_data_source_ID, source_dir=self.handbook_dir,
                        keys_dir=self.keys_dir, user_keys=effective_user_keys))

        while True:
            # GENERATE THE DATA FETCHING PROGRAM
            _stream("\n\n### Generating data fetching code...\n\n")
            handbook_str, code_example_str = _handbook_in_force()
            _, _exported = self._credential_env_plan(
                selected_data_source_ID, effective_user_keys)
            download_prompt_str = self.create_download_prompt(
                data_request, selected_data_source, handbook_str,
                code_example=code_example_str,
                credential_env_names=sorted(_exported))

            data_fetching_code_str = self.generate_data_fetching_code(download_prompt_str=download_prompt_str, stream_callback=stream_callback)
            code = self.extract_code_from_str(data_fetching_code_str)

            # EXECUTE THE CODE
            _stream("\n\n### Executing data download code...\n\n")
            code = code.replace('area({osm_id})->.searchArea;', 'relation({osm_id}); map_to_area->.searchArea;')  # GPT-4o never follow the related instruction!

            # Wrap the exec in a tracker that intercepts common file-write calls
            # (open, GeoDataFrame.to_file, pandas/xarray savers, rasterio.open,
            # shutil). This catches files regardless of where they land on disk
            # and works correctly even when the target path already existed
            # before execution (overwrites).
            cred_error = None
            unavailable_error = None
            # Evaluation mode also logs every HTTP request the code makes,
            # tagged with the execute trial that issued it (L1/L2 scoring).
            http_tracker = HttpRequestTracker() if self.evaluation else None

            def _early_triage(failure_note, failing_code):
                """Let the execute loop ask "is this fixable at all?" once,
                mid-budget, with the same classifier the post-loop ladder
                uses (so the two never disagree on vocabulary)."""
                return self._triage_download_failure(
                    selected_data_source_ID, data_request, failing_code,
                    failure_note,
                    (http_tracker.captured_requests if http_tracker else []),
                    _stream, _warn)

            with FileWriteTracker() as tracker, \
                    (http_tracker or contextlib.nullcontext()), \
                    self._source_keys_in_env(selected_data_source_ID,
                                             effective_user_keys, code) as cred_names:
                try:
                    code = self.execute_complete_program(code=code, try_cnt=DOWNLOAD_TRIALS, task=data_request, model_name=self.model, handbook_str=handbook_str, stream_callback=stream_callback, http_tracker=http_tracker, file_tracker=tracker, attempt_timeout=DOWNLOAD_ATTEMPT_TIMEOUT, triage=_early_triage, credential_names=cred_names)
                except CredentialError as ce:
                    cred_error = ce
                except DataUnavailableError as due:
                    unavailable_error = due
            if eval_rec is not None:
                if http_tracker:
                    eval_rec["http_requests"].extend(http_tracker.captured_requests)
                # Requests accumulate across credential retries; the execution
                # report reflects the latest generate+execute cycle.
                eval_rec["execution"] = dict(self.last_execution_report or {})

            if unavailable_error is not None:
                # The source has nothing for this request. Every rung above
                # this point — handbook revision, re-research, and the
                # caller's whole-request retry — exists to fix wrong
                # KNOWLEDGE about a source, and none of it can conjure data
                # the provider does not hold. The only useful next step is
                # the user's: change the request, or the source. A pending
                # handbook revision is dropped, not blamed: it never got a
                # request it could have satisfied.
                pending_refined_book = None
                _warn(f"Data unavailable for this request: {unavailable_error}")
                return {
                    "status": "failed",
                    "error": (f"The data source has no data for this request: "
                              f"{unavailable_error}"),
                    "data_unavailable": True,
                    "code": code,
                    "source_name": selected_data_source,
                    "source_id": selected_data_source_ID,
                }

            if cred_error is None:
                exec_ok = bool((self.last_execution_report or {}).get("success"))
                produced_files = [p for p in tracker.captured_paths
                                  if os.path.exists(p) and os.path.isfile(p)]
                produced = bool(produced_files)
                # A structurally successful run can still deliver nothing: a
                # header-only CSV or a zero-feature GeoJSON "succeeds" (code
                # ran, file written) while containing no records — observed
                # live when a mid-debug source pivot queried a facilities
                # dataset with filter values that matched nothing. Treat an
                # all-empty result as a failed attempt ONCE (so re-research
                # gets a shot with that evidence); after re-research, accept
                # whatever comes back — a genuinely empty result (e.g. no
                # earthquakes in the window) must terminate, not loop.
                empty_note = ""
                if exec_ok and produced and not research_attempted and not any(
                        self._file_has_records(p) for p in produced_files):
                    empty_note = (
                        "the download completed without errors but the "
                        "produced file(s) contained ZERO records (header-only "
                        "CSV / empty feature set) — likely wrong filter "
                        "values, the wrong endpoint for this kind of query, "
                        "or a source that does not cover the requested "
                        "area or feature type")
                    _warn("Download produced only empty file(s) — treating as "
                          "a failed attempt and repairing the source knowledge.")
                elif (exec_ok and produced) or research_attempted:
                    # Delivered. A revision that got us here has now proven
                    # itself on a real download, which is the only evidence
                    # that justifies writing it over a handbook other tasks
                    # may be relying on.
                    if exec_ok and produced and pending_refined_book is not None:
                        self._persist_refined_skill(
                            selected_data_source_ID, pending_refined_book,
                            _stream, _warn)
                        pending_refined_book = None
                    break  # success — or already re-researched once

                # Non-credential failure with the code-repair budget spent.
                # Escalation from here is expensive, so it goes cheapest-first
                # and only when the failure is the kind an escalation can fix:
                #   triage → revise this handbook → re-research from live docs.
                _tb = str((self.last_execution_report or {}).get("traceback") or "")
                failure_note = (empty_note or _tb[-600:].strip())
                # The execute loop may already have classified this exact
                # error mid-budget (see execute_complete_program's ``triage``).
                # Reuse that verdict rather than paying for a second one —
                # but only when the final error is the same one it judged;
                # a later fix that changed the error deserves a fresh look.
                early = (self.last_execution_report or {}).get("triage") or {}
                final_sig = (_tb.strip().splitlines() or [""])[-1].strip()
                if (not empty_note and early.get("analysis")
                        and early.get("error_signature") == final_sig):
                    analysis = dict(early["analysis"])
                else:
                    analysis = self._triage_download_failure(
                        selected_data_source_ID, data_request, code, failure_note,
                        (http_tracker.captured_requests if http_tracker else []),
                        _stream, _warn)
                _category = str(analysis.get("failure_category") or "").strip().lower()
                if _category == "request_infeasible":
                    # Same stop as DataUnavailableError above, reached via the
                    # classifier instead of the error text.
                    pending_refined_book = None
                    reason = (str(analysis.get("summary") or "").strip()
                              or (failure_note.splitlines() or [""])[-1].strip()
                              or "the request lies outside what this source covers")
                    _stream("\n\n⛔ **Stopping** — the source cannot satisfy "
                            "this request as asked, so revising the knowledge "
                            "would not help. Adjust the date range, area, or "
                            "variable, or choose another source.\n")
                    return {
                        "status": "failed",
                        "error": (f"The data source has no data for this "
                                  f"request: {reason}"),
                        "data_unavailable": True,
                        "code": code,
                        "source_name": selected_data_source,
                        "source_id": selected_data_source_ID,
                    }
                if _category == "external":
                    # Nothing a handbook change can fix. Regenerating the skill
                    # here would spend minutes to rediscover a source that was
                    # already documented correctly, and could replace a working
                    # handbook with one written against a service that happens
                    # to be down right now.
                    _stream("\n\n⛔ **Stopping the repair attempts** — this "
                            "failure is external, so revising the knowledge would "
                            "not help.\n")
                    break

                # Rung 1 — revise the handbook we have. Cheaper than
                # re-research, keeps what already works, and available without
                # an sk- key. Only tried while the handbook is the suspect: a
                # revision cannot fix having picked the wrong source, which is
                # what rung 2 is for.
                if refine_attempts < HANDBOOK_REFINE_ATTEMPTS:
                    refine_attempts += 1
                    revised = self._refine_skill_from_failure(
                        selected_data_source_ID, data_request, analysis,
                        failure_note, exec_ok, _stream, _warn)
                    if revised is not None:
                        pending_refined_book = revised
                        _stream("\n\n🔄 **Retrying the download with the "
                                "revised knowledge...**\n")
                        continue

                # Rung 2 — the handbook could not be repaired in place, so
                # re-research the source's skill from its live docs (needs an
                # sk- key) and retry with that. Hand it the final traceback (or
                # the empty-result evidence) so the regeneration's
                # source-suggestion stage knows what failed (see
                # _re_research_skill docstring). A pending revision is dropped
                # rather than saved: it never delivered data, and re-research
                # replaces the handbook wholesale anyway.
                pending_refined_book = None
                refreshed = self._re_research_skill(
                    selected_data_source, selected_data_source_ID,
                    str(data_request), effective_user_keys, _stream, _warn,
                    failure_note=failure_note)
                research_attempted = True
                if not refreshed:
                    break  # no key / research failed — report failure below
                selected_data_source, selected_data_source_ID = refreshed
                if eval_rec is not None:
                    eval_rec["final_source"] = {
                        "name": selected_data_source,
                        "id": selected_data_source_ID,
                        "is_recognized": True}
                _stream("\n\n🔄 **Retrying the download with the re-researched "
                        "knowledge...**\n")
                continue

            cred_attempts += 1
            # No UI to re-prompt (autonomous run) or retries exhausted → surface
            # the credential failure so the caller can report it.
            if interactive_callback is None or cred_attempts > _MAX_CRED_RETRIES:
                return {
                    "status": "failed",
                    "error": str(cred_error),
                    "credential_error": True,
                    "code": code,
                }

            # Re-show the confirmation card as a credential warning, pre-filling
            # the key(s) the source needs so the user can edit or confirm them.
            # A key the code asked for that the handbook never declared is
            # included too — the code is the ground truth for what it reads.
            _cred_names = list(self.get_required_key_names(
                selected_data_source_ID, keys_dir=self.keys_dir))
            for n in getattr(cred_error, "missing_keys", None) or []:
                if n not in _cred_names:
                    _cred_names.append(n)
            cred_required_keys = [
                {"name": n, "url": _key_url(selected_data_source_ID, n),
                 "value": effective_user_keys.get(n, "")}
                for n in _cred_names
            ]
            if getattr(cred_error, "missing_keys", None):
                _stream("\n\n🔑 **This data source needs an API key. Please "
                        "enter it to continue.**\n")
            else:
                _stream("\n\n⚠️ **Please check your API key for this data source.**\n")
            try:
                cred_response = interactive_callback({
                    "data_request": format_data_request_for_display(data_request),
                    "llm_choice": selected_data_source,
                    "data_to_download": data_to_download,
                    "confidence": confidence,
                    "is_recognized": is_recognized,
                    "available_sources": available_sources,
                    "required_keys": cred_required_keys,
                    "credential_error": True,
                    # Absent vs rejected: the card words its header differently.
                    "credential_missing": bool(
                        getattr(cred_error, "missing_keys", None)),
                    "credential_message": str(cred_error),
                })
            except Exception as cb_err:
                _warn(f"Credential re-prompt callback failed: {cb_err}")
                cred_response = None

            if not isinstance(cred_response, dict):
                return {
                    "status": "failed",
                    "error": str(cred_error),
                    "credential_error": True,
                    "code": code,
                }

            cred_action = (cred_response.get("action") or "").lower()
            if cred_action == "abort":
                _stream("\n\n❌ **User aborted this data request.**\n")
                return {"status": "failed", "error": "User aborted the data request."}
            if cred_action == "skip":
                return {
                    "status": "failed",
                    "error": str(cred_error),
                    "credential_error": True,
                    "code": code,
                }

            # Apply the corrected/confirmed key(s), then loop to regenerate +
            # re-execute. The top of the loop rebuilds the handbook through
            # _handbook_in_force(), so the new key value is injected there --
            # and re-reading the .toml here instead would drop a revision that
            # is still under test.
            entered_keys = cred_response.get("data_source_keys") or {}
            if isinstance(entered_keys, dict):
                for k, v in entered_keys.items():
                    if v:
                        effective_user_keys[k] = v
            _stream("\n\n🔄 **Retrying the download with the updated API key...**\n")

        code = code.replace('area({osm_id})->.searchArea;', 'relation({osm_id}); map_to_area->.searchArea;')  # GPT-4o never follow the related instruction!

        # Keep only paths that actually exist on disk and are regular files
        # (filters out directories, temp handles, and any partial writes that
        # were rolled back).
        new_files = {
            p for p in tracker.captured_paths
            if os.path.exists(p) and os.path.isfile(p)
        }

        if not new_files:
            _warn("No new files were downloaded after executing the code.")
            return {
                "status": "failed",
                "error": "No files were downloaded.",
                "code": code
            }

        # Writing a file is not the same as delivering data, and the attempt
        # loop above breaks out on `research_attempted` even when the code
        # never ran clean -- so a request whose every attempt raised could
        # reach here with nothing but scratch on disk and be reported as a
        # successful download of N files. Both conditions are re-checked.
        _report = self.last_execution_report or {}
        # Only a report that exists and says failure counts against the run;
        # a path that never populated one must not be failed on that basis.
        _exec_failed = bool(_report) and not _report.get("success")
        _datasets, _supporting = classify_delivered_files(new_files)
        if _exec_failed or not _datasets:
            if _exec_failed:
                _warn("Every execution attempt failed; the files on disk are "
                      "scratch from the failed attempts, not the requested data.")
                _err = (str(_report.get("traceback") or "").strip().splitlines()
                        or ["The download code did not run to completion."])[-1]
            else:
                _warn("The code wrote only intermediate files; no dataset was produced.")
                _err = no_dataset_error(_supporting)
            return {
                "status": "failed",
                "error": _err,
                "code": code,
                # Still surfaced: the scratch is the evidence for diagnosing why.
                "downloaded_files": sorted(_supporting or new_files),
                "source_name": selected_data_source,
                "source_id": selected_data_source_ID,
            }

        if eval_rec is not None:
            eval_rec["downloaded_files"] = sorted(new_files)

        logging.info(f"Captured {len(new_files)} file(s) written during execution.")

        # REPROJECTING THE OUTPUT FILE TO EPSG:4326
        # Only reproject vector formats that GeoPandas/Fiona can read and
        # write. Non-spatial (CSV, JSON), raster (TIF/NC) and archive (.zip)
        # files are passed through untouched — the condition used to be
        # inverted, which caused CSVs to be fed into gpd.read_file.
        VECTOR_DRIVERS = {
            '.geojson': 'GeoJSON',
            '.gpkg':    'GPKG',
            '.shp':     'ESRI Shapefile',
            '.kml':     'KML',
            '.gml':     'GML',
        }
        reprojected_files = []
        for f in new_files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VECTOR_DRIVERS:
                try:
                    gdf = gpd.read_file(f)
                    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                        gdf = gdf.to_crs("EPSG:4326")
                        gdf.to_file(f, driver=VECTOR_DRIVERS[ext])
                except Exception as e:
                    _warn(f"Could not reproject {f}: {e}")
            reprojected_files.append(f)
            logging.info(f"Data downloaded and saved to: '{f}'")
        return {
            "status": "success",
            "downloaded_files": list(reprojected_files),
            "code": code,
            # Carried so a post-download re-run can target the SAME source
            # without re-running selection (see rerun_download).
            "source_name": selected_data_source,
            "source_id": selected_data_source_ID,
    }

    def rerun_download(self, data_request, source_ID, source_name=None,
                       user_keys=None, edited_code=None, feedback=None,
                       stream_callback=None):
        """Re-run ONE data download against an already-selected source.

        Entry point for the post-download human review — "Re-run", "Edit code
        & re-run", "Provide feedback & re-run". Unlike ``LLM_Find`` this never
        re-selects the source and never re-prompts for credentials, so a human
        correcting a single request cannot silently change which source the
        run used (which would invalidate the provenance the run reports).

        ``edited_code``: execute exactly this code, skipping generation.
        ``feedback``:    regenerate the code with this instruction appended.
        Neither: regenerate the code as-is (a plain retry).

        Returns the same shape as ``LLM_Find``.
        """
        def _stream(text):
            if stream_callback:
                stream_callback(text)

        handbook_str = self.collect_a_handbook(
            source_ID=source_ID, source_dir=self.handbook_dir,
            keys_dir=self.keys_dir, user_keys=dict(user_keys or {}),
        ) or ""

        if edited_code and edited_code.strip():
            # Accept either a bare script or one still wrapped in a ``` fence.
            code = (self.extract_code_from_str(edited_code)
                    or edited_code.strip())
            _stream("\n\n### Re-running your edited download code...\n\n")
        else:
            task_for_prompt = str(data_request)
            if feedback and feedback.strip():
                task_for_prompt += (
                    f"\n\n[User feedback on the previous attempt — apply it: "
                    f"{feedback.strip()}]")
                _stream("\n\n### Regenerating the download code with your "
                        "feedback...\n\n")
            else:
                _stream("\n\n### Regenerating the download code...\n\n")
            _, _exported = self._credential_env_plan(
                source_ID, dict(user_keys or {}))
            prompt = self.create_download_prompt(
                task_for_prompt, source_name or source_ID, handbook_str,
                code_example=self.collect_code_example(
                    source_ID, source_dir=self.handbook_dir,
                    keys_dir=self.keys_dir, user_keys=dict(user_keys or {})),
                credential_env_names=sorted(_exported))
            reply = self.generate_data_fetching_code(
                download_prompt_str=prompt, stream_callback=stream_callback)
            code = self.extract_code_from_str(reply)

        if not code or not code.strip():
            return {"status": "failed",
                    "error": "No runnable code was produced for the re-run."}

        _stream("\n\n### Executing data download code...\n\n")
        try:
            with FileWriteTracker() as tracker, \
                    self._source_keys_in_env(source_ID, dict(user_keys or {}),
                                             code) as cred_names:
                code = self.execute_complete_program(
                    code=code, try_cnt=DOWNLOAD_TRIALS, task=data_request,
                    model_name=self.model, handbook_str=handbook_str,
                    stream_callback=stream_callback, file_tracker=tracker,
                    attempt_timeout=DOWNLOAD_ATTEMPT_TIMEOUT,
                    credential_names=cred_names)
        except CredentialError as ce:
            # Surfaced verbatim: a re-run has no confirmation card to correct
            # a key on, and retrying the same bad credential never helps.
            return {"status": "failed", "error": str(ce),
                    "credential_error": True, "code": code}
        except DataUnavailableError as due:
            return {"status": "failed",
                    "error": (f"The data source has no data for this request: "
                              f"{due}"),
                    "data_unavailable": True, "code": code}
        except Exception as e:
            return {"status": "failed", "error": str(e), "code": code}

        new_files = sorted(
            p for p in tracker.captured_paths
            if os.path.exists(p) and os.path.isfile(p)
        )
        if not new_files:
            return {"status": "failed",
                    "error": "The re-run produced no files.", "code": code}

        # Same rule as the first pass: scratch is not a delivered dataset.
        _datasets, _supporting = classify_delivered_files(new_files)
        if not _datasets:
            return {"status": "failed",
                    "error": no_dataset_error(_supporting),
                    "code": code,
                    "downloaded_files": sorted(_supporting)}

        # Same EPSG:4326 normalization the first-pass download applies, so a
        # re-run's output stays map-renderable.
        VECTOR_DRIVERS = {
            '.geojson': 'GeoJSON', '.gpkg': 'GPKG', '.shp': 'ESRI Shapefile',
            '.kml': 'KML', '.gml': 'GML',
        }
        for f in new_files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in VECTOR_DRIVERS:
                continue
            try:
                gdf = gpd.read_file(f)
                if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                    gdf.to_crs("EPSG:4326").to_file(f, driver=VECTOR_DRIVERS[ext])
            except Exception as e:
                logging.warning(f"Could not reproject {f}: {e}")

        return {"status": "success", "downloaded_files": new_files,
                "code": code}


##=========================MAIN FUNCTIONS====================================================
    def create_select_prompt(self, task):
        select_role = r'''A professional Python programmer in geographic information science (GIScience). You have worked on GIScience for more than 20 years and know every detail and pitfall when collecting data and coding. You know which websites you can get suitable spatial data and know the methods or tricks to download data, such as OpenStreetMap, Census Bureau, or various APIs. You are also experienced in processing the downloaded data, including saving them in suitable formats, map projections, and creating detailed and useful meta-data.
        '''
        select_task_prefix = """reason about the ideal data source for the requested geo-spatial task, and then finalize a selection from the given list of available data sources"""
        selection_reply_example = """
        {'Ideal source': 'US Census Bureau TIGER/Line', 'Ideal source reasoning': 'For authoritative US state boundaries, Census Bureau TIGER/Line is the canonical source — it is maintained by the federal government and is used in official statistics.', 'Explanation': "TIGER/Line would be ideal, but it is not in the available handbooks. OpenStreetMap is available and also provides complete US state admin boundary polygons via its admin_level=4 relations, so it is the best available fallback.", 'Selected data source': 'OpenStreetMap', 'Data to download': 'US state administrative boundary polygons (all 50 states)', 'Confidence': 8}
        """
        select_requirements = [
            "First, think about the IDEAL data source for this task — what you would pick if you had unrestricted access to any real-world geospatial data source (e.g., Census Bureau TIGER/Line, USGS, NASA EarthData, OpenStreetMap, HydroSHEDS, Copernicus, etc.). Put that in the 'Ideal source' key, and give the reasoning for why it is ideal in 'Ideal source reasoning'.",
            "Then, look at the AVAILABLE data sources listed below and finalize your selection. The 'Selected data source' MUST be the exact name of a source from the given list (or 'Unknown' if none fit).",
            "In the 'Explanation' key, explicitly compare the ideal source to the available sources: if the ideal source is in the available list, say so; if not, explain why the selected available source is an acceptable substitute (or why it is not, in which case use 'Unknown').",
            "If a data source is given in the task, e.g., OpenStreetMap or Census Bureau, you need to select that given data source.",
            "If you need to download the administrative boundary of a place without mentioning the data sources, you can get data from OpenStreetMap.",
            "If you need to download the US Census tract and block group boundaries, download them from Census Bureau.",
            "Follow the given JSON format.",
            "DO NOT make fake data source names. If no available source is a reasonable substitute for the ideal, return 'Unknown' for 'Selected data source'. DO NOT use ```json and ```.",
            "Include a 'Data to download' key in the reply describing, in one sentence, what specific data will be downloaded (dataset name, variables, spatial/temporal extent).",
            "Include a 'Confidence' key in the reply with an integer from 1 to 10 indicating how confident you are that the SELECTED available source can actually fulfil the request. 10 = the ideal source is directly available; lower values reflect fallback distance from the ideal. 1 = pure guess.",
        ]


        select_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(select_requirements)])
        handbook_files = self.collect_handbook_files(source_dir=self.handbook_dir) # NEW CHANGE
        descriptions_str, data_source_dict = self.assemble_handbook_description(handbook_files) #NEW CHANGE
        prompt = f"Your role: {select_role} \n" + \
                 f"Your mission: {select_task_prefix}: " + f"{task}\n\n" + \
                 f"Requirements: \n{select_requirement_str} \n\n" + \
                 f"Data sources:{descriptions_str} \n" + \
                 f'Your reply example: {selection_reply_example}'
        return prompt
    
    
    

    def create_download_prompt(self, task, selected_data_source, handbook_str,
                               code_example="", credential_env_names=None):
        # select_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(constants.select_requirements)])
        from datetime import datetime as _datetime_cls
        current_datetime = _datetime_cls.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")
        
        download_role = r'''A professional Python programmer in geographic information science (GIScience). You have worked on GIScience for more than 20 years and know every detail and pitfall when collecting data and coding. You know which websites you can get suitable spatial data and know the methods or tricks to download data, such as OpenStreetMap, Census Bureau, or various APIs. You are also experienced in processing the downloaded data, including saving them in suitable formats, map projections, and creating detailed and useful meta-data. When downloading geo-spatial data, the technical handbook for a particular data source is provided; you can follow it, and write Python code carefully to download the data. 
        '''

        # Control arm: drop what the prompt says about handbooks rather than
        # leaving the model told that one "is provided" when none is. Only the
        # handbook clause goes -- "write Python code carefully to download the
        # data" is an instruction about coding, so the control keeps it and the
        # two arms still differ by the handbook alone.
        has_handbook = bool((handbook_str or "").strip())
        if not has_handbook:
            download_role = download_role.replace(
                _HANDBOOK_ROLE_CLAUSE, _NO_HANDBOOK_ROLE_CLAUSE)

        download_task_prefix = r'download geo-spatial data from the given data source for this task'

        download_reply_example = """
        ```python
        import geopandas as gpd
        import osmnx as ox
        def download_data():
            # data downloading code 
            # downloaded code 
        download_data()
        ```
        """

        """
        1. Think step by step.
        2. If you need to download the administrative boundary of a place and without mentioning the data sources, you can get data from OSM using OSM package by `ox.geocode_to_gdf(query, which_result=None, by_osmid=False, buffer_dist=None)`. This method is fast. 
        3.If the place of boundaries request is in the USA, you can download boundaries from Census Bureau, which is official and better than OSM. An example link is: https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_{year}_{extend}_{level}_500k.zip. You can change the year and administrative level (state/county) in link accordingly. "year" is 4-digit. 'extend' can be 'us' or 2-digit state FIPS; when 'extend' = 'us', 'level' can be 'state' and 'county' only, and the downloaded data is national. When 'extend' is 2-digit state FIPS, 'level' can be 'tract' and 'bg' only. 'bg' refers to block groups. E.g., do not set 'extend' to 2-digit FIPS code when download county boundaries for a state. If you need to download counties boundaries, 'extend' must be 'us'.
        4. If the mentioned the saving file format, save the downloaded data in GeoPackage format. 
        5. You need to create Python code to download and save the data. Another program will execute your code directly.
        6. You can use various technical ways to download the data, such as Overpass QL, Overpass API, OSMnx Python package, Census file downloading link, or Census Python packages.
        7. Put your reply into a Python code block, Explanation or conversation can be Python comments at the begining of the code block(enclosed by ```python and ```).
        8. The download code is only in a function named 'download_data()'. The last line is to execute this function.
        9. When downloading OSM data, no need to use 'building' tags if it is not asked for.
        10. If using GeoPandas to load a zipped ESRI shapefile from a URL, the correct method is "gpd.read_file(URL)". DO NOT download and unzip the file.
        11. Note Python package 'pandas' has no attribute or method of 'StringIO'.,
        12. If a data source is given in the task, e.g., OSM or Census Bureau, you need to download data from that data source.

        """
        

        # Requirements that MUST reach the model. The block above (the loose
        # docstring) was never wired into the prompt; these are. The HTTP-status
        # rule is what lets an invalid API key surface as a real exception
        # (instead of the provider silently returning a 401/403 body that gets
        # written to disk), so the credential-error detector can catch it.
        download_requirements = [
            "After EVERY HTTP request (requests.get/post, urllib, etc.), immediately check the response status and raise on failure — e.g. call `response.raise_for_status()` for the `requests` library — BEFORE using or saving the response. Never write the response body to a file without first confirming the request succeeded. This ensures an invalid API key (HTTP 401/403) fails loudly instead of saving an error page. On any non-2xx status, PRINT the status code and the first ~500 characters of the response body BEFORE raising — the body often states the real reason (e.g. an invalid parameter, or the provider saying the endpoint/feature is disabled or unavailable), and that reason is lost forever if the code raises without ever having read/logged it.",
            f"Request EXACTLY what the task asks for — its dates, area, variables, and product — never a 'safer' substitute (a shorter or shifted date range, a neighbouring region, a different variable) chosen to make the call succeed. If the provider's response says it has no data for the requested period, area, or variable (e.g. the dates are outside the dataset's coverage, the region has no records, the product/release does not exist yet), raise `RuntimeError('{DATA_UNAVAILABLE_MARKER}: <the reason the provider gave, and which requested parameter is not covered>')` so the failure is reported to the user as a coverage problem instead of being debugged as a code bug. Do NOT raise this marker merely because a well-formed response came back with zero records — write that result out and print a clear message; only raise it when the provider or the documented coverage explicitly says the request cannot be served.",
            f"Never raise `{DATA_UNAVAILABLE_MARKER}` because of the dataset's SHAPE — a missing column/field/parameter name is not a coverage statement. Many datasets carry the requested period or variable implicitly: a release-year dataset (e.g. a '2025 release' built from 2023 data) has no year column because every row is that period; a wide-format table has no 'measure' column because each variable is its own column (e.g. `obesity_crudeprev`); a tiled or product-specific endpoint has no product parameter because the endpoint IS the product. When a probe row lacks a field named the way the task is phrased, read the actual field names (and any source guidance you were given) and pick the column(s) that carry the requested variable and the release/reference period that corresponds to the requested one; print the mapping you chose (e.g. 'year 2023 = 2025 release; obesity = obesity_crudeprev') as an explicit note and proceed. Raise the marker only when the provider's response body, its documented coverage, or a documented release list says the requested period/area/variable is not served — quote that statement in the reason.",
            "Put your reply into a single Python code block enclosed by ```python and ```. Any explanation must be Python comments at the beginning of the block.",
            "The download code must live in a function named 'download_data()', and the last line must call that function.",
            "Do NOT wrap the initial/setup request(s) in try/except — let a genuine bug fail loudly so it can be diagnosed. But when the code loops over many pages, batches, tiles, or items (pagination, chunked/batched downloads, per-feature fetches), wrap EACH iteration's request in its own narrow try/except: on failure, retry once or twice with a short backoff, and if it still fails, print/log which page/batch/item was skipped and why, then continue the loop — never let one bad page or batch abort an otherwise-successful bulk download, and never use a blanket try/except around the whole function to hide errors.",
            "Never require a local file, pre-downloaded dataset, or any input the code doesn't fetch or create itself at runtime — the same rule as credentials: no manual out-of-band setup. If a reference or boundary dataset is needed (e.g. an administrative boundary polygon for spatial filtering), fetch it at runtime via geocoding (e.g. `ox.geocode_to_gdf(place)`) or a real public API — never assume a local file already exists on disk.",
            "For geocoding a place name (resolving it to coordinates or a boundary), prefer `ox.geocode_to_gdf(place)` / `ox.geocode(place)` from the osmnx package over a hand-written request to nominatim.openstreetmap.org — osmnx already sends the descriptive User-Agent Nominatim's usage policy requires. If you must call Nominatim directly, set a real, descriptive `User-Agent` header (e.g. `{'User-Agent': 'your-app-name/1.0 (contact-or-purpose)'}`) on every request; an unauthenticated Nominatim call with no/generic User-Agent gets a 403 that has nothing to do with this source's own credentials.",
            "If the task asks for a SPECIFIC layer, feature class, table, or field, and the source only provides it bundled inside an archive (a zip, a multi-layer geodatabase, a multi-sheet file), downloading that archive is NOT sufficient — you must also extract/open it and read the specific requested layer or table. Print the name of the layer/table you opened and its row or feature count as evidence it was actually found and read, not just that the archive was downloaded.",
            "If the exact filename or resource path isn't given directly and must be discovered (e.g. parsing a directory listing or catalog page), do NOT guess or hard-code a filename, abbreviation, or suffix — parse the actual listing content and match it against the exact naming pattern documented in the handbook. Print the filenames actually discovered before selecting one. If nothing matches, that is a real failure: raise with the discovered filenames included in the message, never return silently with no output.",
            "Any download that could be large (a bulk archive, a full-resolution raster, or anything without a known-small size) must stream: use `requests.get(url, stream=True, timeout=(connect_timeout, read_timeout))` and write via `response.iter_content(chunk_size=...)` in a loop, never `response.content`/`.text` on the whole body at once, which buffers the entire file in memory before returning. The code runs under an overall execution budget of a few minutes — set the connect timeout short (~10-30s) and choose a read timeout that fails clearly within that budget rather than an arbitrarily long one (e.g. 900s) that the harness will never actually let complete; a request that cannot plausibly finish in the available time should fail fast and clearly, not hang.",
        ]
        # Only when the caller has actually exported credentials: the trial
        # path (and therefore the measured treatment prompt) passes none and
        # is unchanged. Naming the exact variables stops the model reading a
        # near-miss (NASA_FIRMS_KEY for FIRMS_MAP_KEY) that is not set.
        if credential_env_names:
            download_requirements.append(
                "The credential(s) this source needs are ALREADY exported as "
                "environment variable(s) named exactly: "
                + ", ".join(str(n) for n in credential_env_names)
                + ". Read them with os.environ[\"NAME\"] under exactly those "
                "names — never a different or shortened name, never a "
                "hard-coded or invented value, and never a keyless fallback "
                "endpoint.")
        download_requirement_str = '\n'.join(
            f"{idx + 1}. {line}" for idx, line in enumerate(download_requirements)
        )
        # One requirement points the model at "the naming pattern documented in
        # the handbook". With no handbook that is a dangling reference, so the
        # control is pointed at the source's own documentation instead. The
        # requirement itself -- parse the listing, don't guess a filename --
        # is unchanged; only where it says to look for the pattern.
        if not has_handbook:
            download_requirement_str = download_requirement_str.replace(
                _HANDBOOK_PATTERN_CLAUSE, _NO_HANDBOOK_PATTERN_CLAUSE)

        prompt =    f"Your role: {download_role} \n" + \
                    f"Your mission: {download_task_prefix}: " + f"{task}" + "And save the downloaded data in this output directory: " + f"{self.output_dir}\n\n" + \
                    f"Current date-time: {formatted_datetime} \n\n" + \
                    f"Requirements: \n{download_requirement_str} \n\n" + \
                    f"Data source:{selected_data_source} \n" + \
                    f'Your reply example: {download_reply_example}\n'
        # An empty heading is still a statement -- it tells the model a
        # handbook section exists and happens to be blank. The control gets no
        # heading at all.
        if has_handbook:
            prompt += f"Technical handbook: \n{handbook_str}"

        # The handbook's code_example is a script that was actually RUN and
        # verified against this source when the handbook was authored. Until
        # now it never reached this prompt — collect_a_handbook() returns only
        # the prose `handbook` field, and the example was injected solely if
        # that prose happened to contain a literal "{code_example}" token
        # (true for 0 of the handbooks in practice). So every download was
        # written from scratch from prose: the model re-derived endpoints and
        # parameters the example already had right, which is the main source
        # of both the sprawling generated scripts and the repeated 4xx
        # failures the debugger then cannot fix (it only ever sees the same
        # prose again).
        if code_example and code_example.strip():
            prompt += (
                "\n\nVERIFIED REFERENCE IMPLEMENTATION for this source — this "
                "script was executed and confirmed to work against this exact "
                "source when its handbook was written:\n"
                "```python\n" + code_example.strip() + "\n```\n"
                "ADAPT this reference to the task above rather than writing a "
                "new script from scratch. Its base URL, endpoint paths, "
                "parameter names, authentication, response parsing and "
                "pagination are known-correct for this source — reuse them "
                "verbatim and change only what the task actually requires "
                "(the area, time range, variables, filters and output path). "
                "Do NOT rebuild machinery the reference already solves — do "
                "not add catalog/dataset discovery, field-name guessing, or "
                "endpoint probing when the reference already names the "
                "correct dataset, fields and endpoint. Deviate from it only "
                "where the handbook rules or this task explicitly require it, "
                "and keep the result as short as the task allows."
            )

        return prompt
    
    
    def get_debug_prompt(self, exception, code, task, handbook_str,
                         credential_env_names=None):
        
        debug_role = r'''A professional geo-information scientist and programmer who is good at Python. You have worked on Geographic information science for over 20 years and know every detail and pitfall when processing spatial data and coding. You have significant experience in code debugging. You like to find out debugs and fix code. Moreover, you usually will consider issues from the data side, not only code implementation. Your current job is to debug the code for map generation.
        '''
        debug_task_prefix = r"You need to correct a program's code based on the given error information and then return the complete corrected code."

        debug_requirement = [
            'Think step by step. Elaborate your reasons for revision before returning the code. E.g., Explaination for the revision: xxxx \n The reivsed code is: ```pyhon xxxx  ```.',
            f"FIRST decide whether this is a code bug at all. If the error or the provider's response says the source has NO DATA for what the task asked — the requested dates are outside the dataset's period, the area/station/region has no records, the variable/product/layer is not offered, the release does not exist yet (phrases like 'no data available', 'out of range', 'outside the available date range', 'not available for this period', or a 400/404 whose body says the period, area, or product is not covered) — then it is NOT a code bug and you must NOT 'fix' it by changing the dates, bounding box, place, variables, product, or filters to something the task did not ask for; data the user did not request is worse than no data. In that case do not return any code: reply with a single line `{DATA_UNAVAILABLE_MARKER}: <what the provider said, and which requested parameter (period/area/variable) it cannot serve>` and nothing else — no code block. Only when the failure is genuinely in the code (wrong endpoint, wrong parameter name or format, parsing, a missing import, a typo, pagination, timeouts, ...) return the corrected program as described below.",
            f"A missing column/field/parameter name is a CODE problem, never a `{DATA_UNAVAILABLE_MARKER}` verdict — a 400 on a `$select`/`$where` naming a field that does not exist, or the code's own check failing to find a 'year'/'measure'/'variable' field, means the code guessed the schema wrong, not that the source lacks the data. Datasets often carry the requested period or variable implicitly (a release-year dataset has no year column; a wide-format table has one column per variable, e.g. `obesity_crudeprev`, and no 'measure' column). Fix it by probing one unfiltered row to read the real field names, mapping the requested period/variable onto the columns and release that actually exist, printing that mapping as a note, and proceeding — do NOT keep a self-imposed 'raise if no year field' check, and do NOT answer with the marker unless the provider's response or documented coverage explicitly says the period/area/variable is not served.",
            'Correct the code. Revise the buggy parts, but need to keep program structure, i.e., the function name, its arguments, and returns.',
            'You must return the entire corrected program in only one Python code block(enclosed by ```python and ```); DO NOT return the revised part only.',
            'If using GeoPandas to load a zipped ESRI shapefile from a URL, the correct method is "gpd.read_file(URL)". DO NOT download and unzip the file.',
            'Make necessary revisions only. Do not change the structure of the given code or program; keep all functions.',
            "Non-regression: only fix what the error actually requires. Do NOT remove or weaken existing resilience code (per-item retry/skip loops, extracted-layer checks) while fixing an unrelated bug elsewhere — preserve it. If the error is a raised exception coming from the code's OWN self-check on its output (e.g. a hard-coded correctness/verification assertion that isn't the actual bug), prefer loosening that self-check to a printed warning over leaving it as a hard failure — a self-imposed check should never be the reason a real, working retrieval aborts. Never introduce a NEW hard requirement the code didn't have before (a local file path, a pre-downloaded dataset, or any input the code doesn't fetch/create itself) to work around the error; a fix that trades a runnable script for one that needs manual setup is not a fix.",
            "If the error happened inside a loop over many pages, batches, tiles, or items (pagination, chunked/batched downloads, per-feature fetches), wrap that iteration's request in its own narrow try/except: retry once or twice with a short backoff, and if it still fails, print/log which page/batch/item was skipped and continue the loop — do not let one bad page or batch abort an otherwise-successful bulk download. Do not wrap the initial/setup request(s) this way; those should still fail loudly if genuinely broken.",
            "If the task asks for a SPECIFIC layer, feature class, table, or field and the source only provides it bundled inside an archive (a zip, a multi-layer geodatabase, a multi-sheet file), the code must extract/open that archive and read the specific requested layer, not just save the archive — print the layer/table name and its row or feature count as evidence it was actually found and read.",
            "If the error shows a discovered/selected filename that doesn't match anything, or a request to a guessed/hard-coded filename or abbreviation, do not just try another guess — parse the actual directory listing content and match against the exact naming pattern documented in the handbook, and print the filenames actually discovered so a wrong guess is diagnosable next time.",
            "If the error is a timeout (or a hang) on a download, switch to a streaming download (`requests.get(url, stream=True, timeout=(connect_timeout, read_timeout))` with `response.iter_content(chunk_size=...)`, not `response.content`/`.text` on the whole body at once) so the file isn't fully buffered in memory before returning. The code runs under an overall execution budget of a few minutes — use a short connect timeout (~10-30s) and a read timeout that fails clearly within that budget, not an arbitrarily long one the harness will never let complete.",
            "If the error is a 401/403 from nominatim.openstreetmap.org, this is NOT this source's credential being wrong — Nominatim is only used as a geocoding helper and requires a descriptive User-Agent header on direct requests. Fix it by switching to `ox.geocode_to_gdf(place)`/`ox.geocode(place)` (osmnx already sets a proper User-Agent), or by adding a real `User-Agent` header to the direct request.",
            "Note module 'pandas' has no attribute or method of 'StringIO'",
            "When doing spatial analysis, convert the involved spatial layers into the same map projection, if they are not in the same projection.",
            "DO NOT reproject or set spatial data(e.g., GeoPandas Dataframe) if only one layer involved.",
            "Map projection conversion is only conducted for spatial data layers such as GeoDataFrame. DataFrame loaded from a CSV file does not have map projection information.",
            "If join DataFrame and GeoDataFrame, using common columns, DO NOT convert DataFrame to GeoDataFrame.",
            "Remember the variable, column, and file names used in ancestor functions when using them, such as joining tables or calculating.",
            "You can use OSMnx Python package to download a city, neighborhood, borough, county, state, or country. The code is: `gdf = ox.geocode_to_gdf(place)`. The Overpass API `area['name'='target_placename']` might return empty results.",
            'If a Python package is not installed, add the install command such as "pip" at the beginning of the revised code.',
            "If using GeoPandas for spatial analysis, when doing overlay analysis, carefully think about use Geopandas.GeoSeries.intersects() or geopandas.sjoin(). ",
            "Geopandas.GeoSeries.intersects(other, align=True) returns a Series of dtype('bool') with value True for each aligned geometry that intersects others. other:GeoSeries or geometric object. ",
            "If using GeoPandas for spatial joining, the arguements are: geopandas.sjoin(left_df, right_df, how='inner', predicate='intersects', lsuffix='left', rsuffix='right', **kwargs), how: the type of join, default ‘inner’, means use intersection of keys from both dfs while retain only left_df geometry column. If 'how' is 'left': use keys from left_df; retain only left_df geometry column, and similarly when 'how' is 'right'. ",
            "Note geopandas.sjoin() returns all joined pairs, i.e., the return could be one-to-many. E.g., the intersection result of a polygon with two points inside it contains two rows; in each row, the polygon attribute is the same. If you need of extract the polygons intersecting with the points, please remember to remove the duplicated rows in the results.",
            "FIPS or GEOID columns may be str type with leading zeros (digits: state: 2, county: 5, tract: 11, block group: 12), or integer type without leading zeros. Thus, when joining using they, you can convert the integer colum to str type with leading zeros to ensure the success.",
            "If you use `ox.geocode_to_gdf(place_name)` to a place's boundary and get a type error of 'Nominatim could not geocode query place_name to a geometry of type (Multi)Polygon'; it is caused by a place name not in OpenStreetMap; you need to change the place name to address this error. E.g., using 'Penn State University' instead of 'Penn State University, State College, PA'.",
            "Carefully check whether the Overpass query is using `relation({osm_id}); map_to_area->.rel;` to get the filtering area. `area(osm_id)->.rel` is wrong",
            "NEVER using `area(osm_id)->.rel` to filter data in Overpass queries.",
            "You must replace 'area({osm_id})->.rel;' by 'relation({osm_id}); map_to_area->.rel;'. Only the latter is correct!",
            "If the error is a 504 Gateway Timeout or 429 Too Many Requests from the Overpass API, add `[timeout:600]` to the Overpass query header (e.g., `[out:json][timeout:600];`), use POST instead of GET (`requests.post(url, data={'data': query}, timeout=660)`), and retry with the fallback endpoint `https://overpass.kumi.systems/api/interpreter` if the primary endpoint fails.",
        ]
                
        # Same note as the download prompt, same condition: only when the
        # caller exported credentials. If the code failed reading a name that
        # is NOT in this list while one that is IS set, the fix is to read
        # the right name — not to invent a value or drop the key.
        if credential_env_names:
            debug_requirement.append(
                "The credential(s) for this source are exported as environment "
                "variable(s) named exactly: "
                + ", ".join(str(n) for n in credential_env_names)
                + ". If the error is a missing/unset credential under some "
                "OTHER name, change the code to read one of these exact names "
                "via os.environ. Never hard-code, invent, or drop a credential.")

        etype, exc, tb = sys.exc_info()
        exttb = traceback.extract_tb(tb)  # Do not quite understand this part.
        # https://stackoverflow.com/questions/39625465/how-do-i-retain-source-lines-in-tracebacks-when-running-dynamically-compiled-cod/39626362#39626362

      
        # print("code in get_debug_prompt:", code)
        ## Fill the missing data:
        exttb2 = [(fn, lnnr, funcname,
                   (code.splitlines()[lnnr - 1] if fn == 'Complete program'
                    else line))
                  for fn, lnnr, funcname, line in exttb]

        # Print:
        error_info_str = 'Traceback (most recent call last):\n'
        for line in traceback.format_list(exttb2[1:]):
            error_info_str += line
        for line in traceback.format_exception_only(etype, exc):
            error_info_str += line

        # print(f"Error_info_str: \n{error_info_str}")

        debug_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(debug_requirement)])
        # Control arm: one debug requirement also points at "the naming pattern
        # documented in the handbook". Omitting the guidelines block alone left
        # that dangling reference in every retry prompt.
        if not (handbook_str or "").strip():
            debug_requirement_str = debug_requirement_str.replace(
                _HANDBOOK_PATTERN_CLAUSE, _NO_HANDBOOK_PATTERN_CLAUSE)

        # Same rule as the generation prompt: in the control arm the guidelines
        # heading has nothing behind it, so it is omitted rather than left
        # standing empty. Everything else about the retry is identical.
        guidelines_block = (
            f"The technical guidelines for the code: \n {handbook_str} \n\n"
            if (handbook_str or "").strip() else "")

        debug_prompt = f"Your role: {debug_role} \n" + \
                          f"Your task: correct the code of a program according to the error information, then return the corrected and completed program. \n\n" + \
                          f"Requirement: \n {debug_requirement_str} \n\n" + \
                          f"The given code is used for this task: {task} \n\n" + \
                          guidelines_block + \
                          f"The error information for the code is: \n{str(error_info_str)} \n\n" + \
                          f"The code is: \n{code}"
        return debug_prompt

    def get_feedback_revision_prompt(self, feedback, code, task, handbook_str,
                                     last_error=""):
        """Prompt to revise retrieval code from a reviewer's written feedback.

        The sibling of ``get_debug_prompt``, for the case where nothing
        crashed: the code ran, a person read it (or its output) and said what
        is wrong with it. The evidence here is a human instruction rather than
        a traceback, so the requirements differ in one important way -- the
        feedback is the specification. Where it contradicts the handbook the
        feedback wins, and the reply says so, because the reviewer can see
        things the handbook never recorded (a deprecated endpoint, a wrong
        area, an output that is technically valid but not what was asked for).
        """
        revise_role = r'''A professional geo-information scientist and programmer who is good at Python. You have worked on Geographic information science for over 20 years and know every detail and pitfall when processing spatial data and coding. You are revising a data-retrieval script for a reviewer who has read it and told you what to change.'''

        revise_requirement = [
            'Think step by step. State your reasons for the revision as prose ABOVE the code block, not as comments inside the code. E.g., Explanation for the revision: xxxx \n The revised code is: ```python xxxx ```.',
            'The reviewer feedback is the specification. Do exactly what it asks, and nothing it did not ask for.',
            'You must return the entire revised program in only one Python code block (enclosed by ```python and ```); DO NOT return the changed part only.',
            'Keep the program structure: the function names, their arguments and returns, and the file(s) and directory the code writes to, unless the feedback explicitly asks for those to change.',
            "Non-regression: do NOT remove or weaken existing resilience code (per-item retry/skip loops, pagination handling, extracted-layer checks) while making the requested change. Never introduce a NEW hard requirement the code did not have before -- a local file path, a pre-downloaded dataset, a manual step, or any input the code does not fetch or create itself.",
            _FEEDBACK_GUIDELINES_CLAUSE,
            "If the feedback is not specific enough to act on with confidence, make the smallest change that satisfies the most likely reading, and say in your explanation what you assumed.",
            "If the feedback asks for a narrower selection (an area, a date range, a layer, a set of columns), filter at the source where the API supports it rather than downloading everything and discarding it afterwards.",
            "Print evidence that the requested change took effect -- the filter applied, the row or feature count written, the file path -- so the next reviewer can confirm it from the output alone.",
            "MINIMAL DIFF. This code already runs; you are changing one thing about it. Touch only the lines the feedback requires. Do NOT reformat, restyle, re-wrap or re-indent anything, do NOT rename variables, do NOT reorder or rewrite imports, and do NOT delete or reword existing comments -- they are the reviewer's notes on their own code. A diff containing anything the feedback did not ask for is a failed revision, however tidy it looks.",
            "Do not add dependency installation (pip, subprocess, importlib) that the code did not already have. It ran as it is, so its packages are present; adding an installer is a new runtime behaviour nobody asked for.",
        ]

        # Control arm: with no handbook there are no "technical guidelines" for
        # the feedback to contradict, so that requirement is reworded rather
        # than left pointing at a block this prompt never includes.
        if not (handbook_str or "").strip():
            revise_requirement = [
                _FEEDBACK_NO_GUIDELINES_CLAUSE if line is _FEEDBACK_GUIDELINES_CLAUSE
                else line for line in revise_requirement]
        requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(revise_requirement)])

        # Same rule as the generation and debug prompts: in the control arm the
        # guidelines heading has nothing behind it, so it is omitted rather
        # than left standing empty.
        guidelines_block = (
            f"The technical guidelines for the code: \n {handbook_str} \n\n"
            if (handbook_str or "").strip() else "")
        # The recorded error is context, not the instruction. It is labelled as
        # such so the model fixes what the reviewer asked about rather than
        # quietly re-litigating an old traceback instead.
        error_block = (
            f"For context, the last recorded error from this code was: \n{last_error}\n\n"
            if (last_error or "").strip() else "")

        return (
            f"Your role: {revise_role} \n"
            f"Your task: revise the code of a program according to the reviewer's "
            f"feedback, then return the revised and complete program. \n\n"
            f"Requirement: \n {requirement_str} \n\n"
            f"The code is used for this task: {task} \n\n"
            f"{guidelines_block}"
            f"{error_block}"
            f"The reviewer's feedback is: \n{feedback}\n\n"
            f"The code is: \n{code}"
        )

    def revise_code_from_feedback(self, feedback, code, task, handbook_str="",
                                  last_error="", stream_callback=None):
        """Rewrite retrieval code to satisfy a reviewer's written feedback.

        Returns ``{"code": str, "explanation": str, "raw": str}``. ``code`` is
        empty when the model replied without a Python block, which the caller
        should surface rather than silently storing nothing.
        """
        prompt = self.get_feedback_revision_prompt(
            feedback, code, task, handbook_str, last_error=last_error)

        if self.provider == "claude":
            # Tool-free Agent SDK session, matching how every other
            # single-completion call reaches Claude in this codebase.
            result = claude_agent_provider.run_agent(
                prompt, tools=[], model=self.model,
                api_key=self.anthropic_api_key, on_activity=stream_callback)
            usage_ledger.record(self.model, getattr(result, "usage", None),
                                phase="retrieval", source="revise_code")
            if not result.success:
                raise RuntimeError(result.error or "Claude Agent SDK call failed.")
            reply = result.text or ""
        else:
            reply = self.generate_data_fetching_code(
                prompt, stream_callback=stream_callback) or ""

        revised = self.extract_code_from_str(reply)
        return {"code": revised,
                "explanation": self._revision_explanation(reply, revised),
                "raw": reply}

    @staticmethod
    def _revision_explanation(reply, revised_code):
        """The model's account of what it changed, wherever it chose to put it.

        Asked for prose above the code block, models oblige in three different
        ways, and a reviewer who is shown nothing cannot tell "the model said
        nothing" from "we failed to find what it said". So: text before the
        fence, else text after it, else the code's own leading comment block
        when that block is clearly about the revision -- gpt-5.2 writes its
        explanation there as a header comment rather than above the fence. The
        comment is copied, never removed: it is part of the code the reviewer
        is about to run.
        """
        if "```" in reply:
            before = reply.split("```")[0].strip()
            if before:
                return before
            after = reply.rsplit("```", 1)[-1].strip()
            if after:
                return after

        lead = []
        for line in (revised_code or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                lead.append(stripped.lstrip("#").strip())
            elif not stripped and lead:
                break
            elif stripped:
                break
        text = "\n".join(lead).strip()
        # Long enough to be an account rather than a one-line file banner, and
        # actually about the change rather than about the script in general.
        if len(text) > 40 and re.search(r"revis|feedback|chang", text, re.I):
            return text
        return ""


    ##=========================SUPPORTING FUNCTIONS====================================================
     
    def select_source(self, select_prompt_str, stream_callback=None):

        messages = [{"role": "system", "content": select_prompt_str}]

        # Let exceptions propagate — swallowing them and returning None hides
        # the real cause (usually an invalid GIBD key) and leaves the caller
        # to crash inside ast.literal_eval(None) later. LLM_Find now reports
        # any None/parse failure as a clean RuntimeError.
        response = _stream_with_usage(
            messages=messages,
            model=self.model,
            user_key=self.api_key,
            base_url=self.base_url,
            **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}),
        )
        reply_content = ""
        for chunk in response:
            # A final usage-only chunk arrives when the backend honours
            # stream_options; it carries no choices, so it must be read before
            # the choices guard below or the cost of this call is lost.
            if getattr(chunk, "usage", None) is not None:
                usage_ledger.record(self.model, chunk.usage,
                                    phase="retrieval", source="generate_code")
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
            if delta:
                reply_content += delta
                if stream_callback:
                    stream_callback(delta)
        logging.info("Successfully got the reply from LLM.")
        return reply_content

    def create_substitute_prompt(self, data_request, available_sources, descriptions_str):
        """Prompt for picking the best AVAILABLE source when the LLM's original
        choice was unrecognized. Unlike create_select_prompt, this forces a pick
        from the available handbooks (no 'Unknown') so an autonomous run can keep
        going instead of dead-ending — mirroring a human's dropdown choice."""
        role = (
            "A professional GIScience data engineer. You match a data request to "
            "the single best data source from a fixed list of available handbooks."
        )
        requirements = [
            "You MUST choose exactly one source NAME from the AVAILABLE list below.",
            "Do NOT invent names and do NOT return 'Unknown' — pick the closest "
            "usable source even if it is an imperfect substitute.",
            "The name must match one of the available source names EXACTLY.",
            "Reply ONLY with a JSON object: {\"Selected data source\": \"<exact available name>\"}.",
        ]
        req_str = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(requirements))
        return (
            f"Your role: {role}\n"
            f"Data request: {data_request}\n\n"
            f"Requirements:\n{req_str}\n\n"
            f"Available data sources (choose one of these):\n{descriptions_str}\n"
            f"Available source names: {available_sources}\n"
        )

    def select_available_substitute(self, data_request, available_sources, descriptions_str):
        """Ask the LLM to pick the best AVAILABLE handbook source for the request
        and return its exact name, or None if nothing valid could be selected.
        Used by autonomous runs to recover from an unrecognized source."""
        if not available_sources:
            return None
        try:
            prompt = self.create_substitute_prompt(
                data_request, available_sources, descriptions_str
            )
            # Don't stream the substitution's raw JSON into the UI — keep it clean.
            reply = self.select_source(select_prompt_str=prompt, stream_callback=None)
            parsed = parse_llm_object(reply) if reply else None
            name = ""
            if isinstance(parsed, dict):
                name = (parsed.get("Selected data source") or "").strip()
            if name in available_sources:
                return name
            # Tolerate case differences in the LLM's echo of the name.
            lower_map = {s.lower(): s for s in available_sources}
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        except Exception as e:
            logging.warning(f"Auto-substitute source selection failed: {e}")
        return None

    def synthesize_source_skill(self, query, website="", stream_callback=None):
        """Need-driven autonomous synthesis of a data-source skill (handbook).

        Given a free-form source description (``query``) the user approved at the
        confirmation card, draft a complete handbook with the LLM, self-test it by
        running its sample download, then persist it to the caller's private
        handbook library so it is immediately selectable for this run and reusable
        in future ones. This is the same generate-and-verify pipeline the Web UI's
        "Add data source → Generate with AI" uses, invoked inline during retrieval.

        Returns ``{"name", "id", "verification"}`` for the new source, or ``None``
        if generation/persistence failed (the caller fails the data request).
        """
        def _stream(text):
            if stream_callback:
                stream_callback(text)

        query = (query or "").strip()
        if not query:
            logging.warning("synthesize_source_skill: no source description given.")
            return None

        # Persisting under the user's private library requires their identity,
        # which we derive from the API key (same hash the Web UI uses).
        from agents.data_agent.user_data_sources import write_user_source, uid_for_key
        from agents.data_agent import handbook_generator

        uid = uid_for_key(self.api_key)
        if not uid:
            _stream("\n⚠️ **Cannot save new knowledge:** no user identity "
                    "(missing API key).\n")
            return None

        try:
            _stream(f"\n\n### ⚡ Synthesizing data-source knowledge for "
                    f"**{query}**...\n\n")
            # Mirror the generator's progress log (doc fetches, model call,
            # each self-verify run / auto-fix cycle) into the workflow stream
            # live — the same lines the "Generate with AI" panel shows. Lines
            # are forwarded RAW (not backtick-wrapped) so the Web UI can parse
            # the "[stage:key:status] message" markers into stage rows inside
            # the skill-synthesis card, the same markers Handbook Studio's
            # Generate & Test parses server-side.
            handbook_generator.start_log_capture(
                callback=lambda line: _stream(line + "\n"))
            try:
                # Generate + self-verify. model=None uses the handbook
                # generator's strong default; route the LLM call through the
                # caller's key.
                new_source = handbook_generator.generate_handbook(
                    query=query, website=website, verify=True,
                    user_key=self.api_key, model=None,
                )
            finally:
                handbook_generator.stop_log_capture()
        except Exception as gen_err:
            logging.warning(f"synthesize_source_skill: generation failed: {gen_err}")
            _stream(f"\n⚠️ **Knowledge generation failed:** {gen_err}\n")
            return None

        report = new_source.pop("_verification", {}) or {}
        v_status = report.get("status", "")
        v_note = report.get("note", "")
        if v_status in ("passed", "passed_smoke"):
            _stream(f"\n✅ **Knowledge verified** ({v_status}). {v_note}\n")
        else:
            _stream(f"\n⚠️ **Knowledge saved but the self-test did not pass** "
                    f"({v_status or 'unknown'}). {v_note} You may need to refine "
                    f"it, or supply an API key, before it downloads.\n")

        try:
            slug = write_user_source(uid, new_source)
        except Exception as save_err:
            logging.warning(f"synthesize_source_skill: save failed: {save_err}")
            _stream(f"\n⚠️ **Could not save the generated knowledge:** {save_err}\n")
            return None

        new_name = (new_source.get("data_source_name") or query).strip()
        # The source ID is the file slug; resolve via the refreshed catalog when
        # possible so a name/slug mismatch can't desync selection.
        handbook_files = self.collect_handbook_files(source_dir=self.handbook_dir)
        _, data_source_dict = self.assemble_handbook_description(handbook_files)
        new_id = data_source_dict.get(new_name, {}).get("ID", slug)
        logging.info(f"synthesize_source_skill: saved '{new_name}' (id={new_id}, "
                     f"verify={v_status}).")
        return {"name": new_name, "id": new_id, "verification": report}

    def _re_research_skill(self, source_name, source_ID, data_request,
                           user_keys, _stream, _warn, failure_note=""):
        """Last-resort self-healing after the download debug loop exhausts its
        retries: regenerate the source's skill from its live documentation
        (needs an sk- key), verify it with the user's data-source keys, and
        save it to the user's sources. Returns (name, id) of the fresh skill,
        or None when re-research is unavailable or failed.

        ``failure_note``: the tail of the failed download's final traceback.
        It is embedded in the generation query so the SOURCE-SUGGESTION stage
        sees what already failed — without it, each re-research round re-picks
        the same source/endpoint and only the downstream code reviser ever
        learns from the failure (observed live: OSM API v0.6, the map-editing
        API, was re-selected round after round for a feature-query task that
        needed Overpass)."""
        from agents.data_agent import handbook_generator
        from agents.data_agent.user_data_sources import (
            write_user_source, uid_for_key, slugify)

        key_ok = (str(self.api_key or "").startswith("sk-")
                  or helper._usable_openai_fallback_key())
        uid = uid_for_key(self.api_key)
        if not key_ok or not uid:
            return None

        old_website = ""
        path = self._resolve_handbook_file(source_ID)
        if path:
            try:
                old_website = str(self._load_book(path).get("website", "") or "")
            except Exception:
                pass

        # Build the re-research query with the failure evidence inline. The
        # old handbook's website is deliberately NOT passed as the ``website=``
        # hint: that hint reads as "User-provided website" to the generator
        # and re-anchors every round to the site whose endpoints just failed.
        # It is mentioned in the query as failed context instead.
        query = f"{source_name} — needed for: {data_request}"
        context_bits = []
        if failure_note:
            context_bits.append(
                f"A previous handbook for this source was already generated "
                f"and its downloads kept failing with: {failure_note}")
        if old_website:
            context_bits.append(
                f"That failing handbook was based on {old_website}.")
        if context_bits:
            query += (
                "\n\n[Re-research context: " + " ".join(context_bits) +
                " If this failure pattern suggests the wrong endpoint or "
                "service was chosen for this kind of request (e.g. an editing "
                "API where a query/search API is needed), pick the correct "
                "one this time rather than re-documenting the same one.]")

        _stream("\n\n🔬 **The download kept failing — re-researching this "
                "source's knowledge from its live documentation...**\n")
        handbook_generator.start_log_capture(
            callback=lambda line: _stream(f"`{line}`  \n"))
        try:
            new_source = handbook_generator.generate_handbook(
                query=query,
                website="", user_key=self.api_key, verify=True,
                data_source_keys=user_keys)
        except Exception as e:
            _warn(f"Re-research failed: {e}")
            return None
        finally:
            handbook_generator.stop_log_capture()

        report = new_source.pop("_verification", {}) or {}
        try:
            write_user_source(uid, new_source)
        except Exception as e:
            _warn(f"Could not save the re-researched knowledge: {e}")
            return None

        name = (new_source.get("data_source_name") or source_name).strip()
        handbook_files = self.collect_handbook_files(source_dir=self.handbook_dir)
        _, data_source_dict = self.assemble_handbook_description(handbook_files)
        new_id = data_source_dict.get(name, {}).get("ID") or slugify(name)
        icon = "✅" if report.get("verified") else "⚠️"
        _stream(f"\n{icon} **Re-researched knowledge saved** "
                f"({report.get('status', '')}). {report.get('note', '')}\n")
        return name, new_id

    def _triage_download_failure(self, source_ID, data_request, code,
                                 failure_note, http_requests, _stream, _warn):
        """Classify an exhausted download failure before spending an expensive
        repair on it: a fixable handbook deficiency, or something external?

        The debug loop has already retried DOWNLOAD_TRIALS times, so what is
        left is either wrong knowledge about the source or a condition no
        handbook change can fix — an outage, a rate limit, a deprecated
        endpoint. Escalating regardless is what this exists to stop:
        re-researching a source from its documentation because the server
        happened to return 503 discards a handbook that works, and costs the
        run a full regeneration to arrive back where it started.

        Fails OPEN. Triage is an optimization, never a gate — if the classifier
        errors, returns nothing usable, or drifts off its two literal
        categories, the caller escalates exactly as it did before this
        existed. Only the literal "external" stops the ladder.

        Returns the analysis dict (see ``analyze_execution_trace``), or {} when
        triage could not be performed.
        """
        from agents.data_agent import handbook_generator

        book_path = self._resolve_handbook_file(source_ID)
        if not book_path:
            return {}
        try:
            book = self._load_book(book_path)
        except Exception as e:
            _warn(f"Failure triage skipped (unreadable handbook): {e}")
            return {}

        _stream("\n\n🧭 **Diagnosing the failure before trying to repair the "
                "knowledge...**\n")
        try:
            analysis = handbook_generator.analyze_execution_trace(
                book, str(data_request),
                {
                    "status": "failed",
                    "error": failure_note,
                    "generated_code": code,
                    "http_requests": list(http_requests or []),
                },
                user_key=self.api_key)
        except Exception as e:
            _warn(f"Failure triage failed to run: {e}")
            return {}

        category = str(analysis.get("failure_category") or "").strip().lower()
        summary = str(analysis.get("summary") or "").strip()
        if category == "external":
            _stream("\n🌐 **This looks like an external failure, not a problem "
                    f"with the knowledge.** {summary}\n")
        elif category == "request_infeasible":
            _stream("\n📭 **The source has no data for this request as "
                    f"asked** (period, area, or variable not covered). {summary}\n")
        elif category == "handbook_deficiency":
            _stream(f"\n🔎 **The skill looks wrong for this task.** {summary}\n")
        return analysis

    def _refine_skill_from_failure(self, source_ID, data_request, analysis,
                                   failure_note, exec_ok, _stream, _warn):
        """Revise the EXISTING handbook against the recorded failure evidence —
        the cheap rung below ``_re_research_skill``.

        Re-research discards the handbook and regenerates it from live
        documentation. That is the right move when the source or the endpoint
        itself was the wrong choice, and the wrong one when the handbook is
        very nearly right: it is slower, it needs an sk- key, and it can re-pick
        a different service altogether (the OSM API v0.6 vs Overpass case in
        ``_re_research_skill``'s docstring). A refinement keeps everything that
        already worked and changes only what the evidence implicates.

        Unlike re-research this DOES pass the source's own website as the
        grounding hint. Re-research withholds it because that hint re-anchors
        the source-suggestion stage to the site whose endpoints just failed;
        here there is no source to re-choose — the whole point is to fix this
        source's handbook — so its documentation is exactly what the fix
        should be read out of.

        Returns the refined handbook RECORD, or None. It is deliberately NOT
        persisted: LLM_Find writes it back only once the retry it enables
        actually delivers data. A revision that makes things worse must not
        outlive the request that produced it — the saved handbook may be
        serving other tasks perfectly well.
        """
        from agents.data_agent import handbook_generator

        book_path = self._resolve_handbook_file(source_ID)
        if not book_path:
            return None
        try:
            book = self._load_book(book_path)
        except Exception as e:
            _warn(f"Knowledge refinement skipped (unreadable handbook): {e}")
            return None

        feedback = handbook_generator.build_refinement_feedback(
            str(data_request), analysis or {}, error=failure_note,
            prior_execution_ok=bool(exec_ok))

        _stream("\n\n🔧 **Revising this source's knowledge from the failure "
                "evidence...**\n")
        handbook_generator.start_log_capture(
            callback=lambda line: _stream(f"`{line}`  \n"))
        try:
            refined = handbook_generator.refine_handbook(
                book, feedback, website=str(book.get("website", "") or ""),
                fetch_docs=True, user_key=self.api_key)
        except Exception as e:
            _warn(f"Knowledge refinement failed: {e}")
            return None
        finally:
            handbook_generator.stop_log_capture()

        next_book = (refined or {}).get("source") or {}
        if not handbook_generator.refinement_changed(book, next_book):
            _stream("\n↩️ **The revision left the knowledge unchanged** — retrying "
                    "it would fail the same way, so moving on.\n")
            return None
        _stream(f"\n📘 **Revised skill ready.** "
                f"{str((refined or {}).get('assistant_message') or '').strip()}\n")
        return next_book

    def _persist_refined_skill(self, source_ID, refined_book, _stream, _warn):
        """Save a refinement that earned it — called only after the retry it
        enabled actually delivered data, so a revision can never poison a
        handbook that other tasks depend on by merely having been attempted.
        Best-effort: the data is already downloaded, and failing to file the
        improvement is not a reason to fail the request."""
        from agents.data_agent.user_data_sources import write_user_source, uid_for_key

        uid = uid_for_key(self.api_key)
        if not uid or not refined_book:
            return
        try:
            write_user_source(uid, refined_book)
        except Exception as e:
            _warn(f"Could not save the revised knowledge: {e}")
            return
        _stream("\n📗 **The revised skill worked and has been saved for future "
                "runs.**\n")

    def _verify_skill_with_keys(self, source_ID, user_keys, _stream, _warn):
        """Verify a just-generated skill with the credentials the user entered
        on the confirmation card, BEFORE the download starts. Auto-fixes the
        handbook on failure (when an sk- key is available) and re-saves it.
        Never blocks the flow: any failure is reported and the download
        proceeds (its own debug loop takes over from there)."""
        from agents.data_agent import handbook_generator
        from agents.data_agent.user_data_sources import write_user_source, uid_for_key

        book_path = self._resolve_handbook_file(source_ID)
        if not book_path:
            return
        try:
            book = self._load_book(book_path)
        except Exception as e:
            _warn(f"Pre-download verification skipped (unreadable handbook): {e}")
            return

        _stream("\n\n🔍 **Verifying the new skill with your credentials before "
                "downloading...**\n")
        handbook_generator.start_log_capture(
            callback=lambda line: _stream(f"`{line}`  \n"))
        try:
            revised, report = handbook_generator.verify_handbook(
                book, data_source_keys=user_keys, user_key=self.api_key)
        except Exception as e:
            _warn(f"Pre-download verification failed to run: {e}")
            return
        finally:
            handbook_generator.stop_log_capture()

        if report.get("verified"):
            _stream(f"\n✅ **Knowledge verified.** {report.get('note', '')}\n")
        else:
            _stream(f"\n⚠️ **Verification did not pass:** {report.get('note', '')} "
                    "Proceeding — the downloader will retry and debug.\n")
        if revised.get("code_example") != str(book.get("code_example", "")):
            uid = uid_for_key(self.api_key)
            if uid:
                try:
                    write_user_source(uid, revised)
                    _stream("\n📘 **The skill was auto-fixed and re-saved.**\n")
                except Exception as e:
                    _warn(f"Could not re-save the fixed knowledge: {e}")

    def _all_handbook_dirs(self):
        """Curated global catalog first, then any per-user dirs. Order matters:
        callers resolve a source ID by taking the FIRST match, so the global
        catalog wins on a name collision."""
        return [self.handbook_dir] + self.extra_handbook_dirs

    @staticmethod
    def _load_book(path):
        """Load a handbook from either TOML (.toml, bundled) or JSON (.json,
        user-contributed). Returns a dict."""
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        with open(path, "rb") as f:
            return tomli.load(f)

    @staticmethod
    def _source_id(path):
        """A source's ID is its filename without the .toml/.json extension."""
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def _file_has_records(path):
        """Best-effort check that a downloaded file actually contains data
        records, not just structure (a header-only CSV, a zero-feature
        GeoJSON, an empty JSON array). Only formats we can inspect cheaply
        and unambiguously are checked; anything else — and any read error —
        counts as having records, so this can only ever flag files that are
        provably empty, never misjudge an unusual-but-valid one."""
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".csv":
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    non_blank = 0
                    for line in f:
                        if line.strip():
                            non_blank += 1
                        if non_blank > 1:  # header + at least one data row
                            return True
                return False
            if ext in (".json", ".geojson"):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return len(data) > 0
                if isinstance(data, dict):
                    if "features" in data:  # GeoJSON FeatureCollection
                        return bool(data.get("features"))
                    # A dict without features: any non-empty payload counts.
                    return bool(data)
                return True
        except Exception:
            return True
        return True

    def _dirs_with_extra(self, source_dir=None):
        """The base catalog dir (the one passed by callers, or the global
        catalog) followed by any per-user dirs. The per-user dirs are ALWAYS
        included — even when a caller passes an explicit source_dir — so a
        user's contributed sources are visible everywhere the global ones are."""
        base = source_dir or self.handbook_dir
        dirs = [base]
        for d in self.extra_handbook_dirs:
            if d and d != base:
                dirs.append(d)
        return dirs

    def collect_handbook_files(self, source_dir=None):
        # Scan the base catalog plus any per-user dirs. Sources are deduped by
        # ID with earlier dirs (global) taking precedence.
        seen = set()
        handbooks = []
        for d in self._dirs_with_extra(source_dir):
            if not d or not os.path.isdir(d):
                continue
            for book in sorted(glob(os.path.join(d, "*.toml")) + glob(os.path.join(d, "*.json"))):
                sid = self._source_id(book)
                if sid == "template" or sid in seen:
                    continue
                seen.add(sid)
                handbooks.append(book)
        # logging.info(f"Successfully collected {len(handbooks)}")
        return handbooks

    def _resolve_handbook_file(self, source_ID, source_dir=None):
        """Find the .toml or .json file for a source ID across the base dir and
        any per-user dirs (base first). Returns the path or None."""
        for d in self._dirs_with_extra(source_dir):
            if not d:
                continue
            for ext in (".toml", ".json"):
                candidate = os.path.join(d, f"{source_ID}{ext}")
                if os.path.exists(candidate):
                    return candidate
        return None


    def assemble_handbook_description(self, handbook_files ):
        descriptions = []
        data_source_dict = {}
        for idx, book in enumerate(handbook_files):
            handbook = self._load_book(book)
            data_source_ID = self._source_id(book)  # data_source_ID is the file name
            data_source_name = handbook['data_source_name'].strip()
            description = f"{idx + 1}. {data_source_name}. {handbook['brief_description'].strip()}"
            # print(description)
            descriptions.append(description)
            data_source_dict[data_source_name] = {"ID": data_source_ID}
        data_source_dict['Unknown'] = {"ID": "Unknown"}
        descriptions_str = "\n".join(descriptions)
        return descriptions_str, data_source_dict

    @staticmethod
    def _normalize_source_name(name):
        """Key for matching a source name the LLM echoed back to the catalog.

        Lowercased, whitespace collapsed, and trailing sentence punctuation
        removed — the LLM routinely returns the catalog's own name with a
        full stop appended ("... Jan 1 2025 vintage)." vs the catalog's
        "... Jan 1 2025 vintage)"), which an exact dict lookup misses.
        """
        s = unicodedata.normalize("NFKC", str(name or ""))
        s = s.replace(" ", " ")          # non-breaking space
        s = re.sub(r"\s+", " ", s).strip()
        s = s.rstrip(" .,;:")                  # trailing sentence punctuation
        return s.casefold()

    def resolve_source_name(self, selected_name, data_source_dict):
        """Map an LLM-returned source name onto a catalog entry.

        Returns ``(canonical_name, source_id)``, or ``(selected_name,
        "Unknown")`` when it genuinely isn't in the catalog.

        Exact match wins. Otherwise a normalized comparison is tried, and is
        accepted ONLY when exactly one catalog entry matches: several entries
        in a real library differ just by vintage or level ("TIGER/Line
        Shapefiles (2025) ..." vs "TIGER/Line Shapefiles (Census Tracts) ..."),
        so an ambiguous match must stay unrecognized rather than silently pick
        the wrong source and download the wrong data.
        """
        if not selected_name:
            return selected_name, "Unknown"
        entry = data_source_dict.get(selected_name)
        if entry and entry.get("ID") != "Unknown":
            return selected_name, entry["ID"]

        target = self._normalize_source_name(selected_name)
        if not target:
            return selected_name, "Unknown"
        matches = [
            (name, meta.get("ID"))
            for name, meta in data_source_dict.items()
            if name != "Unknown"
            and self._normalize_source_name(name) == target
        ]
        if len(matches) == 1:
            name, sid = matches[0]
            logging.info(
                "Source name matched after normalization: "
                f"{selected_name!r} -> {name!r}")
            return name, sid
        if len(matches) > 1:
            logging.warning(
                f"Source name {selected_name!r} matches {len(matches)} catalog "
                "entries after normalization — treating as unrecognized rather "
                "than guessing between them.")
        return selected_name, "Unknown"
    
    
    def effective_keys(self, source_ID, keys_dir=None, user_keys=None):
        """The credentials a source should be rendered with: the bundled
        ``.keys`` file provides defaults, and user-supplied keys (entered in
        the Web UI) override them. This lets users provide their own
        data-source API keys without editing the .keys files on disk.
        """
        keys = {}
        kd = keys_dir or self.keys_dir
        key_file = os.path.join(kd, f"{source_ID}.keys") if kd else ""
        if key_file and os.path.exists(key_file):
            try:
                keys = self.load_keys(source_ID, kd)
            except Exception as e:
                logging.warning(f"Could not read keys for {source_ID}: {e}")
                keys = {}
        for k, v in (user_keys or {}).items():
            if v:  # only override with a non-empty value
                keys[k] = v
        return keys

    @staticmethod
    def _substitute_keys(text, keys):
        """Replace ``{PLACEHOLDER}`` credential tokens with their values.

        Case-insensitive: handbooks sometimes use a different casing for the
        placeholder than the .keys file declares (e.g. .keys 'EPA_AQS_KEY' vs
        handbook '{EPA_AQS_key}'). Match on lowercase so the entered key is
        always injected. Tokens that are not credentials are left alone for
        the structural {field} pass in render_handbook_text.
        """
        if not keys or not text:
            return text
        keys_ci = {k.lower(): str(v) for k, v in keys.items()}

        def _sub_key(m):
            nm = m.group(1)
            if nm.lower() in keys_ci:
                logging.info(f"Replacing key placeholder: {nm}")
                return keys_ci[nm.lower()]
            return m.group(0)

        return re.sub(r"\{([A-Za-z0-9_]+)\}", _sub_key, text)

    def render_handbook_text(self, book, keys=None):
        """Render a handbook RECORD (the dict form) into the numbered prose
        block the download prompt consumes.

        Split out of ``collect_a_handbook`` so a handbook that exists only in
        memory goes through exactly the same substitution, numbering and
        {field} expansion as one loaded off disk. That matters for
        ``_refine_skill_from_failure``, whose revision is deliberately not
        written to disk until it has proven itself: with two renderers, the
        handbook under test would not be quite the artifact that later gets
        saved, and a retry could pass or fail for reasons that have nothing to
        do with the revision.
        """
        text = self._substitute_keys(str(book.get("handbook", "") or ""), keys)
        numbered_handbook_str = ''
        for idx, line in enumerate(text.strip().split('\n')):
            line = line.strip(' ')
            numbered_handbook_str += f"{idx + 1}. {line}\n"

        for variable in book.keys():
            numbered_handbook_str = numbered_handbook_str.replace(
                f"{{{variable}}}", str(book[variable]))
        return numbered_handbook_str

    def render_code_example(self, book, keys=None):
        """Companion to ``render_handbook_text`` for a record's worked example.
        Same reason for existing: an in-memory revision has to render the same
        way a saved one does."""
        example = str(book.get("code_example", "") or "")
        if not example.strip():
            return ""
        return self._substitute_keys(example, keys)

    def collect_a_handbook(self, source_ID, source_dir=None, keys_dir=None, user_keys=None):
        handbook_file = self._resolve_handbook_file(source_ID, source_dir)

        # Check if handbook file exists
        if not handbook_file:
            logging.warning(f"Warning: Handbook file not found for source: {source_ID}")
            return None

        try:
            handbook = self._load_book(handbook_file)
            if 'handbook' not in handbook:
                raise KeyError('handbook')
        except Exception as e:
            logging.error(f"Error loading handbook for {source_ID}: {e}")
            return None

        keys = self.effective_keys(source_ID, keys_dir, user_keys)
        if not keys:
            logging.warning(f"No keys found for source: {source_ID}")
        return self.render_handbook_text(handbook, keys)

    def collect_code_example(self, source_ID, source_dir=None, keys_dir=None,
                             user_keys=None):
        """Return a source's verified ``code_example`` script, or "".

        Companion to ``collect_a_handbook`` (which returns only the prose
        ``handbook`` field). Kept separate so callers decide whether to spend
        the extra prompt tokens, and so any credential placeholders get the
        same substitution the handbook text receives.
        """
        handbook_file = self._resolve_handbook_file(source_ID, source_dir)
        if not handbook_file:
            return ""
        try:
            book = self._load_book(handbook_file)
        except Exception as e:
            logging.warning(f"Could not read code_example for {source_ID}: {e}")
            return ""
        return self.render_code_example(
            book, self.effective_keys(source_ID, keys_dir, user_keys))
    
    
    def load_keys(self, source_ID, keys_dir=None):
        key_file = os.path.join(keys_dir, f"{source_ID}.keys")
        config = CaseSensitiveConfigParser()
        config.read(key_file)
        # keys = config['API_Key'].keys()
        # print("config['API_Key'].keys():", config['API_Key'].keys())
        keys_dict = {}
        for key in config['API_Key'].keys():
            keys_dict[key] = config.get("API_Key", key)
            # print("Key:", key)

        # print("keys_dict:", keys_dict)
        return keys_dict

    def load_key_links(self, source_ID, keys_dir=None):
        """Where to apply for a source's key: the ``[Links]`` section of its
        ``.keys`` file (searched in the global Keys dir, then the per-user
        handbook dirs, where generated skills store theirs). Supports a
        per-source ``website``/``signup_url`` and per-key-name overrides.
        Falls back to the handbook's own ``key_signup_url``/``website`` fields;
        returns {} when nothing is known (no link shown)."""
        dirs = [keys_dir or self.keys_dir] + [d for d in self._all_handbook_dirs() if d]
        for d in dirs:
            key_file = os.path.join(d, f"{source_ID}.keys")
            if not os.path.exists(key_file):
                continue
            try:
                config = CaseSensitiveConfigParser()
                config.read(key_file)
                if config.has_section('Links'):
                    return {k: config.get('Links', k) for k in config['Links'].keys()}
            except Exception as e:
                logging.warning(f"Could not read [Links] for {source_ID}: {e}")
        # No .keys [Links] anywhere — use the handbook's own fields.
        path = self._resolve_handbook_file(source_ID)
        if path:
            try:
                book = self._load_book(path)
                url = str(book.get("key_signup_url", "") or book.get("website", "") or "").strip()
                if url:
                    return {"website": url}
            except Exception:
                pass
        return {}

    def load_source_caveats(self, source_ID, source_dir=None):
        """Optional ``caveats`` field of a source's handbook — short user-facing
        warnings (cost, registration effort, large downloads, licensing) shown
        as an agent advisory on the source-confirmation card. '' when absent."""
        path = self._resolve_handbook_file(source_ID, source_dir)
        if not path:
            return ""
        try:
            return str(self._load_book(path).get("caveats", "") or "").strip()
        except Exception as e:
            logging.warning(f"Could not read caveats for {source_ID}: {e}")
            return ""

    @staticmethod
    def key_value_is_set(value):
        """True when a key value is a real, usable secret (not blank and not a
        placeholder like 'XXXX'/'Example_Key')."""
        if not value:
            return False
        v = str(value).strip().lower()
        if not v or 'xxxx' in v or v in ('none', 'your_key', 'example_key'):
            return False
        return True

    def _handbook_placeholders(self, source_ID, source_dir=None):
        """Set of ``{tokens}`` used in a source's handbook text, excluding the
        handbook's own structural fields. Used to know which declared keys are
        actually referenced (so we don't prompt for keys the handbook ignores)."""
        handbook_file = self._resolve_handbook_file(source_ID, source_dir)
        if not handbook_file:
            return set()
        try:
            handbook = self._load_book(handbook_file)
            text = handbook.get('handbook', '') or ''
        except Exception:
            return set()
        structural = set(handbook.keys())
        return {m for m in re.findall(r"\{([A-Za-z0-9_]+)\}", text) if m not in structural}

    def _credential_env_plan(self, source_ID, user_keys=None, code=None):
        """What the download code may read from the environment for this
        source, and what we can actually put there.

        Returns ``(names, values)``: ``names`` is every credential name the
        code could legitimately read — the handbook's declared keys plus any
        ``os.environ[...]`` name the generated code itself reads — and
        ``values`` is the subset we hold a real value for (bundled ``.keys``
        file, overridden by keys entered in the Web UI), keyed by the exact
        name the code will look up.

        Why both: the download prompt promises "credentials are already in
        the environment", the verification trial honours that promise, and
        until this existed the production path did not — it only substituted
        ``{PLACEHOLDER}`` tokens in the handbook prose, so a synthesized
        handbook whose code reads ``os.environ["FIRMS_MAP_KEY"]`` (which is
        exactly how the generator tells it to) failed with a KeyError on a
        key the user had already entered.
        """
        names = list(self.get_required_key_names(source_ID, keys_dir=self.keys_dir))
        for n in credential_names_in_code(code or ""):
            if n not in names:
                names.append(n)
        keys = self.effective_keys(source_ID, self.keys_dir, user_keys)
        keys_ci = {str(k).lower(): v for k, v in keys.items()}
        values = {}
        for n in names:
            v = keys_ci.get(n.lower())
            if self.key_value_is_set(v):
                values[n] = str(v)
        return names, values

    @contextlib.contextmanager
    def _source_keys_in_env(self, source_ID, user_keys=None, code=None):
        """Export this source's credentials as environment variables for the
        duration of a download, exactly as ``run_controlled_handbook_trial``
        does for verification. Yields the credential names the code may read.
        Scoped and restored in ``finally`` — os.environ is process-global and
        the app serves requests on several threads."""
        names, values = self._credential_env_plan(source_ID, user_keys, code)
        previous = {n: os.environ.get(n) for n in values}
        os.environ.update(values)
        try:
            yield names
        finally:
            for n, old in previous.items():
                if old is None:
                    os.environ.pop(n, None)
                else:
                    os.environ[n] = old

    def get_required_key_names(self, source_ID, keys_dir=None, source_dir=None):
        """Return the API-key field names a source actually needs: the keys
        declared in its ``.keys`` file ([API_Key]) that are ALSO referenced as
        ``{placeholders}`` in the handbook (matched case-insensitively).

        This ensures we only prompt for keys the generated code will actually
        use — a key declared in .keys but never used by the handbook (e.g.
        CDC_PLACES) is skipped. Sources with no ``.keys`` file need no key."""
        keys_dir = keys_dir or self.keys_dir
        source_dir = source_dir or self.handbook_dir
        key_file = os.path.join(keys_dir, f"{source_ID}.keys")
        if not os.path.exists(key_file):
            # No bundled .keys file → a user-contributed source. Its required
            # credentials are whatever the contributor declared (requires_key +
            # key_name, which may list several names). We do NOT fall back to all
            # handbook {tokens} because some sources use {scale}/{category}/etc.
            # for non-credential templating.
            book_path = self._resolve_handbook_file(source_ID, source_dir)
            if not book_path:
                return []
            try:
                book = self._load_book(book_path)
            except Exception:
                return []
            if str(book.get("requires_key", "")).strip().lower() not in ("true", "1", "yes"):
                return []
            return [n.strip() for n in re.split(r"[,\n]", book.get("key_name", "") or "")
                    if n.strip()]
        try:
            declared = [n for n in self.load_keys(source_ID, keys_dir).keys()
                        if n.strip().lower() != 'example_key']
        except Exception as e:
            logging.error(f"Error reading key names for {source_ID}: {e}")
            return []
        placeholders = self._handbook_placeholders(source_ID, source_dir)
        # If the handbook is unreadable, fall back to all declared keys.
        if not placeholders:
            return declared
        ph_lower = {p.lower() for p in placeholders}
        return [n for n in declared if n.lower() in ph_lower]

    def generate_data_fetching_code(self, download_prompt_str, stream_callback=None):
        messages = [{"role": "system", "content": download_prompt_str}]

        # Let exceptions propagate so UI/logs see the real GIBD error instead
        # of a silent None return followed by an opaque downstream crash.
        response = _stream_with_usage(
            messages=messages,
            model=self.model,
            user_key=self.api_key,
            base_url=self.base_url,
            **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}),
        )
        reply_content = ""
        for chunk in response:
            # A final usage-only chunk arrives when the backend honours
            # stream_options; it carries no choices, so it must be read before
            # the choices guard below or the cost of this call is lost.
            if getattr(chunk, "usage", None) is not None:
                usage_ledger.record(self.model, chunk.usage,
                                    phase="retrieval", source="generate_code")
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
            if delta:
                reply_content += delta
                if stream_callback:
                    stream_callback(delta)
        logging.info("Successfully got the reply from LLM.")
        return reply_content


    def run_control_trial_without_handbook(
            self, data_request, source_ID, source_name=None, user_keys=None,
            stream_callback=None, output_dir=None, try_count=10,
            attempt_timeout=None, required_key_names=None):
        """The control arm: ``run_controlled_handbook_trial`` with the handbook
        withheld, and nothing else changed.

        This delegates rather than duplicating the trial body on purpose. A
        copied implementation would drift from the treatment path as that path
        is maintained, and the moment the two diverge the comparison stops
        being an ablation and starts being a comparison of two programs. One
        implementation, one flag, one variable.

        What differs, and all that differs: the handbook text and the worked
        code example are empty strings in the prompt that generates the
        retrieval code, and in the debug prompt used on each retry. The task,
        the source, the model, the reasoning effort, the exported credentials,
        the execution harness, the retry budget and the output inspection are
        the treatment path unchanged.

        The source must still HAVE a generated handbook for this to run: the
        control is meaningful only on a case the treatment also covers, and
        requiring it keeps the two arms paired on the same cases.
        """
        return self.run_controlled_handbook_trial(
            data_request=data_request, source_ID=source_ID,
            source_name=source_name, user_keys=user_keys,
            stream_callback=stream_callback, output_dir=output_dir,
            try_count=try_count, attempt_timeout=attempt_timeout,
            use_handbook=False, required_key_names=required_key_names)

    def run_controlled_handbook_trial(
            self, data_request, source_ID, source_name=None, user_keys=None,
            stream_callback=None, output_dir=None, try_count=10,
            attempt_timeout=None, use_handbook=True,
            required_key_names=None):
        """Run one retrieval trial against an explicitly selected handbook.

        Unlike ``LLM_Find``, this path never asks the model to select or replace
        the source.  It is intended for controlled experiments where changing
        the source would invalidate the trial condition.

        ``use_handbook=False`` runs the ABLATED control condition: identical
        task, source, model and execution harness, but the handbook's
        operational knowledge is withheld. The model is still told WHICH source
        to use -- withholding that would change the task into source discovery
        and stop measuring the handbook -- and still receives the credentials,
        since a control that fails for want of an API key measures nothing
        about handbooks. What it loses is the handbook body and the worked code
        example, in BOTH the generation prompt and the debug prompt used on
        every retry; leaking it into the latter would let the control recover
        the very knowledge the condition removes.
        """
        previous_output_dir = self.output_dir
        if output_dir:
            self.output_dir = os.path.abspath(output_dir)
            os.makedirs(self.output_dir, exist_ok=True)

        previous_env = {}
        try:
            if use_handbook:
                handbook_str = self.collect_a_handbook(
                    source_ID=source_ID,
                    source_dir=self.handbook_dir,
                    keys_dir=self.keys_dir,
                    user_keys=dict(user_keys or {}),
                )
                if not handbook_str:
                    return {
                        "status": "needs_handbook",
                        "error": f"No saved handbook is linked to source '{source_name or source_ID}'.",
                        "source_id": source_ID,
                    }
                required_keys = self.get_required_key_names(
                    source_ID, keys_dir=self.keys_dir, source_dir=self.handbook_dir)
            else:
                # The control condition may run on a session that never
                # generated a handbook, so there is nothing to read the
                # required credential names out of. They are supplied by the
                # caller instead. Credential VALUES still arrive through
                # user_keys and are still never persisted.
                handbook_str = ""
                required_keys = [str(n).strip() for n in (required_key_names or [])
                                 if str(n).strip()]
            missing_keys = [
                name for name in required_keys
                if not self.key_value_is_set((user_keys or {}).get(name))
            ]
            if missing_keys:
                return {
                    "status": "needs_credentials",
                    "error": "This source needs credentials before it can run.",
                    "required_keys": missing_keys,
                    "source_id": source_ID,
                }

            # Handbooks are generated with key_name documented to the model as
            # an UPPER_SNAKE environment-variable name, and code-generation is
            # told to read credentials via os.environ/os.getenv rather than
            # inventing a value — so the generated code expects these to
            # actually be exported, not just substituted into the handbook
            # text. Scoped as tightly as possible and restored in `finally`
            # since os.environ is process-global and this app serves requests
            # on multiple threads.
            env_overrides = {
                name: str((user_keys or {}).get(name)) for name in required_keys
                if self.key_value_is_set((user_keys or {}).get(name))
            }
            previous_env = {name: os.environ.get(name) for name in env_overrides}
            os.environ.update(env_overrides)

            if self.provider == "claude":
                # A fundamentally different execution path, not just a
                # different model: the Claude Agent SDK drives its own
                # write/run/debug loop via its own Bash tool, so this
                # class's in-process exec()+tracker harness (below) never
                # runs for a Claude-driven trial -- see
                # _run_claude_agent_trial's docstring for the trade-off.
                return self._run_claude_agent_trial(
                    data_request=data_request, source_id=source_ID,
                    source_name=source_name or source_ID,
                    handbook_str=(handbook_str if use_handbook else ""),
                    output_dir=self.output_dir,
                    stream_callback=stream_callback,
                    try_count=max(1, min(int(try_count), 10)),
                    extra_env=env_overrides, required_keys=required_keys)

            if use_handbook:
                effective_handbook = handbook_str
                effective_example = self.collect_code_example(
                    source_ID, source_dir=self.handbook_dir,
                    keys_dir=self.keys_dir, user_keys=dict(user_keys or {}))
                if stream_callback:
                    stream_callback("Generating retrieval code from the pinned handbook.")
            else:
                # A true ablation, not a prompted condition: the handbook slot
                # and the worked example are simply empty. No substitute text
                # is injected, because any wording added here would itself be
                # an intervention and the two arms would differ by more than
                # the one variable under study. Everything else -- prompt
                # template, model, source name, exported credentials, execution
                # harness, retry budget -- is byte-for-byte the treatment path.
                effective_handbook = ""
                effective_example = ""
                if stream_callback:
                    stream_callback("Control condition: handbook withheld.")
            prompt = self.create_download_prompt(
                str(data_request), source_name or source_ID, effective_handbook,
                code_example=effective_example)
            reply = self.generate_data_fetching_code(
                download_prompt_str=prompt, stream_callback=stream_callback)
            code = self.extract_code_from_str(reply)
            if not code.strip():
                return {
                    "status": "failed",
                    "error": "The model did not return executable Python code.",
                    "source_id": source_ID,
                }
            # The control arm has no handbook to declare required credentials,
            # so the generated code is the only place they can be learned. If
            # it reads a variable that is not set, stop BEFORE executing and
            # ask for it by the exact name the code will look for -- running
            # first would just produce a confusing 401 or a KeyError, and the
            # user would have to infer the name from a traceback.
            if not use_handbook:
                wanted = credential_names_in_code(code)
                unset = [n for n in wanted if not str(os.environ.get(n) or "").strip()]
                if unset:
                    if stream_callback:
                        stream_callback(
                            "The generated code needs credentials that are not "
                            "set: " + ", ".join(unset) + ". Pausing to collect them.")
                    return {
                        "status": "needs_credentials",
                        "error": "The generated retrieval code reads "
                                 + ", ".join(unset)
                                 + " from the environment, but "
                                 + ("it is" if len(unset) == 1 else "they are")
                                 + " not set. Enter "
                                 + ("it" if len(unset) == 1 else "them")
                                 + " to continue.",
                        "required_keys": unset,
                        "source_id": source_ID,
                        "generated_code": code,
                    }

            if stream_callback:
                stream_callback("Executing the generated retrieval program.")
            http_tracker = HttpRequestTracker()
            credential_error = None
            with FileWriteTracker() as file_tracker, http_tracker:
                try:
                    final_code = self.execute_complete_program(
                        code=code,
                        try_cnt=max(1, min(int(try_count), 10)),
                        task=str(data_request),
                        model_name=self.model,
                        handbook_str=effective_handbook,
                        stream_callback=stream_callback,
                        http_tracker=http_tracker,
                        file_tracker=file_tracker,
                        # None means "caller didn't specify" -- fall through
                        # to execute_complete_program's own default (300).
                        # 0 is a deliberate, distinct value meaning "no
                        # limit" and must still be forwarded, not treated as
                        # falsy-and-omitted.
                        **({"attempt_timeout": int(attempt_timeout)}
                           if attempt_timeout is not None else {}),
                    )
                except CredentialError as exc:
                    credential_error = exc
                    final_code = code

            files = []
            for path in sorted(file_tracker.captured_paths):
                if not (os.path.isfile(path) and os.path.exists(path)):
                    continue
                try:
                    size = os.path.getsize(path)
                except OSError:
                    size = 0
                files.append({
                    "name": os.path.basename(path),
                    "path": path,
                    "size_bytes": size,
                })

            report = dict(self.last_execution_report or {})
            if credential_error is not None:
                # Prefer the declared names; fall back to whatever the code
                # actually read, so the control arm never asks for a credential
                # without saying which one.
                asked_for = required_keys or credential_names_in_code(final_code)
                return {
                    "status": "needs_credentials",
                    "error": str(credential_error),
                    "required_keys": asked_for,
                    "source_id": source_ID,
                    "generated_code": final_code,
                    "execution": report,
                    "http_requests": http_tracker.captured_requests,
                    "downloaded_files": files,
                }

            success = bool(report.get("success")) and any(
                item["size_bytes"] > 0 for item in files)
            return {
                "status": "passed" if success else "failed",
                "error": "" if success else (
                    report.get("traceback")
                    or "Execution completed without a non-empty output file."),
                "source_id": source_ID,
                "generated_code": final_code,
                "execution": report,
                "http_requests": http_tracker.captured_requests,
                "downloaded_files": files,
            }
        finally:
            self.output_dir = previous_output_dir
            for name, value in previous_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _run_claude_agent_trial(self, data_request, source_id, source_name,
                                handbook_str, output_dir, stream_callback,
                                try_count, extra_env, required_keys):
        """Claude Agent SDK path for ``run_controlled_handbook_trial``.

        Instead of this class's own generate -> exec (in-process, tracker-
        instrumented) -> debug loop, hand the entire write/run/fix cycle to
        one bounded agent session with Read/Write/Edit/Bash scoped to
        ``output_dir``.

        Accepted trade-off: ``FileWriteTracker``/``HttpRequestTracker`` only
        see traffic from code exec'd in THIS process (see their docstrings),
        so they can't observe a script the agent ran via its own Bash tool.
        ``downloaded_files`` is still populated -- real files on disk,
        listed after the session ends -- but ``http_requests`` is always
        empty for a Claude-driven trial.
        """
        # An empty handbook_str IS the control condition -- the caller passes ""
        # for the ablated arm. The worked code example has to be withheld along
        # with it: collecting it unconditionally would have handed the control
        # the very reference implementation the condition removes, on this path
        # only, which is exactly the kind of leak that makes an ablation
        # meaningless without anyone noticing.
        control = not str(handbook_str or "").strip()
        if stream_callback:
            stream_callback("Control condition: handbook withheld." if control
                            else "Generating retrieval code from the pinned handbook.")
        download_prompt = self.create_download_prompt(
            str(data_request), source_name, handbook_str,
            code_example=("" if control else self.collect_code_example(
                source_id, source_dir=self.handbook_dir,
                keys_dir=self.keys_dir)))
        script_name = "retrieve.py"
        agent_prompt = (f"{download_prompt}\n\n"
                        + _claude_agent_instructions(script_name, try_count, control))

        if stream_callback:
            stream_callback("Executing the generated retrieval program.")
        # on_activity streams the agent's real activity (its own commentary,
        # each tool call it makes, and a preview of each tool result) as it
        # happens -- e.g. "Running: python retrieve.py" followed by
        # "  -> Traceback ...". This is the actual work, not an
        # approximation of it, so there's no need to separately synthesize
        # "Executing code (trial N/M)..." progress messages the way the
        # OpenAI path does.
        result = claude_agent_provider.run_agent(
            agent_prompt, tools=["Read", "Write", "Edit", "Bash"],
            cwd=output_dir, max_turns=max(12, try_count * 6),
            model=self.model, api_key=self.anthropic_api_key,
            extra_env=extra_env, on_activity=stream_callback)
        # The whole agent session is one billable unit, and the SDK reports its
        # cost only in the final ResultMessage the provider already captured.
        # The OpenAI path records per streamed chunk; this path had no
        # equivalent, so a Claude-provider retrieval silently reported zero
        # tokens and zero cost -- and provider usage cannot be looked up after
        # the fact, so the figure would have been lost for good.
        usage_ledger.record(self.model, getattr(result, "usage", None),
                            phase="retrieval", source="claude_agent_trial")
        attempt_count = sum(1 for e in result.tool_events if e["name"] == "Bash")

        script_path = os.path.join(output_dir, script_name)
        final_code = ""
        if os.path.isfile(script_path):
            try:
                with open(script_path, encoding="utf-8") as f:
                    final_code = f.read()
            except OSError:
                pass

        files = []
        for name in sorted(os.listdir(output_dir)):
            if name == script_name:
                continue
            path = os.path.join(output_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            files.append({"name": name, "path": path, "size_bytes": size})

        report_text = result.text or ""
        # No exception object to walk here (failures surface as the agent's
        # own prose report, not a Python exception) -- match the same
        # credential-error phrases against that text instead.
        if not result.success and _text_has_credential_error_signs(report_text):
            if stream_callback:
                stream_callback(
                    "\n\n⚠️ **Please check your API key.** The API key for "
                    "this data source appears to be invalid.\n")
            return {
                "status": "needs_credentials",
                "error": ("The API key for this data source appears to be "
                          "invalid. Please check your API key and make "
                          "sure it is correct, then try again."),
                "required_keys": required_keys,
                "source_id": source_id,
                "generated_code": final_code,
                "execution": {"attempts_used": attempt_count, "success": False,
                             "try_budget": try_count, "traceback": report_text,
                             "timed_out": False},
                "http_requests": [],
                "downloaded_files": files,
            }

        success = bool(result.success) and any(f["size_bytes"] > 0 for f in files)
        if stream_callback:
            stream_callback(
                "\n**Code executed successfully.**\n" if success else
                f"\n⚠️ **Failed to execute code after {attempt_count} "
                "attempt(s).**\n")
        return {
            "status": "passed" if success else "failed",
            "error": "" if success else (
                result.error or report_text
                or "Execution completed without a non-empty output file."),
            "source_id": source_id,
            "generated_code": final_code,
            "execution": {"attempts_used": attempt_count, "success": success,
                         "try_budget": try_count,
                         "traceback": "" if success else report_text,
                         "timed_out": False},
            "http_requests": [],
            "downloaded_files": files,
        }

    def extract_code_from_str(self, code_str):
        python_code = ""
        python_code_match = re.search(r"```(?:python)?(.*?)```", code_str, re.DOTALL)
        if python_code_match:
            python_code = python_code_match.group(1).strip()
        return python_code
    
    def execute_complete_program(self, code, try_cnt, task, model_name, handbook_str,
                                  stream_callback=None, http_tracker=None,
                                  file_tracker=None, attempt_timeout=300,
                                  triage=None, credential_names=None):
        """Run ``code``; on failure hand the traceback to the LLM debugger and
        run its fix, up to ``try_cnt`` times.

        Three failure classes never reach the debugger, because no code
        change can fix them: invalid credentials (``CredentialError``),
        throttling (same code is re-run after a backoff), and the source
        having no data for the request (``DataUnavailableError``). The last
        is caught three ways — from the exception/response text, from the
        debugger declining to fix it (``DATA_UNAVAILABLE: ...`` reply), and,
        when ``triage`` is given, from a one-off classification of the error
        after the debugger's first fix has failed. ``triage(failure_note,
        code)`` returns an ``analyze_execution_trace`` dict; a verdict of
        ``request_infeasible`` raises ``DataUnavailableError`` and
        ``external`` stops the loop, while anything else lets debugging
        continue. The verdict is stored on ``last_execution_report["triage"]``
        so the caller's own post-loop triage can reuse it instead of paying
        for a second classification of the same error.

        ``credential_names`` are the environment-variable names this source's
        credentials are exported under (see ``_source_keys_in_env``). A
        failure that says one of them — or any name the code reads via
        ``os.environ`` — is missing is a ``CredentialError`` carrying
        ``missing_keys``, so the caller can ask the user for it instead of
        debugging code that is correct.
        """
        count = 0
        # Structured report for the evaluation harness (L3): how many execute
        # trials were used (repairs = attempts_used - 1), whether the final
        # attempt succeeded, and the final traceback (error-taxonomy material).
        # "attempts" is the full per-trial record (the code that actually ran,
        # its outcome, and — for attempts 2+ — the debugger's stated reason for
        # the change) so the UI can show every attempt, not just the last one.
        exec_report = {"attempts_used": 0, "success": False,
                       "try_budget": try_cnt, "traceback": "", "timed_out": False,
                       "attempts": []}
        self.last_execution_report = exec_report
        # Tracks the (code, error) from the previous failed attempt so we can
        # detect a debugger that made no progress (returned the same code, or
        # the fix didn't change the error) and stop early instead of burning
        # the rest of the retry budget on a loop that isn't converging.
        prev_code = None
        prev_error_signature = None
        # The debugger's stated reasoning for the fix that produced the code
        # about to run next (everything the debug reply said before its code
        # fence) — attached to that next attempt's record once it runs.
        pending_fix_explanation = ""
        while count < try_cnt:
            logging.info(f"Execute the code (trial # {count + 1}/{try_cnt})...")
            if stream_callback:
                stream_callback(f"\nExecuting code (trial {count + 1}/{try_cnt})...\n")
            attempt_record = {
                "attempt": count + 1, "code": code, "success": False,
                "error": "", "timed_out": False, "running": True,
                "fix_explanation": pending_fix_explanation,
            }
            # Published BEFORE the code runs, not after it finishes. The record
            # is appended now and mutated in place when the attempt settles, so
            # a reader holding exec_report (the UI runner polls
            # last_execution_report) can show the code the moment it starts
            # executing. Appending on completion instead meant no code was
            # visible until a 300s download or the whole retry chain had
            # finished -- exactly when it stopped being useful to watch.
            exec_report["attempts"].append(attempt_record)
            pending_fix_explanation = ""
            try:
                count += 1
                exec_report["attempts_used"] = count
                if http_tracker:
                    # Tag this trial's HTTP requests — evaluation L1/L2 score
                    # on the requests of the final attempt.
                    http_tracker.set_attempt(count)
                if file_tracker is not None:
                    # Discard files left behind by earlier failed attempts so
                    # a stale/partial write from a bug we already fixed isn't
                    # reported as a "downloaded file" of the final attempt.
                    file_tracker.reset()
                compiled_code = compile(code, 'Complete program', 'exec')
                # Fresh globals dict per attempt: exec(code, globals()) used
                # to run every attempt — and every concurrent download in this
                # process — against the SAME module-level namespace, so a
                # variable left behind by a failed attempt (or another user's
                # concurrent download) could leak into the next exec() and
                # mask a real bug. Seed a copy so imports/helpers already in
                # scope still work, but nothing written by this attempt
                # escapes it.
                exec_globals = dict(globals())
                # Deliberately NOT a `with ThreadPoolExecutor() as pool:` block —
                # Executor.__exit__ calls shutdown(wait=True), which would block
                # until the hung worker thread actually finishes, silently
                # defeating the timeout below. Shut down with wait=False instead
                # so a timed-out call returns control immediately; the orphaned
                # thread is a deliberate, documented trade-off (Python threads
                # cannot be force-killed without a subprocess).
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = pool.submit(exec, compiled_code, exec_globals)
                # attempt_timeout of 0/None means "no limit" -- Future.result's
                # own timeout=None already means "wait indefinitely", so this
                # is just making that sentinel explicit rather than passing a
                # falsy 0 straight through (which concurrent.futures treats as
                # "return immediately", not "unlimited").
                effective_timeout = attempt_timeout if attempt_timeout else None
                try:
                    future.result(timeout=effective_timeout)
                except concurrent.futures.TimeoutError:
                    exec_report["timed_out"] = True
                    pool.shutdown(wait=False)
                    raise TimeoutError(
                        f"Code did not finish within {effective_timeout}s — "
                        "likely a network call with no timeout that hung. "
                        "Add explicit timeouts to HTTP/download calls."
                    )
                pool.shutdown(wait=False)
                logging.info("Done. Code executed successfully without error.")
                exec_report["success"] = True
                exec_report["traceback"] = ""
                attempt_record["success"] = True
                attempt_record["running"] = False
                if stream_callback:
                    stream_callback("\n**Code executed successfully.**\n")
                return code
            except Exception as err:
                exec_report["traceback"] = traceback.format_exc()
                attempt_record["error"] = str(err)
                attempt_record["timed_out"] = isinstance(err, TimeoutError)
                attempt_record["running"] = False
                logging.error(f"Error on trial {count}/{try_cnt}: {err}")
                print(f"Error on trial {count}/{try_cnt}: {err}")

                # A credential the code needs is simply not there. Not a bug
                # — the debugger's only move would be to hard-code or invent
                # one — and not "invalid" either: the user never got a chance
                # to supply it. Raise with the names so the caller can ask.
                missing = _missing_credential_names(
                    err, list(credential_names or []) + credential_names_in_code(code))
                if missing:
                    shown = ", ".join(missing)
                    msg = (f"This data source needs the credential(s) {shown}, "
                           "which the download code could not find. Enter the "
                           "key below and retry — it is also kept under "
                           "Settings → Data-source keys for next time.")
                    logging.error(
                        f"Missing credential(s) {shown}, aborting retries: {err}")
                    if stream_callback:
                        stream_callback(
                            f"\n\n🔑 **This source needs an API key ({shown}) "
                            "that was not available to the download code.** "
                            "Not a code problem — please provide the key.\n")
                    raise CredentialError(msg, missing_keys=missing) from err

                # Invalid / unauthorized credentials are not code bugs: retrying
                # and LLM-debugging can never fix them. Stop immediately and tell
                # the user clearly instead of silently burning every attempt.
                if _looks_like_credential_error(err):
                    # Keep the user-facing message short and actionable — just
                    # tell them to check the key. The raw provider error (with
                    # the request URL) goes to the logs only, not the UI.
                    msg = (
                        "The API key for this data source appears to be invalid. "
                        "Please check your API key and make sure it is correct, "
                        "then try again."
                    )
                    logging.error(f"Credential error detected, aborting retries: {err}")
                    if stream_callback:
                        stream_callback(f"\n\n⚠️ **Please check your API key.** {msg}\n")
                    raise CredentialError(msg) from err

                # Rate limiting / throttling is not a code bug either: the
                # code is fine, the source just wants us to slow down. Retry
                # the SAME code after a short backoff instead of spending an
                # LLM debug round-trip "fixing" code that was never broken.
                if _looks_like_rate_limit_error(err):
                    if count == try_cnt:
                        logging.error(f"Still rate-limited after {try_cnt} attempts.")
                        if stream_callback:
                            stream_callback(f"\n⚠️ **Still rate-limited after {try_cnt} attempts.**\n")
                        return code
                    backoff_s = min(5 * count, 30)
                    logging.warning(
                        f"Rate-limited on trial {count}/{try_cnt}: {err}. "
                        f"Retrying same code in {backoff_s}s...")
                    if stream_callback:
                        stream_callback(
                            f"\n⏳ Rate-limited by the data source — retrying the "
                            f"same code in {backoff_s}s...\n")
                    time.sleep(backoff_s)
                    continue

                # The source has no data for what was asked (period/area/
                # variable outside its coverage). Not a code bug: the only
                # "fix" the debugger could make is to quietly change the
                # request until something comes back, and that data would
                # not be what the user asked for. Stop and say so.
                if _looks_like_data_unavailable_error(err):
                    reason = _data_unavailable_reason(err)
                    logging.error(
                        f"Data-unavailable error detected, aborting retries: {err}")
                    if stream_callback:
                        stream_callback(
                            "\n\n⛔ **The data source has no data for this "
                            f"request** — {reason}\n\nThis is not a code "
                            "problem, so the download code will not be "
                            "debugged further. Adjust the date range, area, "
                            "or variable in the request, or choose another "
                            "source.\n")
                    raise DataUnavailableError(reason) from err

                if stream_callback:
                    stream_callback(f"\n⚠️ Error on trial {count}/{try_cnt}: {err}\n")

                # Stuck-loop detection: if the debugger's last fix produced the
                # exact same code, or the fix changed nothing about the error,
                # another round-trip won't help — stop instead of exhausting
                # the remaining attempts (and their LLM calls) for nothing.
                error_signature = f"{type(err).__name__}: {err}"
                if code == prev_code or error_signature == prev_error_signature:
                    logging.error(
                        "Debugger made no progress between attempts "
                        f"(same code/error) — stopping early at {count}/{try_cnt}."
                    )
                    if stream_callback:
                        stream_callback(
                            "\n⚠️ **The debugger's fix didn't change the outcome "
                            f"— stopping early instead of using the remaining "
                            f"{try_cnt - count} attempt(s).**\n"
                        )
                    return code
                prev_code = code
                prev_error_signature = error_signature

                # One-off early triage. The debugger has now had one go and
                # the request still fails, and the wording did not match a
                # known coverage phrase — so ask the classifier once whether
                # this is even fixable before spending the remaining
                # attempts. Runs at exactly one trial so a bug the second
                # fix would have solved costs one extra LLM call, not one
                # per trial. Fails open: no callback, an error, or an
                # unrecognised verdict all mean "keep debugging".
                if (triage is not None and count == EARLY_TRIAGE_AFTER_TRIAL
                        and count < try_cnt):
                    failure_note = exec_report["traceback"][-600:].strip()
                    try:
                        analysis = triage(failure_note, code) or {}
                    except Exception as triage_err:
                        logging.warning(f"Early failure triage failed: {triage_err}")
                        analysis = {}
                    # Keyed by the traceback's last line (module-qualified
                    # exception + message), which is what the post-loop
                    # triage in LLM_Find compares against to decide whether
                    # this verdict still describes the final failure.
                    exec_report["triage"] = {
                        "analysis": dict(analysis),
                        "error_signature": (exec_report["traceback"].strip()
                                            .splitlines() or [""])[-1].strip(),
                    }
                    category = str(analysis.get("failure_category")
                                   or "").strip().lower()
                    if category == "request_infeasible":
                        reason = (str(analysis.get("summary") or "").strip()
                                  or error_signature[:400])
                        logging.error(
                            "Triage says the request is infeasible for this "
                            f"source, aborting retries: {reason}")
                        if stream_callback:
                            stream_callback(
                                "\n\n⛔ **The data source cannot satisfy this "
                                f"request** — {reason}\n\nThis is not a code "
                                "problem, so the download code will not be "
                                "debugged further. Adjust the date range, "
                                "area, or variable in the request, or choose "
                                "another source.\n")
                        raise DataUnavailableError(reason) from err
                    if category == "external":
                        logging.error(
                            "Triage says the failure is external — stopping "
                            f"the debug loop early at {count}/{try_cnt}.")
                        if stream_callback:
                            stream_callback(
                                "\n⚠️ **This failure looks external (the "
                                "service, not the code) — stopping instead of "
                                f"using the remaining {try_cnt - count} "
                                "attempt(s).**\n")
                        return code

                if count == try_cnt:
                    logging.error(f"Failed to execute and debug the code within {try_cnt} times.")
                    if stream_callback:
                        stream_callback(f"\n⚠️ **Failed to execute code after {try_cnt} attempts.**\n")
                    return code

                debug_prompt = self.get_debug_prompt(
                    exception=err, code=code, task=task, handbook_str=handbook_str,
                    credential_env_names=[n for n in (credential_names or [])
                                          if n in os.environ])
                logging.info("Sending error information to LLM for debugging...")
                if stream_callback:
                    stream_callback(f"\n**Debugging code (attempt {count + 1})...**\n")
                debug_response = _stream_with_usage(
                    messages=[{"role": "system", "content": debug_prompt}],
                    model=model_name,
                    user_key=self.api_key,
                    base_url=self.base_url,
                    **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}),
                )
                debug_content = ""
                for chunk in debug_response:
                    if getattr(chunk, "usage", None) is not None:
                        usage_ledger.record(model_name, chunk.usage,
                                            phase="retrieval", source="debug_code")
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                    if delta:
                        debug_content += delta
                        if stream_callback:
                            stream_callback(delta)
                # The debugger may decline to "fix" the code because the
                # provider has no data for the request (the debug prompt
                # asks it to say so with a marker line and no code). Honour
                # that instead of extracting an empty program and running it.
                verdict = _data_unavailable_verdict(debug_content)
                if verdict:
                    logging.error(
                        f"Debugger reports the data is unavailable, aborting retries: {verdict}")
                    if stream_callback:
                        stream_callback(
                            "\n\n⛔ **The data source has no data for this "
                            f"request** — {verdict}\n\nThis is not a code "
                            "problem, so the download code will not be "
                            "debugged further. Adjust the date range, area, "
                            "or variable in the request, or choose another "
                            "source.\n")
                    raise DataUnavailableError(verdict) from err
                fixed_code = self.extract_code_from_str(debug_content)
                if not fixed_code.strip():
                    # No verdict honoured and no code either (typically a
                    # schema-based DATA_UNAVAILABLE reply that was demoted).
                    # Keep the current code: re-running it reproduces the
                    # error and the stuck-loop guard above ends the attempt
                    # honestly, instead of executing an empty program.
                    logging.warning(
                        "Debugger returned no code; keeping the previous version.")
                    if stream_callback:
                        stream_callback(
                            "\n⚠️ **The debugger returned no code** — it blamed a "
                            "missing field/column, which is a schema-reading "
                            "problem rather than missing data. Keeping the "
                            "previous code.\n")
                    continue
                code = fixed_code
                # Everything the debugger said before its code fence is its
                # stated reasoning for the change (the debug prompt asks for
                # "Explaination for the revision: ..." first) — keep it for
                # the attempt record this fix produces, since
                # extract_code_from_str discards it.
                pending_fix_explanation = re.sub(
                    r"```.*", "", debug_content, count=1, flags=re.DOTALL).strip()
                logging.info("Received debugged code from LLM, retrying execution...")
        return code  # Return the last version of code after exhausting all tries
    
    

class CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr
    
        
        
