"""Persistent storage for "Generate & Test" handbook sessions.

Sessions are isolated by the caller's hashed user ID and stored as JSON
files, mirroring ``experiment2_store.py``'s mechanics exactly (same ID
validation, atomic writes). This module intentionally keeps its own
whitelist rather than reusing Experiment 2's ``_clean_record`` — that one
only knows about Experiment 2's multi-mechanism schema and would silently
strip fields (like ``generated_source``/``test_result``) it doesn't
recognize.

No Flask imports, so it can be tested and reused independently.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from copy import deepcopy
from datetime import datetime, timezone


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_ROOT = os.path.join(
    PROJECT_ROOT, "Data_retrieval_skill_project", "Handbook_Studio_Sessions"
)
_SAFE_ID = re.compile(r"^[a-f0-9-]{16,64}$")

MECHANISMS = ("rest", "stac", "ogc", "sql", "bulk", "http_file", "arcgis_featureserver")
MAX_CUSTOM_MECHANISM_LENGTH = 80

# Which LLM backend runs handbook generation/refinement/self-verification/
# evaluation (agents/data_agent/handbook_generator.py) and the retrieval
# code-gen/execute/debug loop (DataRetrieverAgent). "claude" requires an
# Anthropic API key and drives every step through Claude Agent SDK sessions
# (see agents/data_agent/claude_agent_provider.py) instead of plain chat
# completions -- including the code-generation/execution/debug step, which
# runs as its own bounded agent session rather than this app's in-process
# exec()+tracker harness (see DataRetrieverAgent._run_claude_agent_trial's
# docstring for what that trades away: HTTP-request evidence, specifically).
PROVIDERS = ("openai", "claude")
DEFAULT_PROVIDER = "openai"

# handbook_gen_model drives handbook generation/refinement/self-verification/
# evaluation; data_retrieval_model drives the code that's actually generated
# and executed for a retrieval test (DataRetrieverAgent). Matches the app's
# main chat model picker in index.html's #model-select for the OpenAI list.
MODEL_OPTIONS = (
    "gpt-5.6-sol", "gpt-5.2", "gpt-5.3", "gpt-5.4",
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
)
DEFAULT_HANDBOOK_GEN_MODEL = "gpt-5.2"
DEFAULT_DATA_RETRIEVAL_MODEL = "gpt-4o"

CLAUDE_MODEL_OPTIONS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
DEFAULT_CLAUDE_HANDBOOK_MODEL = "claude-sonnet-5"
DEFAULT_CLAUDE_DATA_RETRIEVAL_MODEL = "claude-sonnet-5"

# Which of MODEL_OPTIONS support variable reasoning effort, and which
# effort levels each one accepts (mirrors WebUI/script.js's
# REASONING_EFFORT_MAP for the main chat model picker -- keep both in
# sync). Models not listed here (the gpt-4.x family) don't support it at
# all -- the UI hides the control and the backend never sends the param.
REASONING_EFFORT_MODELS = {
    # gpt-5.6-sol's "max" is real (verified live) but ONLY on the Responses
    # API -- the Chat Completions surface this app also uses (refine/
    # evaluate/analyze, and all of DataRetrieverAgent) 400s on "max" with
    # "Supported values are: none, low, medium, high, and xhigh." Offered
    # here anyway since it's a genuine capability for part of the
    # pipeline; agents/data_agent/handbook_generator.py's _call_model and
    # DataRetrieverAgent's Chat Completions call sites downgrade
    # "max" -> "xhigh" rather than erroring.
    "gpt-5.6-sol": ("none", "low", "medium", "high", "xhigh", "max"),
    "gpt-5.2": ("none", "low", "medium", "high", "xhigh"),
    "gpt-5.3": ("low", "medium", "high", "xhigh"),
    "gpt-5.4": ("none", "low", "medium", "high", "xhigh"),
}
DEFAULT_REASONING_EFFORT = "medium"

# Per-attempt wall-clock budget for the generated retrieval code
# (DataRetrieverAgent.execute_complete_program's attempt_timeout). Sources
# vary widely in how long a genuine, non-hung attempt takes -- a few
# seconds for a REST/SQL query, tens of minutes for a bulk/STAC download --
# so this is configurable per session, and the default is unlimited so a
# slow-but-working source is never cut off before the user has a reason to
# set a limit. 0 is a distinct sentinel meaning "no limit"
# (DataRetrieverAgent.execute_complete_program treats 0/None as unlimited),
# not just a very large number -- kept separate from
# MAX_EXECUTION_TIMEOUT_SECONDS so "unlimited" is an explicit choice rather
# than whatever the current numeric ceiling happens to be.
MIN_EXECUTION_TIMEOUT_SECONDS = 60
MAX_EXECUTION_TIMEOUT_SECONDS = 21600  # 6 hours
UNLIMITED_EXECUTION_TIMEOUT = 0
DEFAULT_EXECUTION_TIMEOUT_SECONDS = UNLIMITED_EXECUTION_TIMEOUT


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_dir(user_id: str) -> str:
    safe = re.sub(r"[^a-fA-F0-9_-]", "", str(user_id or ""))[:80]
    if not safe:
        raise ValueError("A valid user ID is required.")
    path = os.path.join(STORE_ROOT, safe)
    os.makedirs(path, exist_ok=True)
    return path


def _session_path(user_id: str, session_id: str) -> str:
    if not _SAFE_ID.match(str(session_id or "")):
        raise ValueError("Invalid session ID.")
    return os.path.join(_user_dir(user_id), f"{session_id}.json")


def new_session(name: str = "Untitled Handbook Test") -> dict:
    now = _now()
    return {
        "schema_version": 1,
        "id": str(uuid.uuid4()),
        "name": str(name or "Untitled Handbook Test").strip()[:160],
        "source_name": "",
        "documentation_url": "",
        "retrieval_task": "",
        # Pre-registered, human-authored TOML/JSON that makes the retrieval
        # task's requirements machine-checkable (see
        # agents/data_agent/validation.py). Free text alone can't be
        # validated mechanically.
        "validation_spec": "",
        "access_mechanism": "",
        "access_mechanism_source": "",
        "access_mechanism_why": "",
        "provider": DEFAULT_PROVIDER,
        "handbook_gen_model": DEFAULT_HANDBOOK_GEN_MODEL,
        "data_retrieval_model": DEFAULT_DATA_RETRIEVAL_MODEL,
        "handbook_gen_reasoning_effort": DEFAULT_REASONING_EFFORT,
        "data_retrieval_reasoning_effort": DEFAULT_REASONING_EFFORT,
        "execution_timeout_seconds": DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        "status": "draft",
        "generated_source": {},
        "test_result": {},
        "test_runs": [],
        "source_slug": "",
        "pipeline_state": {},
        # Control condition. When False the session never generates a handbook
        # and the retrieval agent is given none -- the ablation arm.
        "use_handbook": True,
        # Credential NAMES only. Values are never persisted: they are collected
        # per run in the browser and posted with the request, exactly as the
        # handbook path already does. Without a handbook there is nothing to
        # derive these names from, so the author supplies them.
        "control_key_names": [],
        # Cumulative LLM token usage for the session. Only knowable at call
        # time -- providers expose no way to look it up afterwards -- so it
        # must be part of the persisted schema; a field missing from this
        # whitelist is silently dropped on every save.
        "usage": {},
        # Set while a pipeline is running, cleared when it ends. A browser that
        # reloads mid-run cannot rejoin the SSE stream (it is one-shot), so
        # without this the page has no way to know work is still happening and
        # shows a stage stuck at "running" forever. Carries a heartbeat so a
        # marker orphaned by a killed process can be recognised as stale
        # instead of blocking the session permanently.
        "active_run": {},
        "created_at": now,
        "updated_at": now,
    }


def _clean_record(record: dict, existing: dict | None = None) -> dict:
    if not isinstance(record, dict):
        raise ValueError("Session data must be an object.")
    base = deepcopy(existing) if existing else new_session(record.get("name", ""))

    for field in ("name", "source_name", "documentation_url", "retrieval_task", "status",
                  "access_mechanism_source", "access_mechanism_why", "source_slug",
                  "validation_spec"):
        if field in record:
            base[field] = str(record.get(field) or "").strip()
    base["validation_spec"] = base.get("validation_spec", "")[:20000]
    if base["validation_spec"]:
        # Reject a malformed spec at save time rather than at validation
        # time: a spec that silently fails to parse would leave the run
        # looking unconstrained when the author believed it was constrained.
        from agents.data_agent.validation import parse_task_spec
        parse_task_spec(base["validation_spec"])
    if not base["name"]:
        raise ValueError("Session name is required.")
    base["name"] = base["name"][:160]
    base["source_name"] = base.get("source_name", "")[:300]
    base["documentation_url"] = base.get("documentation_url", "")[:3000]
    base["retrieval_task"] = base.get("retrieval_task", "")[:10000]
    base["access_mechanism_why"] = base.get("access_mechanism_why", "")[:2000]
    base["source_slug"] = base.get("source_slug", "")[:100]

    if "access_mechanism" in record:
        # A recognized fixed option is stored lowercase (the canonical key);
        # anything else is a user-typed custom service type ("Other") --
        # kept as free text (original casing, length-capped) rather than
        # blanked out, the same way source_name/retrieval_task are.
        mechanism = str(record.get("access_mechanism") or "").strip()
        base["access_mechanism"] = (
            mechanism.lower() if mechanism.lower() in MECHANISMS
            else mechanism[:MAX_CUSTOM_MECHANISM_LENGTH])

    if "provider" in record:
        provider = str(record.get("provider") or "").strip().lower()
        base["provider"] = provider if provider in PROVIDERS else DEFAULT_PROVIDER
    elif base.get("provider") not in PROVIDERS:
        base["provider"] = DEFAULT_PROVIDER

    # Model validity/defaults depend on the (now-resolved) provider -- an
    # OpenAI model id is meaningless once provider="claude" and vice versa,
    # so switching provider without also picking a new model falls back to
    # that provider's default rather than keeping a stale, mismatched id.
    if base["provider"] == "claude":
        model_options, model_defaults = CLAUDE_MODEL_OPTIONS, (
            DEFAULT_CLAUDE_HANDBOOK_MODEL, DEFAULT_CLAUDE_DATA_RETRIEVAL_MODEL)
    else:
        model_options, model_defaults = MODEL_OPTIONS, (
            DEFAULT_HANDBOOK_GEN_MODEL, DEFAULT_DATA_RETRIEVAL_MODEL)

    for field, default in zip(
            ("handbook_gen_model", "data_retrieval_model"), model_defaults):
        if field in record:
            chosen = str(record.get(field) or "").strip()
            base[field] = chosen if chosen in model_options else default
        elif field not in base or base.get(field) not in model_options:
            base[field] = default

    # Reasoning effort is only meaningful for the models in
    # REASONING_EFFORT_MODELS -- forced to "" for any model that doesn't
    # support it (the gpt-4.x family), and validated against that specific
    # model's allowed effort levels otherwise (e.g. gpt-5.3 has no "none").
    for model_field, effort_field in (
        ("handbook_gen_model", "handbook_gen_reasoning_effort"),
        ("data_retrieval_model", "data_retrieval_reasoning_effort"),
    ):
        allowed_efforts = REASONING_EFFORT_MODELS.get(base[model_field])
        if not allowed_efforts:
            base[effort_field] = ""
        elif effort_field in record:
            chosen = str(record.get(effort_field) or "").strip().lower()
            base[effort_field] = (
                chosen if chosen in allowed_efforts else DEFAULT_REASONING_EFFORT)
        elif base.get(effort_field) not in allowed_efforts:
            base[effort_field] = DEFAULT_REASONING_EFFORT

    if "execution_timeout_seconds" in record:
        try:
            timeout = int(record.get("execution_timeout_seconds"))
        except (TypeError, ValueError):
            timeout = DEFAULT_EXECUTION_TIMEOUT_SECONDS
        if timeout <= UNLIMITED_EXECUTION_TIMEOUT:
            base["execution_timeout_seconds"] = UNLIMITED_EXECUTION_TIMEOUT
        else:
            base["execution_timeout_seconds"] = max(
                MIN_EXECUTION_TIMEOUT_SECONDS,
                min(MAX_EXECUTION_TIMEOUT_SECONDS, timeout))
    elif not isinstance(base.get("execution_timeout_seconds"), int):
        base["execution_timeout_seconds"] = DEFAULT_EXECUTION_TIMEOUT_SECONDS

    if "use_handbook" in record:
        base["use_handbook"] = bool(record.get("use_handbook"))
    elif not isinstance(base.get("use_handbook"), bool):
        base["use_handbook"] = True

    if "control_key_names" in record:
        names = record.get("control_key_names") or []
        if not isinstance(names, list):
            names = []
        cleaned_names, seen = [], set()
        for name in names:
            text = str(name or "").strip()[:120]
            # Reject anything that looks like a value rather than a name, so a
            # pasted secret cannot end up persisted through this field.
            if text and text not in seen and " " not in text:
                seen.add(text)
                cleaned_names.append(text)
        base["control_key_names"] = cleaned_names[:20]
    elif not isinstance(base.get("control_key_names"), list):
        base["control_key_names"] = []

    for field in ("generated_source", "test_result", "pipeline_state", "usage",
                  "active_run"):
        value = record.get(field, base.get(field, {}))
        base[field] = value if isinstance(value, dict) else {}

    # Public sharing: a read-only link anyone can open (manuscript appendix).
    # Only the owner's requests reach this code, so accepting the flag here
    # is safe; the timestamp is what a reader sees as "shared since".
    if "public" in record:
        was_public = bool(base.get("public"))
        base["public"] = bool(record.get("public"))
        if base["public"] and not was_public:
            base["public_since"] = _now()
        elif not base["public"]:
            base.pop("public_since", None)
    else:
        base["public"] = bool(base.get("public"))

    test_runs = record.get("test_runs", base.get("test_runs", []))
    base["test_runs"] = [_clean_test_run(run) for run in test_runs
                         if isinstance(run, dict)] if isinstance(test_runs, list) else []

    base["updated_at"] = _now()
    return base


def _clean_test_run(run: dict) -> dict:
    """Guarantee every run carries an append-only ``validations`` log.

    Validation used to be a single ``result.semantic_validation`` slot, so
    running a second method over the same run silently destroyed the first
    verdict -- an LLM-judge run after a manual review erased the human's
    judgment, which is the one result the experiment can't reconstruct.
    Records are now appended and never overwritten; ``semantic_validation``
    survives as a derived pointer to the highest-precedence record so the
    existing rendering keeps working.
    """
    result = run.get("result")
    if not isinstance(result, dict):
        return run
    validations = result.get("validations")
    if not isinstance(validations, list):
        validations = []
        legacy = result.get("semantic_validation")
        if isinstance(legacy, dict):
            # Pre-migration data: keep the one verdict that was stored.
            legacy = dict(legacy)
            legacy.setdefault("method", result.get("validation_method") or "llm")
            legacy.setdefault(
                "verdict", "pass" if legacy.get("task_completed") else "fail")
            legacy.setdefault("migrated_from_legacy_slot", True)
            validations = [legacy]
        result["validations"] = validations
    return run


def save_session(user_id: str, record: dict, session_id: str | None = None) -> dict:
    existing = get_session(user_id, session_id) if session_id else None
    cleaned = _clean_record(record, existing=existing)
    if session_id:
        cleaned["id"] = session_id
        cleaned["created_at"] = existing.get("created_at", cleaned["created_at"])
    path = _session_path(user_id, cleaned["id"])
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(prefix=".session_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(cleaned, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return cleaned


def get_session(user_id: str, session_id: str | None) -> dict | None:
    if not session_id:
        return None
    path = _session_path(user_id, session_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def duplicate_session(user_id: str, session_id: str) -> dict:
    """Clone an existing session's generated handbook into a brand-new
    session, skipping generation entirely. Lets an experiment reuse the
    exact same handbook across multiple retrieval tasks (e.g. difficulty
    tiers of the same source) as a controlled variable, rather than
    regenerating a possibly-different handbook for each one."""
    source = get_session(user_id, session_id)
    if source is None:
        raise ValueError("Session not found.")
    if not (source.get("generated_source") or {}).get("data_source_name"):
        raise ValueError("This session has no generated handbook to duplicate yet.")

    now = _now()
    base_name = source.get("name") or source.get("source_name") or "Untitled Handbook Test"
    new = new_session(f"{base_name} (copy)")
    for field in ("source_name", "documentation_url", "retrieval_task",
                  "access_mechanism", "access_mechanism_source", "access_mechanism_why"):
        new[field] = source.get(field, "")
    new["generated_source"] = deepcopy(source.get("generated_source") or {})
    new["status"] = "awaiting_review"

    # The duplicate must run under the SAME configuration as the handbook it
    # reuses. Falling back to the module defaults here meant a handbook
    # generated on one model was reported -- and re-tested -- under another,
    # which silently breaks the "same handbook, different task" control the
    # duplicate exists to provide. The user can still change any of it in
    # Setup before running.
    for field in ("provider", "handbook_gen_model", "data_retrieval_model",
                  "handbook_gen_reasoning_effort", "data_retrieval_reasoning_effort",
                  "execution_timeout_seconds"):
        if source.get(field) not in (None, ""):
            new[field] = source[field]

    # The handbook was generated ONCE, and this session reuses that exact
    # artifact -- so what it cost belongs here too, or a duplicated session
    # reports a handbook that appeared out of nowhere for free.
    #
    # Each carried entry is flagged `inherited` and keeps the id of the session
    # that actually spent the tokens, because the other half of being honest
    # about this is that these tokens must be counted ONCE PER HANDBOOK, not
    # once per session: summing raw totals across a parent and its duplicates
    # would multiply one generation into several. Retrieval runs are never
    # carried -- those tested the parent's task, not this one's.
    from agents.data_agent import usage_ledger

    inherited = []
    for entry in ((source.get("usage") or {}).get("runs") or []):
        if entry.get("phase") != "generation":
            continue
        carried = deepcopy(entry)
        carried["inherited"] = True
        # setdefault, not assignment: duplicating a duplicate must keep
        # pointing at the session that really generated the handbook.
        carried.setdefault("inherited_from", source.get("id") or session_id)
        inherited.append(carried)
    if inherited:
        new["usage"] = {
            "runs": inherited,
            "total": usage_ledger.merge(*inherited),
            "priced": bool((source.get("usage") or {}).get("priced")),
        }

    # Mirrors DataRetrieverAgent.get_required_key_names' no-.keys-file branch,
    # so the credential field (and any carried-over value) shows immediately
    # instead of only appearing after a failed test attempt.
    generated = new["generated_source"]
    if str(generated.get("requires_key", "")).strip().lower() in ("true", "1", "yes"):
        required_keys = [
            n.strip() for n in re.split(r"[,\n]", generated.get("key_name", "") or "")
            if n.strip()
        ]
    else:
        required_keys = []

    new["pipeline_state"] = {
        "status": "awaiting_review",
        "required_keys": required_keys,
        "stages": [
            {
                "key": "duplicated",
                "label": "Handbook reused",
                "description": (
                    "Copied from an earlier session instead of generating a "
                    "new one."),
                "status": "complete",
                "message": (
                    f'Reused the handbook from "{base_name}" without '
                    "regenerating it."),
                "details": [],
                "created_at": now,
            },
            {
                "key": "review",
                "label": "Review the handbook",
                "description": (
                    "Wait for the human to review and, if needed, edit the "
                    "handbook and retrieval task before testing."),
                "status": "warning",
                "message": (
                    "Update the retrieval task for this run, then run the "
                    "retrieval test."),
                "details": [],
                "created_at": now,
            },
        ],
    }
    # Creation-style save (no session_id) so a fresh id is assigned, matching
    # how every other new session is created.
    return save_session(user_id, new)


def find_public_session(session_id: str) -> tuple[str, dict] | None:
    """(owner_user_id, session) for a session its owner has made public.

    Sessions are filed by owner, and a shared link carries only the session
    id, so this looks across every owner directory. Returns None for an
    unknown id, a malformed id, or a session that is not flagged public --
    the caller must not distinguish those cases to the client.
    """
    if not session_id or not _SAFE_ID.match(session_id):
        return None
    if not os.path.isdir(STORE_ROOT):
        return None
    for user_id in os.listdir(STORE_ROOT):
        path = os.path.join(STORE_ROOT, user_id, f"{session_id}.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                session = json.load(handle)
        except (OSError, ValueError):
            return None
        return (user_id, session) if session.get("public") else None
    return None


def list_sessions(user_id: str) -> list[dict]:
    directory = _user_dir(user_id)
    sessions = []
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, filename), encoding="utf-8") as handle:
                session = json.load(handle)
            sessions.append({
                "id": session.get("id", ""),
                "name": session.get("name", "Untitled Handbook Test"),
                "status": session.get("status", "draft"),
                "source_name": session.get("source_name", ""),
                "access_mechanism": session.get("access_mechanism", ""),
                # Carried so the picker can search and preview what a session
                # was actually asking for -- the source name alone doesn't
                # separate "fires in California last week" from "fires in
                # Portugal in 2023". Sent whole rather than truncated: a
                # clipped copy would make the search quietly miss matches
                # past the cut. The field is capped at 10 KB on save.
                "retrieval_task": session.get("retrieval_task", ""),
                "updated_at": session.get("updated_at", ""),
                "public": bool(session.get("public")),
            })
        except (OSError, ValueError, TypeError):
            continue
    sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return sessions


def delete_session(user_id: str, session_id: str) -> bool:
    path = _session_path(user_id, session_id)
    if not os.path.exists(path):
        return False
    os.unlink(path)
    return True
