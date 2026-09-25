"""Pipeline for the "Generate & Test" mode: generate one OpenAI-only handbook,
pause for human review, then run one controlled retrieval trial against it.

Deliberately separate from Experiment 2's multi-mechanism research harness.
This module reuses Experiment 2's generic, dict-shape-agnostic plumbing (log-
capture streaming, stage bookkeeping, result sanitization) by importing it
directly — none of it is modified, and nothing here touches Experiment 2's
own persistence schema.

No Flask imports; every function yields plain event dicts for SSE.
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
import uuid

from WebUI import handbook_studio_store as store
from WebUI.studio_common import (
    _now, _safe, _redact_text, _infer_expected_format, _sanitize_result,
    _run_logged_work, _run_quiet_work, _pipeline_state, _record_stage,
    _stage_event, _source_snapshot, _GEN_STAGE_RE, _GEN_CHAT_RE,
    _GEN_ARTIFACT_RE, _MECHANISM_LABELS, _MECHANISM_GUIDANCE,
)
from agents.data_agent import validation
from agents.data_agent.DataRetrieverAgent import DataRetrieverAgent
from agents.data_agent import user_data_sources
from agents.data_agent import usage_ledger
from WebUI import output_retention


def _default_output_root() -> str:
    """A durable home for retrieved data.

    This used to be /tmp, which cost real work: macOS purges /tmp on its own
    schedule and empties it on reboot, so a run's session record survived while
    the files it downloaded quietly disappeared underneath it. The record still
    listed the file, so the loss only showed up when someone tried to open it.

    Deliberately NOT under the project directory: that tree is synced to
    OneDrive, and retrieved geospatial data runs to tens of gigabytes -- it
    would consume the user's cloud quota to store files that are reproducible
    by re-running the code. A per-user application-data directory is durable,
    local, and outside both git and the sync client.
    """
    override = os.environ.get("GIS_COSCI_OUTPUT_ROOT", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    home = os.path.expanduser("~")
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
    elif sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")
    return os.path.join(base, "gis-co-scientist", "outputs")


OUTPUT_ROOT = _default_output_root()

# Where outputs used to be written. Kept as a READ path only -- nothing new is
# written here -- so files from earlier runs that /tmp has not yet purged stay
# openable in the UI instead of vanishing the moment this constant changed.
LEGACY_OUTPUT_ROOT = (
    os.path.join(tempfile.gettempdir(), "GISHBStudio")
    if os.name == "nt"
    else os.path.join("/tmp", "gis-co-scientist-hbstudio")
)

# A failure analyze_execution_trace categorizes as "external" (network/
# server-side, e.g. an outage or a transient error) isn't a handbook defect,
# so the refinement loop retries the SAME handbook this many times before
# giving up -- no LLM revision call, no H(n+1) version, doesn't count against
# max_refinements or "revisions to converge".
_MAX_EXTERNAL_RETRIES = 2

# How often a running retrieval checks free disk on a quota-limited host.
_DISK_WATCHDOG_SECONDS = 5

# Marker log lines from generate_handbook map onto these Studio-local stages.
_GEN_STAGE_MAP = {
    "source": ("documentation", "Documentation discovery",
               "Locate official documentation for this source."),
    "docs": ("documentation", "Documentation discovery",
             "Read the official documentation and access requirements."),
    "draft": ("draft", "Initial handbook draft",
              "Create a handbook grounded in the research."),
    "verify": ("verification", "Self-verification",
               "Run the sample-download verification."),
}
_GEN_STATUS_MAP = {
    "start": "running", "progress": "running", "complete": "complete",
    "warning": "warning", "error": "error",
}


def _session_output_roots(user_id: str, session_id: str) -> list[str]:
    """The active output root, then the legacy one, for READING a saved file.

    Order matters: a file present in both wins from the active root. Callers
    that write must use _session_output_root() (the first entry) -- the legacy
    root is never created or written to.
    """
    tail = (_safe(user_id)[:12], _safe(session_id)[:12])
    return [os.path.abspath(os.path.join(OUTPUT_ROOT, *tail)),
            os.path.abspath(os.path.join(LEGACY_OUTPUT_ROOT, *tail))]


def _session_output_root(user_id: str, session_id: str) -> str:
    root = _session_output_roots(user_id, session_id)[0]
    os.makedirs(root, exist_ok=True)
    return root


def _handbook_dir(user_id: str, session_id: str) -> str:
    path = os.path.join(_session_output_root(user_id, session_id), "_handbook")
    os.makedirs(path, exist_ok=True)
    return path


def _materialize_handbook(user_id: str, session_id: str, source: dict) -> tuple[str, str]:
    source_id = f"hbstudio_{_safe(session_id)[:16]}"
    directory = _handbook_dir(user_id, session_id)
    user_data_sources._write_source(
        os.path.join(directory, f"{source_id}.toml"), source)
    return source_id, directory


def _persist(user_id: str, session_id: str, session: dict) -> None:
    store.save_session(user_id, session, session_id=session_id)


# A marker older than this with no heartbeat is treated as abandoned (the
# process died mid-run) rather than as work still in progress.
ACTIVE_RUN_STALE_SECONDS = 180
# How often the marker's heartbeat is written while a run is going. Every
# heartbeat is a disk write, and the UI polls far slower than 1 Hz, so there is
# nothing to gain from writing one per second.
ACTIVE_RUN_TOUCH_SECONDS = 10

_APP_COMMIT: str | None = None


def _app_commit() -> str:
    """Short git revision of the running code, or "" outside a checkout.

    Pinned onto every run so a result can be tied to the code that produced it
    -- the difference between "this case failed" and "this case failed on the
    build before the windowing fix".
    """
    global _APP_COMMIT
    if _APP_COMMIT is None:
        import subprocess
        try:
            _APP_COMMIT = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=store.PROJECT_ROOT, capture_output=True, text=True,
                timeout=5, check=True).stdout.strip()
        except Exception:
            _APP_COMMIT = ""
    return _APP_COMMIT


def _run_environment(session: dict, use_handbook: bool, try_count: int = 3,
                     output_dir: str = "") -> dict:
    """What produced this run, pinned to the run itself.

    Session-level settings are editable between runs, so reading the model off
    the session tells you what is configured NOW -- not what generated a result
    recorded weeks ago. Recording the conditions per run is what lets a results
    table stand on its own instead of needing the model attribution
    reconstructed by hand afterwards.
    """
    return {
        "provider": session.get("provider") or "",
        "handbook_gen_model": session.get("handbook_gen_model") or "",
        "handbook_gen_reasoning_effort":
            session.get("handbook_gen_reasoning_effort") or "",
        "data_retrieval_model": session.get("data_retrieval_model") or "",
        "data_retrieval_reasoning_effort":
            session.get("data_retrieval_reasoning_effort") or "",
        "execution_timeout_seconds": session.get("execution_timeout_seconds"),
        "access_mechanism": session.get("access_mechanism") or "",
        "use_handbook": bool(use_handbook),
        "try_count": int(try_count),
        "app_commit": _app_commit(),
        "output_dir": os.path.basename(output_dir) if output_dir else "",
        "recorded_at": _now(),
    }


def _mark_run_active(user_id: str, session_id: str, session: dict, kind: str,
                     label: str, stage_key: str = "") -> None:
    session["active_run"] = {
        "kind": kind, "label": label, "stage_key": stage_key,
        "started_at": _now(), "heartbeat": _now(),
    }
    _persist(user_id, session_id, session)


def _touch_run_active(user_id: str, session_id: str, session: dict,
                      stage_key: str = "") -> None:
    marker = session.get("active_run")
    if not isinstance(marker, dict) or not marker:
        return
    marker["heartbeat"] = _now()
    if stage_key:
        marker["stage_key"] = stage_key
    _persist(user_id, session_id, session)


def _clear_run_active(session: dict) -> None:
    """Drop the marker. Callers persist -- every exit path already does."""
    if isinstance(session, dict):
        session["active_run"] = {}


def _model_prices() -> dict:
    """Per-model USD rates, from HANDBOOK_MODEL_PRICES, or {} when unset.

    Rates are configuration, never a constant in the source: they differ per
    account and change over time, and a stale hardcoded number produces a
    confidently wrong cost. With no table the token counts are still recorded
    and cost is simply reported as unpriced.

        HANDBOOK_MODEL_PRICES='{"gpt-5.2": {"input": 1.25, "output": 10.0}}'
    """
    raw = os.environ.get("HANDBOOK_MODEL_PRICES", "").strip()
    if not raw:
        return {}
    try:
        table = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return table if isinstance(table, dict) else {}


def _record_usage(session: dict, ledger, phase: str) -> dict:
    """Fold one ledger's totals into the session's cumulative usage.

    Stored per phase as well as in total, so "what did generation cost versus
    refinement" is answerable later without re-running anything.
    """
    summary = ledger.totals(_model_prices())
    summary["phase"] = phase
    usage = session.setdefault("usage", {"runs": [], "total": {}})
    usage.setdefault("runs", []).append(summary)
    usage["total"] = usage_ledger.merge(*usage["runs"])
    usage["priced"] = bool(_model_prices())
    return summary


def _cancel_path(user_id: str, session_id: str) -> str:
    directory = os.path.join(store.STORE_ROOT, _safe(user_id))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f".cancel-{_safe(session_id)}.flag")


def request_cancel(user_id: str, session_id: str) -> None:
    with open(_cancel_path(user_id, session_id), "w", encoding="utf-8") as handle:
        handle.write(_now())


def _clear_cancel(user_id: str, session_id: str) -> None:
    path = _cancel_path(user_id, session_id)
    if os.path.exists(path):
        os.unlink(path)


def _is_cancelled(user_id: str, session_id: str) -> bool:
    return os.path.exists(_cancel_path(user_id, session_id))


def _provider_and_key(session, api_key, anthropic_api_key):
    """Which LLM backend this session uses, and the matching credential.

    ``provider`` is persisted per-session (set via the normal session
    save/update endpoint, same as ``handbook_gen_model`` etc.) rather than
    passed as a call argument -- this just resolves it plus the right key
    for the caller's ``provider=`` kwarg on generate_handbook/
    select_access_mechanism/DataRetrieverAgent/analyze_execution_trace/
    evaluate_retrieval_result/refine_handbook.
    """
    provider = str((session or {}).get("provider") or "openai")
    if provider not in ("openai", "claude"):
        provider = "openai"
    key = anthropic_api_key if provider == "claude" else api_key
    return provider, key


def _failure_signature(result):
    """A coarse signature for 'is this the same underlying failure as
    another attempt' -- lets a refinement-cap stop distinguish 'kept fixing
    one bug and finding a new one underneath' (real progress; more revisions
    might have converged) from 'stuck repeating the same failure' (more
    revisions would not help). Mirrors handbook_generator._error_signature's
    idea (strip the volatile URL, keep ExceptionType:HTTPStatus) but kept as
    a separate, small copy since this reads a controlled-test result dict,
    not a raw subprocess capture. Returns None when there's no error text to
    compare (e.g. a semantic-validation-only failure)."""
    err = str((result or {}).get("error") or "").strip()
    if not err:
        return None
    last_line = err.splitlines()[-1]
    last_line = re.split(r"\s+for url:", last_line, maxsplit=1)[0]
    match = re.match(r"^([\w.]+(?:Error|Exception)):\s*(.*)$", last_line)
    if not match:
        return last_line
    exc_type, detail = match.groups()
    status_match = re.match(r"^(\d{3})\b", detail)
    return f"{exc_type}:{status_match.group(1)}" if status_match else exc_type


def run_generate_stream(user_id: str, api_key: str, session_id: str,
                        anthropic_api_key: str | None = None):
    """Generate one handbook, then pause for human review. Backend (OpenAI
    or Claude) is whatever the session's ``provider`` field says."""
    from agents.data_agent.handbook_generator import (
        HANDBOOK_GEN_MODEL, generate_handbook, select_access_mechanism)

    session = None
    # Token usage is only knowable at call time -- providers expose no way to
    # look it up afterwards -- so the whole generation phase runs inside a
    # ledger and its totals are written onto the session before it is saved.
    _gen_ctx = usage_ledger.collect("generation")
    _gen_ledger = _gen_ctx.__enter__()
    try:
        session = store.get_session(user_id, session_id)
        if session is None:
            raise ValueError("Session not found.")
        provider, provider_key = _provider_and_key(
            session, api_key, anthropic_api_key)
        source_name = str(session.get("source_name") or "").strip()
        website = str(session.get("documentation_url") or "").strip()
        task_text = str(session.get("retrieval_task") or "").strip()
        if not (source_name or website):
            raise ValueError("Enter a data source name or documentation URL.")

        # Handbook generation is a long LLM call; mark it so a page reloaded
        # while it runs knows to wait rather than showing a dead stage.
        _mark_run_active(user_id, session_id, session, "generate",
                         "Handbook generation")

        if session.get("use_handbook", True) is False:
            # Control condition: there is no handbook to generate. The session
            # still needs a source record, because the retrieval agent is
            # addressed by source id and the source NAME is what tells the
            # model which source to use -- that is not handbook knowledge, it
            # is the task. Everything the handbook would have carried
            # (instructions, code example) is left empty.
            # Only the source name. The TOML writer defaults every other
            # stored field to empty already, so spelling out blank handbook /
            # code_example keys would just be noise in the record -- and a
            # record that carries empty handbook fields reads as "a handbook
            # that happens to be blank" rather than "no handbook".
            session["generated_source"] = {
                "data_source_name": source_name or website,
            }
            session["status"] = "awaiting_review"
            state = _pipeline_state(session)
            state.update({"status": "awaiting_review", "finished_at": _now()})
            stage = _record_stage(
                session, "control_condition", "Control condition",
                "No handbook is generated for this session.", "complete",
                "This session runs without a handbook. Go to Review and test "
                "to run the retrieval.")
            _persist(user_id, session_id, session)
            yield _stage_event(stage)
            yield {
                "type": "handbook_ready_for_review",
                "message": "Control condition: no handbook generated. "
                           "Run the retrieval from the Review and test tab.",
                "session": session,
            }
            return
        # The retrieval task is optional at generation time — a general-
        # purpose handbook can be drafted from the source/mechanism alone,
        # and the task can be added or edited on the Review & test step
        # before the handbook is actually tested (which does require one).

        _clear_cancel(user_id, session_id)
        session["status"] = "running"
        state = _pipeline_state(session)
        state["status"] = "running"
        _persist(user_id, session_id, session)
        yield {
            "type": "pipeline_started",
            "message": "Starting handbook generation.",
            "session": session,
        }

        base_query = source_name or website
        # Preserve the raw casing for a custom (free-text "Other") service
        # type — mechanism_key is the lowercased form used only for looking
        # up the fixed-option dicts; mechanism is what gets stored/displayed/
        # sent to the model, so a custom type like "Custom SOAP API" doesn't
        # get flattened to lowercase.
        mechanism = str(session.get("access_mechanism") or "").strip()
        mechanism_key = mechanism.lower()
        if not mechanism:
            stage = _record_stage(
                session, "mechanism_selection", "Choosing a service type",
                "Select the most appropriate way to access this source.",
                "running",
                "Choosing the best service type for this source and task.")
            _persist(user_id, session_id, session)
            yield _stage_event(stage)

            work = _run_quiet_work(
                lambda: select_access_mechanism(
                    base_query, website, user_key=provider_key,
                    provider=provider),
                "Still choosing the best service type…")
            mech_result = None
            while True:
                try:
                    event = next(work)
                except StopIteration as done:
                    mech_result = done.value
                    break
                if _is_cancelled(user_id, session_id):
                    raise InterruptedError("Handbook generation cancelled.")
                yield event
            mechanism = str((mech_result or {}).get("mechanism") or "rest")
            mechanism_key = mechanism
            session["access_mechanism"] = mechanism
            session["access_mechanism_source"] = "auto"
            session["access_mechanism_why"] = str((mech_result or {}).get("why") or "")
            stage = _record_stage(
                session, "mechanism_selection", "Choosing a service type",
                "Select the most appropriate way to access this source.",
                "complete",
                f"Selected {_MECHANISM_LABELS.get(mechanism_key, mechanism)} as "
                "the service type.",
                artifact={
                    "mechanism": mechanism,
                    "why": session["access_mechanism_why"],
                })
            _persist(user_id, session_id, session)
            yield _stage_event(stage)
        else:
            session["access_mechanism_source"] = (
                session.get("access_mechanism_source") or "user")

        mechanism_label = _MECHANISM_LABELS.get(mechanism_key, mechanism)
        task_context = (
            f"Retrieval task for capability context: {task_text}\n"
            if task_text else
            "No specific retrieval task was given yet — draft a general-"
            "purpose handbook covering this source's main capabilities "
            "under the required mechanism; a specific task will be added "
            "before it's tested.\n")
        mechanism_query = (
            f"{base_query}\n\nRequired access mechanism: {mechanism_label}. "
            f"{_MECHANISM_GUIDANCE.get(mechanism_key, '')}\n{task_context}"
            "Generate and verify a handbook specifically for this mechanism; "
            "do not silently substitute another mechanism.")

        documentation_key = "documentation"
        stage = _record_stage(
            session, documentation_key, "Documentation discovery",
            "Locate official documentation for this source.", "running",
            f"Finding official documentation and access requirements for "
            f"{mechanism_label}.")
        _persist(user_id, session_id, session)
        yield _stage_event(stage)

        generation = _run_logged_work(lambda: generate_handbook(
            mechanism_query,
            website=website,
            doc_urls=[website] if website else [],
            user_key=provider_key,
            model=session.get("handbook_gen_model") or HANDBOOK_GEN_MODEL,
            reasoning_effort=session.get("handbook_gen_reasoning_effort") or None,
            verify=True,
            data_source_keys={},
            provider=provider,
        ))
        generated = None
        while True:
            try:
                event = next(generation)
            except StopIteration as done:
                generated, _logs = done.value
                break
            if _is_cancelled(user_id, session_id):
                raise InterruptedError("Handbook generation cancelled.")
            if event.get("kind") == "heartbeat":
                yield {
                    "type": "heartbeat", "stage_key": documentation_key,
                    "message": f"{mechanism_label} handbook generation is "
                               "still working…",
                }
                continue
            line = event.get("line") or ""
            stage_match = _GEN_STAGE_RE.match(line)
            chat_match = _GEN_CHAT_RE.match(line)
            artifact_match = _GEN_ARTIFACT_RE.match(line)
            if stage_match:
                raw_stage, raw_status, message = stage_match.groups()
                if raw_stage not in _GEN_STAGE_MAP:
                    continue
                key, label, description = _GEN_STAGE_MAP[raw_stage]
                status = _GEN_STATUS_MAP.get(raw_status, "running")
                if raw_stage == "source" and status == "complete":
                    status = "running"
                stage = _record_stage(
                    session, key, label, description, status, message,
                    detail=message if raw_status == "progress" else "")
                yield _stage_event(stage)
            elif chat_match:
                current = next((
                    item for item in reversed(_pipeline_state(session)["stages"])
                    if item.get("status") == "running"), None)
                if current:
                    stage = _record_stage(
                        session, current["key"], current["label"],
                        current["description"], current["status"],
                        current.get("message", ""), detail=chat_match.group(1))
                    yield _stage_event(stage)
            elif artifact_match:
                raw_stage, raw_json = artifact_match.groups()
                key = {"source": "documentation", "docs": "documentation",
                       "draft": "draft"}.get(raw_stage)
                current = next((
                    item for item in _pipeline_state(session)["stages"]
                    if item.get("key") == key), None) if key else None
                if current:
                    try:
                        artifact = json.loads(raw_json)
                    except ValueError:
                        artifact = {}
                    stage = _record_stage(
                        session, key, current["label"], current["description"],
                        current["status"], current.get("message", ""),
                        artifact=artifact)
                    yield _stage_event(stage)
            else:
                # generate_handbook() logs plenty of finer-grained narration
                # ("Docs fetch: read 4,213 characters from ...", "LLM call:
                # retrying...", "[verify] attempt 2 failed — revising", ...)
                # that isn't tagged with a [stage:...] marker. Without this
                # branch those lines are silently dropped instead of
                # streaming onto the currently running stage's card.
                text = _redact_text(
                    re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", line).strip(),
                    [k for k in (api_key, anthropic_api_key) if k])
                if text:
                    current = next((
                        item for item in reversed(_pipeline_state(session)["stages"])
                        if item.get("status") == "running"), None)
                    if current:
                        stage = _record_stage(
                            session, current["key"], current["label"],
                            current["description"], current["status"],
                            current.get("message", ""), detail=text)
                        yield _stage_event(stage)

        source = dict(generated or {})
        verification = source.pop("_verification", None) or {}
        # Sample-output previews are raw text from the sandbox run; scrub the
        # provider keys before they are persisted on the stage card.
        samples = verification.get("samples")
        if isinstance(samples, dict):
            secrets = [k for k in (api_key, anthropic_api_key) if k]
            samples["stdout"] = _redact_text(samples.get("stdout", ""), secrets)
            for item in samples.get("files") or []:
                item["preview"] = _redact_text(item.get("preview", ""), secrets)
        doc_stage = _record_stage(
            session, documentation_key, "Documentation discovery",
            "Locate official documentation for this source.", "complete",
            "The documentation has been collected.")
        yield _stage_event(doc_stage)
        draft_stage = _record_stage(
            session, "draft", "Initial handbook draft",
            "Create a handbook grounded in the research.", "complete",
            "The handbook draft has been created.",
            artifact={"source": _source_snapshot(source)})
        yield _stage_event(draft_stage)
        verify_stage = _record_stage(
            session, "verification", "Self-verification",
            "Run the sample-download verification.",
            "complete" if verification.get("verified") else "warning",
            verification.get("note") or "Self-verification finished.",
            artifact={"verification": verification})
        yield _stage_event(verify_stage)

        source["_verification"] = verification
        source_id, _handbook_directory = _materialize_handbook(
            user_id, session_id, source)
        session["generated_source"] = source
        session["source_slug"] = source_id
        session["status"] = "awaiting_review"
        state = _pipeline_state(session)
        state["status"] = "awaiting_review"

        review_message = (
            "The handbook is ready to review. Edit it if needed, then run "
            "the retrieval test.")
        review_stage = _record_stage(
            session, "review", "Review the handbook",
            "Wait for the human to review and, if needed, edit the handbook "
            "and retrieval task before testing.", "warning", review_message)
        if _gen_ledger is not None:
            _record_usage(session, _gen_ledger, "generation")
        _persist(user_id, session_id, session)
        yield _stage_event(review_stage)
        yield {
            "type": "handbook_ready_for_review",
            "message": review_message,
            "session": session,
        }
    except InterruptedError:
        session = session or store.get_session(user_id, session_id)
        if session:
            state = _pipeline_state(session)
            state.update({"status": "interrupted", "finished_at": _now()})
            session["status"] = "draft"
            _persist(user_id, session_id, session)
        yield {
            "type": "pipeline_cancelled",
            "message": "Handbook generation stopped at a safe checkpoint.",
            "session": session,
        }
    except Exception as exc:
        session = session or store.get_session(user_id, session_id)
        if session:
            state = _pipeline_state(session)
            state.update({
                "status": "error", "finished_at": _now(), "error": str(exc)})
            session["status"] = "error"
            stage = _record_stage(
                session, "pipeline_error", "Pipeline error",
                "An unexpected error stopped handbook generation.",
                "error", str(exc))
            _persist(user_id, session_id, session)
            yield _stage_event(stage)
        yield {"type": "error", "error": str(exc), "session": session}
    finally:
        if session is not None:
            _clear_run_active(session)
            _persist(user_id, session_id, session)
        # Tokens spent before a failure still cost money, so the ledger closes
        # on every path -- including the interrupted and error ones.
        _gen_ctx.__exit__(None, None, None)


def run_test_stream(user_id: str, api_key: str, session_id: str,
                    edited_task: str | None = None,
                    edited_source: dict | None = None,
                    user_keys: dict | None = None,
                    anthropic_api_key: str | None = None,
                    use_handbook: bool | None = None):
    """Run one controlled retrieval trial against the reviewed handbook.

    The condition comes from the session (``use_handbook``), set once at setup,
    so every run in a session belongs to the same arm and a control session
    cannot accidentally be tested with a handbook. ``use_handbook`` here is an
    explicit override for callers that need one; None means "use the session's".
    """
    session = None
    # Retrieval spends tokens too (code generation, and a debugger rewrite per
    # failed attempt), so the test phase gets its own ledger and its cost is
    # attributed to the run rather than lumped in with generation.
    _test_ctx = usage_ledger.collect("retrieval")
    _test_ledger = _test_ctx.__enter__()
    try:
        session = store.get_session(user_id, session_id)
        if session is None:
            raise ValueError("Session not found.")
        control_session = session.get("use_handbook", True) is False
        if not control_session and not session.get("generated_source"):
            raise ValueError("Generate a handbook before running the retrieval test.")
        provider, provider_key = _provider_and_key(
            session, api_key, anthropic_api_key)
        if use_handbook is None:
            use_handbook = session.get("use_handbook", True) is not False

        _clear_cancel(user_id, session_id)
        keys = {
            str(name): str(value) for name, value in (user_keys or {}).items()
            if name and value
        }

        source = dict(session.get("generated_source") or {})
        if isinstance(edited_source, dict):
            for field in user_data_sources.STORED_FIELDS:
                if field in edited_source:
                    source[field] = str(edited_source.get(field) or "")
        task_text = str(
            edited_task if edited_task is not None
            else session.get("retrieval_task") or "").strip()
        if not task_text:
            raise ValueError("Enter the retrieval task before running the test.")

        session["generated_source"] = source
        session["retrieval_task"] = task_text
        session["status"] = "testing"
        state = _pipeline_state(session)
        state["status"] = "testing"
        source_id, handbook_dir = _materialize_handbook(
            user_id, session_id, source)
        session["source_slug"] = source_id
        _persist(user_id, session_id, session)
        yield {
            "type": "pipeline_started",
            "message": "Starting the retrieval test.",
            "session": session,
        }

        expected_format = _infer_expected_format(task_text)
        test_number = len(session.get("test_runs") or []) + 1
        stage_key = f"retrieval_test_{test_number}"
        stage_label = f"Retrieval test #{test_number}"
        test_started_at = _now()
        stage = _record_stage(
            session, stage_key, stage_label,
            "Execute the retrieval task against this handbook.", "running",
            "Preparing the retrieval program.")
        _mark_run_active(user_id, session_id, session, "test", stage_label,
                         stage_key)
        _persist(user_id, session_id, session)
        yield _stage_event(stage)

        agent = DataRetrieverAgent(
            api_key=api_key, extra_handbook_dirs=[handbook_dir],
            evaluation=True, model=session.get("data_retrieval_model"),
            reasoning_effort=session.get("data_retrieval_reasoning_effort") or None,
            provider=provider, anthropic_api_key=anthropic_api_key)

        events: "queue.Queue" = queue.Queue()
        box: dict = {}
        last_progress = {"message": ""}

        def stream_message(message):
            if _is_cancelled(user_id, session_id):
                raise InterruptedError("Retrieval test cancelled.")
            # Raw LLM code/log tokens are intentionally not surfaced live —
            # only the translated, human-readable sentences below are.
            plain = re.sub(r"[*`#]", "", str(message or "")).strip()
            progress = ""
            attempt = re.search(
                r"Executing code \(trial\s+(\d+)/(\d+)\)", plain, re.I)
            repair = re.search(
                r"Debugging code \(attempt\s+(\d+)\)", plain, re.I)
            failed_attempt = re.search(
                r"Error on trial\s+(\d+)/(\d+)", plain, re.I)
            rate_limit_wait = re.search(
                r"retrying the same code in\s+(\d+)s", plain, re.I)
            rate_limit_exhausted = re.search(
                r"Still rate-limited after\s+(\d+)\s+attempts", plain, re.I)

            if plain == "Generating retrieval code from the pinned handbook.":
                progress = "Reading the task and preparing retrieval code from the handbook."
            elif plain == "Control condition: handbook withheld.":
                # The control arm must never claim to be reading a handbook.
                progress = ("Control condition: no handbook. Reading the task "
                            "and preparing retrieval code from the model's own "
                            "knowledge of this source.")
            elif plain.startswith("The generated code needs credentials"):
                progress = plain
            elif plain == "Executing the generated retrieval program.":
                progress = "The retrieval program is ready. Starting execution."
            elif attempt:
                progress = (
                    f"Executing retrieval attempt {attempt.group(1)} of "
                    f"{attempt.group(2)} and monitoring the data response.")
            elif "Code executed successfully." in plain:
                progress = (
                    "Execution succeeded. Checking downloaded files and "
                    "validation evidence.")
            elif "Please check your API key." in plain:
                progress = (
                    "The data source rejected the supplied credential. "
                    "Execution has paused safely.")
            elif rate_limit_wait:
                progress = (
                    "The data source is rate-limiting requests. Waiting "
                    f"{rate_limit_wait.group(1)}s before retrying the same "
                    "code (no code changes needed).")
            elif rate_limit_exhausted:
                progress = (
                    f"Still rate-limited after {rate_limit_exhausted.group(1)} "
                    "attempts. Returning the last response received.")
            elif "fix didn't change the outcome" in plain:
                progress = (
                    "The last repair attempt didn't change the outcome — "
                    "stopping early instead of repeating a fix that isn't "
                    "converging.")
            elif "Failed to execute code after" in plain:
                progress = (
                    "All permitted execution attempts have finished. "
                    "Preserving the failure evidence for analysis.")
            elif repair:
                progress = (
                    "Analyzing the execution error and preparing repair "
                    f"attempt {repair.group(1)}.")
            elif failed_attempt:
                progress = (
                    f"Execution attempt {failed_attempt.group(1)} did not "
                    "pass. Collecting the error evidence for automatic repair.")
            elif provider == "claude" and plain:
                # Claude Agent SDK activity (tool calls, truncated tool
                # result previews, assistant commentary from
                # claude_agent_provider.run_agent's on_activity callback)
                # isn't shaped like the OpenAI-path phrases above -- surface
                # it directly rather than silently dropping it. Unlike the
                # OpenAI path, DataRetrieverAgent._run_claude_agent_trial
                # never streams raw generated code token-by-token here, so
                # there's no equivalent noise to filter out; gated to the
                # Claude provider so the OpenAI path's raw-code-stream
                # suppression (see the comment above) is unaffected.
                progress = plain if len(plain) <= 300 else plain[:300] + "…"

            progress = _redact_text(progress, list(keys.values())).strip()
            if progress and progress != last_progress["message"]:
                last_progress["message"] = progress
                events.put(("progress", progress))

        output_dir = os.path.join(
            _session_output_root(user_id, session_id),
            f"test_{uuid.uuid4().hex[:8]}")

        # On a quota-limited host, refuse to start a retrieval that the disk
        # can't hold rather than fail halfway through with a full volume.
        refusal = output_retention.preflight()
        if refusal:
            raise RuntimeError(refusal)

        def worker():
            # The trial -- and every LLM call inside it -- runs on this thread,
            # while the ledger was opened on the request thread. Adopting it
            # here is what makes the retrieval tokens land somewhere; without
            # it the run reports zero cost despite plainly having spent it.
            with usage_ledger.adopt(_test_ledger):
              try:
                # Two named entry points rather than a flag at the call site,
                # so which arm a run belongs to is obvious from the code.
                trial = (agent.run_controlled_handbook_trial if use_handbook
                         else agent.run_control_trial_without_handbook)
                extra = ({} if use_handbook else
                         {"required_key_names": session.get("control_key_names") or []})
                box["result"] = trial(
                    data_request=task_text,
                    source_ID=source_id,
                    source_name=source.get("data_source_name"),
                    **extra,
                    user_keys=keys,
                    stream_callback=stream_message,
                    output_dir=output_dir,
                    try_count=3,
                    attempt_timeout=session.get("execution_timeout_seconds"),
                )
              except InterruptedError:
                box["interrupted"] = True
              except Exception as exc:
                box["result"] = {"status": "failed", "error": str(exc)}
              finally:
                events.put(("done", None))

        # Which attempts have already been sent, and in what state, so each is
        # emitted once when it starts and once more when it settles.
        seen_attempts: dict = {}
        secrets = list(keys.values())

        def attempt_events():
            """The trial's attempts so far, as UI events.

            The agent publishes its execution report the moment its attempt
            loop begins and mutates it in place, so the code for attempt N is
            readable here while N is still running. Reading it from the worker
            thread's report rather than waiting for the trial to return is the
            whole point: a return-value-only view shows nothing until every
            retry and every download has finished.

            The code is redacted before it leaves this process -- generated
            retrieval code routinely contains the user's key inline, and this
            is the first path that puts that code on screen mid-run.
            """
            report = getattr(agent, "last_execution_report", None) or {}
            for record in list(report.get("attempts") or []):
                number = record.get("attempt")
                if number is None:
                    continue
                status = ("running" if record.get("running")
                          else "timed_out" if record.get("timed_out")
                          else "succeeded" if record.get("success")
                          else "failed")
                if seen_attempts.get(number) == status:
                    continue
                seen_attempts[number] = status
                yield {
                    "type": "attempt_code",
                    "stage_key": stage_key,
                    "attempt": number,
                    "index": number - 1,
                    "status": status,
                    "code": _redact_text(str(record.get("code") or ""), secrets)[:60000],
                    "note": str(record.get("fix_explanation") or ""),
                    "error": _redact_text(str(record.get("error") or ""), secrets)[:4000],
                }

        started = time.monotonic()
        last_touch = [started]
        last_disk_check = [started]
        # Ignore a report left on the agent by any earlier trial; only attempts
        # from the run started below should reach this session's UI.
        agent.last_execution_report = None
        threading.Thread(target=worker, daemon=True).start()
        while True:
            try:
                kind, payload = events.get(timeout=1)
            except queue.Empty:
                yield from attempt_events()
                now = time.monotonic()
                if now - last_touch[0] >= ACTIVE_RUN_TOUCH_SECONDS:
                    last_touch[0] = now
                    # Tells a reloaded page this run is alive, not abandoned.
                    _touch_run_active(user_id, session_id, session, stage_key)
                if now - last_disk_check[0] >= _DISK_WATCHDOG_SECONDS:
                    last_disk_check[0] = now
                    # A download that would fill the volume is stopped through
                    # the same cancel flag the user's Stop button uses; the
                    # worker sees it on its next progress message.
                    abort = output_retention.watchdog()
                    if abort and not box.get("disk_abort"):
                        box["disk_abort"] = abort
                        request_cancel(user_id, session_id)
                yield {
                    "type": "heartbeat", "stage_key": stage_key,
                    "elapsed_seconds": round(now - started),
                    "message": last_progress["message"]
                               or "Preparing the retrieval program…",
                }
                continue
            if kind == "done":
                break
            yield from attempt_events()
            yield {"type": "stage_progress", "stage_key": stage_key,
                   "message": payload}
        # Final states: the last attempt settles as the worker exits, so its
        # closing status would otherwise never be sent.
        yield from attempt_events()

        if box.get("disk_abort"):
            # Our own cancel, not the user's: clear the flag so the session
            # isn't left looking cancelled, and surface the disk message as
            # the run's error instead of "cancelled".
            _clear_cancel(user_id, session_id)
            raise RuntimeError(box["disk_abort"])
        if box.get("interrupted"):
            raise InterruptedError("Retrieval test cancelled.")

        raw_result = box.get("result") or {"status": "failed"}
        if raw_result.get("status") == "needs_credentials":
            required = list(raw_result.get("required_keys") or [])
            session["status"] = "awaiting_review"
            state.update({
                "status": "awaiting_credentials", "required_keys": required})
            stage = _record_stage(
                session, stage_key, stage_label,
                "Execute the retrieval task against this handbook.",
                "warning",
                "Valid credentials are required before the retrieval test "
                "can run.")
            _persist(user_id, session_id, session)
            yield _stage_event(stage)
            yield {
                "type": "credential_required",
                "required_keys": required,
                "message": (
                    "This source needs credentials before the retrieval "
                    "test can run."),
                "session": session,
            }
            return

        result = _sanitize_result(
            raw_result, keys, expected_format,
            artifact_root=_session_output_root(user_id, session_id))

        trace_analysis = None
        if result.get("status") != "passed":
            # The run didn't structurally pass (crash, empty output, wrong
            # format). Diagnose *why* — fixable handbook gap vs. external
            # failure — the same evidence Experiment 2's automated refinement
            # loop uses, so failed trials aren't left with only a raw
            # traceback.
            from agents.data_agent.handbook_generator import analyze_execution_trace

            analysis_key = f"trace_analysis_{test_number}"
            analysis_label = f"Failure analysis #{test_number}"
            # The control arm has no handbook, so it is asked whether the
            # failure was external or a gap in the model's own knowledge of
            # the source. Framing it as a handbook question -- against the
            # empty handbook a control session stores -- returned
            # "handbook_deficiency" for essentially every control failure.
            analysis_desc = (
                "Distinguish a gap in the model's own knowledge of the source "
                "from an external failure." if control_session else
                "Identify handbook deficiencies and distinguish them from "
                "external failures.")
            analysis_stage = _record_stage(
                session, analysis_key, analysis_label, analysis_desc, "running",
                "Analyzing the execution trace for a gap in the model's "
                "knowledge of this source." if control_session else
                "Analyzing the execution trace for a fixable handbook gap.")
            _persist(user_id, session_id, session)
            yield _stage_event(analysis_stage)

            analysis_work = _run_quiet_work(
                lambda: analyze_execution_trace(
                    source, task_text, result, user_key=provider_key,
                    model=session.get("handbook_gen_model"),
                    reasoning_effort=session.get("handbook_gen_reasoning_effort") or None,
                    provider=provider, control=control_session),
                "Analyzing endpoints, parameters, and operational evidence…")
            while True:
                try:
                    event = next(analysis_work)
                except StopIteration as done:
                    trace_analysis = done.value
                    break
                event["stage_key"] = analysis_key
                yield event

            result["trace_analysis"] = trace_analysis
            result["failure_category"] = (
                trace_analysis.get("failure_category") or "Unknown")
            analysis_stage = _record_stage(
                session, analysis_key, analysis_label, analysis_desc, "complete",
                trace_analysis.get("summary") or "Failure analysis finished.",
                artifact={"analysis": trace_analysis})
            _persist(user_id, session_id, session)
            yield _stage_event(analysis_stage)

        outcome = "passed" if result.get("status") == "passed" else "failed"
        finish_message = (
            "Retrieval test succeeded." if outcome == "passed"
            else "Retrieval test did not pass — review the evidence below.")
        stage = _record_stage(
            session, stage_key, stage_label,
            "Execute the retrieval task against this handbook.",
            "complete" if outcome == "passed" else "warning", finish_message,
            artifact={"execution_result": result})
        # Snapshot the handbook actually under test, and diff it against the
        # previous test run, so refinement iterations and handbook revisions
        # are reconstructable later instead of being silently overwritten.
        existing_runs = session.get("test_runs") or []
        previous_run = existing_runs[-1] if existing_runs else None
        handbook_snapshot = _source_snapshot(source)
        handbook_changed = (
            previous_run is not None
            and previous_run.get("handbook_snapshot") != handbook_snapshot)
        task_changed = (
            previous_run is not None
            and previous_run.get("retrieval_task") != task_text)
        if previous_run is None:
            refinement_iteration = 0
        elif handbook_changed:
            refinement_iteration = int(
                previous_run.get("refinement_iteration", 0)) + 1
        else:
            refinement_iteration = int(
                previous_run.get("refinement_iteration", 0))

        run_record = {
            "id": str(uuid.uuid4()),
            "test_number": test_number,
            "refinement_iteration": refinement_iteration,
            "retrieval_task": task_text,
            "handbook_snapshot": handbook_snapshot,
            "handbook_changed": handbook_changed,
            "task_changed": task_changed,
            "manual_intervention": bool(handbook_changed or task_changed),
            # The experimental condition this run belongs to. Recorded on the
            # run rather than the session so both arms can live in one case and
            # be compared pairwise.
            "condition": "with_handbook" if use_handbook else "no_handbook",
            "usage": _test_ledger.totals(_model_prices()),
            # What produced this run: provider, models, efforts, timeout,
            # retry budget and the app revision. Pinned here because the
            # session's settings can change before anyone reads the result.
            "environment": _run_environment(
                session, use_handbook, try_count=3, output_dir=output_dir),
            "outcome": outcome,
            "started_at": test_started_at,
            "finished_at": _now(),
            "result": result,
            # Size of what this run left on disk and, on a quota-limited host,
            # how long its bulk files are guaranteed to stay. Small files
            # (handbook, manifests, samples) are never evicted.
            "retention": output_retention.after_run(output_dir),
        }
        session.setdefault("test_runs", []).append(run_record)
        session["test_result"] = result
        session["status"] = "complete"
        state["status"] = "complete"
        _persist(user_id, session_id, session)
        yield _stage_event(stage)
        yield {
            "type": "test_finished", "outcome": outcome,
            "message": finish_message, "session": session,
        }
    except InterruptedError:
        session = session or store.get_session(user_id, session_id)
        if session:
            state = _pipeline_state(session)
            state.update({"status": "interrupted", "finished_at": _now()})
            session["status"] = "awaiting_review"
            _persist(user_id, session_id, session)
        yield {
            "type": "pipeline_cancelled",
            "message": "The retrieval test stopped at a safe checkpoint.",
            "session": session,
        }
    except Exception as exc:
        session = session or store.get_session(user_id, session_id)
        if session:
            state = _pipeline_state(session)
            state.update({
                "status": "error", "finished_at": _now(), "error": str(exc)})
            session["status"] = "error"
            stage = _record_stage(
                session, "pipeline_error", "Pipeline error",
                "An unexpected error stopped the retrieval test.",
                "error", str(exc))
            _persist(user_id, session_id, session)
            yield _stage_event(stage)
        yield {"type": "error", "error": str(exc), "session": session}
    finally:
        if session is not None:
            _record_usage(session, _test_ledger, "retrieval")
            # Whatever happened -- finished, cancelled, errored, or the client
            # walked away and this ran on the detached thread -- the run is no
            # longer active, and a stale marker would leave the page waiting
            # for something that already ended.
            _clear_run_active(session)
            _persist(user_id, session_id, session)
        _test_ctx.__exit__(None, None, None)


def _perform_refinement(user_id, session_id, api_key, session, state,
                        current_source, task_text, analysis, result,
                        refine_key, refine_label, next_version,
                        anthropic_api_key=None):
    """Shared refine_handbook invocation used by both the automatic
    execution-failure refinement loop (run_test_with_refinement_stream) and
    the manual, Validation-tab-triggered refinement
    (run_validation_refine_stream). Yields the same progress/stage events
    either caller relies on; returns (via StopIteration.value, since this is
    a generator) one of: {"status": "refined", "source": dict} |
    {"status": "no_change"} | {"status": "cancelled"} |
    {"status": "error", "error": str}.
    """
    from agents.data_agent.handbook_generator import (
        HANDBOOK_GEN_MODEL, CLAUDE_HANDBOOK_MODEL, build_refinement_feedback,
        refine_handbook, refinement_changed)

    provider, provider_key = _provider_and_key(session, api_key, anthropic_api_key)

    stage = _record_stage(
        session, refine_key, refine_label,
        "Revise the handbook using only the recorded execution evidence.",
        "running",
        f"Creating {next_version} from the previous attempt's failure "
        "analysis.")
    _persist(user_id, session_id, session)
    yield {
        "type": "refinement_started", "version": next_version,
        "message": f"Creating {next_version} from the failure analysis.",
        "session": session,
    }
    yield _stage_event(stage)

    # Executed successfully means this is a correctness/completeness gap,
    # not a crash -- build_refinement_feedback flags it explicitly so the
    # reviser doesn't trade a working, data-producing script for one that
    # fails to run at all (see _REFINE_INSTRUCTIONS' non-regression rule).
    # The wording lives in handbook_generator because LLM_Find's mid-retrieval
    # repair asks for the same revision and must ask for it in the same words.
    feedback = build_refinement_feedback(
        task_text, analysis, error=result.get("error", ""),
        mechanism=session.get("access_mechanism", ""),
        prior_execution_ok=bool((result.get("execution") or {}).get("success")))

    try:
        default_model = (
            CLAUDE_HANDBOOK_MODEL if provider == "claude" else HANDBOOK_GEN_MODEL)
        refine_work = _run_logged_work(lambda: refine_handbook(
            current_source, feedback,
            website=str(session.get("documentation_url") or ""),
            fetch_docs=True, user_key=provider_key,
            model=session.get("handbook_gen_model") or default_model,
            reasoning_effort=session.get("handbook_gen_reasoning_effort") or None,
            provider=provider))
        refined = None
        while True:
            try:
                event = next(refine_work)
            except StopIteration as done:
                refined, _logs = done.value
                break
            if _is_cancelled(user_id, session_id):
                raise InterruptedError("Handbook refinement cancelled.")
            if event.get("kind") == "heartbeat":
                yield {
                    "type": "heartbeat", "stage_key": refine_key,
                    "message": f"{next_version} refinement is still "
                               "working…",
                }
                continue
            line = event.get("line") or ""
            detail = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", line).strip()
            if detail:
                stage = _record_stage(
                    session, refine_key, refine_label,
                    "Revise the handbook using only the recorded "
                    "execution evidence.", "running", detail=detail)
                yield {"type": "stage_progress", "stage_key": refine_key,
                       "message": detail}
    except InterruptedError:
        state.update({"status": "interrupted", "finished_at": _now()})
        session["status"] = "awaiting_review"
        _persist(user_id, session_id, session)
        yield {
            "type": "pipeline_cancelled",
            "message": "Handbook refinement stopped at a safe checkpoint.",
            "session": session,
        }
        return {"status": "cancelled"}
    except Exception as exc:
        state.update({
            "status": "error", "finished_at": _now(), "error": str(exc)})
        session["status"] = "error"
        stage = _record_stage(
            session, "pipeline_error", "Pipeline error",
            "An unexpected error stopped automatic refinement.",
            "error", str(exc))
        _persist(user_id, session_id, session)
        yield _stage_event(stage)
        yield {"type": "error", "error": str(exc), "session": session}
        return {"status": "error", "error": str(exc)}

    next_source = refined["source"]
    if not refinement_changed(current_source, next_source):
        stage = _record_stage(
            session, refine_key, refine_label,
            "Revise the handbook using only the recorded execution "
            "evidence.", "warning",
            "The proposed refinement made no change to the handbook — "
            "stopping instead of retesting an identical handbook.")
        session["status"] = "awaiting_review"
        state["status"] = "awaiting_review"
        _persist(user_id, session_id, session)
        yield _stage_event(stage)
        yield {
            "type": "refinement_finished", "outcome": "no_change",
            "message": "The refinement produced no handbook change.",
            "session": session,
        }
        return {"status": "no_change"}

    session["generated_source"] = next_source
    stage = _record_stage(
        session, refine_key, refine_label,
        "Revise the handbook using only the recorded execution evidence.",
        "complete",
        refined.get("assistant_message") or f"{next_version} is ready.",
        artifact={"version": next_version, "source": _source_snapshot(next_source)})
    _persist(user_id, session_id, session)
    yield _stage_event(stage)
    return {"status": "refined", "source": next_source}


def run_test_with_refinement_stream(user_id: str, api_key: str, session_id: str,
                                    edited_task: str | None = None,
                                    edited_source: dict | None = None,
                                    user_keys: dict | None = None,
                                    max_refinements: int = 3,
                                    anthropic_api_key: str | None = None):
    """Run a controlled retrieval trial; if it fails for a reason the
    execution-trace or semantic-validation analysis identifies as a fixable
    handbook gap, revise the handbook (H1, H2, H3, ...) using only that
    evidence and retry — up to ``max_refinements`` times, stopping as soon as
    a test passes, a failure isn't a handbook deficiency, or a revision makes
    no change to the handbook.

    Each attempt still goes through run_test_stream, so refinement_iteration,
    handbook_changed, and the full test_runs history already reflect H0..Hn
    correctly with no additional bookkeeping — this function only decides
    *whether* to keep going and performs the revision itself.
    """
    session = store.get_session(user_id, session_id)
    if session is not None and session.get("use_handbook", True) is False:
        # A control session has no handbook, so there is nothing to refine --
        # and revising one would quietly turn the control back into the
        # treatment. The UI already routes control sessions to the plain
        # test endpoint; this is the same rule enforced where it cannot be
        # bypassed by calling the API directly.
        yield from run_test_stream(
            user_id, api_key, session_id, edited_task=edited_task,
            edited_source=edited_source, user_keys=user_keys,
            anthropic_api_key=anthropic_api_key)
        return

    task_override = edited_task
    source_override = edited_source
    iteration = 0
    external_retries = 0

    while True:
        if _is_cancelled(user_id, session_id):
            return

        latest_outcome = None
        for event in run_test_stream(
                user_id, api_key, session_id,
                edited_task=task_override, edited_source=source_override,
                user_keys=user_keys, anthropic_api_key=anthropic_api_key):
            yield event
            event_type = event.get("type")
            if event_type == "test_finished":
                latest_outcome = event.get("outcome")
            elif event_type in ("credential_required", "pipeline_cancelled", "error"):
                return
        # Only apply an edit on the first attempt — subsequent attempts test
        # whatever run_test_stream just persisted (H0, then each Hn).
        task_override = None
        source_override = None

        if latest_outcome == "passed" or latest_outcome is None:
            return

        session = store.get_session(user_id, session_id)
        if session is None:
            return
        test_runs = session.get("test_runs") or []
        if not test_runs:
            return
        latest_run = test_runs[-1]
        result = latest_run.get("result") or {}

        # A failure categorized as "external" (network/server-side, e.g. an
        # outage or a transient error) isn't a handbook defect, so retry the
        # SAME handbook a bounded number of times before deciding whether to
        # spend a revision on it -- this costs no LLM call, creates no H(n+1)
        # version, and doesn't count against max_refinements or "revisions
        # to converge" (run_test_stream only advances refinement_iteration
        # when the handbook snapshot actually changes).
        # Exact match is the contract (analyze_execution_trace's prompt now
        # constrains the model to the literal strings "handbook_deficiency"/
        # "external"), but tolerate case/whitespace variance defensively
        # rather than silently falling through to the revision path if the
        # model ever drifts from the literal value.
        if str(result.get("failure_category") or "").strip().lower() == "external":
            if external_retries < _MAX_EXTERNAL_RETRIES:
                external_retries += 1
                stage = _record_stage(
                    session, "refinement_external_retry",
                    f"Retrying after an external failure "
                    f"({external_retries}/{_MAX_EXTERNAL_RETRIES})",
                    "The failure analysis identified this as an "
                    "external/transient failure, not a handbook defect.",
                    "warning",
                    "Retrying the same handbook without spending a revision "
                    "— an external failure doesn't mean the handbook needs "
                    "to change.")
                _persist(user_id, session_id, session)
                yield _stage_event(stage)
                yield {
                    "type": "refinement_retry",
                    "message": "External failure — retrying the same "
                               f"handbook ({external_retries}/"
                               f"{_MAX_EXTERNAL_RETRIES}).",
                    "session": session,
                }
                continue

            stage = _record_stage(
                session, "refinement_external_stopped",
                "Refinement stopped — external failure",
                "Automatic refinement does not spend revisions chasing "
                "external/transient failures.", "warning",
                f"Still failing externally after {_MAX_EXTERNAL_RETRIES} "
                "retries with the same handbook. This looks like a network "
                "or server-side issue outside the handbook's control — "
                "review manually or try again later.")
            _persist(user_id, session_id, session)
            yield _stage_event(stage)
            yield {
                "type": "refinement_finished", "outcome": "external_failure",
                "message": "Stopped after repeated external/transient "
                           "failures — not a handbook defect.",
                "session": session,
            }
            return

        if iteration >= max_refinements:
            # Tell "each revision fixed one bug and uncovered a new one"
            # (real progress, just out of budget) apart from "the same
            # failure kept recurring" (more revisions likely wouldn't help)
            # -- a fixed max_refinements can be right for one case and too
            # low for another, so this is reported per-case rather than by
            # raising the shared default.
            failed_runs = [r for r in test_runs if r.get("outcome") != "passed"]
            signatures = [_failure_signature(r.get("result") or {})
                         for r in failed_runs]
            signatures = [s for s in signatures if s]
            distinct_failures = len(set(signatures))
            made_progress = distinct_failures >= 2
            progress_note = (
                f" {distinct_failures} distinct failures were seen across "
                "attempts, each fixed before the next appeared — this case "
                "may need a higher refinement limit to converge."
                if made_progress and signatures else
                " The same underlying failure recurred despite revisions — "
                "a higher refinement limit likely would not help; review "
                "the handbook manually."
                if signatures else "")
            stage = _record_stage(
                session, "refinement_limit", "Automatic refinement stopped",
                f"Refinement stops after {max_refinements} revisions.",
                "warning",
                f"Stopped after {iteration} automatic revision"
                f"{'s' if iteration != 1 else ''} — the handbook still did "
                f"not pass.{progress_note} Review the evidence above and "
                "edit it manually if needed.")
            _persist(user_id, session_id, session)
            yield _stage_event(stage)
            yield {
                "type": "refinement_finished", "outcome": "stopped",
                "message": "Reached the automatic refinement limit.",
                "distinct_failures": distinct_failures,
                "made_progress": made_progress,
                "session": session,
            }
            return

        # Decide refinability + build feedback from whichever analysis this
        # attempt produced: trace_analysis for a structural failure,
        # semantic_validation for a "ran fine but didn't satisfy the task"
        # failure. An empty deficiencies list / should_refine_handbook=False
        # means the analysis judged this not a fixable handbook gap.
        trace_analysis = result.get("trace_analysis")
        semantic = result.get("semantic_validation")
        if trace_analysis is not None:
            analysis = trace_analysis
            should_refine = bool(trace_analysis.get("deficiencies"))
        elif semantic is not None:
            analysis = semantic
            should_refine = bool(semantic.get("should_refine_handbook"))
        else:
            analysis = None
            should_refine = False

        if not should_refine or analysis is None:
            stage = _record_stage(
                session, "refinement_stopped", "Refinement not attempted",
                "Automatic refinement only runs for evidence-grounded "
                "handbook deficiencies.", "warning",
                "The failure analysis did not identify a fixable handbook "
                "gap, so the handbook was not revised automatically.")
            _persist(user_id, session_id, session)
            yield _stage_event(stage)
            yield {
                "type": "refinement_finished", "outcome": "not_refinable",
                "message": "No refinable handbook deficiency was identified.",
                "session": session,
            }
            return

        # Undo the "complete" status run_test_stream just set — from the
        # user's view this session is still actively working.
        session["status"] = "testing"
        state = _pipeline_state(session)
        state["status"] = "testing"

        next_iteration = iteration + 1
        next_version = f"H{next_iteration}"
        refine_key = f"handbook_refinement_{next_iteration}"
        refine_label = f"Handbook refinement · {next_version}"
        current_source = session.get("generated_source") or {}
        task_text = str(session.get("retrieval_task") or "")

        refine_work = _perform_refinement(
            user_id, session_id, api_key, session, state, current_source,
            task_text, analysis, result, refine_key, refine_label,
            next_version, anthropic_api_key=anthropic_api_key)
        outcome_state = None
        while True:
            try:
                event = next(refine_work)
            except StopIteration as done:
                outcome_state = done.value
                break
            yield event

        if outcome_state["status"] != "refined":
            return

        iteration = next_iteration
        # Loop back: the next run_test_stream call re-reads the session from
        # disk, so it will pick up the generated_source we just saved.


def _apply_validation(run, record):
    """Append one validation record and refresh the derived pointers.

    Append-only on purpose: each method's verdict is a separate observation,
    and the human verdict in particular must survive a later LLM-judge run
    (see _clean_test_run). ``semantic_validation``/``validation_method``
    stay as derived pointers to the highest-precedence record so existing
    rendering and refinement code needs no change.
    """
    result = run.get("result") or {}
    record = dict(record)
    record["id"] = str(uuid.uuid4())
    record["created_at"] = _now()
    validations = result.get("validations")
    if not isinstance(validations, list):
        validations = []
    validations.append(record)
    result["validations"] = validations
    primary = validation.primary_validation(validations)
    result["semantic_validation"] = primary
    result["validation_method"] = primary.get("method")
    run["result"] = result
    run["validated_at"] = _now()
    return primary


def run_validation_stream(user_id: str, api_key: str, session_id: str,
                          run_id: str, method: str,
                          manual_verdict: dict | None = None,
                          anthropic_api_key: str | None = None):
    """Judge whether one completed retrieval test run's output actually
    satisfies the task -- separate from, and triggered independently of,
    Review & Test (which only judges "did this handbook retrieve real,
    non-empty data", per run_test_stream). ``method`` is one of:

    "mechanical" -- deterministic checks over harness-captured evidence,
        driven by the session's pre-registered validation spec. No LLM. Every
        dimension is pass/fail/undecidable, and "undecidable" routes to a
        human instead of rolling up to a pass ("metadata" is accepted as the
        old name for this method).
    "manual" -- a human adjudication rubric: per-dimension verdicts with the
        evidence the reviewer cited for each.
    "llm" -- the evidence-grounded LLM judge. Kept so its agreement with the
        human gold standard can be measured; it is never the reported
        measurement when a mechanical or manual record exists.

    Each call APPENDS a record to run["result"]["validations"]; nothing is
    overwritten.
    """
    session = None
    try:
        session = store.get_session(user_id, session_id)
        if session is None:
            raise ValueError("Session not found.")
        runs = session.get("test_runs") or []
        run = next((item for item in runs if item.get("id") == run_id), None)
        if run is None:
            raise ValueError("Test run not found.")
        result = run.get("result") or {}
        if not (result.get("validation") or {}).get("output_present"):
            raise ValueError("This run has no retrieved output to validate.")

        method = str(method or "").strip().lower()
        if method == "metadata":
            method = "mechanical"
        if method not in ("mechanical", "llm", "manual"):
            raise ValueError(f"Unknown validation method: {method!r}")

        provider, provider_key = _provider_and_key(
            session, api_key, anthropic_api_key)
        source = dict(session.get("generated_source") or {})
        task_text = str(
            run.get("retrieval_task") or session.get("retrieval_task") or "")
        stage_key = f"validation_{run.get('test_number')}"
        stage_label = f"Validation · Test #{run.get('test_number')}"

        yield {"type": "pipeline_started", "message": "Starting validation.",
               "session": session}

        if method == "manual":
            record = validation.manual_record(manual_verdict)
            stage = _record_stage(
                session, stage_key, stage_label,
                "Human adjudication against the pre-registered rubric.",
                "complete", record["summary"])
            _persist(user_id, session_id, session)
            yield _stage_event(stage)

        elif method == "mechanical":
            description = ("Deterministic checks against the pre-registered "
                           "spec, using harness-captured URLs and the "
                           "artifact's own computed properties.")
            stage = _record_stage(session, stage_key, stage_label, description,
                                  "running", "Running deterministic checks…")
            _persist(user_id, session_id, session)
            yield _stage_event(stage)
            record = validation.evaluate_mechanical(
                task_text, result,
                spec_text=session.get("validation_spec") or "",
                mechanism=str(session.get("access_mechanism") or ""),
                artifact_root=_session_output_root(user_id, session_id))
            stage = _record_stage(session, stage_key, stage_label, description,
                                  "complete", record["summary"])
            _persist(user_id, session_id, session)
            yield _stage_event(stage)

        else:  # method == "llm"
            from agents.data_agent.handbook_generator import (
                evaluate_retrieval_result)
            mechanism = str(session.get("access_mechanism") or "").strip()
            mechanism_label = _MECHANISM_LABELS.get(mechanism, mechanism)
            description = ("LLM judge: kept as a comparison arm against the "
                           "mechanical and human verdicts, not as the "
                           "reported measurement.")
            stage = _record_stage(
                session, stage_key, stage_label, description, "running",
                "Checking whether the output actually satisfies the task.")
            _persist(user_id, session_id, session)
            yield _stage_event(stage)

            semantic_work = _run_quiet_work(
                lambda: evaluate_retrieval_result(
                    source, task_text, result, mechanism=mechanism_label,
                    user_key=provider_key, model=session.get("handbook_gen_model"),
                    reasoning_effort=session.get(
                        "handbook_gen_reasoning_effort") or None,
                    provider=provider),
                "Inspecting request parameters and output evidence…")
            analysis = None
            while True:
                try:
                    event = next(semantic_work)
                except StopIteration as done:
                    analysis = done.value
                    break
                event["stage_key"] = stage_key
                yield event

            record = validation.llm_record(analysis)
            stage = _record_stage(
                session, stage_key, stage_label, description, "complete",
                record.get("summary") or "Validation finished.")
            _persist(user_id, session_id, session)
            yield _stage_event(stage)

        primary = _apply_validation(run, record)
        _persist(user_id, session_id, session)
        yield {
            "type": "validation_finished",
            "verdict": record.get("verdict"),
            "task_completed": bool(record.get("task_completed")),
            "primary_method": primary.get("method"),
            "message": f"Validation finished: {record.get('verdict')}.",
            "session": session,
        }
    except Exception as exc:
        yield {"type": "error", "error": str(exc), "session": session}


def run_validation_refine_stream(user_id: str, api_key: str, session_id: str,
                                 run_id: str, anthropic_api_key: str | None = None):
    """Send a test run that Validation judged incomplete back for a single
    handbook revision, using its semantic_validation as the evidence --
    triggered deliberately from the Validation tab rather than
    automatically. Deliberately does NOT auto-retest afterward: the user
    re-runs Review & Test explicitly on the new handbook version, keeping
    the two tabs' actions distinct.
    """
    session = None
    try:
        session = store.get_session(user_id, session_id)
        if session is None:
            raise ValueError("Session not found.")
        runs = session.get("test_runs") or []
        run = next((item for item in runs if item.get("id") == run_id), None)
        if run is None:
            raise ValueError("Test run not found.")
        result = run.get("result") or {}
        analysis = result.get("semantic_validation")
        if not analysis:
            raise ValueError("This run has not been validated yet.")
        # Only a determinate failure is evidence a handbook change could act
        # on. An "undecidable" verdict means nothing was established -- if it
        # were allowed through, the model would be asked to revise a handbook
        # against a non-finding, which is how a requirement gets rationalized
        # away rather than fixed. Adjudicate it manually first.
        if validation.normalize_verdict(analysis.get("verdict")) == validation.UNDECIDABLE:
            raise ValueError(
                "This run's validation is undecidable, not failed. Adjudicate "
                "it manually before sending it back for refinement.")

        state = _pipeline_state(session)
        session["status"] = "testing"
        state["status"] = "testing"

        existing_runs = session.get("test_runs") or []
        last_run = existing_runs[-1] if existing_runs else None
        next_iteration = int((last_run or {}).get("refinement_iteration") or 0) + 1
        next_version = f"H{next_iteration}"
        refine_key = f"handbook_refinement_{next_iteration}"
        refine_label = f"Handbook refinement · {next_version} (from validation)"
        current_source = session.get("generated_source") or {}
        task_text = str(
            run.get("retrieval_task") or session.get("retrieval_task") or "")

        refine_work = _perform_refinement(
            user_id, session_id, api_key, session, state, current_source,
            task_text, analysis, result, refine_key, refine_label,
            next_version, anthropic_api_key=anthropic_api_key)
        outcome_state = None
        while True:
            try:
                event = next(refine_work)
            except StopIteration as done:
                outcome_state = done.value
                break
            yield event

        if outcome_state["status"] == "refined":
            session["status"] = "awaiting_review"
            state["status"] = "awaiting_review"
            _persist(user_id, session_id, session)
    except Exception as exc:
        yield {"type": "error", "error": str(exc), "session": session}
