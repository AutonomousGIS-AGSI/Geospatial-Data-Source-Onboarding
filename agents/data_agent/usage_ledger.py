"""Token accounting for the Generate & Test pipeline.

Why this exists: the cost of onboarding a data source automatically is the
number the framework is judged on, and it was not being recorded anywhere. Once
a run finishes, per-request usage is unrecoverable -- providers do not expose
historical usage keyed by anything this application knows, and the prompts are
not retained. So it has to be captured at call time or not at all.

Design notes:

* A ledger is scoped to one pipeline run with ``collect()``. Nesting is safe;
  the innermost active ledger receives the records. Calls made with no ledger
  active are silently ignored, so instrumenting a call site never changes
  behaviour outside a collection scope.
* State is thread-local. The Flask app runs pipelines on worker threads, and
  usage from one session must never land in another's total.
* Records are kept individually, not just summed, so cost can be recomputed
  later at different prices and broken down by phase -- "what did refinement
  cost separately from generation" is the interesting question for the paper.
* Prices are NOT hardcoded. Rates change and vary per account; a wrong constant
  buried in code produces confidently wrong figures. ``price_table`` is
  supplied by the caller (see HANDBOOK_MODEL_PRICES in the environment or the
  UI settings), and cost is reported as null when a model has no rate rather
  than being silently treated as free.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

_local = threading.local()


def _stack():
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = []
        _local.stack = stack
    return stack


@contextmanager
def collect(phase=None):
    """Open a ledger for the duration of the block.

    >>> with collect("generation") as ledger:
    ...     ...                       # LLM calls here are recorded
    >>> ledger.totals()["total_tokens"]
    """
    ledger = Ledger(phase=phase)
    _stack().append(ledger)
    try:
        yield ledger
    finally:
        stack = _stack()
        if stack and stack[-1] is ledger:
            stack.pop()
        elif ledger in stack:            # defensive: out-of-order exit
            stack.remove(ledger)


def active():
    stack = _stack()
    return stack[-1] if stack else None


@contextmanager
def adopt(ledger):
    """Attach an existing ledger to THIS thread for the duration of the block.

    Thread-local state is the right default -- two sessions running
    concurrently must never pool their tokens -- but it means a ledger opened
    on the request thread is invisible to a worker thread. The retrieval agent
    runs its whole trial, and therefore every LLM call it makes, on a worker
    (handbook_studio_runner.py starts one), so without this the calls record
    into nothing and the run reports zero tokens while plainly having cost
    money.

    The worker calls this with the ledger the caller opened:

        with usage_ledger.adopt(parent_ledger):
            ...                      # records land in parent_ledger
    """
    if ledger is None:
        yield None
        return
    _stack().append(ledger)
    try:
        yield ledger
    finally:
        stack = _stack()
        if stack and stack[-1] is ledger:
            stack.pop()
        elif ledger in stack:
            stack.remove(ledger)


def record(model, usage, phase=None, source=""):
    """Record one call's usage against the active ledger, if any.

    ``usage`` is whatever the provider returned; the shape differs between
    OpenAI's Responses API (input_tokens/output_tokens), Chat Completions
    (prompt_tokens/completion_tokens) and the Claude Agent SDK (a cost figure
    and no token counts at all), so all three are normalised here.
    """
    ledger = active()
    if ledger is None:
        return None
    return ledger.add(model=model, usage=usage, phase=phase, source=source)


def _int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def normalize(usage):
    """Provider-specific usage -> {input_tokens, output_tokens, total_tokens,
    cached_tokens, reasoning_tokens, cost_usd}. Missing values stay None."""
    if usage is None:
        return {}
    if not isinstance(usage, dict):
        # SDK objects expose the same names as attributes.
        usage = {name: getattr(usage, name, None) for name in (
            "input_tokens", "output_tokens", "total_tokens",
            "prompt_tokens", "completion_tokens",
            "input_tokens_details", "output_tokens_details",
            "prompt_tokens_details", "completion_tokens_details",
            "cost_usd", "total_cost_usd")}

    def _detail(*names):
        for name in names:
            block = usage.get(name)
            if block is None:
                continue
            if not isinstance(block, dict):
                block = {k: getattr(block, k, None)
                         for k in ("cached_tokens", "reasoning_tokens")}
            for key in ("cached_tokens", "reasoning_tokens"):
                value = _int(block.get(key))
                if value is not None:
                    yield key, value

    out = {}
    input_tokens = _int(usage.get("input_tokens"))
    if input_tokens is None:
        input_tokens = _int(usage.get("prompt_tokens"))
    output_tokens = _int(usage.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _int(usage.get("completion_tokens"))
    total = _int(usage.get("total_tokens"))
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens

    if input_tokens is not None:
        out["input_tokens"] = input_tokens
    if output_tokens is not None:
        out["output_tokens"] = output_tokens
    if total is not None:
        out["total_tokens"] = total
    for key, value in _detail("input_tokens_details", "prompt_tokens_details",
                              "output_tokens_details", "completion_tokens_details"):
        out[key] = value

    cost = usage.get("cost_usd")
    if cost is None:
        cost = usage.get("total_cost_usd")
    try:
        if cost is not None:
            out["cost_usd"] = float(cost)
    except (TypeError, ValueError):
        pass
    return out


def cost_for(record_dict, price_table):
    """USD for one record, or None when the model has no published rate.

    ``price_table`` maps a model name to {"input": <usd per 1M input tokens>,
    "output": <usd per 1M output tokens>}. A provider-reported cost (the Claude
    Agent SDK gives one directly) always wins over a computed estimate.
    """
    reported = record_dict.get("cost_usd")
    if reported is not None:
        return float(reported)
    rates = (price_table or {}).get(record_dict.get("model"))
    if not rates:
        return None
    input_tokens = record_dict.get("input_tokens")
    output_tokens = record_dict.get("output_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    total = 0.0
    if input_tokens is not None:
        total += input_tokens / 1_000_000 * float(rates.get("input") or 0)
    if output_tokens is not None:
        total += output_tokens / 1_000_000 * float(rates.get("output") or 0)
    return total


class Ledger:
    """Usage for one pipeline run, kept per call so it can be re-priced."""

    def __init__(self, phase=None):
        self.phase = phase
        self.records = []
        self.started_at = time.time()
        # A ledger adopted by a worker thread receives records from more than
        # one thread, so the append is guarded.
        self._lock = threading.Lock()
        # The chat modes run many phases under ONE ledger (a phase-per-ledger
        # split is not available there -- the workflow handlers are generators
        # whose phases are only identifiable from the events they yield). The
        # consumer of those events sets this as it advances, so each call is
        # attributed to whatever phase was running when it was made.
        self._current_phase = None

    def set_phase(self, phase):
        """Attribute subsequent calls to *phase*. Safe to call repeatedly.

        Deliberately NOT thread-local: the phase is a property of the run, and
        the LLM calls it should be attributed to happen on worker threads the
        phase spawned. Phases in these workflows are sequential, so the value
        read by a worker is the phase that spawned it.
        """
        if phase:
            self._current_phase = phase

    def add(self, model, usage, phase=None, source=""):
        # An explicit `phase` is the caller's own label for the work -- the
        # retrieval agent tags its calls "retrieval", the handbook generator
        # tags its own "generation". Those are the right names in Generate &
        # Test, where a ledger is opened per phase and nothing calls
        # set_phase(). In the chat modes the phases are the WORKFLOW CARDS, and
        # a call labelled "retrieval" matches no card -- so a whole download
        # run recorded its tokens against a phase nothing displays. Where a
        # current phase has been set, it therefore wins, and the caller's label
        # is kept as `stage` rather than discarded.
        current = self._current_phase
        entry = {
            "model": str(model or "unknown"),
            "phase": current or phase or self.phase or "unattributed",
            "stage": phase or "",
            "source": source,
            "at": time.time(),
        }
        entry.update(normalize(usage))
        with self._lock:
            self.records.append(entry)
        return entry

    def records_since(self, offset):
        """Records added after index *offset*, plus the new offset.

        Used to stream newly-recorded calls to the browser without resending
        the whole ledger on every event.
        """
        with self._lock:
            return list(self.records[offset:]), len(self.records)

    def totals(self, price_table=None):
        """Aggregate totals, plus per-phase and per-model breakdowns.

        ``calls_missing_usage`` matters: a provider that returns no counts
        would otherwise make a run look cheaper than it was, so the number of
        unaccounted calls is reported rather than hidden.
        """
        summary = {
            "calls": len(self.records),
            "calls_missing_usage": 0,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "cached_tokens": 0, "reasoning_tokens": 0,
            "cost_usd": 0.0, "cost_is_partial": False,
            "elapsed_seconds": round(time.time() - self.started_at, 1),
            "by_phase": {}, "by_model": {},
        }
        for entry in self.records:
            has_tokens = entry.get("total_tokens") is not None
            if not has_tokens:
                summary["calls_missing_usage"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens",
                        "cached_tokens", "reasoning_tokens"):
                summary[key] += entry.get(key) or 0
            cost = cost_for(entry, price_table)
            if cost is None:
                if has_tokens:
                    summary["cost_is_partial"] = True
            else:
                summary["cost_usd"] += cost

            for axis, key in (("by_phase", entry["phase"]), ("by_model", entry["model"])):
                bucket = summary[axis].setdefault(
                    key, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                          "total_tokens": 0, "cost_usd": 0.0,
                          "elapsed_seconds": 0.0})
                bucket["calls"] += 1
                for field in ("input_tokens", "output_tokens", "total_tokens"):
                    bucket[field] += entry.get(field) or 0
                if cost is not None:
                    bucket["cost_usd"] += cost
        summary["cost_usd"] = round(summary["cost_usd"], 6)
        # Wall-clock belongs to the ledger, not to an individual call: the
        # phase's real duration includes the gaps between calls (downloading,
        # running generated code), which is exactly what "how long did this
        # take" means. A ledger is opened per phase, so its elapsed time is
        # attributed to that phase's bucket -- never spread across calls,
        # which would invent per-call timings nothing measured.
        if self.phase and self.phase in summary["by_phase"]:
            summary["by_phase"][self.phase]["elapsed_seconds"] = (
                summary["elapsed_seconds"])
        for axis in ("by_phase", "by_model"):
            for bucket in summary[axis].values():
                bucket["cost_usd"] = round(bucket["cost_usd"], 6)
        return summary


def merge(*summaries):
    """Combine several totals() results (e.g. generation + each test run)."""
    out = {
        "calls": 0, "calls_missing_usage": 0,
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "cached_tokens": 0, "reasoning_tokens": 0,
        "cost_usd": 0.0, "cost_is_partial": False,
        # Summed, not max'd: the phases run one after another (generate, then
        # each test), so their durations add up to the session's real cost in
        # time. Concurrent phases within one session would break that, and
        # none exist.
        "elapsed_seconds": 0.0,
        "by_phase": {}, "by_model": {},
    }
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        for key in ("calls", "calls_missing_usage", "input_tokens",
                    "output_tokens", "total_tokens", "cached_tokens",
                    "reasoning_tokens"):
            out[key] += summary.get(key) or 0
        out["cost_usd"] += summary.get("cost_usd") or 0.0
        out["elapsed_seconds"] += float(summary.get("elapsed_seconds") or 0)
        out["cost_is_partial"] |= bool(summary.get("cost_is_partial"))
        for axis in ("by_phase", "by_model"):
            for key, bucket in (summary.get(axis) or {}).items():
                target = out[axis].setdefault(
                    key, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                          "total_tokens": 0, "cost_usd": 0.0,
                          "elapsed_seconds": 0.0})
                for field in ("calls", "input_tokens", "output_tokens", "total_tokens"):
                    target[field] += bucket.get(field) or 0
                target["cost_usd"] += bucket.get("cost_usd") or 0.0
                target["elapsed_seconds"] += float(bucket.get("elapsed_seconds") or 0)
    out["cost_usd"] = round(out["cost_usd"], 6)
    out["elapsed_seconds"] = round(out["elapsed_seconds"], 1)
    for axis in ("by_phase", "by_model"):
        for bucket in out[axis].values():
            bucket["cost_usd"] = round(bucket["cost_usd"], 6)
            bucket["elapsed_seconds"] = round(bucket["elapsed_seconds"], 1)
    return out
