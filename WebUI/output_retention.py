"""Disk retention for Handbook Studio outputs on hosts with a finite quota.

Retrieved data is a deliverable, not a store: the handbook and the code that
produced a file are what make it reproducible, so bulk files can be evicted
once the user has had a fair chance to download them. This module decides
what stays and what goes; the runner calls it after every run and a
scheduled task calls it once a day.

Enabled only when ``GIS_COSCI_OUTPUT_QUOTA_MB`` is set (> 0). Without it --
the desktop install -- nothing here ever deletes anything, so a local machine
keeps today's unlimited behaviour.

Policy (all sizes in MB, all env-configurable):

    GIS_COSCI_OUTPUT_QUOTA_MB          total bulk data allowed under OUTPUT_ROOT
    GIS_COSCI_OUTPUT_KEEP_SMALL_MB     files at or under this are kept forever (1)
    GIS_COSCI_OUTPUT_RETENTION_DAYS    bulk files older than this are evicted (7)
    GIS_COSCI_RUN_CAP_MB               a run larger than this is "large" (500)
    GIS_COSCI_RUN_CAP_RETENTION_HOURS  claim window for a large run's files (24)
    GIS_COSCI_OUTPUT_MIN_FREE_MB       never let free disk drop below this (200)

Guarantees:

* Files <= KEEP_SMALL are never touched: handbooks, manifests, logs, samples.
* No bulk file is evicted inside the claim window after it was written, even
  under quota pressure. If a run completed, the user has at least that long
  to download its result.
* ``_handbook/`` directories and the runner's own bookkeeping files are never
  touched.
* A run that would push free disk below MIN_FREE is refused or aborted rather
  than filling the disk (which would break the conversation DB and logs).

Run as a script for the scheduled task::

    python -m WebUI.output_retention            # sweep
    python -m WebUI.output_retention --dry-run  # report only
    python -m WebUI.output_retention --status   # usage and policy
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

_MB = 1024 * 1024

# Runner bookkeeping written next to a run's data; small, and the UI relies
# on some of them, so they are excluded from eviction regardless of size.
_INTERNAL_NAMES = {
    "_rerun_user.py", "_rerun_boot.py", "_rerun_http.jsonl",
    "_reexec_boot.py", "_reexec_user.py",
}
_EVICTED_MANIFEST = ".evicted.json"
_RUN_MARKER = ".retention.json"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def policy() -> dict:
    """The active policy. ``enabled`` is False unless a quota is configured."""
    quota_mb = _env_float("GIS_COSCI_OUTPUT_QUOTA_MB", 0)
    return {
        "enabled": quota_mb > 0,
        "quota_bytes": int(quota_mb * _MB),
        "keep_small_bytes": int(_env_float("GIS_COSCI_OUTPUT_KEEP_SMALL_MB", 1) * _MB),
        "retention_seconds": _env_float("GIS_COSCI_OUTPUT_RETENTION_DAYS", 7) * 86400,
        "run_cap_bytes": int(_env_float("GIS_COSCI_RUN_CAP_MB", 500) * _MB),
        "claim_seconds": _env_float("GIS_COSCI_RUN_CAP_RETENTION_HOURS", 24) * 3600,
        "min_free_bytes": int(_env_float("GIS_COSCI_OUTPUT_MIN_FREE_MB", 200) * _MB),
    }


def _output_root() -> str:
    from WebUI.handbook_studio_runner import OUTPUT_ROOT
    return OUTPUT_ROOT


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def free_disk_bytes(path: str | None = None) -> int:
    """Free bytes on the volume holding ``path`` (or the output root)."""
    target = path or _output_root()
    while target and not os.path.exists(target):
        target = os.path.dirname(target)
    try:
        return shutil.disk_usage(target or os.getcwd()).free
    except OSError:
        return 1 << 62  # unknown: treat as unlimited rather than block runs


def _run_dirs(root: str):
    """Yield (user_dir, session_dir, run_dir) for every run under root."""
    if not os.path.isdir(root):
        return
    for user in sorted(os.listdir(root)):
        user_dir = os.path.join(root, user)
        if not os.path.isdir(user_dir):
            continue
        for session in sorted(os.listdir(user_dir)):
            session_dir = os.path.join(user_dir, session)
            if not os.path.isdir(session_dir):
                continue
            for run in sorted(os.listdir(session_dir)):
                run_dir = os.path.join(session_dir, run)
                if os.path.isdir(run_dir) and run != "_handbook":
                    yield user_dir, session_dir, run_dir


def _files(run_dir: str):
    """(path, size, mtime) for every data file in a run, skipping bookkeeping."""
    for base, _dirs, names in os.walk(run_dir):
        for name in names:
            if name in _INTERNAL_NAMES or name in (_EVICTED_MANIFEST, _RUN_MARKER):
                continue
            path = os.path.join(base, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            yield path, st.st_size, st.st_mtime


def run_size_bytes(run_dir: str) -> int:
    return sum(size for _p, size, _m in _files(run_dir))


def usage(root: str | None = None) -> dict:
    """Total, bulk and small bytes under the output root."""
    root = root or _output_root()
    pol = policy()
    total = bulk = small = 0
    runs = 0
    for _u, _s, run_dir in _run_dirs(root):
        runs += 1
        for _p, size, _m in _files(run_dir):
            total += size
            if size > pol["keep_small_bytes"]:
                bulk += size
            else:
                small += size
    return {"root": root, "runs": runs, "total_bytes": total,
            "bulk_bytes": bulk, "small_bytes": small,
            "free_bytes": free_disk_bytes(root)}


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    except OSError:
        pass


def mark_run(run_dir: str, finished_at: float | None = None) -> dict:
    """Record a run's size and its retention deadline; return the record.

    Called by the runner when a run completes. The record is what the UI
    shows ("kept until ...") and what the sweep uses to find the claim window,
    so the deadline is fixed at completion time rather than recomputed from
    file mtimes that a later re-run could refresh.
    """
    pol = policy()
    finished_at = finished_at or time.time()
    size = run_size_bytes(run_dir)
    over_cap = size > pol["run_cap_bytes"]
    keep_for = pol["claim_seconds"] if over_cap else pol["retention_seconds"]
    record = {
        "policy_enabled": pol["enabled"],
        "run_size_bytes": size,
        "over_cap": over_cap,
        "run_cap_bytes": pol["run_cap_bytes"],
        "keep_small_bytes": pol["keep_small_bytes"],
        "finished_at": _iso(finished_at),
        # Bulk files are guaranteed until this moment; small files forever.
        "bulk_kept_until": _iso(finished_at + keep_for) if pol["enabled"] else None,
    }
    if pol["enabled"]:
        _write_json(os.path.join(run_dir, _RUN_MARKER), record)
    return record


def run_marker(run_dir: str) -> dict:
    return _read_json(os.path.join(run_dir, _RUN_MARKER))


def evicted_files(run_dir: str) -> dict:
    """{filename: {"at": iso, "reason": str, "size_bytes": int}} for a run."""
    return _read_json(os.path.join(run_dir, _EVICTED_MANIFEST))


def evicted_reason(run_dir: str, name: str) -> str | None:
    """A user-facing sentence if ``name`` was evicted from ``run_dir``."""
    entry = evicted_files(run_dir).get(name)
    if not entry:
        return None
    when = str(entry.get("at", ""))[:10]
    return (f"Removed by the storage retention policy on {when} "
            f"({entry.get('reason', 'expired')}). Re-run this session to "
            "regenerate it, or run the handbook locally to keep large outputs.")


def _evict(path: str, run_dir: str, size: int, reason: str, dry_run: bool) -> None:
    name = os.path.basename(path)
    if not dry_run:
        try:
            os.remove(path)
        except OSError:
            return
        manifest = evicted_files(run_dir)
        manifest[name] = {"at": _iso(time.time()), "reason": reason,
                          "size_bytes": size}
        _write_json(os.path.join(run_dir, _EVICTED_MANIFEST), manifest)


def sweep(root: str | None = None, now: float | None = None,
          dry_run: bool = False) -> dict:
    """Apply the policy. Returns a report; a no-op unless the policy is enabled.

    Pass 1 -- expiry: evict bulk files whose run has passed its deadline
    (claim window for large runs, retention period otherwise).
    Pass 2 -- quota: if bulk usage is still above the quota, evict the oldest
    eligible bulk files first until it fits. "Eligible" excludes anything
    inside its claim window, so a file the user just produced is safe.
    """
    pol = policy()
    root = root or _output_root()
    now = now or time.time()
    report = {"enabled": pol["enabled"], "dry_run": dry_run,
              "expired": [], "quota": [], "freed_bytes": 0}
    if not pol["enabled"]:
        return report

    candidates = []  # (mtime, size, path, run_dir) bulk files still eligible
    bulk_total = 0
    for _u, _s, run_dir in _run_dirs(root):
        marker = run_marker(run_dir)
        deadline = None
        if marker.get("bulk_kept_until"):
            try:
                deadline = datetime.fromisoformat(marker["bulk_kept_until"]).timestamp()
            except ValueError:
                deadline = None
        for path, size, mtime in _files(run_dir):
            if size <= pol["keep_small_bytes"]:
                continue
            # Unmarked runs (from before this policy existed) fall back to the
            # file's own age against the ordinary retention period.
            file_deadline = deadline if deadline is not None else mtime + pol["retention_seconds"]
            if now >= file_deadline:
                _evict(path, run_dir, size, "expired", dry_run)
                report["expired"].append(os.path.relpath(path, root))
                report["freed_bytes"] += size
                continue
            bulk_total += size
            if now - mtime >= pol["claim_seconds"]:
                candidates.append((mtime, size, path, run_dir))

    if bulk_total > pol["quota_bytes"]:
        for mtime, size, path, run_dir in sorted(candidates):
            if bulk_total <= pol["quota_bytes"]:
                break
            _evict(path, run_dir, size, "over storage quota", dry_run)
            report["quota"].append(os.path.relpath(path, root))
            report["freed_bytes"] += size
            bulk_total -= size
    report["bulk_bytes_after"] = bulk_total
    return report


def preflight(root: str | None = None) -> str | None:
    """A refusal message if a run must not start, else None."""
    pol = policy()
    if not pol["enabled"]:
        return None
    free = free_disk_bytes(root or _output_root())
    if free < pol["min_free_bytes"]:
        return (f"Not enough free disk on this server to run a retrieval "
                f"({free // _MB} MB free, {pol['min_free_bytes'] // _MB} MB "
                "required). Older outputs will be cleared by the retention "
                "policy; try again later, or run this handbook locally.")
    return None


def watchdog(root: str | None = None) -> str | None:
    """Called periodically during a run: a message if it must be aborted."""
    pol = policy()
    if not pol["enabled"]:
        return None
    free = free_disk_bytes(root or _output_root())
    if free < pol["min_free_bytes"]:
        return (f"Aborted: this retrieval was about to fill the server's disk "
                f"({free // _MB} MB left). The dataset is too large for this "
                "server -- the handbook and code are saved, so run it locally "
                "to retrieve the data.")
    return None


def after_run(run_dir: str) -> dict:
    """Runner hook: mark the finished run, then sweep. Returns the marker."""
    record = mark_run(run_dir)
    if record["policy_enabled"]:
        try:
            sweep()
        except Exception:  # a janitor failure must never fail the run itself
            pass
    return record


def _fmt_mb(n: int) -> str:
    return f"{n / _MB:,.1f} MB"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Handbook Studio output retention")
    parser.add_argument("--dry-run", action="store_true", help="report, delete nothing")
    parser.add_argument("--status", action="store_true", help="show usage and policy")
    args = parser.parse_args(argv)

    pol = policy()
    if args.status or not pol["enabled"]:
        use = usage()
        print(f"Output root : {use['root']}")
        print(f"Policy      : {'ENABLED' if pol['enabled'] else 'disabled (set GIS_COSCI_OUTPUT_QUOTA_MB to enable)'}")
        if pol["enabled"]:
            print(f"  quota {_fmt_mb(pol['quota_bytes'])} | keep-small <= {_fmt_mb(pol['keep_small_bytes'])} | "
                  f"retention {pol['retention_seconds'] / 86400:g} d | run cap {_fmt_mb(pol['run_cap_bytes'])} "
                  f"(claim {pol['claim_seconds'] / 3600:g} h) | min free {_fmt_mb(pol['min_free_bytes'])}")
        print(f"Usage       : {use['runs']} runs, total {_fmt_mb(use['total_bytes'])} "
              f"(bulk {_fmt_mb(use['bulk_bytes'])}, small {_fmt_mb(use['small_bytes'])}), "
              f"free disk {_fmt_mb(use['free_bytes'])}")
        if not pol["enabled"]:
            return 0
    report = sweep(dry_run=args.dry_run)
    label = "Would evict" if args.dry_run else "Evicted"
    print(f"{label}: {len(report['expired'])} expired, {len(report['quota'])} over quota, "
          f"{_fmt_mb(report['freed_bytes'])} freed; bulk now {_fmt_mb(report.get('bulk_bytes_after', 0))}")
    for path in report["expired"] + report["quota"]:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
