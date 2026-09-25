"""Bulk-manage public (read-only) links for Handbook Studio sessions.

Flipping the Share button on dozens of sessions by hand is tedious; this
works straight on the session store, so run it on the machine that serves
the sessions (e.g. inside ~/mysite3 on PythonAnywhere).

    python -m WebUI.studio_share_cli --list
    python -m WebUI.studio_share_cli --on  ids.txt          # ids or URLs, any format
    python -m WebUI.studio_share_cli --off ids.txt
    python -m WebUI.studio_share_cli --links ids.txt --base https://<host>   # appendix table (CSV)

``ids.txt`` may contain bare session ids, /handbook-studio/<id> links, or the
older /handbook-generator/generate-test/<id> links, one or many per line;
UUIDs are extracted from whatever is there. Ids may also be passed directly
as arguments. Duplicates are ignored.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

from WebUI import handbook_studio_store as store

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _collect_ids(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        text = item
        if os.path.isfile(item):
            with open(item, encoding="utf-8") as handle:
                text = handle.read()
        for match in _UUID.findall(text.lower()):
            seen.setdefault(match, None)
    return list(seen)


def _locate(session_id: str) -> tuple[str, dict] | None:
    """(owner_user_id, session) wherever the session lives, public or not."""
    if not os.path.isdir(store.STORE_ROOT):
        return None
    for user_id in os.listdir(store.STORE_ROOT):
        path = os.path.join(store.STORE_ROOT, user_id, f"{session_id}.json")
        if os.path.isfile(path):
            session = store.get_session(user_id, session_id)
            return (user_id, session) if session else None
    return None


def _set_public(session_ids: list[str], public: bool) -> tuple[int, list[str]]:
    changed, missing = 0, []
    for sid in session_ids:
        found = _locate(sid)
        if not found:
            missing.append(sid)
            continue
        user_id, session = found
        if bool(session.get("public")) == public:
            continue
        store.save_session(user_id, {**session, "public": public}, session_id=sid)
        changed += 1
    return changed, missing


def _list_public() -> list[tuple[str, dict]]:
    out = []
    if not os.path.isdir(store.STORE_ROOT):
        return out
    for user_id in sorted(os.listdir(store.STORE_ROOT)):
        for session in store.list_sessions(user_id):
            if session.get("public"):
                out.append((user_id, session))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bulk public-link management for Studio sessions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="show every public session")
    group.add_argument("--on", nargs="+", metavar="ID|FILE", help="make these public")
    group.add_argument("--off", nargs="+", metavar="ID|FILE", help="make these private")
    group.add_argument("--links", nargs="+", metavar="ID|FILE",
                       help="print an appendix table (CSV) with the public link of each")
    parser.add_argument("--base", default="", help="site origin for --links, e.g. https://x.pythonanywhere.com")
    args = parser.parse_args(argv)

    if args.list:
        rows = _list_public()
        print(f"{len(rows)} public session(s)")
        for user_id, s in rows:
            print(f"  {s['id']}  {s.get('name', '')[:50]:50}  {s.get('source_name', '')[:40]}")
        return 0

    ids = _collect_ids(args.on or args.off or args.links)
    if not ids:
        print("No session ids found in the arguments.")
        return 1

    if args.links:
        base = args.base.rstrip("/")
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(["session_id", "name", "source", "mechanism", "status", "public", "link"])
        for sid in ids:
            found = _locate(sid)
            if not found:
                writer.writerow([sid, "(not found)", "", "", "", "", ""])
                continue
            _u, s = found
            writer.writerow([sid, s.get("name", ""), s.get("source_name", ""),
                             s.get("access_mechanism", ""), s.get("status", ""),
                             "yes" if s.get("public") else "no",
                             f"{base}/handbook-studio/shared/{sid}"])
        return 0

    changed, missing = _set_public(ids, public=bool(args.on))
    verb = "public" if args.on else "private"
    print(f"{len(ids)} id(s): {changed} changed to {verb}, "
          f"{len(ids) - changed - len(missing)} already {verb}, {len(missing)} not found")
    for sid in missing:
        print(f"  not found: {sid}")
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
