"""User-contributed data sources (handbooks).

Lets end users register a *retrievable* data source from the Web UI without
editing the bundled handbooks on disk. The design mirrors the data-source-key
flow: a source is stored on disk so it survives restarts and is shared across
all PythonAnywhere workers, but it is scoped per user so one user's source
(and its executable ``code_example``) is never visible or runnable for anyone
else.

Three tiers of storage, all under ``DataRetriever_Handbooks/``:

* ``Handbooks/``                  - curated global catalog (bundled .toml). Read
                                    by every task for every user.
* ``Handbooks_user/<user_id>/``   - a user's private sources. Read only for that
                                    user's own tasks (Tier 1: instant value).
* ``Handbooks_pending/``          - community submissions awaiting admin review.
                                    NEVER read by the agent; an admin promotes an
                                    entry into ``Handbooks/`` to publish it.

Sources are stored as TOML (``<slug>.toml``), matching the bundled catalog.
``code_example`` is user-authored and frequently mixes ``'''`` and ``\"\"\"``
quoting, so the serializer (``_to_toml``) writes multiline *basic* strings and
escapes backslashes and double-quotes — which round-trips losslessly through
``tomllib``/``tomli``. Legacy ``<slug>.json`` files are still read (and replaced
with TOML on the next save). The agent (``DataRetrieverAgent``) reads both.

The agent iterates *every* key of a loaded source and treats each value as a
string (it substitutes ``{field}`` placeholders), so a source file read by the
agent must contain ONLY the string fields below. All bookkeeping (contributor,
timestamps, attachments) lives in a ``<slug>.meta`` sidecar that the agent's
``*.toml``/``*.json`` glob never picks up.
"""

import hashlib
import json
import os
import re
import time

# For reading the bundled .toml handbooks in the global catalog.
try:
    import tomllib as _toml      # Python 3.11+
except ImportError:              # pragma: no cover
    try:
        import tomli as _toml
    except ImportError:
        _toml = None

# The four string fields the agent understands. Keep this exact set in any
# file the agent reads (user dir + global dir) — see module docstring.
SOURCE_FIELDS = ("data_source_name", "brief_description", "handbook", "code_example")

# Extra string fields stored with a source for display only (the agent ignores
# them; they're plain strings so they're safe in agent-read files).
# caveats: user-facing warnings shown as an agent advisory on the source card.
# key_signup_url: where to apply for the source's credentials ([Links] in .keys).
DISPLAY_FIELDS = ("website", "requires_key", "key_name", "caveats", "key_signup_url")

# Everything persisted in a source file / queue record.
STORED_FIELDS = SOURCE_FIELDS + DISPLAY_FIELDS

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HANDBOOKS_ROOT = os.path.join(_BASE_DIR, "DataRetriever_Handbooks")
GLOBAL_DIR = os.path.join(HANDBOOKS_ROOT, "Handbooks")
USER_ROOT = os.path.join(HANDBOOKS_ROOT, "Handbooks_user")
PENDING_DIR = os.path.join(HANDBOOKS_ROOT, "Handbooks_pending")
KEYS_DIR = os.path.join(HANDBOOKS_ROOT, "Keys")


def _global_required_keys(slug, handbook_text):
    """Credentials a bundled source needs: the key names declared in its
    ``.keys`` file that are referenced as ``{placeholders}`` in the handbook
    (mirrors DataRetrieverAgent.get_required_key_names, for display only)."""
    import configparser
    kf = os.path.join(KEYS_DIR, f"{slug}.keys")
    if not os.path.exists(kf):
        return []
    cfg = configparser.ConfigParser()
    cfg.optionxform = str          # case-sensitive key names
    try:
        cfg.read(kf)
    except Exception:
        return []
    if not cfg.has_section("API_Key"):
        return []
    declared = [k for k in cfg["API_Key"].keys() if k.strip().lower() != "example_key"]
    placeholders = {m.lower() for m in re.findall(r"\{([A-Za-z0-9_]+)\}", handbook_text or "")}
    if not placeholders:
        return declared
    return [n for n in declared if n.lower() in placeholders]


# ── Path helpers ─────────────────────────────────────────────────────────────

def slugify(name):
    """Turn a human source name into a safe filename stem (no path separators,
    no traversal). Always returns a non-empty slug."""
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = s[:64]
    return s or f"source_{int(time.time())}"


def _safe_user_id(user_id):
    """user_id is a hash from the caller, but never trust it as a path."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "", str(user_id or ""))
    return s or "anon"


def user_handbook_dir(user_id, create=False):
    """Absolute path to a user's private handbook directory."""
    path = os.path.join(USER_ROOT, _safe_user_id(user_id))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def uid_for_key(api_key):
    """Derive the user_id from an API key. MUST match WebUI.conversation_db
    .hash_api_key (SHA-256, first 16 hex chars) so a source written under the
    web user_id is found again at download time."""
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def user_handbook_dir_for_key(api_key):
    """The caller's private handbook directory, derived straight from their API
    key. Used at download time so the agent finds the user's own sources in any
    workflow path (autonomous or manual), without relying on per-task state."""
    uid = uid_for_key(api_key)
    return user_handbook_dir(uid) if uid else None


def _user_files_dir(user_id, slug, create=False):
    path = os.path.join(user_handbook_dir(user_id), "files", slug)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _meta_path(directory, slug):
    return os.path.join(directory, f"{slug}.meta")


def _source_path(directory, slug):
    """Path to a queue/JSON record (pending submissions). Source FILES (user +
    global) are TOML and resolved via _find_source / written via _write_source."""
    return os.path.join(directory, f"{slug}.json")


def _find_source(directory, slug):
    """Locate a source file by slug — TOML preferred, legacy JSON as fallback."""
    for ext in (".toml", ".json"):
        p = os.path.join(directory, f"{slug}{ext}")
        if os.path.exists(p):
            return p
    return None


def _load_source_file(path):
    """Load a source file (TOML or JSON) into a dict."""
    if path.endswith(".toml"):
        if _toml is None:
            return None
        with open(path, "rb") as f:
            return _toml.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _toml_quote(value):
    """Serialize a string as a TOML value. Multiline content uses a basic
    multiline string with backslashes and double-quotes escaped, which
    round-trips losslessly (so embedded ''' and \"\"\" in code are safe)."""
    s = str(value)
    esc = s.replace("\\", "\\\\").replace('"', '\\"')
    if "\n" in s:
        # A newline right after the opening delimiter is trimmed by TOML, so the
        # leading "\n" we add is consumed and the content is preserved exactly.
        return '"""\n' + esc + '"""'
    return '"' + esc + '"'


def _write_source(path, fields):
    """Write a source dict to a TOML file (only the stored string fields)."""
    body = "".join(f"{k} = {_toml_quote(fields.get(k, '') or '')}\n" for k in STORED_FIELDS)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def _credential_names(fields):
    """Parsed list of credential names a source declares (requires_key +
    key_name, comma/newline separated). Empty when the source needs none."""
    if str(fields.get("requires_key", "")).strip().lower() not in ("true", "1", "yes"):
        return []
    return [n.strip() for n in re.split(r"[,\n]", fields.get("key_name", "") or "") if n.strip()]


def _write_keys_file(path, names, signup_url=""):
    """Write a .keys file declaring each credential under [API_Key] with a blank
    value (the same format as the bundled catalog's .keys files). When a signup
    URL is known, a [Links] section is added so the UI can show an
    "apply for a key" link under the credential inputs."""
    content = "[API_Key]\n" + "".join(f"{n} = \n" for n in names)
    if (signup_url or "").strip():
        content += f"\n[Links]\nwebsite = {signup_url.strip()}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ── Read / write a single source ─────────────────────────────────────────────

def _clean_fields(fields):
    """Keep the stored fields (agent fields + display fields), as stripped strings
    with normalized newlines (so TOML multiline serialization is well-formed)."""
    out = {}
    for k in STORED_FIELDS:
        v = str(fields.get(k, "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        out[k] = v
    return out


def _read_meta(directory, slug):
    try:
        with open(_meta_path(directory, slug), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_source(directory, slug):
    path = _find_source(directory, slug)
    if not path:
        return None
    try:
        data = _load_source_file(path)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    out = {"slug": slug}
    out.update({k: data.get(k, "") for k in STORED_FIELDS})
    out.update(_read_meta(directory, slug))
    return out


def write_user_source(user_id, fields, attachments=None, share=False):
    """Create/overwrite a user's private source. ``attachments`` is a list of
    ``(filename, bytes)``. Returns the slug.

    The source is usable immediately for that user's own tasks (Tier 1). It is
    submitted to the community review queue (Tier 2) only when ``share`` is true
    — sharing is opt-in.
    """
    fields = _clean_fields(fields)
    if not fields["data_source_name"]:
        raise ValueError("data_source_name is required")
    slug = slugify(fields["data_source_name"])

    udir = user_handbook_dir(user_id, create=True)

    # Save attachments first so we can reference their absolute paths in the
    # handbook text the generated code will see.
    saved_attachments = []
    for filename, blob in (attachments or []):
        safe_name = os.path.basename(filename or "").replace("\\", "_")
        if not safe_name:
            continue
        fdir = _user_files_dir(user_id, slug, create=True)
        dest = os.path.join(fdir, safe_name)
        with open(dest, "wb") as f:
            f.write(blob)
        saved_attachments.append(dest)

    if saved_attachments:
        note = "\nLocal attached file(s) for this source are available on the server at:\n" + \
               "\n".join(saved_attachments)
        fields["handbook"] = (fields["handbook"] + note).strip()

    # If the source declares credentials, make sure each appears as a {placeholder}
    # in the handbook so the agent can detect it, prompt for it, and substitute the
    # value at runtime. Names not already referenced get a documented line added.
    cred_names = _credential_names(fields)
    if cred_names:
        combined = (fields.get("handbook", "") + " " + fields.get("code_example", ""))
        missing = [n for n in cred_names if ("{" + n + "}") not in combined]
        if missing:
            block = "\nCredentials for this source (values are supplied at runtime):\n" + \
                    "\n".join(f"{n}: {{{n}}}" for n in missing)
            fields["handbook"] = (fields["handbook"].rstrip() + "\n" + block).strip()

    # Agent-facing file: TOML with the agent's fields plus display-only strings.
    # All values are strings, so this stays agent-safe.
    _write_source(os.path.join(udir, f"{slug}.toml"), fields)
    legacy = os.path.join(udir, f"{slug}.json")   # replace any older JSON copy
    if os.path.exists(legacy):
        os.remove(legacy)

    # Companion .keys file (named after the source) declaring each credential
    # with a blank value — matching the bundled catalog. Removed if no longer
    # needed. Promoted to the global Keys/ folder when the source is approved.
    keys_path = os.path.join(udir, f"{slug}.keys")
    if cred_names:
        _write_keys_file(keys_path, cred_names,
                         fields.get("key_signup_url") or fields.get("website", ""))
    elif os.path.exists(keys_path):
        os.remove(keys_path)

    # Sidecar metadata (never globbed by the agent).
    meta = {
        "contributor": _safe_user_id(user_id),
        "updated_at": _now_iso(),
        "attachments": [os.path.basename(p) for p in saved_attachments],
    }
    with open(_meta_path(udir, slug), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Tier 2: submit to the review queue only if the user opted to share.
    # Otherwise withdraw any earlier submission for this slug.
    if share:
        submit_to_queue(user_id, slug, fields)
    else:
        ppath = _source_path(PENDING_DIR, _pending_id(user_id, slug))
        if os.path.exists(ppath):
            os.remove(ppath)
    return slug


def _source_status(user_id, slug):
    """Status of a user's source: 'approved' (published to the shared catalog),
    'pending' (awaiting admin review), 'rejected' (an admin declined it), or
    'private' (saved, not shared)."""
    if (os.path.exists(_source_path(GLOBAL_DIR, slug)) or
            os.path.exists(os.path.join(GLOBAL_DIR, f"{slug}.toml"))):
        return "approved"
    if os.path.exists(_source_path(PENDING_DIR, _pending_id(user_id, slug))):
        return "pending"
    if _read_meta(user_handbook_dir(user_id), slug).get("rejected"):
        return "rejected"
    return "private"


def _set_rejected(user_id, slug, reason=""):
    """Mark a user's source as rejected (with an optional reason) in its meta."""
    udir = user_handbook_dir(user_id)
    if not os.path.isdir(udir):
        return
    meta = _read_meta(udir, slug)
    meta["rejected"] = True
    meta["rejection_reason"] = (reason or "").strip()
    meta["rejected_at"] = _now_iso()
    with open(_meta_path(udir, slug), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _clear_rejected(user_id, slug):
    """Clear any prior rejection flag (e.g. when the source is re-shared)."""
    udir = user_handbook_dir(user_id)
    meta = _read_meta(udir, slug)
    if meta.pop("rejected", None) is not None:
        meta.pop("rejection_reason", None)
        meta.pop("rejected_at", None)
        with open(_meta_path(udir, slug), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def list_user_sources(user_id):
    """List a user's private sources (newest first), each tagged with its
    sharing status (approved / pending / private)."""
    udir = user_handbook_dir(user_id)
    if not os.path.isdir(udir):
        return []
    out = []
    seen = set()
    for name in sorted(os.listdir(udir)):
        if not (name.endswith(".toml") or name.endswith(".json")):
            continue
        slug = os.path.splitext(name)[0]
        if slug in seen:
            continue
        seen.add(slug)
        src = _read_source(udir, slug)   # prefers .toml
        if src:
            src["status"] = _source_status(user_id, slug)
            out.append(src)
    out.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return out


def get_user_source(user_id, slug):
    return _read_source(user_handbook_dir(user_id), slug)


def list_global_sources():
    """List the curated global catalog (the bundled Handbooks/). Reads both
    .toml and .json handbooks, returning name + description for display. The
    template is skipped."""
    if not os.path.isdir(GLOBAL_DIR):
        return []
    out = []
    for name in sorted(os.listdir(GLOBAL_DIR)):
        if name == "template.toml":
            continue
        path = os.path.join(GLOBAL_DIR, name)
        try:
            if name.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif name.endswith(".toml") and _toml is not None:
                with open(path, "rb") as f:
                    data = _toml.load(f)
            else:
                continue
        except Exception:
            continue
        slug = os.path.splitext(name)[0]
        handbook = (data.get("handbook") or "").strip()
        # Credentials: an approved user source carries its own key_name; a bundled
        # .toml source derives them from its .keys file.
        key_name = (data.get("key_name") or "").strip()
        requires_key = (data.get("requires_key") or "").strip()
        if not key_name:
            req = _global_required_keys(slug, handbook)
            if req:
                key_name = ", ".join(req)
                requires_key = "true"
        out.append({
            "slug": slug,
            "data_source_name": (data.get("data_source_name") or slug).strip(),
            "brief_description": (data.get("brief_description") or "").strip(),
            "handbook": handbook,
            "code_example": (data.get("code_example") or "").strip(),
            "website": (data.get("website") or "").strip(),
            "requires_key": requires_key,
            "key_name": key_name,
        })
    out.sort(key=lambda s: s["data_source_name"].lower())
    return out


def delete_global_source(slug):
    """Admin: remove a published source from the global (shared) catalog.

    Deletes the handbook (.toml/.json) and its .keys sidecar from GLOBAL_DIR.
    The bundled ``template`` is protected. Returns True if anything was removed.
    """
    if not slug or slug == "template":
        return False
    removed = False
    for path in (os.path.join(GLOBAL_DIR, f"{slug}.toml"),
                 os.path.join(GLOBAL_DIR, f"{slug}.json"),
                 os.path.join(GLOBAL_DIR, f"{slug}.keys")):
        if os.path.exists(path):
            os.remove(path)
            removed = True
    return removed


def delete_user_source(user_id, slug):
    """Remove a user's private source, its sidecar, attachments, and any pending
    submission. Does not touch the published global copy."""
    udir = user_handbook_dir(user_id)
    removed = False
    for path in (os.path.join(udir, f"{slug}.toml"),
                 os.path.join(udir, f"{slug}.json"),
                 os.path.join(udir, f"{slug}.keys"),
                 _meta_path(udir, slug)):
        if os.path.exists(path):
            os.remove(path)
            removed = True
    fdir = _user_files_dir(user_id, slug)
    if os.path.isdir(fdir):
        import shutil
        shutil.rmtree(fdir, ignore_errors=True)
    # Withdraw the pending submission too.
    ppath = _source_path(PENDING_DIR, _pending_id(user_id, slug))
    if os.path.exists(ppath):
        os.remove(ppath)
    return removed


# ── Community review queue (Tier 2) ──────────────────────────────────────────

def _pending_id(user_id, slug):
    return f"{_safe_user_id(user_id)}__{slug}"


def submit_to_queue(user_id, slug, fields):
    """Upsert a submission keyed by (user, slug) so repeated edits collapse into
    one queue entry rather than piling up duplicates. Re-sharing clears any prior
    rejection."""
    _clear_rejected(user_id, slug)
    os.makedirs(PENDING_DIR, exist_ok=True)
    pid = _pending_id(user_id, slug)
    record = {k: fields.get(k, "") for k in STORED_FIELDS}
    record.update({
        "pending_id": pid,
        "slug": slug,
        "contributor": _safe_user_id(user_id),
        "submitted_at": _now_iso(),
        "status": "pending",
    })
    with open(_source_path(PENDING_DIR, pid), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return pid


def list_pending():
    """All submissions awaiting review (admin only)."""
    if not os.path.isdir(PENDING_DIR):
        return []
    out = []
    for name in os.listdir(PENDING_DIR):
        if name.endswith(".json"):
            try:
                with open(os.path.join(PENDING_DIR, name), "r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception:
                continue
    out.sort(key=lambda s: s.get("submitted_at", ""), reverse=True)
    return out


def approve_pending(pending_id):
    """Publish a submission into the curated global catalog. Writes only the
    four agent fields (no contributor/metadata) and clears it from the queue."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "", str(pending_id or ""))
    ppath = _source_path(PENDING_DIR, safe)
    if not os.path.exists(ppath):
        return False
    with open(ppath, "r", encoding="utf-8") as f:
        record = json.load(f)
    slug = record.get("slug") or safe
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    _write_source(os.path.join(GLOBAL_DIR, f"{slug}.toml"),
                  {k: record.get(k, "") for k in STORED_FIELDS})
    legacy = os.path.join(GLOBAL_DIR, f"{slug}.json")   # replace any older JSON copy
    if os.path.exists(legacy):
        os.remove(legacy)

    # Promote the credential declaration into the global Keys/ folder so the
    # approved source behaves like a bundled one.
    cred_names = _credential_names(record)
    if cred_names:
        os.makedirs(KEYS_DIR, exist_ok=True)
        _write_keys_file(os.path.join(KEYS_DIR, f"{slug}.keys"), cred_names)

    # Clear any prior rejection on the contributor's copy.
    if record.get("contributor") and slug:
        _clear_rejected(record["contributor"], slug)

    os.remove(ppath)
    return True


def reject_pending(pending_id, reason=""):
    """Remove a submission from the queue and mark the contributor's copy as
    rejected (with an optional reason they'll see)."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "", str(pending_id or ""))
    ppath = _source_path(PENDING_DIR, safe)
    if not os.path.exists(ppath):
        return False
    try:
        with open(ppath, "r", encoding="utf-8") as f:
            record = json.load(f)
    except Exception:
        record = {}
    os.remove(ppath)
    contributor, slug = record.get("contributor"), record.get("slug")
    if contributor and slug:
        _set_rejected(contributor, slug, reason)
    return True


def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
