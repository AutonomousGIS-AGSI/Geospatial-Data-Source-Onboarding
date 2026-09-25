
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

try:
    import tomllib as _toml      # Python 3.11+
except ImportError:              # pragma: no cover
    import tomli as _toml

from openai import OpenAI

import utils.agm_helper as helper
from agents.data_agent import claude_agent_provider
from agents.data_agent import usage_ledger
from utils.python_exe import python_executable

HANDBOOK_GEN_MODEL = "gpt-5.2"
CLAUDE_HANDBOOK_MODEL = "claude-sonnet-5"
# Claude Agent SDK "turns" cover both a tool call and its result, so a full
# write -> run -> read-error -> fix cycle costs several turns, not one. Sized
# generously above _MAX_VERIFY_ATTEMPTS so the turn budget is never the
# thing that cuts a verify loop short before its own attempt cap does.
_CLAUDE_VERIFY_MAX_TURNS = 40

# Documentation-fetch budgets.
_DOC_TEXT_BUDGET = 14000
_DOC_TEXT_PER_URL = 8000
_FETCH_TIMEOUT = 15  # seconds per URL
_URL_RE = re.compile(r"https?://[^\s)>\]\"'{}]+", re.IGNORECASE)

_VERIFY_TIMEOUT = 300       # seconds per sample-download run
_SAME_ERROR_LIMIT = 3       # stop revising after N identical errors in a row
_MAX_VERIFY_ATTEMPTS = 6    # hard cap regardless of error-signature matching —
                            # a model that keeps trying superficially
                            # different variants of the same broken approach
                            # (different URLs, query-param casing, alternating
                            # between two dead endpoints) must not be able to
                            # loop unbounded and burn API calls/time

_SAMPLE_PREVIEW_CHARS = 2500   # per produced file, shown on the review card
_SAMPLE_PREVIEW_FILES = 6
_TEXT_SAMPLE_EXTS = {
    ".csv", ".tsv", ".txt", ".json", ".geojson", ".xml", ".kml", ".gml",
    ".html", ".md", ".log", ".wkt", ".yaml", ".yml", ".toml",
}


def _file_has_data(path):
    """Whether one sample-download output carries at least one data record.

    A run that exits 0 and writes a header-only CSV, ``[]``, or an empty
    FeatureCollection has NOT demonstrated a working query -- a CDC PLACES
    handbook once passed verification on a 2-byte CSV, then failed every
    real retrieval because the query it encoded matched nothing. Text
    formats are judged by content; binary outputs (GeoTIFF, zip, shapefile
    parts) only by being non-empty, since a real header alone is already
    tens of bytes."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    if size == 0:
        return False
    ext = os.path.splitext(path)[1].lower()
    if ext not in _TEXT_SAMPLE_EXTS:
        return True
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(_SAMPLE_PREVIEW_CHARS * 4)
    except OSError:
        return False
    if ext in (".json", ".geojson"):
        # Only the whole document can be parsed; a truncated read means the
        # file is large, which is itself evidence of real content.
        if len(text) > _SAMPLE_PREVIEW_CHARS * 4 - 1:
            return True
        try:
            doc = json.loads(text)
        except ValueError:
            return bool(text.strip())
        if isinstance(doc, dict):
            if "features" in doc:
                return bool(doc.get("features"))
            return bool(doc)
        return bool(doc)
    if ext in (".csv", ".tsv"):
        # Header plus at least one row.
        return len([ln for ln in text.splitlines() if ln.strip()]) >= 2
    return bool(text.strip())


def _empty_output_error(run_dir, files):
    """The error to feed back to the reviser when a sample run 'succeeded'
    without producing any data, or "" when at least one file has records.
    Formatted like an exception's last line so _error_signature collapses
    every empty-output failure to one signature and the same-error limit
    still applies."""
    if not files:
        return ("EmptyOutputError: the sample download exited without error "
                "but wrote NO files. A verified handbook must save at least "
                "one file containing real records.")
    sizes = []
    for name in files:
        path = os.path.join(run_dir, name)
        if os.path.isfile(path) and _file_has_data(path):
            return ""
        try:
            sizes.append(f"{name} ({os.path.getsize(path)} bytes)")
        except OSError:
            sizes.append(name)
    return ("EmptyOutputError: the sample download exited without error but "
            "every file it wrote is empty or holds no records: "
            f"{', '.join(sizes)}. The query/filters in code_example matched "
            "nothing. Probe the source for its real field names and values "
            "(e.g. fetch one unfiltered row) and fix the filters so the "
            "sample returns at least one record; do not paper over this by "
            "writing an empty file.")


def _sample_previews(run_dir, files, stdout=""):
    """What the sample download actually produced, captured before the
    sandbox is deleted: a short head of each text file (size + first
    _SAMPLE_PREVIEW_CHARS chars) and the run's stdout tail. Binary outputs
    (GeoTIFF, shapefile parts, zips...) are listed by name and size only.
    Shown on the Handbook Studio review card so a reviewer can see the
    verification evidence, not just a pass/fail flag."""
    previews = []
    for name in list(files)[:_SAMPLE_PREVIEW_FILES]:
        path = os.path.join(run_dir, name)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        item = {"name": name, "size": size, "preview": "", "truncated": False}
        if os.path.splitext(name)[1].lower() in _TEXT_SAMPLE_EXTS:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read(_SAMPLE_PREVIEW_CHARS + 1)
                item["preview"] = text[:_SAMPLE_PREVIEW_CHARS]
                item["truncated"] = len(text) > _SAMPLE_PREVIEW_CHARS
            except OSError:
                pass
        previews.append(item)
    return {"files": previews, "more": max(0, len(files) - len(previews)),
            "stdout": (stdout or "")[-_SAMPLE_PREVIEW_CHARS:].strip()}


# Mirrors user_data_sources.STORED_FIELDS so drafts round-trip through the
# form and write_user_source without remapping.
_OUTPUT_FIELDS = (
    "data_source_name", "brief_description", "handbook", "code_example",
    "website", "requires_key", "key_name", "caveats", "key_signup_url",
)

# In-context example anchoring the writer on the house handbook style.
# Embedded directly rather than read from Handbooks/ at generation time: this
# used to open agents/data_agent/DataRetriever_Handbooks/Handbooks/
# openaq_api_global_air_quality_observations.toml straight off disk, so
# deleting that one catalog entry (e.g. to force "unknown source" testing, or
# just removing OpenAQ as a data source) crashed EVERY handbook draft with
# FileNotFoundError, not just OpenAQ's. The example's content is otherwise
# unchanged in spirit from that file — same schema, same style.
_TEMPLATE_HANDBOOK_TEXT = '''
data_source_name = 'OpenAQ API (Global Air Quality Observations)'

brief_description = \'\'\'
OpenAQ provides globally aggregated, station-level air quality observations (PM2.5, PM10, O3, NO2, SO2, CO, and other pollutants) contributed by government and research monitoring networks worldwide. Coverage and historical depth vary by country and station; some stations have only a few months of data, others go back a decade or more.
\'\'\'

handbook = \'\'\'
OpenAQ data is accessed via the REST API at https://api.openaq.org/v3/.
IMPORTANT: v3 requires an API key on every request, sent as the header 'X-API-Key: {OPENAQ_API_KEY}'. Requests without this header return 401/403.
Register for a free API key from the account settings page at https://explore.openaq.org.
Monitoring stations are called "locations". Use GET /v3/locations to discover stations; each has an id, name, coordinates, and the list of sensors (parameter + sensor id) it reports.
Spatial filtering on /v3/locations supports either 'coordinates=LAT,LON' combined with 'radius' (meters, max 25000), or 'bbox=MIN_LON,MIN_LAT,MAX_LON,MAX_LAT'. Only one of the two spatial filters may be used per request.
Filter by pollutant with 'parameters_id' (numeric id) — call GET /v3/parameters first to look up the id for a pollutant name (e.g. pm25, pm10, o3, no2, so2, co) if it is not already known.
To get actual measurement values for a station, read its sensor ids from the location record's 'sensors' list, then call GET /v3/sensors/{sensor_id}/measurements. Each sensor corresponds to one parameter at one location.
Temporal filtering on measurement endpoints uses 'date_from' and 'date_to' as ISO 8601 timestamps (e.g. 2024-01-01T00:00:00Z).
Pagination uses 'limit' (max 1000 per page) and 'page' (1-indexed). Keep incrementing 'page' until a page returns an empty 'results' array.
The API rate-limits requests (roughly 60 requests/minute per key on the free tier); on a 429 response, back off and retry rather than failing immediately.
Responses are JSON shaped {"meta": {...}, "results": [...]}; the fields of interest differ between /locations (station metadata) and /sensors/{id}/measurements (value, parameter, period.datetimeFrom/datetimeTo, coordinates).
Not every station reports every pollutant continuously — verify a station actually has recent data for the requested parameter before treating a gap as an error.
You need to create Python code to download and save the data. Another program will execute your code directly.
Put your reply into a Python code block. Explanation or conversation can be Python comments at the beginning of the code block (enclosed by ```python and ```).
The download code is only in a function named 'download_data()'. The last line is to execute this function.
Throw an error if the program fails to download the data; no need to handle the exceptions.
\'\'\'

code_example = \'\'\'
import os
import requests
import pandas as pd

def download_data():
    API_KEY = os.environ["OPENAQ_API_KEY"]
    BASE_URL = "https://api.openaq.org/v3"
    HEADERS = {"X-API-Key": API_KEY}

    # 1) Find stations near a point (illustrative — substitute your own
    #    coordinates/radius/bbox and parameter).
    locations_resp = requests.get(
        f"{BASE_URL}/locations",
        headers=HEADERS,
        params={"coordinates": "40.4406,-79.9959", "radius": 25000, "limit": 100},
        timeout=30,
    )
    locations_resp.raise_for_status()
    locations = locations_resp.json()["results"]

    # 2) Collect every sensor id that measures the target pollutant across
    #    the matched stations.
    target_param = "pm25"
    sensor_ids = []
    for loc in locations:
        for sensor in loc.get("sensors", []):
            if sensor.get("parameter", {}).get("name") == target_param:
                sensor_ids.append(sensor["id"])

    # 3) Pull measurements for each sensor, paginating until a page comes
    #    back empty. Each sensor's page is wrapped in its own try/except so
    #    one bad sensor doesn't abort the whole download.
    rows = []
    for sensor_id in sensor_ids:
        page = 1
        while True:
            try:
                resp = requests.get(
                    f"{BASE_URL}/sensors/{sensor_id}/measurements",
                    headers=HEADERS,
                    params={
                        "date_from": "2024-01-01T00:00:00Z",
                        "date_to": "2024-12-31T23:59:59Z",
                        "limit": 1000,
                        "page": page,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                results = resp.json()["results"]
            except Exception as e:
                print(f"Skipping sensor {sensor_id} page {page}: {e}")
                break
            if not results:
                break
            for r in results:
                rows.append({
                    "sensor_id": sensor_id,
                    "parameter": target_param,
                    "value": r.get("value"),
                    "datetime_from": r.get("period", {}).get("datetimeFrom", {}).get("utc"),
                    "datetime_to": r.get("period", {}).get("datetimeTo", {}).get("utc"),
                })
            page += 1

    df = pd.DataFrame(rows)
    df.to_csv("openaq_measurements.csv", index=False)
    return df

download_data()
\'\'\'

website = 'https://openaq.org'
requires_key = 'true'
key_name = 'OPENAQ_API_KEY'
'''


# ── Per-request log capture ──────────────────────────────────────────────────
_log_state = threading.local()


def start_log_capture(callback=None):
    """Begin capturing this thread's log lines; ``callback`` receives each line
    live (how the Web UI streams progress). Returns the live buffer."""
    _log_state.buffer = []
    _log_state.callback = callback
    return _log_state.buffer


def stop_log_capture():
    """Stop capturing and return this thread's captured lines."""
    logs = list(getattr(_log_state, "buffer", None) or [])
    _log_state.buffer = None
    _log_state.callback = None
    return logs


def _log(msg, *args):
    text = (msg % args) if args else msg
    logging.info("%s", text)
    buf = getattr(_log_state, "buffer", None)
    if buf is not None:
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        buf.append(line)
        cb = getattr(_log_state, "callback", None)
        if cb:
            try:
                cb(line)
            except Exception:
                pass  # a broken live stream must not kill the run


# ── Shared helpers ───────────────────────────────────────────────────────────

def _normalize_source(data, fallback_name="", website=""):
    """Coerce a model reply into the canonical handbook fields: stripped
    strings, ``requires_key`` forced to "true"/"false"."""
    out = {k: str(data.get(k, "") or "").strip() for k in _OUTPUT_FIELDS}
    out["data_source_name"] = out["data_source_name"] or (fallback_name or "").strip()
    out["website"] = out["website"] or (website or "").strip()
    out["requires_key"] = "true" if out["requires_key"].lower() in ("true", "1", "yes") else "false"
    if out["requires_key"] != "true":
        out["key_name"] = ""
    return out


def _parse_json_object(reply):
    """Parse a model reply into a dict, tolerating code fences and prose."""
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("empty reply from the model")
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", reply.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


def _extract_urls(text):
    # Strip trailing punctuation AND quote chars — callers now include JSON
    # blobs (e.g. context strings with embedded "docs_url": "https://...")
    # where the regex would otherwise swallow the closing quote.
    return [u.rstrip('.,;\'"') for u in _URL_RE.findall(text)] if text else []


def _html_to_text(html):
    """HTML -> readable text; crude tag strip if bs4 is unavailable."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "header",
                         "footer", "nav", "form"]):
            tag.decompose()
        text = soup.get_text("\n")
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)
    return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())


# Request headers for documentation pages. The identifying UA is tried
# first; agencies behind bot filters (cdc.gov returned 403 to it) get one
# retry with a browser-like UA and the Accept headers a browser sends.
_DOC_HEADERS = {
    "User-Agent": "AGSI-HandbookGenerator/1.0",
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_DOC_HEADERS_BROWSER = {
    **_DOC_HEADERS,
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"),
}

# A Socrata dataset id ("4x4") anywhere in a URL: /resource/i46a-9kgh.json,
# /d/i46a-9kgh, /dataset/Name/i46a-9kgh, dev.socrata.com/foundry/<domain>/i46a-9kgh.
_SOCRATA_ID_RE = re.compile(r"(?<![a-z0-9])([a-z0-9]{4}-[a-z0-9]{4})(?![a-z0-9])", re.I)
_SOCRATA_FOUNDRY_RE = re.compile(
    r"^https?://dev\.socrata\.com/foundry/([^/]+)/([a-z0-9]{4}-[a-z0-9]{4})", re.I)
_SOCRATA_META_COLUMNS = 250
_SOCRATA_META_DESC_CHARS = 90
_SOCRATA_META_BUDGET = 24000   # its own budget: a full column list is the
                               # point, and it must not crowd out the docs page


def _socrata_dataset_ref(url):
    """(domain, dataset_id) when ``url`` points at a Socrata dataset, else
    None. The foundry docs page names the data domain in its path; every
    other form lives on the data domain itself."""
    m = _SOCRATA_FOUNDRY_RE.match(url)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    m = re.match(r"^https?://([^/]+)(/.*)?$", url)
    if not m:
        return None
    domain, path = m.group(1).lower(), m.group(2) or ""
    m = re.search(r"/(?:resource|api/views|api/id)/([a-z0-9]{4}-[a-z0-9]{4})(?:\.\w+)?(?:[/?#]|$)",
                  path, re.I)
    if not m:
        # Catalog links end in the id: /d/<id>, /<Category>/<Name>/<id>
        m = re.search(r"/([a-z0-9]{4}-[a-z0-9]{4})/?(?:[?#]|$)", path, re.I)
    return (domain, m.group(1).lower()) if m else None


def _socrata_metadata_text(domain, dataset_id):
    """The dataset's own column list from Socrata's views endpoint, rendered
    for the handbook writer. The endpoint is served by the data domain
    itself (no bot filter, no key) and is the one place the real field
    names live -- a wide-format table like CDC PLACES has no 'year' or
    'measure' column, and a writer that only saw the narrative docs page
    (or nothing, when that page 403s) will guess that it does."""
    import requests
    url = f"https://{domain}/api/views/{dataset_id}.json"
    resp = requests.get(url, timeout=_FETCH_TIMEOUT, headers=_DOC_HEADERS)
    resp.raise_for_status()
    meta = resp.json()
    lines = [f"Dataset: {meta.get('name', '')} (id {dataset_id})",
             f"SODA endpoint: https://{domain}/resource/{dataset_id}.json"]
    desc = " ".join(str(meta.get("description") or "").split())
    if desc:
        lines.append(f"Description: {desc[:1200]}")
    columns = meta.get("columns") or []
    lines.append(f"Columns ({len(columns)}; use these exact field names in "
                 "$select/$where -- there are no others):")
    for col in columns[:_SOCRATA_META_COLUMNS]:
        field = col.get("fieldName") or ""
        if not field or field.startswith(":"):
            continue
        # One compact line per column: wide tables run to 150+ columns and
        # the whole list has to fit the metadata budget. The description is
        # kept because it often carries what the field name does not
        # (PLACES: "...prevalence of obesity among adults, 2023").
        entry = f"  {field} ({col.get('dataTypeName', '')})"
        cdesc = " ".join(str(col.get("description") or "").split())
        label = str(col.get("name") or "")
        if cdesc:
            entry += f": {cdesc[:_SOCRATA_META_DESC_CHARS]}"
        elif label and label.lower() != field.lower():
            entry += f": {label[:_SOCRATA_META_DESC_CHARS]}"
        lines.append(entry)
    if len(columns) > _SOCRATA_META_COLUMNS:
        lines.append(f"  ... {len(columns) - _SOCRATA_META_COLUMNS} more")
    return "\n".join(lines)


def _fetch_doc_page(url):
    """GET a docs page, retrying once with browser headers on 403/406.
    Returns the page text, or raises."""
    import requests
    resp = requests.get(url, timeout=_FETCH_TIMEOUT, headers=_DOC_HEADERS)
    if resp.status_code in (403, 406):
        resp = requests.get(url, timeout=_FETCH_TIMEOUT, headers=_DOC_HEADERS_BROWSER)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    if not any(t in ctype for t in ("html", "text", "json", "xml")):
        _log("Docs fetch: skipped %s (content type %s)", url, ctype or "unknown")
        return ""
    return _html_to_text(resp.text) if "html" in ctype else resp.text


def _fetch_doc_text(urls):
    """Fetch documentation pages (bounded by the text budgets); failures are
    skipped so the run still proceeds. Any URL that names a Socrata dataset
    also contributes that dataset's column list from the views API, so the
    writer sees the real schema even when the narrative page is blocked.
    Returns "" when nothing fetched."""
    seen, chunks, used = set(), [], 0
    socrata_seen = set()

    def _add(label, text):
        nonlocal used
        if not text.strip() or used >= _DOC_TEXT_BUDGET:
            return
        snippet = text[:min(_DOC_TEXT_PER_URL, _DOC_TEXT_BUDGET - used)]
        used += len(snippet)
        _log("Docs fetch: read %s characters from %s", f"{len(snippet):,}", label)
        chunks.append(f"----- Documentation from {label} -----\n{snippet}")

    def _add_metadata(ref, text):
        # Not charged to the page budget: the column list IS the schema and
        # is cut only by its own cap, never by how long the docs page was.
        snippet = text[:_SOCRATA_META_BUDGET]
        _log("Docs fetch: read %s characters of Socrata metadata for %s (%d columns)",
             f"{len(snippet):,}", ref[1], text.count("\n  "))
        chunks.append(f"----- Dataset schema for {ref[1]} on {ref[0]} "
                      f"(from https://{ref[0]}/api/views/{ref[1]}.json) -----\n{snippet}")

    for url in urls:
        if not url or not url.lower().startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        if used >= _DOC_TEXT_BUDGET:
            break
        ref = _socrata_dataset_ref(url)
        if ref and ref not in socrata_seen:
            socrata_seen.add(ref)
            try:
                _add_metadata(ref, _socrata_metadata_text(*ref))
            except Exception as e:
                _log("Docs fetch: could not read Socrata metadata for %s (%s)", ref[1], e)
        try:
            text = _fetch_doc_page(url)
        except Exception as e:
            _log("Docs fetch: could not fetch %s (%s)", url, e)
            continue
        _add(url, text)
    return "\n\n".join(chunks)


def _messages_to_claude(messages):
    """Convert an OpenAI-style chat ``messages`` list into the Claude Agent
    SDK's shape: a single ``system_prompt`` (every ``role: "system"`` turn,
    concatenated) plus one ``prompt`` string carrying the rest of the
    conversation (the Agent SDK's ``query()`` takes one prompt per session,
    not a message list) -- earlier user/assistant turns are rendered as a
    plain transcript so multi-turn context (e.g. refine_handbook's chat
    history) still reaches the model."""
    system_parts, turns = [], []
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = str(msg.get("content") or "")
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        else:
            turns.append(f"{role.upper()}: {content}")
    return "\n\n".join(system_parts), "\n\n".join(turns)


def _call_model(messages, user_key, model, stream_callback=None,
                reasoning_effort=None, provider="openai", phase="generation"):
    """One completion via the selected provider (used by refine/evaluate/
    analyze).

    ``phase`` is what the tokens are booked against. It used to be hard-coded
    to "generation" for all three callers, which put the failure-analysis and
    semantic-validation calls -- both made DURING a test run -- into a
    "generation" bucket. In a control session, where no handbook is ever
    generated, that bucket was the only thing in the token table and read as
    though a handbook had been drafted after all. OpenAI: streaming chat completion via the shared router, JSON
    mode first, falling back to plain text since GIBD/older backends may
    reject response_format. Claude: a tool-free Agent SDK session (still a
    real agent session per the chosen integration style, just with no tools
    to reach for) -- ``model``/``user_key`` are then the Claude model id and
    Anthropic API key rather than their OpenAI equivalents."""
    if provider == "claude":
        system_prompt, prompt = _messages_to_claude(messages)
        result = claude_agent_provider.run_agent(
            prompt, system_prompt=system_prompt or None, tools=[],
            model=model or CLAUDE_HANDBOOK_MODEL, api_key=user_key,
            on_activity=stream_callback)
        usage_ledger.record(model or CLAUDE_HANDBOOK_MODEL,
                            getattr(result, "usage", None),
                            phase=phase, source="llm_call_claude")
        if not result.success:
            raise RuntimeError(result.error or "Claude Agent SDK call failed.")
        return result.text

    def consume(stream):
        full = ""
        for chunk in stream:
            # With stream_options={"include_usage": True} the provider sends a
            # final chunk carrying usage and an empty choices list. Older
            # backends (GIBD, local models) simply never send one, so this is
            # best-effort and its absence is counted as an unmeasured call
            # rather than as zero cost.
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                usage_ledger.record(model, usage, phase=phase,
                                    source="llm_call_stream")
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if content:
                print(content, end="", flush=True)
                full += content
                if stream_callback:
                    stream_callback(content)
        print()
        return full

    # "max" is a real level for some models (e.g. gpt-5.6-sol) but only on
    # the Responses API -- Chat Completions (this function) 400s on it, so
    # downgrade to the next-highest tier it does support rather than fail.
    if reasoning_effort == "max":
        reasoning_effort = "xhigh"
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    # Only OpenAI-compatible backends understand stream_options; the retry
    # below drops it along with response_format if the server rejects either.
    usage_opt = {"stream_options": {"include_usage": True}}
    try:
        return consume(helper.client_chat_completion_stream(
            model=model, messages=messages, user_key=user_key,
            response_format={"type": "json_object"}, **usage_opt, **extra))
    except Exception as e:
        _log("LLM call: retrying without response_format/stream usage (%s)", e)
        return consume(helper.client_chat_completion_stream(
            model=model, messages=messages, user_key=user_key, **extra))


# ── Generation: the skill pipeline (in-process) ──────────────────────────────

def _research_json(client, model, prompt, web_search=True, reasoning_effort=None):
    """One Responses-API call (optionally web-search-enabled) parsed as JSON."""
    kwargs = {"tools": [{"type": "web_search"}]} if web_search else {}
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    resp = client.responses.create(model=model, input=prompt, **kwargs)
    usage_ledger.record(model, getattr(resp, "usage", None),
                        phase="generation", source="research_json")
    return _parse_json_object(resp.output_text)


def _research_json_claude(model, api_key, prompt, web_search=True):
    """Claude Agent SDK equivalent of ``_research_json``: a session with the
    built-in WebSearch/WebFetch tools (or none) whose final text is the same
    JSON object contract every research prompt here already asks for."""
    tools = ["WebSearch", "WebFetch"] if web_search else []
    result = claude_agent_provider.run_agent(
        prompt, tools=tools, model=model or CLAUDE_HANDBOOK_MODEL,
        api_key=api_key, on_activity=_log)
    usage_ledger.record(model or CLAUDE_HANDBOOK_MODEL, getattr(result, "usage", None),
                        phase="generation", source="research_json_claude")
    if not result.success:
        raise RuntimeError(result.error or "Claude Agent SDK call failed.")
    return _parse_json_object(result.text)


def _suggest_source(client, model, query, website, reasoning_effort=None,
                    provider="openai", api_key=None):
    """Stage 1: pick and confirm the best public data source via web search."""
    hint = f"\nUser-provided website: {website}" if website else ""
    prompt = (
        "Given a data need, source name, or API link, pick the single best "
        "public data source. Use web search to confirm it exists and find the "
        "official website and documentation URL. Reply with ONLY a JSON object: "
        '{"name": str, "provider": str, "website": str, "docs_url": str, "why": str}'
        f"\n\nData need: {query}{hint}")
    if provider == "claude":
        return _research_json_claude(model, api_key, prompt)
    return _research_json(client, model, prompt, reasoning_effort=reasoning_effort)


def _analyze_access(client, model, query, src, reasoning_effort=None,
                    provider="openai", api_key=None):
    """Stages 2-3: fetch the source's docs and pin down the access method."""
    docs = _fetch_doc_text([src.get("docs_url", ""), src.get("website", "")])
    prompt = (
        "Determine exactly how to retrieve this data programmatically. The "
        "documentation below may be thin, generic, or entirely empty — a "
        "user does not always supply a docs URL, and even a found one is "
        "often a marketing page rather than an API reference. Start from "
        "your own well-established knowledge of this source's real API when "
        "you have it, then use web search to verify or fill in anything "
        "you are not confident about, anything likely to have changed "
        "since your training (API versions, deprecated endpoints, current "
        "rate limits or auth schemes), or anything the documentation below "
        "does not cover. Record real endpoints, parameters, auth, formats, "
        "pagination and rate limits in notes — only facts you are actually "
        "confident are current and correct; never fabricate a URL or "
        "parameter you are unsure of. Reply with ONLY a JSON object: "
        '{"method": str, "base_url": str, "requires_key": bool, '
        '"key_name": str (comma-separated UPPER_SNAKE env-var names, "" if none), '
        '"key_signup_url": str, "notes": str}'
        f"\n\nData need: {query}\nSource: {json.dumps(src)}\n\nDOCUMENTATION:\n{docs}")
    if provider == "claude":
        return _research_json_claude(model, api_key, prompt)
    return _research_json(client, model, prompt, reasoning_effort=reasoning_effort)


_ACCESS_MECHANISMS = {
    "rest": "REST API",
    "stac": "STAC",
    "ogc": "OGC Service",
    "sql": "SQL Interface",
    "bulk": "Bulk Download",
    "http_file": "HTTP File Download",
    "arcgis_featureserver": "ArcGIS FeatureServer",
}


def select_access_mechanism(query, website="", user_key=None, model=None,
                            provider="openai"):
    """Classify a data need into the single best-fitting access mechanism,
    for callers (like the "Generate & Test" mode) that let a human skip
    picking one. Requires an OpenAI key (provider="openai", same requirement
    as ``generate_handbook``) or an Anthropic key (provider="claude").
    Returns {"mechanism": one of _ACCESS_MECHANISMS, "why": str}, defaulting
    to "rest" if the model reply can't be parsed."""
    prompt = (
        "Pick the single best-fitting data access mechanism for this source "
        'and task. Reply with ONLY a JSON object: {"mechanism": one of '
        f"{list(_ACCESS_MECHANISMS)}, \"why\": str}}\n\n"
        f"Source/task: {query}\nWebsite: {website}\n\n"
        "Mechanism meanings: rest=documented REST/HTTP endpoints; "
        "stac=STAC catalogs, collections, item search, and asset retrieval; "
        "ogc=OGC service or OGC API interface such as WFS/WCS/WMS or OGC API "
        "Features/Coverages; sql=database or SQL query interface; "
        "bulk=official bulk file, regional extract, archive, or repository "
        "download; http_file=a single, direct HTTP(S) file download (a "
        "plain static file link, not a query API or multi-file archive); "
        "arcgis_featureserver=an Esri ArcGIS FeatureServer/MapServer REST "
        "endpoint specifically (query/queryRelatedRecords over "
        "/FeatureServer or /MapServer), not a generic REST API.")

    if provider == "claude":
        key = user_key or claude_agent_provider.load_anthropic_key()
        if not key:
            raise ValueError(
                "Selecting an access mechanism requires an Anthropic API key.")
        result = claude_agent_provider.run_agent(
            prompt, tools=[], model=model or CLAUDE_HANDBOOK_MODEL,
            api_key=key, on_activity=_log)
        usage_ledger.record(model or CLAUDE_HANDBOOK_MODEL,
                            getattr(result, "usage", None),
                            phase="generation", source="select_mechanism_claude")
        reply_text = result.text if result.success else ""
    else:
        key = (user_key if str(user_key or "").startswith("sk-")
               else helper._usable_openai_fallback_key())
        if not key:
            raise ValueError(
                "Selecting an access mechanism requires an OpenAI API key (sk-...).")
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model or HANDBOOK_GEN_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}])
        usage_ledger.record(model or HANDBOOK_GEN_MODEL,
                            getattr(resp, "usage", None),
                            phase="generation", source="select_mechanism")
        reply_text = resp.choices[0].message.content
    try:
        data = _parse_json_object(reply_text)
        mechanism = str(data.get("mechanism") or "").strip().lower()
        if mechanism not in _ACCESS_MECHANISMS:
            mechanism = "rest"
        return {"mechanism": mechanism, "why": str(data.get("why") or "").strip()}
    except (ValueError, TypeError):
        return {"mechanism": "rest", "why": "Defaulted after a parsing error."}


def _writer_prompt(context):
    template = _TEMPLATE_HANDBOOK_TEXT
    return (
        "Draft a data-retrieval handbook as raw TOML (no code fence) with exactly the structure of the "
        "example below: data_source_name, brief_description, handbook, code_example, website, requires_key, "
        "key_name. In handbook: one requirement per line (base URL, endpoints, parameters, spatial filtering, "
        "response structure, pagination, rate limits, save format, pitfalls), ending with the same four fixed "
        "reply-format lines as the example. Reusability: this handbook documents the SOURCE's general "
        "capabilities and must stay usable for any future retrieval task against it, not just the one "
        "described below — never phrase a line as being 'for the task', and never bake today's specific "
        "parameter values (a place name, coordinates, a radius, a date range, a tag value) into the handbook "
        "as if they were permanent facts about the source. If a worked example helps illustrate query "
        "syntax, label it clearly as illustrative (e.g. 'Example query pattern (illustrative — substitute "
        "your own place/tags/radius/dates)') and use placeholder-style values; phrase any correctness-"
        "verification guidance generically (e.g. 'verify results fall within the requested radius', never a "
        "specific number) so it still applies when a different task supplies different parameters. "
        "Keep downloads fast and bounded: when the source offers any way to estimate result size before "
        "downloading (a count endpoint or parameter, a metadata/summary query, a paged total field), document "
        "it in the handbook and instruct that bulk downloads check the estimate FIRST and split by time window "
        "or sub-region when the estimate is large — never issue one unbounded query for a large area/period "
        "and hope; document server-side filters (fields/columns, tags, date windows, bbox) so retrieval code "
        "requests only what the task needs rather than downloading everything and filtering locally; and when "
        "the source supports a server-side query timeout or result limit (e.g. a [timeout:] setting or "
        "maxRecordCount), document its use with a modest value instead of leaving it unbounded. "
        "code_example: one complete runnable script with a download_data() "
        "function, called on the last line. Do NOT wrap the initial/setup request(s) in try/except — let a "
        "genuine bug fail loudly so it can be diagnosed. But when the code loops over many pages, batches, "
        "tiles, or items (pagination, chunked/batched downloads, per-feature fetches), wrap EACH iteration's "
        "request in its own narrow try/except: on failure, retry once or twice with a short backoff, and if it "
        "still fails, print/log which page/batch/item was skipped and why, then continue the loop — never let "
        "one bad page or batch abort an otherwise-successful bulk download, and never use a blanket try/except "
        "around the whole function to hide errors. "
        "Use only facts from the provided research; never "
        "invent credentials. requires_key: 'true' or ''. key_name: comma-separated UPPER_SNAKE env-var names, "
        "one per credential the source needs (e.g. 'EOG_CLIENT_ID,EOG_CLIENT_SECRET'), '' if none — and "
        "code_example must read exactly these names via os.environ. If the source needs only ONE credential "
        "(the overwhelmingly common case — a single API key), key_name must list exactly ONE name — never "
        "list two or three alternate/alias spellings of the same single credential (e.g. never "
        "'FIRMS_MAP_KEY,NASA_FIRMS_MAP_KEY,MAP_KEY' for one API key): each entry in key_name becomes its own "
        "required-credential field the user is asked to fill in, so listing aliases of one credential as if "
        "they were separate credentials makes the user fill in the same value multiple times under different "
        "names for no reason. Pick the ONE name the source's own documentation uses. If the handbook field's "
        "prose describes an endpoint or parameter that includes a credential placeholder (e.g. in a URL "
        "pattern), that placeholder must be written as the exact key_name in curly braces (e.g. "
        "{EOG_CLIENT_ID}) — never a shortened or different alias — since a real value gets substituted into "
        "exactly that token before this handbook is used to write retrieval code, and any other token is left "
        "unreplaced. code_example must NEVER silently continue, fall back to a keyless/reduced endpoint, or "
        "return placeholder/partial data when a required credential is missing or empty — reading an unset "
        "credential must raise (e.g. os.environ[\"NAME\"], which raises KeyError on its own) so a missing key "
        "surfaces as a clear failure instead of a script that appears to succeed with the wrong data. Also add "
        "two fields not in the example: "
        "caveats — short user-facing warnings, one per line (paid tiers/cost, registration effort, rate limits, "
        "license restrictions, large downloads, coverage gaps), '' if none apply; and key_signup_url — the "
        "exact page where a user registers for the credentials, '' if no key needed.\n\n"
        f"EXAMPLE:\n{template}\n\n{context}")


def _draft_toml(client, model, prompt, reasoning_effort=None,
                provider="openai", api_key=None):
    """Stage 4: write (or fix) the handbook TOML; validated before use."""
    if provider == "claude":
        result = claude_agent_provider.run_agent(
            prompt, tools=[], model=model or CLAUDE_HANDBOOK_MODEL,
            api_key=api_key, on_activity=_log)
        # Recorded BEFORE the success check and before _toml.loads: a draft
        # that comes back malformed still spent the tokens, and dropping it
        # would make the verification loop -- which calls this once per
        # revision -- look free precisely when it is costing the most.
        usage_ledger.record(model or CLAUDE_HANDBOOK_MODEL,
                            getattr(result, "usage", None),
                            phase="generation", source="draft_toml_claude")
        if not result.success:
            raise RuntimeError(result.error or "Claude Agent SDK call failed.")
        text = result.text.strip().removeprefix("```toml").removesuffix("```").strip()
        _toml.loads(text)
        return text
    kwargs = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}
    resp = client.responses.create(model=model, input=prompt, **kwargs)
    usage_ledger.record(model, getattr(resp, "usage", None),
                        phase="generation", source="draft_toml")
    text = resp.output_text.strip().removeprefix("```toml").removesuffix("```").strip()
    _toml.loads(text)
    return text


def _verify_revise_prompt(context, toml_text, err, sig, streak, docs_text=""):
    """Revision prompt for the self-verification loop (_verify_toml).

    Mirrors _REFINE_INSTRUCTIONS' diagnose-first, change-minimally framing
    instead of a bare "here's the error, fix it", and grounds the fix in
    re-fetched documentation instead of a blind guess from the stack trace
    alone — the two gaps that let self-verification cycle through superficial
    retries (e.g. WFS <-> WMS <-> ConnectionError) without ever finding the
    actual root cause."""
    repeat_note = ""
    if streak > 1:
        repeat_note = (
            f"\n\nThis is the SAME underlying failure ({sig}) as your last "
            f"{streak - 1} attempt(s) — your previous fix did not address the "
            "root cause. Do not repeat a change you already tried; find a "
            "different cause, checking the documentation below rather than "
            "guessing another endpoint or parameter.")
    docs_block = (
        f"\n\nRELEVANT DOCUMENTATION (ground your fix in this; do not invent "
        f"URLs or parameters):\n\n{docs_text}" if docs_text else "")
    return (
        f"{_writer_prompt(context)}\n\n"
        "Your previous handbook's code_example failed when executed. Diagnose "
        "the root cause from the error below, then return an IMPROVED "
        "handbook (same TOML structure). Typical fixes: correct a wrong "
        "endpoint or parameter named in the error; add a required parameter; "
        "switch the saved file format; handle pagination, limits, or rate "
        "limits; fix the authentication; validate raw response bytes instead "
        "of hard-failing when an expected header is missing. Change ONLY "
        "what the error requires — keep every instruction and code line that "
        "isn't implicated."
        f"{repeat_note}\n\nPREVIOUS TOML:\n{toml_text}\n\nERROR:\n{err}"
        f"{docs_block}")


def _error_signature(err):
    """A stable signature for 'is this the same failure as before', ignoring
    volatile details (exact URLs, query-param casing/ordering) that make an
    LLM's superficially-different retries look like distinct errors when
    they're really the same underlying problem — e.g. a 404 on a WFS
    endpoint and a 404 on a WMS endpoint at the same host should count as
    the same failure, not reset the stuck-loop counter."""
    last_line = err.strip().splitlines()[-1] if err.strip() else ""
    # Requests' HTTPError/ConnectionError messages embed the full request
    # URL after " for url:" — drop it so different URLs don't look distinct.
    last_line = re.split(r"\s+for url:", last_line, maxsplit=1)[0]
    match = re.match(r"^([\w.]+(?:Error|Exception)):\s*(.*)$", last_line)
    if not match:
        return last_line
    exc_type, detail = match.groups()
    status_match = re.match(r"^(\d{3})\b", detail)
    return f"{exc_type}:{status_match.group(1)}" if status_match else exc_type


def _verify_toml(client, model, toml_text, context, extra_env=None,
                 reasoning_effort=None):
    """Stage 5: run code_example in a sandbox, feeding failures back to the
    writer. Retries until success; stops when the same error repeats
    _SAME_ERROR_LIMIT times in a row, or after _MAX_VERIFY_ATTEMPTS total
    attempts regardless of matching (immediately if ``client`` is None — no
    reviser available). ``extra_env`` supplies data-source credentials to the
    sample run (env vars + {PLACEHOLDER} substitution); credentials the code
    needs but nobody has -> skip. Returns (final_toml_text,
    verified|unverified|skipped_needs_key, attempts, last_error, files) —
    ``last_error`` is the full text of the final failed attempt ("" if
    verified/skipped), ``files`` is what the successful run produced, and
    ``samples`` (see _sample_previews) is a preview of those files plus the
    run's stdout -- or None when nothing ran successfully."""
    extra_env = {k: v for k, v in (extra_env or {}).items() if v}
    attempts, last_sig, streak = 0, None, 0
    last_error = ""
    docs_text = None  # fetched lazily on the first revision, cached after that
    work_dir = tempfile.mkdtemp(prefix="skillverify_")
    try:
        while True:
            code = _toml.loads(toml_text)["code_example"]
            missing = [k for k in re.findall(r"os\.environ\[[\"'](\w+)[\"']\]", code)
                       if k not in os.environ and k not in extra_env]
            if missing:
                _log("[verify] skipped — set env var(s) %s to enable a live run", missing)
                return toml_text, "skipped_needs_key", attempts, "", [], None
            # Legacy handbooks may inline credentials as {NAME} tokens instead
            # of reading os.environ — substitute those too.
            run_code = code
            for k, v in extra_env.items():
                run_code = run_code.replace("{" + k + "}", v)

            run_dir = os.path.join(work_dir, "_verify")
            shutil.rmtree(run_dir, ignore_errors=True)
            os.makedirs(run_dir, exist_ok=True)
            try:
                # Generated code never runs in this process: subprocess only.
                p = subprocess.run([python_executable(), "-c", run_code], cwd=run_dir,
                                   capture_output=True, text=True,
                                   timeout=_VERIFY_TIMEOUT,
                                   env={**os.environ, **extra_env})
                err = "" if p.returncode == 0 else (p.stderr or p.stdout)[-3000:]
                run_stdout = p.stdout or ""
            except subprocess.TimeoutExpired:
                err = (f"Timed out after {_VERIFY_TIMEOUT}s — likely downloading too "
                       "much; shrink the example's area/time window.")
                run_stdout = ""
            attempts += 1

            if not err:
                files = sorted(os.listdir(run_dir))
                _log("[verify] code ran successfully, produced: %s",
                     files or "no files")
                # Exit code 0 is not evidence: the run must have saved records.
                err = _empty_output_error(run_dir, files)
                if err:
                    _log("[verify] %s", err.split(".", 1)[0])
                    # Stdout first: _error_signature reads the LAST line, and
                    # that must stay the EmptyOutputError line.
                    if run_stdout.strip():
                        err = f"RUN STDOUT (tail):\n{run_stdout[-1500:]}\n\n{err}"
            if not err:
                samples = _sample_previews(run_dir, files, run_stdout)
                for item in samples["files"]:
                    _log("[verify] sample output %s (%s bytes)", item["name"], item["size"])
                return toml_text, "verified", attempts, "", files, samples

            sig = _error_signature(err)
            streak = streak + 1 if sig == last_sig else 1
            last_sig = sig
            last_error = err
            if (streak >= _SAME_ERROR_LIMIT or attempts >= _MAX_VERIFY_ATTEMPTS
                    or client is None):
                reason = ("hit the same-error limit" if streak >= _SAME_ERROR_LIMIT
                          else "hit the max-attempts cap" if attempts >= _MAX_VERIFY_ATTEMPTS
                          else "no reviser available")
                _log("[verify] giving up after %d attempt(s) (%s): %s",
                     attempts, reason, sig)
                return toml_text, "unverified", attempts, last_error, [], None
            _log("[verify] attempt %d failed — revising\n  %s", attempts, sig)
            if docs_text is None:
                docs_text = _fetch_doc_text(_extract_urls(context))
            toml_text = _draft_toml(client, model, _verify_revise_prompt(
                context, toml_text, err, sig, streak, docs_text),
                reasoning_effort=reasoning_effort)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _verify_toml_claude(toml_text, context, extra_env=None, model=None,
                        api_key=None):
    """Claude Agent SDK equivalent of ``_verify_toml``: instead of the host
    driving a subprocess.run + redraft loop itself, hand the entire write ->
    run -> diagnose -> fix cycle to one bounded agent session with Bash/
    Write/Read/Edit scoped to a fresh sandbox directory. The agent manages
    its own retries (bounded by ``_CLAUDE_VERIFY_MAX_TURNS``); the host does
    NOT trust the agent's self-reported outcome, only what it can observe
    afterward -- real files produced in the sandbox, and whether the agent's
    final message is a parseable handbook TOML.

    Returns the same shape as ``_verify_toml``: (final_toml_text,
    verified|unverified|skipped_needs_key, attempts, last_error, files).
    ``attempts`` is the literal string "agent-managed" (the agent, not the
    host, is driving individual retries here) -- callers that store/display
    it must tolerate a non-int.
    """
    extra_env = {k: v for k, v in (extra_env or {}).items() if v}
    work_dir = tempfile.mkdtemp(prefix="skillverify_claude_")
    try:
        code = _toml.loads(toml_text)["code_example"]
        missing = [k for k in re.findall(r"os\.environ\[[\"'](\w+)[\"']\]", code)
                   if k not in os.environ and k not in extra_env]
        if missing:
            _log("[verify] skipped — set env var(s) %s to enable a live run", missing)
            return toml_text, "skipped_needs_key", "agent-managed", "", [], None

        prompt = (
            f"{_writer_prompt(context)}\n\n"
            "You are VERIFYING this handbook by actually running it, not just "
            "writing it. Work in the current directory (it is empty and "
            "sandboxed): write the handbook's code_example to a file named "
            "download.py, then run it with `python download.py` via the Bash "
            "tool. If it fails, diagnose the real cause from the error (never "
            "guess), revise the handbook (same TOML structure) to fix it, "
            "overwrite download.py with the corrected code_example, and run "
            "it again. Repeat until it runs successfully and saves real "
            "output files (a header-only CSV, an empty JSON array, or an "
            "empty FeatureCollection does NOT count: check that the file "
            "holds at least one record, and if not, treat the filters as "
            "wrong and fix them), or you are confident further attempts with a "
            "different endpoint/parameter guess would not help. Required "
            "credentials are already present in the environment under the "
            "exact names documented in key_name — read them via os.environ, "
            "never invent or hard-code a value.\n\n"
            "When you are done (whether it succeeded or you are giving up), "
            "your FINAL message must be ONLY the complete, current handbook "
            "as raw TOML (no code fence, no prose before or after) — the "
            "same structure as the example above, reflecting whatever fixes "
            "you made along the way.\n\n"
            f"STARTING TOML:\n{toml_text}")

        _log("[verify] handing the write/run/fix loop to a Claude agent session…")
        result = claude_agent_provider.run_agent(
            prompt, tools=["Read", "Write", "Edit", "Bash"], cwd=work_dir,
            max_turns=_CLAUDE_VERIFY_MAX_TURNS,
            model=model or CLAUDE_HANDBOOK_MODEL, api_key=api_key,
            extra_env=extra_env, on_activity=_log)
        # A write/run/fix loop over many turns -- typically the largest single
        # spend in generation, so leaving it out understated the whole phase.
        usage_ledger.record(model or CLAUDE_HANDBOOK_MODEL,
                            getattr(result, "usage", None),
                            phase="generation", source="verify_toml_claude")

        files = sorted(
            name for name in os.listdir(work_dir)
            if name != "download.py"
            and os.path.isfile(os.path.join(work_dir, name)))

        final_text = (result.text.strip()
                     .removeprefix("```toml").removesuffix("```").strip())
        try:
            _toml.loads(final_text)
            final_toml = final_text
        except Exception:
            # The agent didn't end on a clean TOML-only message -- keep the
            # handbook we started with rather than persist unparseable text.
            final_toml = toml_text

        if not result.success:
            _log("[verify] Claude agent session failed: %s", result.error)
            return final_toml, "unverified", "agent-managed", result.error, files, None

        if files:
            _log("[verify] Claude agent run produced: %s", files)
            empty = _empty_output_error(work_dir, files)
            if empty:
                _log("[verify] %s", empty.split(".", 1)[0])
                return final_toml, "unverified", "agent-managed", empty, files, None
            return (final_toml, "verified", "agent-managed", "", files,
                    _sample_previews(work_dir, files))

        _log("[verify] Claude agent session finished without producing any files")
        return final_toml, "unverified", "agent-managed", (
            "The agent session ended without any output file in the "
            "sandbox — treating this as unverified regardless of what the "
            "agent claimed."), [], None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# Pipeline verify statuses -> the report vocabulary callers expect.
_SKILL_STATUS_MAP = {
    "verified": ("passed", True, "The sample download ran and saved real data."),
    "skipped_needs_key": ("passed_smoke", True,
                          "Drafted from live docs; needs a real API key for a full test run."),
    "unverified": ("failed", False,
                   "The sample download kept failing with the same error; review before use."),
    "skipped": ("skipped", False, "Verification was not run."),
}


def verify_handbook(source, data_source_keys=None, user_key=None, model=None,
                    reasoning_effort=None, provider="openai"):
    """Verify a saved handbook by running its code_example with real
    data-source credentials (``data_source_keys``: {NAME: value}). With an
    sk- key (OpenAI) or an Anthropic key (Claude) available the writer
    auto-fixes failures (same revision loop as generation); otherwise it is
    a single pass/fail run.

    Returns ``(source, report)`` — the possibly revised handbook fields and a
    {status, verified, attempts, error, files, note} report."""
    fields = {k: str(source.get(k, "") or "") for k in _OUTPUT_FIELDS}
    # json.dumps quoting is valid TOML for basic strings.
    toml_text = "\n".join(f"{k} = {json.dumps(fields[k])}" for k in _OUTPUT_FIELDS)
    context = (f"Data need: {fields['data_source_name']}\n"
               f"Source: {json.dumps({'name': fields['data_source_name'], 'website': fields['website']})}")

    _log("Verify: testing '%s' with the provided credentials...",
         fields["data_source_name"][:80])
    if provider == "claude":
        key = user_key or claude_agent_provider.load_anthropic_key()
        toml_text, status, attempts, error, files, samples = _verify_toml_claude(
            toml_text, context, extra_env=data_source_keys,
            model=model, api_key=key)
    else:
        key = (user_key if str(user_key or "").startswith("sk-")
               else helper._usable_openai_fallback_key())
        client = OpenAI(api_key=key) if key else None
        toml_text, status, attempts, error, files, samples = _verify_toml(
            client, model or HANDBOOK_GEN_MODEL, toml_text, context,
            extra_env=data_source_keys, reasoning_effort=reasoning_effort)
    out = _normalize_source(_toml.loads(toml_text),
                            fallback_name=fields["data_source_name"],
                            website=fields["website"])
    st, ok, note = _SKILL_STATUS_MAP.get(status, ("skipped", False, status))
    return out, {"status": st, "verified": ok, "attempts": attempts,
                 "error": error, "files": files, "note": note,
                 "samples": samples}


def generate_handbook(query, website="", doc_urls=None, user_key=None,
                      model=None, verify=True, data_source_keys=None,
                      reasoning_effort=None, provider="openai", **_legacy):
    """Draft a complete handbook: suggest source -> analyze access (docs +
    web search) -> write -> verify.

    ``provider`` selects the backend for every stage: "openai" (default,
    requires an sk- key) or "claude" (requires an Anthropic API key; every
    stage runs as a Claude Agent SDK session — see ``claude_agent_provider``
    and ``_verify_toml_claude``).

    Returns the handbook dict; when ``verify``, the self-test report rides in
    ``source['_verification']`` ({status, verified, attempts, error, files,
    note}). Extra keyword arguments from older callers (fetch_docs,
    verify_attempts, ...) are accepted and ignored.
    """
    if not query or not query.strip():
        raise ValueError("Describe the data source to generate a handbook.")

    client = None
    if provider == "claude":
        key = user_key or claude_agent_provider.load_anthropic_key()
        if not key:
            raise ValueError(
                "Generating a handbook with Claude requires an Anthropic "
                "API key. Enter one in Settings, or set ANTHROPIC_API_KEY "
                "on the server.")
        model = model or CLAUDE_HANDBOOK_MODEL
    else:
        # The hosted web-search tool needs an sk- key: the caller's own, else
        # the server's .env fallback.
        key = (user_key if str(user_key or "").startswith("sk-")
               else helper._usable_openai_fallback_key())
        if not key:
            raise ValueError(
                "Generating knowledge requires an OpenAI API key (sk-...): the live "
                "web-research pipeline cannot run on a GIBD proxy key. Use an "
                "OpenAI key, or set OPENAI_API_KEY in the server's .env.")
        client = OpenAI(api_key=key)  # per-call; never module-global (shared workers)
        model = model or HANDBOOK_GEN_MODEL

    query, site = query.strip(), (website or "").strip() or next(iter(doc_urls or []), "")
    _log("[stage:source:start] Finding the best official source for your request.")

    src = _suggest_source(client, model, query, site,
                          reasoning_effort=reasoning_effort,
                          provider=provider, api_key=key)
    _log("[stage:source:complete] Selected %s from %s.",
         src.get("name") or "the data source",
         src.get("provider") or "its official provider")

    _log("[stage:docs:start] Reading the official API documentation and access requirements.")
    access = _analyze_access(client, model, query, src,
                             reasoning_effort=reasoning_effort,
                             provider=provider, api_key=key)
    auth_note = ("Credentials are required" if access.get("requires_key")
                 else "No API key is required")
    _log("[stage:docs:complete] Confirmed %s access at %s. %s.",
         access.get("method") or "programmatic",
         access.get("base_url") or "the documented endpoint", auth_note)

    context = f"Data need: {query}\nSource: {json.dumps(src)}\nAccess: {json.dumps(access)}"
    _log("[stage:draft:start] Writing retrieval instructions and a runnable Python example.")
    toml_text = _draft_toml(client, model, _writer_prompt(context),
                            reasoning_effort=reasoning_effort,
                            provider=provider, api_key=key)
    _log("[stage:draft:complete] The handbook draft and sample code are ready.")

    status = "skipped"
    verify_attempts, verify_error, verify_files, verify_samples = 0, "", [], None
    if verify:
        _log("[stage:verify:start] Running a small sample download in an isolated process.")
        if provider == "claude":
            (toml_text, status, verify_attempts, verify_error, verify_files,
             verify_samples) = _verify_toml_claude(
                toml_text, context, extra_env=data_source_keys,
                model=model, api_key=key)
        else:
            (toml_text, status, verify_attempts, verify_error, verify_files,
             verify_samples) = _verify_toml(
                client, model, toml_text, context, extra_env=data_source_keys,
                reasoning_effort=reasoning_effort)
        verify_message = {
            "verified": "The sample download completed successfully.",
            "skipped_needs_key": "A full test needs source credentials; the draft passed its credential check.",
            "unverified": "The sample still needs review because its download did not complete.",
        }.get(status, "Verification finished.")
        verify_state = ("complete" if status in ("verified", "skipped_needs_key")
                        else "warning")
        _log("[stage:verify:%s] %s", verify_state, verify_message)

    fields = _toml.loads(toml_text)
    if not str(fields.get("key_signup_url", "")).strip():
        fields["key_signup_url"] = str(access.get("key_signup_url", "") or "")
    source = _normalize_source(fields, fallback_name=query, website=site)
    if verify:
        st, ok, note = _SKILL_STATUS_MAP.get(status, ("skipped", False, status))
        source["_verification"] = {
            "status": st, "verified": ok, "attempts": verify_attempts,
            "error": verify_error, "files": verify_files, "note": note,
            "samples": verify_samples,
        }
    _log("[stage:review:complete] Your handbook is ready to review, edit, and save.")
    return source


# ── Refinement (draft-editing chat) ──────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a professional GIScience data engineer with 20+ years of experience "
    "collecting geospatial data and writing the technical 'handbooks' that let an "
    "automated agent download it. You know the real APIs, endpoints, parameters, "
    "authentication schemes, and the practical pitfalls of OpenStreetMap, the US "
    "Census Bureau, NASA EarthData, USGS, Copernicus, EPA, OpenTopography, and "
    "hundreds of other geospatial data providers.\n\n"
    "Your job: given a data source, write an accurate, self-contained handbook a "
    "code-generating agent can follow to download data WITHOUT any further "
    "research. Be precise and factual. If you are unsure of an exact endpoint or "
    "parameter, describe the documented behavior rather than inventing a URL — "
    "never fabricate API paths, parameter names, or key formats."
)

_REFINE_INSTRUCTIONS = """\
You are REFINING an existing handbook for a data source based on the user's \
feedback — most often an error they hit when running the code_example, or a \
request to change behavior (e.g. the output format, an endpoint, or a \
parameter). You are given the CURRENT handbook as a JSON object plus the user's \
latest message.

Diagnose the problem and return an IMPROVED handbook. Typical fixes: correct a \
wrong endpoint or parameter named in the error; add a required parameter; switch \
the saved file format; handle pagination, limits, or rate limits; fix the \
authentication. Change ONLY what the feedback requires — keep everything that \
already works, and preserve the user's manual edits where they don't conflict.

Non-regression: if the feedback says the CURRENT code already ran successfully \
and produced real output (a correctness/completeness gap, not a crash), do NOT \
make the code more likely to fail to run at all. Never introduce a new hard \
requirement it didn't have before — a credential, a local file path, or any \
other resource the code can't obtain itself — to chase a correctness \
improvement. Any additional check or refinement you add for correctness must \
be best-effort: wrap it so that if it can't be completed, the code logs a \
warning and still saves the output it already has, rather than raising and \
producing nothing. A handbook that reliably produces imperfect output is \
better than one revised into reliably producing none.

Non-generalization: this handbook is reused for many different future tasks \
against the same source, not just the one that triggered this refinement — do \
NOT bake this task's specific parameter values (coordinates, a radius, dates, \
place names, tag values) into the handbook text as if they were permanent \
facts about the source. If the feedback's evidence or error is tied to a \
specific task's parameters, fix the underlying gap generically — e.g. "verify \
results are within the requested radius using Haversine distance", never \
"radius<=2000m". If the CURRENT handbook already has a worked example anchored \
to a stale prior task (e.g. a line reading "for the task" with another task's \
numbers baked in), replace it with one clearly labeled illustrative (e.g. \
"Example query pattern (illustrative — substitute your own place/tags/radius/\
dates)") using placeholder-style values, rather than swapping in this task's \
numbers instead.

Return ONLY a single JSON object with the SAME handbook keys as before \
(data_source_name, brief_description, handbook, code_example, website, \
requires_key, key_name, caveats, key_signup_url) PLUS one extra key:
- "assistant_message": 1-3 plain-text sentences (NO code) explaining what you \
changed and why. This is shown to the user as a chat reply.

Keep the house style: handbook = one requirement per line; code_example = a \
runnable download_data() that calls raise_for_status() after each request and \
saves to a file; when the code loops over many pages/batches/items, each \
iteration retries once or twice on failure, then logs and skips that one item \
rather than aborting the whole download (no blanket try/except elsewhere); any \
step that determines whether the output actually satisfies the request \
(boundary/AOI clipping, joins against a reference dataset, dedup, date/category \
subsetting) prints the row/feature count before and after that step plus which \
reference file or dataset version was used, and the function prints the final \
row count and output path before returning; {PLACEHOLDER} tokens for any \
secret. Do not invent URLs or parameters — prefer what the documentation and \
the error actually show."""


def _build_refine_prompt(current, message, docs_text=""):
    draft = {k: str(current.get(k, "") or "") for k in _OUTPUT_FIELDS}
    prompt = (
        f"{_REFINE_INSTRUCTIONS}\n\n"
        f"CURRENT HANDBOOK (JSON):\n{json.dumps(draft, ensure_ascii=False, indent=2)}\n\n"
        f"USER FEEDBACK / ERROR TO ADDRESS:\n{message.strip()}"
    )
    if docs_text:
        prompt += (
            "\n\nRelevant documentation fetched from URLs in the message / "
            "website (ground your fix in this; do not invent URLs):\n\n"
            f"{docs_text}"
        )
    return prompt


def refine_handbook(current, message, history=None, website="",
                    fetch_docs=True, user_key=None, model=None,
                    stream_callback=None, reasoning_effort=None,
                    provider="openai"):
    """Refine a handbook draft through conversational feedback.

    Stateless: the Web UI holds the working draft and chat history, so each
    call gets the authoritative ``current`` fields plus the new ``message``
    (a change request or a pasted error). Returns ``{"source": <handbook
    fields>, "assistant_message": str}``; raises ValueError on blank input or
    an unparseable model reply.
    """
    if not message or not message.strip():
        raise ValueError("Enter a message describing what to change.")
    current = current if isinstance(current, dict) else {}

    _log("Refine: updating the handbook (%s)...", message.strip()[:160])
    docs_text = ""
    if fetch_docs:
        urls = [website] + _extract_urls(message)
        docs_text = _fetch_doc_text([u for u in urls if u and u.strip()])

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    # Replay recent turns for context; only plain-text replies, never the JSON.
    for turn in (history or [])[-10:]:
        role, content = turn.get("role"), str(turn.get("content", "") or "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user",
                     "content": _build_refine_prompt(current, message, docs_text)})

    default_model = CLAUDE_HANDBOOK_MODEL if provider == "claude" else HANDBOOK_GEN_MODEL
    reply = _call_model(messages, user_key=user_key,
                        model=model or default_model,
                        stream_callback=stream_callback,
                        reasoning_effort=reasoning_effort, provider=provider)
    try:
        data = _parse_json_object(reply)
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"The model did not return a valid handbook JSON object: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("The model reply was not a JSON object.")

    source = _normalize_source(
        data, fallback_name=current.get("data_source_name", ""),
        website=website or current.get("website", ""))
    assistant_message = (str(data.get("assistant_message", "") or "").strip()
                         or "Updated the handbook.")
    return {"source": source, "assistant_message": assistant_message}


# ── Evidence-grounded refinement, shared by both repair paths ──────────────
# Two callers refine a handbook against the evidence of a failed retrieval:
# Handbook Studio's refinement loop, and LLM_Find's mid-retrieval repair. They
# share nothing around the call — one keeps stages and persists a session, the
# other writes to a workflow stream — but the request they make of the model
# has to be the same one, or the two paths quietly become two different
# mechanisms with one name. What is shared is therefore exactly this: how the
# evidence is phrased, and what counts as a real change.

_REFINE_NON_REGRESSION_NOTE = (
    "IMPORTANT: the current handbook's code already ran successfully "
    "and produced real output — this is a correctness/completeness "
    "gap, not a crash. Do not introduce anything that could make the "
    "code fail to run (a new required credential, local file, or "
    "other resource it can't obtain itself); any new correctness "
    "check must be best-effort with a fallback to the existing "
    "behavior if it can't be completed.\n\n"
)


def build_refinement_feedback(task, analysis, error="", mechanism="",
                              prior_execution_ok=False):
    """The ``message`` to hand ``refine_handbook`` for an evidence-grounded
    repair after a retrieval attempt failed.

    ``prior_execution_ok`` says the code RAN and produced output, so the
    failure is a correctness or completeness gap rather than a crash. That
    distinction earns its own paragraph because the obvious repair for a
    wrong-looking result — add a validation step, require a reference file —
    can trade a script that delivers imperfect data for one that delivers
    none.

    ``mechanism`` pins the access mechanism when the caller has one to pin
    (Handbook Studio fixes it per session). The clause is omitted entirely
    when there is none, rather than emitted with an empty parenthetical:
    LLM_Find has no mechanism concept, and "do not change the access
    mechanism ()" is an instruction about nothing.
    """
    mechanism_clause = (
        f" Do not change the access mechanism ({mechanism})."
        if str(mechanism or "").strip() else "")
    return (
        "Refine this handbook using the controlled execution evidence "
        "below. Apply only evidence-grounded corrections and preserve "
        f"instructions that already work.{mechanism_clause}\n\n"
        f"{_REFINE_NON_REGRESSION_NOTE if prior_execution_ok else ''}"
        f"Retrieval task: {task}\n"
        f"Failure analysis: {json.dumps(analysis, ensure_ascii=False)}\n"
        f"Execution error: {error}"
    )


def refinement_changed(before, after):
    """True when a refinement actually altered the handbook's operative text.

    A revision that leaves both the prose and the worked example untouched is
    not worth retesting: the retry would run an identical handbook into an
    identical failure, spending a full attempt to learn nothing. Only those
    two fields are compared, because they are the two the retrieval prompt
    consumes — a reworded description changes no behavior.
    """
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    return not (after.get("handbook") == before.get("handbook")
                and after.get("code_example") == before.get("code_example"))


# ── Evaluation & trace analysis ──────────────────────────────────────────────
# Used by Experiment 2 after a controlled retrieval trial: evaluate_retrieval_
# result judges whether a *passing* trial actually satisfied the task (not
# just "did the code run"), and analyze_execution_trace diagnoses a *failing*
# trial to decide whether refining the handbook could plausibly fix it. Both
# are OpenAI/GIBD-routed through the shared _call_model helper (JSON mode) —
# no web search needed, since they only reason over evidence already in hand.

def _evaluation_status(value):
    value = str(value or "").strip().lower()
    return value if value in ("pass", "fail", "unknown", "not_applicable") else "unknown"


# Deterministic safety net: a model can be over-optimistic about "task
# completed" from a merely non-empty output file. When the task text implies
# a spatial filter, require either the model's own confirmation or actual
# spatial evidence in the output before trusting "completed".
_SPATIAL_HINT_RE = re.compile(
    r"\b(county|counties|state|country|region|city|province|watershed|"
    r"basin|boundary|within|bounding\s*box|bbox|polygon|extent|"
    r"area of interest|aoi)\b", re.I)


def _has_spatial_evidence(execution_result):
    for item in (execution_result or {}).get("output_evidence") or []:
        if item.get("bbox") or item.get("feature_count") or item.get("layers"):
            return True
    return False


# Best-effort real-world grounding for spatial_query_correctness: an LLM
# reasoning over raw coordinate floats can't reliably tell whether a bbox
# actually covers "Los Angeles County" — a real reference extent helps. This
# runs in OUR OWN validation code (never inside generated retrieval code, so
# it can't make a retrieval run fail) and degrades to plain LLM judgment on
# any failure (no network, no match, bad response) — never raises.
_PLACE_AFTER_RE = re.compile(
    r"\b(?:within|inside|in|near|across|over|for)\s+"
    r"((?:[A-Z][\w.'-]*\s*){1,6}"
    r"(?:County|Counties|City|State|Province|Region|Country|Park|Forest|"
    r"Basin|Watershed)?)")


def _extract_place_candidate(task):
    """Best-effort heuristic: pull a capitalized place-like phrase following
    a spatial preposition out of a task description. Returns "" when
    nothing plausible is found -- callers must treat that as 'skip'."""
    match = _PLACE_AFTER_RE.search(str(task or ""))
    if not match:
        return ""
    return match.group(1).strip(" ,.;")


def _place_bbox(place_name):
    """Resolve a place name to its real bounding box via Nominatim, for
    grounding spatial-correctness judgments. Returns None on any failure
    (no network call here is allowed to break evaluation)."""
    if not place_name:
        return None
    try:
        import requests
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place_name, "format": "json", "limit": 1},
            headers={"User-Agent": "AGSI-HandbookGenerator/1.0"},
            timeout=10)
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        south, north, west, east = (float(v) for v in results[0]["boundingbox"])
        return {"west": west, "south": south, "east": east, "north": north}
    except Exception:
        return None


def evaluate_retrieval_result(current, task, execution_result, mechanism="",
                              user_key=None, model=None, stream_callback=None,
                              reasoning_effort=None, provider="openai",
                              **_legacy):
    """Judge whether a completed retrieval attempt actually satisfies the
    task — correct mechanism, parameters, spatial/temporal filtering, and
    output — grounded only in the execution evidence provided.

    Returns {"task_completed": bool, "confidence": float, "summary": str,
    "mechanism_correctness"/"parameter_correctness"/"spatial_query_correctness"/
    "temporal_query_correctness"/"output_correctness": "pass|fail|unknown|
    not_applicable", "checks": [{"name","status","evidence"}],
    "failure_category": str, "should_refine_handbook": bool,
    "handbook_gap": str}.
    """
    fields = {k: str(current.get(k, "") or "") for k in _OUTPUT_FIELDS}
    evidence = {
        "status": execution_result.get("status"),
        "validation": execution_result.get("validation"),
        "output_evidence": execution_result.get("output_evidence"),
        "http_requests": (execution_result.get("http_requests") or [])[-5:],
        "error": str(execution_result.get("error") or "")[:2000],
    }
    # Best-effort real-world grounding (see _place_bbox): an LLM reasoning
    # over raw coordinate floats can't reliably tell whether a bbox actually
    # covers a named place. Silently absent when extraction/geocoding fails
    # -- the judgment just falls back to unaided LLM reasoning, same as
    # before this was added.
    if _SPATIAL_HINT_RE.search(str(task or "")):
        place_candidate = _extract_place_candidate(task)
        place_bbox = _place_bbox(place_candidate)
        if place_bbox:
            evidence["reference_place_bbox"] = {
                "place": place_candidate,
                "source": "OpenStreetMap Nominatim (server-computed, not from the generated code)",
                **place_bbox,
            }
    prompt = (
        "You are auditing a completed data-retrieval attempt against the "
        "task it was supposed to satisfy. The code ran and produced a file, "
        "but that alone does not prove the task is done — check whether the "
        "correct source, mechanism, parameters, spatial filter, and "
        "temporal filter were actually used, using only the evidence below. "
        "Never assume a filter was applied just because the request "
        "succeeded.\n\n"
        "Spatial precision: when the task names an administrative or named "
        "area (e.g. 'within X County/City/Region') and the handbook's access "
        "mechanism only supports a bounding box or similarly coarse spatial "
        "filter (not exact polygon membership), a bounding box that "
        "reasonably covers the named area SATISFIES the spatial requirement "
        "— do not fail spatial_query_correctness or task_completed merely "
        "because the output wasn't clipped to an exact administrative "
        "polygon; that precision is not something the source's own query "
        "mechanism offers. Only fail it if the bbox is clearly wrong (wrong "
        "place, implausibly large/loose, or missing) or the task explicitly "
        "demands boundary-exact exclusion (e.g. 'excluding points outside "
        "the county boundary', 'strictly within the polygon, not just a "
        "bounding box'). If the evidence includes reference_place_bbox (a "
        "server-computed real-world extent for the named place, not from "
        "the generated code), use it to judge whether the bbox actually "
        "used is a reasonable match — don't estimate the place's true "
        "extent from general knowledge when this reference is provided.\n\n"
        "Reply with ONLY a JSON object:\n"
        '{"task_completed": bool, "confidence": number 0-1, "summary": str, '
        '"mechanism_correctness": "pass|fail|unknown|not_applicable", '
        '"parameter_correctness": "pass|fail|unknown|not_applicable", '
        '"spatial_query_correctness": "pass|fail|unknown|not_applicable", '
        '"temporal_query_correctness": "pass|fail|unknown|not_applicable", '
        '"output_correctness": "pass|fail|unknown|not_applicable", '
        '"checks": [{"name": str, "status": "pass|fail|unknown", "evidence": str}], '
        '"failure_category": "handbook_deficiency" | "external" | "", '
        '"should_refine_handbook": bool, "handbook_gap": str}\n\n'
        "failure_category is only meaningful when task_completed is false: "
        "\"external\" means the provider/environment is the blocker (an "
        "outage, disabled/deprecated feature, invalid credentials) and no "
        "handbook change would help — use exactly that literal string, no "
        "qualifiers; otherwise use \"handbook_deficiency\" if refinable, or "
        "\"\" if not applicable.\n\n"
        f"REQUIRED ACCESS MECHANISM:\n{str(mechanism or '').strip()}\n\n"
        f"TASK:\n{task}\n\n"
        f"HANDBOOK:\n{json.dumps(fields, ensure_ascii=False)}\n\n"
        f"EXECUTION EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False, default=str)}")
    default_model = CLAUDE_HANDBOOK_MODEL if provider == "claude" else HANDBOOK_GEN_MODEL
    reply = _call_model([{"role": "user", "content": prompt}],
                       user_key=user_key, model=model or default_model,
                       stream_callback=stream_callback,
                       reasoning_effort=reasoning_effort, provider=provider,
                       phase="evaluation")
    try:
        data = _parse_json_object(reply)
    except (ValueError, json.JSONDecodeError):
        data = {}

    result = {
        "task_completed": bool(data.get("task_completed")),
        "confidence": max(0.0, min(1.0, float(data.get("confidence") or 0) or 0)),
        "summary": str(data.get("summary") or "").strip(),
        "mechanism_correctness": _evaluation_status(data.get("mechanism_correctness")),
        "parameter_correctness": _evaluation_status(data.get("parameter_correctness")),
        "spatial_query_correctness": _evaluation_status(data.get("spatial_query_correctness")),
        "temporal_query_correctness": _evaluation_status(data.get("temporal_query_correctness")),
        "output_correctness": _evaluation_status(data.get("output_correctness")),
        "checks": [
            {
                "name": str(item.get("name") or "").strip(),
                "status": str(item.get("status") or "unknown").strip().lower(),
                "evidence": str(item.get("evidence") or "").strip(),
            }
            for item in (data.get("checks") or []) if isinstance(item, dict)
        ],
        "failure_category": str(data.get("failure_category") or "").strip(),
        "should_refine_handbook": bool(data.get("should_refine_handbook")),
        "handbook_gap": str(data.get("handbook_gap") or "").strip(),
    }

    if (_SPATIAL_HINT_RE.search(str(task or ""))
            and result["spatial_query_correctness"] != "pass"
            and not _has_spatial_evidence(execution_result)):
        result["task_completed"] = False
        result["spatial_query_correctness"] = "fail"
        result["should_refine_handbook"] = True
        result["failure_category"] = result["failure_category"] or "Spatial validation"
        result["summary"] = result["summary"] or (
            "The task requires a spatial filter, but no spatial evidence "
            "(bounding box, feature count, or layer list) was recorded in "
            "the output.")
    return result


def _coerce_bbox(value):
    """Accept either the canonical {"west","south","east","north"} dict or
    the raw GeoJSON/STAC flat-list convention [west, south, east, north]
    (also tolerates a 6-element 3D bbox). Persisted output_evidence from
    before this normalization existed in experiment2_runner._inspect_output_file
    may still contain the raw list shape, so this stays defensive here too
    rather than assuming every caller's evidence was captured post-fix."""
    if isinstance(value, dict) and {"west", "south", "east", "north"} <= value.keys():
        try:
            return {k: float(value[k]) for k in ("west", "south", "east", "north")}
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) in (4, 6):
        try:
            nums = [float(v) for v in value]
            west, south, east, north = (
                (nums[0], nums[1], nums[2], nums[3]) if len(nums) == 4
                else (nums[0], nums[1], nums[3], nums[4]))
            return {"west": west, "south": south, "east": east, "north": north}
        except (TypeError, ValueError):
            return None
    return None


def _bbox_overlaps(a, b):
    """True if two {west,south,east,north} boxes intersect at all."""
    try:
        return (a["west"] <= b["east"] and b["west"] <= a["east"]
                and a["south"] <= b["north"] and b["south"] <= a["north"])
    except (KeyError, TypeError):
        return False


def evaluate_retrieval_result_metadata(current, task, execution_result, mechanism=""):
    """Deterministic, non-LLM alternative to evaluate_retrieval_result: judges
    spatial correctness by comparing the downloaded artifact's OWN computed
    bounding box (from _inspect_output_file's structured file inspection --
    rasterio/geopandas-derived, not self-reported by the generated code)
    against a reference bbox for the task's named place (via _place_bbox,
    the same Nominatim lookup evaluate_retrieval_result uses). No API key,
    no LLM call, no dependency on the generated code having printed anything.

    Same return shape as evaluate_retrieval_result so callers/rendering code
    need no changes. Deliberately narrow in scope: it can confirm or refute
    the file's real spatial extent, but it cannot judge task-specific
    selection logic (e.g. whether this was the best/lowest-cloud match among
    candidates) or mechanism/parameter correctness -- those stay
    "not_applicable" here and are left to the LLM-judge or manual methods.
    """
    evidence_items = [
        item for item in (execution_result.get("output_evidence") or [])
        if isinstance(item, dict)
    ]
    output_present = bool(evidence_items) or bool(execution_result.get("downloaded_files"))

    checks = []
    spatial_status = "not_applicable"
    place_candidate = ""

    if _SPATIAL_HINT_RE.search(str(task or "")):
        place_candidate = _extract_place_candidate(task)
        reference_bbox = _place_bbox(place_candidate) if place_candidate else None
        if reference_bbox is None:
            spatial_status = "unknown"
            checks.append({
                "name": "Spatial extent vs. requested place",
                "status": "unknown",
                "evidence": (
                    f"Could not resolve a reference bounding box for "
                    f"'{place_candidate}'." if place_candidate else
                    "Could not identify a place name in the task text to "
                    "check against."),
            })
        else:
            file_bboxes = [
                (item.get("name"), _coerce_bbox(item.get("bbox")))
                for item in evidence_items if item.get("bbox")
            ]
            file_bboxes = [(name, bbox) for name, bbox in file_bboxes if bbox]
            if not file_bboxes:
                spatial_status = "unknown"
                checks.append({
                    "name": "Spatial extent vs. requested place",
                    "status": "unknown",
                    "evidence": (
                        "A reference bounding box was resolved for "
                        f"'{place_candidate}', but no output file exposed a "
                        "computed spatial extent to compare it against."),
                })
            else:
                any_overlap = False
                for name, file_bbox in file_bboxes:
                    overlap = _bbox_overlaps(reference_bbox, file_bbox)
                    any_overlap = any_overlap or overlap
                    checks.append({
                        "name": f"Spatial extent of {name} vs. requested place",
                        "status": "pass" if overlap else "fail",
                        "evidence": (
                            f"File extent {file_bbox} vs. reference extent "
                            f"for '{place_candidate}' {reference_bbox} -- "
                            f"{'overlaps' if overlap else 'does NOT overlap'}."
                        ),
                    })
                spatial_status = "pass" if any_overlap else "fail"

    task_completed = output_present and spatial_status in ("pass", "not_applicable")
    summary_parts = [
        "Automatic check: compared the downloaded file's own computed "
        "spatial extent against the requested place's real-world extent. "
        "This does not verify task-specific selection logic (e.g. whether "
        "this was the best/lowest-cloud match among candidates) or "
        "mechanism/parameter correctness -- use the LLM judge or manual "
        "review for that."
    ]
    if spatial_status == "fail":
        summary_parts.insert(0, "The downloaded file's real spatial extent "
                                 "does not overlap the requested place.")
    elif spatial_status == "unknown":
        summary_parts.insert(0, "Spatial extent could not be automatically "
                                 "verified.")
    elif spatial_status == "pass":
        summary_parts.insert(0, "The downloaded file's real spatial extent "
                                 "overlaps the requested place.")

    failure_category = ""
    should_refine = False
    handbook_gap = ""
    if not task_completed and spatial_status == "fail":
        failure_category = "handbook_deficiency"
        should_refine = True
        handbook_gap = (
            "The retrieved data's actual spatial extent (computed from the "
            "output file itself) does not cover the requested place -- "
            "check the spatial filter/parameters the handbook instructs "
            "the code to use.")
    elif not output_present:
        failure_category = "handbook_deficiency"
        handbook_gap = "No output evidence was available to validate."

    return {
        "task_completed": bool(task_completed),
        "confidence": 1.0 if spatial_status in ("pass", "fail") else 0.0,
        "summary": " ".join(summary_parts),
        "mechanism_correctness": "not_applicable",
        "parameter_correctness": "not_applicable",
        "spatial_query_correctness": spatial_status,
        "temporal_query_correctness": "not_applicable",
        "output_correctness": "pass" if output_present else "fail",
        "checks": checks,
        "failure_category": failure_category,
        "should_refine_handbook": should_refine,
        "handbook_gap": handbook_gap,
    }


def analyze_execution_trace(current, task, execution_result, user_key=None,
                            model=None, stream_callback=None,
                            reasoning_effort=None, provider="openai",
                            control=False, **_legacy):
    """Diagnose a failing retrieval attempt: is there a fixable handbook
    deficiency (wrong endpoint, missing/incorrect parameter, wrong auth,
    unhandled pagination or rate limit, ...), was the failure external
    (network/service outage, invalid credentials, platform issue), or is
    the request itself infeasible for this source (the period, area, or
    variable asked for lies outside what it holds)? Refining the handbook
    helps only in the first case; the retrieval ladder stops on the other
    two, and ``request_infeasible`` is surfaced to the user as "change the
    request or the source" rather than "the download failed".

    ``control=True`` asks the question that fits the ablated arm instead. That
    run had no handbook, so "does the handbook have a deficiency?" is a
    category error -- and asked against the empty handbook a control session
    stores, the answer came back "handbook_deficiency" essentially by
    construction, since every field genuinely is missing. The control is asked
    whether the failure was external or a KNOWLEDGE GAP: the model's own
    understanding of this source being wrong or incomplete. It is also shown
    only the source NAME, never handbook text -- the arm never saw one, and
    judging it against one would describe a run that never happened.

    Returns {"failure_category": str, "summary": str, "condition": str,
    "deficiencies": [{"handbook_gap", "evidence", "recommended_change"}]}.
    The wire shape is identical in both arms so the UI and the CSV export keep
    working; ``condition`` says how to read it. An empty ``deficiencies`` list
    means nothing fixable was identified.
    """
    fields = {k: str(current.get(k, "") or "") for k in _OUTPUT_FIELDS}
    evidence = {
        "status": execution_result.get("status"),
        "error": str(execution_result.get("error") or "")[:4000],
        "validation": execution_result.get("validation"),
        "http_requests": (execution_result.get("http_requests") or [])[-8:],
        "generated_code": str(execution_result.get("generated_code") or "")[:6000],
    }
    control_prompt = (
        "A data-retrieval attempt did not pass. This attempt ran in the "
        "CONTROL condition of an ablation study: the model was told which "
        "data source to use and was given the task, but NO handbook, "
        "documentation excerpt, or worked example — it had to work from its "
        "own knowledge of this source. There is therefore no handbook to "
        "diagnose and no handbook to edit; do not propose handbook changes "
        "or refer to a handbook at all.\n\n"
        "Decide whether the failure was EXTERNAL (network/service outage, "
        "invalid or missing credential, a platform/environment issue, OR the "
        "response itself indicates the provider has disabled, deprecated, or "
        "marked the endpoint/feature unavailable — look for phrasing in the "
        "response body or error like 'not available', 'deprecated', 'under "
        "maintenance', not just the HTTP status code), or a KNOWLEDGE GAP: "
        "the model's own understanding of this source was wrong or "
        "incomplete (wrong base URL or endpoint path, wrong or missing "
        "parameter, wrong authentication scheme, wrong assumption about the "
        "response format, missed pagination or rate limiting), or "
        "REQUEST_INFEASIBLE: the source simply has no data for what the "
        "task asked (dates outside the dataset's period, an area with no "
        "records, a variable/product it does not offer), which no "
        "knowledge and no code could have changed. Ground every "
        "claim in the evidence provided; never invent endpoints or "
        "parameters.\n\n"
        "Reply with ONLY a JSON object:\n"
        '{"failure_category": "knowledge_gap" | "external" | '
        '"request_infeasible", '
        '"summary": str, "deficiencies": [{"handbook_gap": str, '
        '"evidence": str, "recommended_change": str}]}\n'
        "failure_category MUST be exactly one of those three literal strings. "
        "In each deficiency, \"handbook_gap\" is what the model got WRONG "
        "about this source (the key name is fixed by the record format), "
        "\"evidence\" is the specific trace line showing it, and "
        "\"recommended_change\" is the correct fact that would have avoided "
        "it. An empty deficiencies list means no knowledge gap was "
        "identified; it should be empty whenever failure_category is "
        "\"external\", since no knowledge would have fixed an external "
        "condition.\n\n"
        f"TASK:\n{task}\n\nDATA SOURCE (all the attempt was told):\n"
        f"{fields.get('data_source_name', '')}"
        f"\n\nEXECUTION TRACE:\n{json.dumps(evidence, ensure_ascii=False, default=str)}")

    prompt = (
        "A data-retrieval attempt did not pass. Analyze the trace below and "
        "decide whether the handbook itself has a fixable deficiency (wrong "
        "endpoint, missing or incorrect parameter, wrong authentication, "
        "wrong output format, unhandled pagination or rate limit, ...) or "
        "the failure is external and refining the handbook would not help "
        "(network/service outage, invalid credentials, a platform/"
        "environment issue, OR the response itself indicates the provider "
        "has disabled, deprecated, or marked the endpoint/feature "
        "unavailable right now — look for phrasing in the response body or "
        "error like 'not available', 'feature currently unavailable', "
        "'deprecated', 'under maintenance', 'coming soon', not just the "
        "HTTP status code, since a generic 400/404 alone can be either "
        "category). There is a THIRD possibility: the handbook and the "
        "code are both right, but the REQUEST ITSELF cannot be served by "
        "this source — the task asks for dates outside the dataset's "
        "period, an area/station/region the source has no records for, a "
        "variable, product, or release the source does not offer (yet). "
        "Signs: the provider says 'no data available', 'out of range', "
        "'outside the available date range', 'not available for this "
        "period/area', a 400/404 whose body names the period/area/product "
        "as not covered, or the handbook's documented coverage plainly "
        "excludes what the task asks. That is \"request_infeasible\": no "
        "handbook change and no code change can fix it; only changing the "
        "request or the source can. Do not label it handbook_deficiency "
        "just because a handbook COULD document the coverage limit. Ground "
        "every claim in the evidence provided; never invent endpoints or "
        "parameters.\n\n"
        "Reply with ONLY a JSON object:\n"
        '{"failure_category": "handbook_deficiency" | "external" | '
        '"request_infeasible", '
        '"summary": str, "deficiencies": [{"handbook_gap": str, '
        '"evidence": str, "recommended_change": str}]}\n'
        "failure_category MUST be exactly one of those three literal "
        "strings — no qualifiers or slashes (write \"external\", never "
        "\"external/platform_issue\" or similar). For "
        "\"request_infeasible\", the summary must say WHICH requested "
        "parameter (period, area, variable, product) the source cannot "
        "serve and what the provider/documentation says its coverage is, "
        "so the user can adjust the request. An empty deficiencies "
        "list means no refinable handbook gap was found; deficiencies "
        "should be empty whenever failure_category is \"external\" or "
        "\"request_infeasible\", since no handbook change can fix either "
        "condition.\n\n"
        f"TASK:\n{task}\n\nHANDBOOK:\n{json.dumps(fields, ensure_ascii=False)}"
        f"\n\nEXECUTION TRACE:\n{json.dumps(evidence, ensure_ascii=False, default=str)}")
    default_model = CLAUDE_HANDBOOK_MODEL if provider == "claude" else HANDBOOK_GEN_MODEL
    reply = _call_model([{"role": "user",
                          "content": control_prompt if control else prompt}],
                       user_key=user_key, model=model or default_model,
                       stream_callback=stream_callback,
                       reasoning_effort=reasoning_effort, provider=provider,
                       phase="evaluation")
    try:
        data = _parse_json_object(reply)
    except (ValueError, json.JSONDecodeError):
        data = {}
    # Keep each arm inside its own vocabulary: a control run must never be
    # recorded as a "handbook_deficiency", which is the exact mislabel this
    # branch exists to stop.
    allowed = ({"knowledge_gap", "external", "request_infeasible"} if control
               else {"handbook_deficiency", "external", "request_infeasible"})
    category = str(data.get("failure_category") or "").strip()
    return {
        "failure_category": category if category in allowed else "Unknown",
        "summary": str(data.get("summary") or "").strip(),
        # How to read `handbook_gap` below: in the control arm it is what the
        # model got wrong about the source, not a gap in any handbook.
        "condition": "control" if control else "with_handbook",
        "deficiencies": [
            {
                "handbook_gap": str(item.get("handbook_gap") or "").strip(),
                "evidence": str(item.get("evidence") or "").strip(),
                "recommended_change": str(item.get("recommended_change") or "").strip(),
            }
            for item in (data.get("deficiencies") or []) if isinstance(item, dict)
        ],
    }
