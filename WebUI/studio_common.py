"""Helpers shared by the Handbook Studio runner.

Stage bookkeeping, log capture around generator work, and result
sanitizing (secret redaction, output-file evidence).
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from agents.data_agent import user_data_sources


_SAFE_PART = re.compile(r"[^A-Za-z0-9_-]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: str) -> str:
    return _SAFE_PART.sub("", str(value or ""))[:100] or "unknown"


def _redact_text(value, secrets: list[str]):
    text = str(value or "")
    for secret in secrets:
        secret = str(secret or "")
        if not secret:
            continue
        if len(secret) >= 8:
            text = text.replace(secret, "[REDACTED]")
        elif len(secret) >= 4:
            # Short test values such as "a" or "s" previously destroyed every
            # traceback and URL containing those letters. Redact short values
            # only when they appear as a complete token.
            text = re.sub(
                rf"(?<![A-Za-z0-9]){re.escape(secret)}(?![A-Za-z0-9])",
                "[REDACTED]", text)
    return text


def _safe_url(url: str, secrets: list[str]) -> str:
    try:
        parts = urlsplit(_redact_text(url, secrets))
        query = []
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            if any(token in key.casefold() for token in (
                    "key", "token", "secret", "password", "authorization")):
                value = "[REDACTED]"
            query.append((key, value))
        return urlunsplit((
            parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
    except Exception:
        return _redact_text(url, secrets)


def _expected_extensions(expected_format: str) -> set[str]:
    value = str(expected_format or "").casefold()
    mapping = {
        "geojson": {".geojson", ".json"},
        "geopackage": {".gpkg"},
        "gpkg": {".gpkg"},
        "shapefile": {".shp"},
        "csv": {".csv"},
        "parquet": {".parquet", ".geoparquet"},
        "raster": {".tif", ".tiff", ".nc"},
        "geotiff": {".tif", ".tiff"},
        "netcdf": {".nc", ".netcdf"},
        "json": {".json", ".geojson"},
        "osm xml": {".osm", ".xml"},
        "zip": {".zip"},
    }
    result = set()
    for label, extensions in mapping.items():
        if label in value:
            result.update(extensions)
    return result


def _normalize_bbox(value):
    """Normalize a raw JSON "bbox" value into the canonical
    {"west","south","east","north"} shape every other bbox producer in this
    file (rasterio/geopandas) already emits, so downstream comparisons never
    have to handle two shapes. Accepts the already-canonical dict, or the
    common GeoJSON/STAC flat-list convention [west, south, east, north] (a
    6-element 3D bbox [west,south,elev_min,east,north,elev_max] is also
    accepted, ignoring elevation). Returns None if unparseable."""
    if isinstance(value, dict) and {"west", "south", "east", "north"} <= value.keys():
        try:
            return {k: float(value[k]) for k in ("west", "south", "east", "north")}
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        try:
            if len(value) == 4:
                west, south, east, north = (float(v) for v in value)
            elif len(value) == 6:
                west, south, _, east, north, _ = (float(v) for v in value)
            else:
                return None
            return {"west": west, "south": south, "east": east, "north": north}
        except (TypeError, ValueError):
            return None
    return None


def _inspect_output_file(path: str, display_name: str, size_bytes: int) -> dict:
    """Collect bounded, non-secret evidence for semantic task validation."""
    extension = os.path.splitext(display_name)[1].casefold()
    evidence = {
        "name": display_name,
        "size_bytes": size_bytes,
        "extension": extension,
    }
    try:
        if extension in (".json", ".geojson") and size_bytes <= 10_000_000:
            with open(path, encoding="utf-8", errors="replace") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                evidence["top_level_keys"] = list(data.keys())[:30]
                features = data.get("features")
                if isinstance(features, list):
                    evidence["feature_count"] = len(features)
                    if features and isinstance(features[0], dict):
                        props = features[0].get("properties")
                        if isinstance(props, dict):
                            evidence["sample_property_keys"] = list(
                                props.keys())[:30]
                if "bbox" in data:
                    normalized = _normalize_bbox(data.get("bbox"))
                    if normalized:
                        evidence["bbox"] = normalized
                metadata = data.get("metadata")
                if isinstance(metadata, dict):
                    evidence["metadata"] = {
                        key: metadata.get(key)
                        for key in list(metadata.keys())[:12]
                        if isinstance(metadata.get(key), (
                            str, int, float, bool, type(None)))
                    }
            if "bbox" not in evidence and size_bytes <= 20_000_000:
                # Fallback for the common case where the file has no
                # top-level "bbox" key: compute a real extent from the
                # actual geometries rather than leaving spatial extent
                # unverifiable.
                try:
                    import geopandas as gpd
                    gdf = gpd.read_file(path)
                    if len(gdf):
                        if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
                            gdf = gdf.to_crs(epsg=4326)
                        west, south, east, north = gdf.total_bounds
                        evidence["bbox"] = {
                            "west": float(west), "south": float(south),
                            "east": float(east), "north": float(north),
                        }
                        if gdf.crs is not None:
                            evidence["crs"] = str(gdf.crs)
                except Exception:
                    pass
        elif extension in (".csv", ".tsv") and size_bytes <= 20_000_000:
            import csv
            delimiter = "\t" if extension == ".tsv" else ","
            with open(path, encoding="utf-8", errors="replace",
                      newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                header = next(reader, [])
                lat_idx = next((i for i, col in enumerate(header)
                                if re.fullmatch(r"lat(itude)?", col.strip(), re.I)), None)
                lon_idx = next((i for i, col in enumerate(header)
                                if re.fullmatch(r"lon(gitude)?|lng", col.strip(), re.I)), None)
                row_count = 0
                lat_min = lat_max = lon_min = lon_max = None
                for row in reader:
                    row_count += 1
                    if (lat_idx is not None and lon_idx is not None
                            and lat_idx < len(row) and lon_idx < len(row)):
                        try:
                            lat_val = float(row[lat_idx])
                            lon_val = float(row[lon_idx])
                        except ValueError:
                            continue
                        lat_min = lat_val if lat_min is None else min(lat_min, lat_val)
                        lat_max = lat_val if lat_max is None else max(lat_max, lat_val)
                        lon_min = lon_val if lon_min is None else min(lon_min, lon_val)
                        lon_max = lon_val if lon_max is None else max(lon_max, lon_val)
            evidence["columns"] = header[:50]
            evidence["row_count"] = row_count
            if lat_min is not None and lon_min is not None:
                evidence["bbox"] = {
                    "west": lon_min, "south": lat_min,
                    "east": lon_max, "north": lat_max,
                }
        elif extension in (".tif", ".tiff") and size_bytes <= 500_000_000:
            import rasterio
            from rasterio.warp import transform_bounds
            with rasterio.open(path) as dataset:
                evidence["crs"] = str(dataset.crs) if dataset.crs else None
                evidence["bands"] = dataset.count
                evidence["dtype"] = (
                    str(dataset.dtypes[0]) if dataset.dtypes else None)
                if dataset.crs and dataset.bounds:
                    try:
                        west, south, east, north = transform_bounds(
                            dataset.crs, "EPSG:4326", *dataset.bounds)
                        evidence["bbox"] = {
                            "west": west, "south": south,
                            "east": east, "north": north,
                        }
                    except Exception:
                        pass
        elif extension == ".shp":
            import geopandas as gpd
            gdf = gpd.read_file(path)
            evidence["feature_count"] = len(gdf)
            if len(gdf):
                if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
                    gdf = gdf.to_crs(epsg=4326)
                west, south, east, north = gdf.total_bounds
                evidence["bbox"] = {
                    "west": float(west), "south": float(south),
                    "east": float(east), "north": float(north),
                }
                evidence["crs"] = str(gdf.crs) if gdf.crs is not None else None
        elif extension in (".xml", ".osm"):
            import xml.etree.ElementTree as element_tree
            if size_bytes <= 15_000_000:
                root = element_tree.parse(path).getroot()
                evidence["root_tag"] = root.tag
                if extension == ".osm":
                    evidence["osm_counts"] = {
                        kind: len(root.findall(kind))
                        for kind in ("node", "way", "relation")
                    }
            else:
                with open(path, "rb") as handle:
                    head = handle.read(4096).decode(
                        "utf-8", errors="replace")
                evidence["root_tag"] = (
                    "osm" if "<osm" in head else "xml")
        elif extension == ".gpkg":
            import sqlite3
            with sqlite3.connect(path) as connection:
                rows = connection.execute(
                    "SELECT table_name FROM gpkg_contents LIMIT 30"
                ).fetchall()
            evidence["layers"] = [row[0] for row in rows]
            try:
                import geopandas as gpd
                gdf = gpd.read_file(path)
                if len(gdf):
                    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
                        gdf = gdf.to_crs(epsg=4326)
                    west, south, east, north = gdf.total_bounds
                    evidence["bbox"] = {
                        "west": float(west), "south": float(south),
                        "east": float(east), "north": float(north),
                    }
                    if gdf.crs is not None:
                        evidence["crs"] = str(gdf.crs)
            except Exception:
                pass
    except Exception as exc:
        evidence["inspection_note"] = (
            f"Structured inspection unavailable: {type(exc).__name__}")
    return evidence


def _sanitize_execution(execution: dict, secrets: list) -> dict:
    """Redact secrets out of the per-attempt code/error/fix_explanation text
    (execute_complete_program's "attempts" list) the same way generated_code
    and error already are — these carry the same kind of content (full
    generated code across every attempt, including any substituted
    credential values) and were passed through unredacted otherwise."""
    execution = dict(execution or {})
    attempts = []
    for item in execution.get("attempts") or []:
        if not isinstance(item, dict):
            continue
        attempts.append({
            "attempt": item.get("attempt"),
            "success": bool(item.get("success")),
            "timed_out": bool(item.get("timed_out")),
            "code": _redact_text(item.get("code"), secrets)[:60000],
            "error": _redact_text(item.get("error"), secrets)[:4000],
            "fix_explanation": _redact_text(
                item.get("fix_explanation"), secrets)[:4000],
        })
    execution["attempts"] = attempts
    return execution


def _sanitize_result(result: dict, user_keys: dict, expected_format: str,
                     artifact_root: str = "") -> dict:
    secrets = [str(value) for value in (user_keys or {}).values() if value]
    files = []
    allowed_root = os.path.abspath(artifact_root) if artifact_root else ""
    for item in result.get("downloaded_files") or []:
        file_record = {
            "name": os.path.basename(str(item.get("name") or "")),
            "size_bytes": max(0, int(item.get("size_bytes") or 0)),
        }
        candidate = os.path.abspath(str(item.get("path") or ""))
        if allowed_root and candidate:
            try:
                if (os.path.commonpath([allowed_root, candidate])
                        == allowed_root and os.path.isfile(candidate)):
                    file_record["artifact_ref"] = os.path.relpath(
                        candidate, allowed_root).replace(os.sep, "/")
            except (OSError, ValueError):
                pass
        if os.path.isfile(candidate):
            file_record["evidence"] = _inspect_output_file(
                candidate, file_record["name"], file_record["size_bytes"])
        files.append(file_record)
    requests = [{
        "attempt": item.get("attempt"),
        "method": str(item.get("method") or "GET").upper(),
        "url": _safe_url(str(item.get("url") or ""), secrets),
        "time": item.get("time"),
    } for item in (result.get("http_requests") or [])]

    expected = _expected_extensions(expected_format)
    output_ok = any(item["size_bytes"] > 0 for item in files)
    format_ok = not expected or any(
        os.path.splitext(item["name"])[1].casefold() in expected for item in files)
    status = result.get("status") or "failed"
    validation_message = "A non-empty output file was produced."
    if not output_ok:
        validation_message = "No non-empty output file was produced."
    elif not format_ok:
        status = "failed"
        validation_message = (
            f"Output did not match the expected format ({expected_format}).")

    return {
        "status": status,
        "error": _redact_text(result.get("error"), secrets)[:12000],
        "required_keys": list(result.get("required_keys") or []),
        "generated_code": _redact_text(
            result.get("generated_code"), secrets)[:60000],
        "execution": _sanitize_execution(result.get("execution"), secrets),
        "http_requests": requests,
        "downloaded_files": files,
        "output_evidence": [
            item["evidence"] for item in files if item.get("evidence")
        ],
        "validation": {
            "passed": status == "passed" and output_ok and format_ok,
            "output_present": output_ok,
            "format_match": format_ok,
            "expected_format": str(expected_format or ""),
            "message": validation_message,
        },
    }


_GEN_STAGE_RE = re.compile(
    r"^(?:\[\d{2}:\d{2}:\d{2}\]\s*)?\[stage:(\w+):(\w+)\]\s*(.*)$")
_GEN_CHAT_RE = re.compile(
    r"^(?:\[\d{2}:\d{2}:\d{2}\]\s*)?\[chat\]\s*(.+)$")
_GEN_ARTIFACT_RE = re.compile(
    r"^(?:\[\d{2}:\d{2}:\d{2}\]\s*)?\[artifact:(\w+)\]\s*(\{.*\})\s*$")


def _pipeline_state(study: dict) -> dict:
    state = study.get("pipeline_state")
    if not isinstance(state, dict):
        state = {}
        study["pipeline_state"] = state
    state.setdefault("status", "not_started")
    state.setdefault("stages", [])
    state.setdefault("current_version", "")
    state.setdefault("required_keys", [])
    return state


def _record_stage(study: dict, key: str, label: str, description: str,
                  status: str, message: str = "", detail: str = "",
                  artifact=None) -> dict:
    state = _pipeline_state(study)
    stage = next(
        (item for item in state["stages"] if item.get("key") == key), None)
    if stage is None:
        stage = {
            "key": key,
            "label": label,
            "description": description,
            "status": "waiting",
            "message": "",
            "details": [],
            "created_at": _now(),
        }
        state["stages"].append(stage)
    stage["status"] = status
    if message:
        stage["message"] = str(message)[:4000]
    if detail:
        details = stage.setdefault("details", [])
        text = str(detail).strip()
        if text and (not details or details[-1] != text):
            details.append(text[:3000])
            del details[:-30]
    if artifact is not None:
        stage["artifact"] = artifact
    if status == "running" and not stage.get("started_at"):
        stage["started_at"] = _now()
    if status in ("complete", "warning", "error", "stopped"):
        stage["finished_at"] = _now()
    return dict(stage)


def _stage_event(stage: dict) -> dict:
    return {"type": "stage", "stage": stage}



def _run_logged_work(work):
    """Yield captured handbook-generator logs and return the worker result."""
    from agents.data_agent.handbook_generator import (
        start_log_capture, stop_log_capture)
    from agents.data_agent import usage_ledger

    events = queue.Queue()
    box = {}
    # Usage ledgers are thread-local -- two sessions generating at once must
    # never pool their tokens -- but work() runs on the thread started below,
    # so a ledger the caller opened is invisible to every LLM call the
    # generator makes, and record() silently no-ops against it. Carrying the
    # caller's ledger across is what makes generation tokens land anywhere at
    # all; without it the phase reports zero while plainly having spent money.
    # active() is read HERE, on the caller's thread, for the same reason.
    parent_ledger = usage_ledger.active()

    def worker():
        start_log_capture(callback=lambda line: events.put(("log", line)))
        try:
            with usage_ledger.adopt(parent_ledger):
                box["result"] = work()
        except Exception as exc:
            box["error"] = exc
        finally:
            box["logs"] = stop_log_capture()
            events.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        try:
            kind, payload = events.get(timeout=1)
        except queue.Empty:
            yield {"kind": "heartbeat"}
            continue
        if kind == "done":
            break
        yield {"kind": kind, "line": payload}
    if "error" in box:
        raise box["error"]
    return box.get("result"), box.get("logs", [])


def _run_quiet_work(work, message: str):
    """Run model work off-thread while yielding useful heartbeat messages."""
    from agents.data_agent import usage_ledger

    box = {}
    # Same thread-local ledger problem as _run_logged_work: mechanism
    # selection, trace analysis and the LLM judge all run through here, and
    # their tokens vanish unless the caller's ledger comes along.
    parent_ledger = usage_ledger.active()

    def worker():
        try:
            with usage_ledger.adopt(parent_ledger):
                box["result"] = work()
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(timeout=1)
        if thread.is_alive():
            yield {"type": "heartbeat", "message": message}
    if "error" in box:
        raise box["error"]
    return box.get("result")



def _source_snapshot(source: dict) -> dict:
    return {
        field: str(source.get(field, "") or "")
        for field in user_data_sources.STORED_FIELDS
    }



def _infer_expected_format(task_text: str) -> str:
    lowered = str(task_text or "").casefold()
    formats = (
        ("GeoPackage", ("geopackage", "gpkg")),
        ("GeoJSON", ("geojson",)),
        ("Shapefile", ("shapefile", ".shp")),
        ("GeoTIFF", ("geotiff", ".tif", ".tiff")),
        ("NetCDF", ("netcdf", ".nc")),
        ("Parquet", ("parquet", "geoparquet")),
        ("CSV", ("csv",)),
        ("JSON", ("json",)),
        ("OSM XML", (".osm", "osm xml")),
    )
    return next((
        label for label, terms in formats
        if any(term in lowered for term in terms)
    ), "")



_MECHANISM_LABELS = {
    "rest": "REST API",
    "stac": "STAC",
    "ogc": "OGC Service",
    "sql": "SQL Interface",
    "bulk": "Bulk Download",
    "http_file": "HTTP File Download",
    "arcgis_featureserver": "ArcGIS FeatureServer",
}
_MECHANISM_GUIDANCE = {
    "rest": (
        "Use documented REST/HTTP query endpoints only. Do not substitute a "
        "bulk archive when evaluating this condition."),
    "stac": (
        "Use STAC catalogs, collections, item search, and asset retrieval. "
        "Do not substitute a non-STAC download method."),
    "ogc": (
        "Use the documented OGC service or OGC API interface such as WFS, "
        "WCS, WMS, or OGC API Features/Coverages."),
    "sql": (
        "Use the documented database or SQL query interface. Do not replace "
        "it with an unrelated REST or bulk workflow."),
    "bulk": (
        "Use an official bulk file, regional extract, archive, or repository "
        "appropriate to the requested scale. Do not use a small-area editing "
        "API to simulate a state- or country-scale bulk extract."),
    "http_file": (
        "Use a single, direct HTTP(S) file download (a plain static file "
        "link), not a query API, catalog search, or a multi-file bulk "
        "archive/repository."),
    "arcgis_featureserver": (
        "Use an Esri ArcGIS FeatureServer/MapServer REST endpoint "
        "(query/queryRelatedRecords over /FeatureServer or /MapServer), not "
        "a generic REST API, OGC service, or file download."),
}
