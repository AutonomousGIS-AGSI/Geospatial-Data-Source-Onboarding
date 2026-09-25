"""Deterministic (non-LLM) validation for Generate & Test, plus the verdict
algebra shared by the mechanical checker, the human adjudication rubric, and
the LLM judge.

Why this exists as a separate instrument: the Generate & Test loop is an LLM
writing a handbook, an LLM writing retrieval code, and an LLM refining that
handbook on failure. If an LLM also decides whether the result is correct,
the measurement sits inside the system being measured. Everything here is
either deterministic or human-entered, so it can serve as the reported ground
truth while ``evaluate_retrieval_result`` (the LLM judge) is demoted to an
arm that gets *compared against* it.

Two rules distinguish this from the previous metadata check:

1. Verdicts are three-valued -- ``pass``/``fail``/``undecidable`` -- and
   ``undecidable`` NEVER rolls up to a pass. "We could not check this" is a
   routing decision (send it to a human), not a clean bill of health. The
   old ``evaluate_retrieval_result_metadata`` folded "not applicable" into
   "completed", so a task whose only real requirement was never examined
   still came back ``task_completed: true``.
2. Every check is grounded in harness-captured evidence -- the URLs the
   sandbox actually observed, and file properties read off the artifact with
   rasterio/geopandas -- never in what the generated code printed about
   itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime
from urllib.parse import urlparse, parse_qs

PASS, FAIL, UNDECIDABLE, NA = "pass", "fail", "undecidable", "not_applicable"
_VERDICTS = (PASS, FAIL, UNDECIDABLE, NA)

# The five dimensions the existing UI/schema already renders, plus the two
# the experimental-design metrics list calls for but had no home:
# "correct dataset selection" and byte-level provenance.
DIMENSIONS = (
    "dataset_selection",
    "mechanism_correctness",
    "parameter_correctness",
    "spatial_query_correctness",
    "temporal_query_correctness",
    "output_correctness",
    "provenance",
)

DIMENSION_LABELS = {
    "dataset_selection": "Dataset / product selection",
    "mechanism_correctness": "Access mechanism",
    "parameter_correctness": "Query parameters",
    "spatial_query_correctness": "Spatial query",
    "temporal_query_correctness": "Temporal query",
    "output_correctness": "Output validity",
    "provenance": "Provenance",
}

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "validation_reference")
_PLACE_CACHE = os.path.join(REFERENCE_DIR, "reference_places.json")


def normalize_verdict(value):
    value = str(value or "").strip().lower()
    return value if value in _VERDICTS else UNDECIDABLE


def roll_up(dimensions):
    """Combine per-dimension verdicts into one overall verdict.

    Any ``fail`` fails the run. Otherwise any ``undecidable`` makes the run
    undecidable -- this is the rule the old metadata checker lacked. A run
    where every dimension is ``not_applicable`` is ``undecidable``, not a
    pass: it means nothing was actually checked.
    """
    values = [normalize_verdict(v) for v in (dimensions or {}).values()]
    applicable = [v for v in values if v != NA]
    if any(v == FAIL for v in applicable):
        return FAIL
    if any(v == UNDECIDABLE for v in applicable):
        return UNDECIDABLE
    return PASS if applicable else UNDECIDABLE


# --------------------------------------------------------------------------
# Task specification
# --------------------------------------------------------------------------

def parse_task_spec(text):
    """Parse a human-authored, pre-registered task spec (TOML or JSON).

    Free text cannot be checked mechanically; this is what makes the task's
    requirements machine-decidable. Authored BEFORE the run, so it is a
    pre-registration rather than a post-hoc rationalization of whatever came
    back. Returns {} for empty input and raises ValueError on malformed
    input (callers surface that to the author rather than silently ignoring
    a spec that was meant to apply).
    """
    text = str(text or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            spec = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Validation spec is not valid JSON: {exc}") from exc
    else:
        import tomllib
        try:
            spec = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"Validation spec is not valid TOML: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError("Validation spec must be a table/object.")
    # Tolerate the spec being nested under [expect], which reads naturally
    # in TOML ("[expect.spatial]") and is what the UI placeholder shows.
    if "expect" in spec and isinstance(spec["expect"], dict):
        merged = {k: v for k, v in spec.items() if k != "expect"}
        merged.update(spec["expect"])
        spec = merged
    return spec


SPEC_TEMPLATE = """\
# Pre-registered validation spec — author this BEFORE running the test.
# TOML (or JSON). Every section is optional; omitted sections are reported
# as "not applicable" rather than silently passing.

[expect.output]
format = "GeoTIFF"      # extension family the artifact must have
count_min = 1           # how many matching artifacts must exist
bands = 1               # raster band count (rasters only)
min_bytes = 1024

[expect.spatial]
place = "Kenya"         # resolved against a local reference geometry
mode = "covers"         # covers | within | overlaps
min_coverage = 0.95     # for covers/within: required area fraction
# IoU is reported for every artifact but only ENFORCED when you set it.
# Leave it unset for whole-country raster downloads: a rectangular raster
# against a real country polygon scores a low IoU even when it covers the
# country perfectly. It is useful for clipped/vector outputs.
# min_iou = 0.5

[expect.temporal]
year = 2025
# start = "2025-01-01"
# end   = "2025-12-31"

[expect.dataset]
url_must_match = ["Global_2015_2030/R2024B/2025/KEN"]
filename_must_match = ["_UC_"]
filename_must_not_match = ["/constrained/"]
# Requirements that provably CANNOT be settled from the URL, the filename or
# the artifact — these are routed to a human instead of being guessed at.
undecidable_claims = ["UN-adjusted (release-level attribute, not in the path)"]

[expect.parameters]
required = { }          # query-string key -> expected value, or "*" for "present"
forbidden = []

[expect.provenance]
verify_remote = false   # re-fetch a byte range and compare against the artifact
"""


# --------------------------------------------------------------------------
# Reference geometries
# --------------------------------------------------------------------------

def _load_reference_index():
    """Index every GeoJSON dropped into ``validation_reference/`` by the
    name properties it exposes. Ship Natural Earth / GADM extracts here to
    get real polygon geometry (and therefore real IoU) instead of bboxes."""
    index = {}
    if not os.path.isdir(REFERENCE_DIR):
        return index
    try:
        import geopandas as gpd
    except ImportError:
        return index
    for entry in sorted(os.listdir(REFERENCE_DIR)):
        if not entry.lower().endswith((".geojson", ".json", ".gpkg", ".shp")):
            continue
        if entry == os.path.basename(_PLACE_CACHE):
            continue
        try:
            frame = gpd.read_file(os.path.join(REFERENCE_DIR, entry))
            if frame.crs is not None and str(frame.crs) != "EPSG:4326":
                frame = frame.to_crs(epsg=4326)
        except Exception:
            continue
        name_cols = [c for c in frame.columns
                     if c.lower() in ("name", "name_en", "admin", "name_long",
                                      "shapename", "namelsad", "state_name")]
        for _, row in frame.iterrows():
            for column in name_cols:
                key = str(row.get(column) or "").strip().lower()
                if key and key not in index and row.geometry is not None:
                    index[key] = (row.geometry, f"{entry}:{column}")
    return index


_REFERENCE_INDEX = None


def _read_place_cache():
    try:
        with open(_PLACE_CACHE, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _write_place_cache(cache):
    try:
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        with open(_PLACE_CACHE, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)
    except OSError:
        pass


def reference_geometry(place, allow_network=True):
    """Resolve a place name to a shapely geometry plus a provenance string.

    Resolution order is deliberately local-first, because a live geocoder
    call inside the reported metric makes the experiment non-reproducible
    (rate limits, silent result drift between runs). A network lookup is a
    last resort and its result is written to ``reference_places.json``, which
    should be committed alongside the results: after the first run the
    measurement is frozen and offline. Returns (geometry, provenance) or
    (None, reason).
    """
    global _REFERENCE_INDEX
    place = str(place or "").strip()
    if not place:
        return None, "no place named in the spec"

    if _REFERENCE_INDEX is None:
        _REFERENCE_INDEX = _load_reference_index()
    hit = _REFERENCE_INDEX.get(place.lower())
    if hit:
        return hit[0], f"local reference geometry ({hit[1]})"

    cache = _read_place_cache()
    entry = cache.get(place.lower())
    if entry:
        return _geometry_from_entry(entry), (
            f"cached reference ({entry.get('source', 'unknown')}, "
            f"retrieved {entry.get('retrieved_at', 'unknown')})")

    if not allow_network:
        return None, f"'{place}' is not in the local reference set"

    entry = _geocode(place)
    if entry is None:
        return None, f"could not resolve a reference geometry for '{place}'"
    cache[place.lower()] = entry
    _write_place_cache(cache)
    return _geometry_from_entry(entry), (
        f"{entry['source']}, retrieved {entry['retrieved_at']} "
        f"(now cached for reproducibility)")


def _geometry_from_entry(entry):
    from shapely.geometry import box, shape
    if entry.get("geometry"):
        try:
            return shape(entry["geometry"])
        except Exception:
            pass
    bbox = entry.get("bbox")
    if bbox:
        return box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    return None


def _geocode(place):
    try:
        import requests
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place, "format": "json", "limit": 1,
                    "polygon_geojson": 1},
            headers={"User-Agent": "AGSI-HandbookGenerator/1.0"}, timeout=15)
        response.raise_for_status()
        results = response.json()
        if not results:
            return None
        south, north, west, east = (float(v) for v in results[0]["boundingbox"])
        return {
            "source": "OpenStreetMap Nominatim",
            "retrieved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "display_name": results[0].get("display_name", ""),
            "bbox": {"west": west, "south": south, "east": east, "north": north},
            "geometry": results[0].get("geojson"),
        }
    except Exception:
        return None


def _artifact_geometry(item):
    from shapely.geometry import box
    bbox = _coerce_bbox(item.get("bbox"))
    if not bbox:
        return None
    if bbox["east"] <= bbox["west"] or bbox["north"] <= bbox["south"]:
        return None
    return box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])


def _coerce_bbox(value):
    """Accept the canonical dict shape or the flat GeoJSON/STAC list."""
    if isinstance(value, dict) and {"west", "south", "east", "north"} <= value.keys():
        try:
            return {k: float(value[k]) for k in ("west", "south", "east", "north")}
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) in (4, 6):
        try:
            nums = [float(v) for v in value]
        except (TypeError, ValueError):
            return None
        west, south, east, north = (
            (nums[0], nums[1], nums[2], nums[3]) if len(nums) == 4
            else (nums[0], nums[1], nums[3], nums[4]))
        return {"west": west, "south": south, "east": east, "north": north}
    return None


# --------------------------------------------------------------------------
# Individual mechanical checks
# --------------------------------------------------------------------------

def _check(name, status, evidence):
    return {"name": name, "status": normalize_verdict(status),
            "evidence": str(evidence)}


_FORMAT_EXTENSIONS = {
    "geotiff": {".tif", ".tiff"}, "tiff": {".tif", ".tiff"},
    "geojson": {".geojson", ".json"}, "json": {".json", ".geojson"},
    "csv": {".csv"}, "tsv": {".tsv"}, "shapefile": {".shp", ".zip"},
    "netcdf": {".nc", ".nc4"}, "zip": {".zip"}, "parquet": {".parquet"},
    "grib": {".grib", ".grib2"}, "hdf": {".hdf", ".h5", ".hdf5"},
}


def _primary_artifacts(evidence_items, expected_extensions):
    """The artifacts a format-scoped check should look at: those matching
    the expected format when one is declared, else everything. Keeps a
    sidecar metadata .json from being mistaken for the deliverable."""
    if not expected_extensions:
        return evidence_items
    matched = [item for item in evidence_items
               if str(item.get("extension", "")).lower() in expected_extensions]
    return matched or evidence_items


def check_output(evidence_items, spec):
    rules = spec.get("output") or {}
    checks = []
    if not evidence_items:
        return FAIL, [_check("Output present", FAIL,
                             "No output artifact was recorded for this run.")]

    expected_format = str(rules.get("format") or "").strip().lower()
    expected_extensions = _FORMAT_EXTENSIONS.get(
        expected_format, {f".{expected_format}"} if expected_format else set())
    if expected_extensions:
        matched = [item for item in evidence_items
                   if str(item.get("extension", "")).lower() in expected_extensions]
        checks.append(_check(
            "Output format", PASS if matched else FAIL,
            f"Expected {rules['format']} ({'/'.join(sorted(expected_extensions))}); "
            f"artifacts present: "
            f"{', '.join(str(i.get('name')) for i in evidence_items) or 'none'}."))
    primary = _primary_artifacts(evidence_items, expected_extensions)

    count_min = rules.get("count_min")
    if isinstance(count_min, int):
        checks.append(_check(
            "Artifact count", PASS if len(primary) >= count_min else FAIL,
            f"{len(primary)} matching artifact(s); spec requires at least {count_min}."))

    min_bytes = rules.get("min_bytes")
    if isinstance(min_bytes, int):
        sizes = [int(item.get("size_bytes") or 0) for item in primary]
        biggest = max(sizes) if sizes else 0
        checks.append(_check(
            "Artifact size", PASS if biggest >= min_bytes else FAIL,
            f"Largest matching artifact is {biggest} bytes; "
            f"spec requires at least {min_bytes}."))

    expected_bands = rules.get("bands")
    if isinstance(expected_bands, int):
        observed = [item.get("bands") for item in primary if item.get("bands")]
        if not observed:
            checks.append(_check("Raster band count", UNDECIDABLE,
                                 "No artifact reported a band count."))
        else:
            ok = any(int(b) == expected_bands for b in observed)
            checks.append(_check("Raster band count", PASS if ok else FAIL,
                                 f"Observed bands {observed}; spec requires "
                                 f"{expected_bands}."))

    if not checks:
        # An artifact exists, but the spec declared no rules about it, so
        # nothing about its correctness was established. Reporting this as
        # a pass is precisely the error this module exists to prevent: "a
        # file came back" is not evidence the right file came back. It is
        # not-applicable rather than undecidable because the author chose
        # not to constrain output; when NO section of the spec constrains
        # anything, every dimension lands here and roll_up returns
        # undecidable for the run as a whole.
        return NA, [_check(
            "Output rules", NA,
            f"{len(evidence_items)} artifact(s) recorded: "
            f"{', '.join(str(i.get('name')) for i in evidence_items)}. "
            "No [expect.output] rules were declared, so nothing about the "
            "output's correctness was checked.")]
    return roll_up({f"c{i}": c["status"] for i, c in enumerate(checks)}), checks


def check_spatial(evidence_items, spec, allow_network=True):
    rules = spec.get("spatial") or {}
    place = str(rules.get("place") or "").strip()
    if not place:
        return NA, []

    geometry, provenance = reference_geometry(place, allow_network=allow_network)
    if geometry is None:
        return UNDECIDABLE, [_check(
            f"Spatial extent vs. {place}", UNDECIDABLE,
            f"No reference geometry available: {provenance}. Drop a GeoJSON "
            f"for this place into {REFERENCE_DIR} to make this decidable.")]

    artifacts = [(item.get("name"), _artifact_geometry(item))
                 for item in evidence_items]
    artifacts = [(name, geom) for name, geom in artifacts if geom is not None]
    if not artifacts:
        return UNDECIDABLE, [_check(
            f"Spatial extent vs. {place}", UNDECIDABLE,
            f"A reference geometry for '{place}' was resolved ({provenance}), "
            "but no artifact exposed a computed spatial extent to compare "
            "against.")]

    mode = str(rules.get("mode") or "overlaps").strip().lower()
    min_coverage = float(rules.get("min_coverage") or 0.95)
    min_iou = rules.get("min_iou")
    checks = []
    for name, geom in artifacts:
        intersection = geom.intersection(geometry).area
        union = geom.union(geometry).area
        iou = intersection / union if union else 0.0
        covered = intersection / geometry.area if geometry.area else 0.0
        contained = intersection / geom.area if geom.area else 0.0
        if mode == "covers":
            status = PASS if covered >= min_coverage else FAIL
            requirement = (f"must cover >= {min_coverage:.0%} of {place} "
                           f"(covers {covered:.1%})")
        elif mode == "within":
            status = PASS if contained >= min_coverage else FAIL
            requirement = (f">= {min_coverage:.0%} of the artifact must fall "
                           f"inside {place} (inside {contained:.1%})")
        else:
            status = PASS if intersection > 0 else FAIL
            requirement = ("must intersect " + place +
                           (" (intersects)" if intersection else " (disjoint)"))
        if status == PASS and isinstance(min_iou, (int, float)) and iou < min_iou:
            status = FAIL
            requirement += f"; IoU {iou:.3f} is below the required {min_iou}"
        checks.append(_check(
            f"Spatial extent of {name} vs. {place}", status,
            f"Mode '{mode}': {requirement}. IoU {iou:.3f}, covers "
            f"{covered:.1%} of the reference, {contained:.1%} of the "
            f"artifact falls inside it. Reference: {provenance}."))
    return roll_up({f"c{i}": c["status"] for i, c in enumerate(checks)}), checks


_YEAR_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")


def check_temporal(evidence_items, http_requests, spec):
    rules = spec.get("temporal") or {}
    year = rules.get("year")
    start, end = rules.get("start"), rules.get("end")
    if not year and not (start or end):
        return NA, []

    checks = []
    haystack = " ".join(
        [str(item.get("name") or "") for item in evidence_items] +
        [str(request.get("url") or "") for request in http_requests])

    if year:
        found = sorted(set(match.group(0) for match in _YEAR_RE.finditer(haystack)))
        if not found:
            checks.append(_check(
                "Requested year in the resolved URL/filename", UNDECIDABLE,
                f"No four-digit year appears in the requested URLs or artifact "
                f"names, so the {year} requirement cannot be settled from "
                "them."))
        else:
            ok = str(year) in found
            checks.append(_check(
                "Requested year in the resolved URL/filename", PASS if ok else FAIL,
                f"Years present in the requested URLs/artifact names: "
                f"{', '.join(found)}; spec requires {year}."))

    if start or end:
        observed = _artifact_time_range(evidence_items)
        if observed is None:
            checks.append(_check(
                "Artifact time range", UNDECIDABLE,
                "No artifact exposed a temporal range to compare against the "
                "requested interval."))
        else:
            low, high = observed
            ok = ((not start or high >= _as_date(start))
                  and (not end or low <= _as_date(end)))
            checks.append(_check(
                "Artifact time range", PASS if ok else FAIL,
                f"Artifact covers {low} to {high}; spec requires "
                f"{start or '-inf'} to {end or '+inf'}."))
    return roll_up({f"c{i}": c["status"] for i, c in enumerate(checks)}), checks


def _as_date(value):
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def _artifact_time_range(evidence_items):
    """Pull a min/max date out of whatever temporal fields the inspector
    captured. Returns None when nothing temporal was recorded."""
    stamps = []
    for item in evidence_items:
        for key in ("time_range", "datetime", "start_datetime", "end_datetime"):
            value = item.get(key)
            if isinstance(value, (list, tuple)):
                stamps.extend(value)
            elif value:
                stamps.append(value)
    parsed = []
    for stamp in stamps:
        try:
            parsed.append(_as_date(stamp))
        except (ValueError, TypeError):
            continue
    return (min(parsed), max(parsed)) if parsed else None


def check_dataset(evidence_items, http_requests, spec):
    """Did the run select the right product/variant?

    Matched against the URLs the sandbox actually observed, not against what
    the generated code claimed it downloaded. ``undecidable_claims`` is the
    important half: a requirement the author knows cannot be settled from
    the path or the artifact is declared up front and routed to a human,
    rather than being quietly dropped (which is exactly how the WorldPop
    "UN-adjusted" requirement disappeared).
    """
    rules = spec.get("dataset") or {}
    if not rules:
        return NA, []

    urls = [str(request.get("url") or "") for request in http_requests]
    names = [str(item.get("name") or "") for item in evidence_items]
    # The last requested URL is the one that produced the artifact; earlier
    # entries are directory listings and probes.
    corpus = "\n".join(urls + names)
    checks = []

    def _patterns(key):
        value = rules.get(key) or []
        return [value] if isinstance(value, str) else list(value)

    for pattern in _patterns("url_must_match") + _patterns("filename_must_match"):
        hit = bool(re.search(pattern, corpus, re.I))
        checks.append(_check(
            f"Selection requires '{pattern}'", PASS if hit else FAIL,
            f"Pattern {'found in' if hit else 'NOT found in'} the requested "
            f"URLs/artifact names."))
    for pattern in (_patterns("url_must_not_match")
                    + _patterns("filename_must_not_match")):
        hit = bool(re.search(pattern, corpus, re.I))
        checks.append(_check(
            f"Selection must avoid '{pattern}'", FAIL if hit else PASS,
            f"Pattern {'FOUND in' if hit else 'absent from'} the requested "
            f"URLs/artifact names."))
    for claim in _patterns("undecidable_claims"):
        checks.append(_check(
            f"Product claim: {claim}", UNDECIDABLE,
            "Declared in the spec as not settleable from the URL, the "
            "filename or the artifact. Routed to human adjudication — it is "
            "deliberately NOT treated as satisfied."))

    if not checks:
        return NA, []
    return roll_up({f"c{i}": c["status"] for i, c in enumerate(checks)}), checks


def check_parameters(http_requests, spec):
    rules = spec.get("parameters") or {}
    required = rules.get("required") or {}
    forbidden = rules.get("forbidden") or []
    if not required and not forbidden:
        return NA, []
    if not http_requests:
        return UNDECIDABLE, [_check(
            "Query parameters", UNDECIDABLE,
            "No HTTP requests were captured, so query parameters cannot be "
            "checked.")]

    observed = {}
    for request in http_requests:
        for key, values in parse_qs(urlparse(str(request.get("url") or "")).query).items():
            observed.setdefault(key.lower(), set()).update(values)

    checks = []
    for key, expected in (required.items() if isinstance(required, dict)
                          else ((k, "*") for k in required)):
        values = observed.get(str(key).lower())
        if values is None:
            checks.append(_check(f"Parameter '{key}'", FAIL,
                                 f"Not present in any captured request. "
                                 f"Observed parameters: "
                                 f"{', '.join(sorted(observed)) or 'none'}."))
        elif expected == "*":
            checks.append(_check(f"Parameter '{key}'", PASS,
                                 f"Present with value(s) {sorted(values)}."))
        else:
            ok = any(str(v).lower() == str(expected).lower() for v in values)
            checks.append(_check(f"Parameter '{key}'", PASS if ok else FAIL,
                                 f"Expected {expected!r}; observed "
                                 f"{sorted(values)}."))
    for key in forbidden:
        present = str(key).lower() in observed
        checks.append(_check(f"Parameter '{key}' must be absent",
                             FAIL if present else PASS,
                             "Present in a captured request." if present
                             else "Absent, as required."))
    return roll_up({f"c{i}": c["status"] for i, c in enumerate(checks)}), checks


def check_provenance(downloaded_files, http_requests, spec, artifact_root=None):
    """Prove the file on disk is the file at the URL that was requested.

    This is the only check that catches an artifact the generated code
    assembled locally rather than downloaded. Hashing is local and always
    runs when the file is reachable; ``verify_remote`` additionally re-fetches
    a byte range and compares, which is the stronger claim.
    """
    rules = spec.get("provenance") or {}
    if not artifact_root or not downloaded_files:
        return NA, []

    checks = []
    for record in downloaded_files:
        ref = str(record.get("artifact_ref") or "")
        if not ref:
            continue
        path = os.path.join(artifact_root, *[p for p in ref.replace("\\", "/").split("/") if p])
        if not os.path.isfile(path):
            checks.append(_check(f"Artifact hash for {record.get('name')}",
                                 UNDECIDABLE,
                                 "The artifact is no longer on disk, so it "
                                 "cannot be hashed."))
            continue
        digest, size = _sha256(path)
        checks.append(_check(
            f"Artifact hash for {record.get('name')}", PASS,
            f"sha256 {digest}, {size} bytes (recomputed from the file on "
            "disk, not read from what the generated code reported)."))

        if rules.get("verify_remote"):
            url = _download_url_for(record, http_requests)
            checks.append(_verify_remote_bytes(path, url, record.get("name")))
    if not checks:
        return NA, []
    return roll_up({f"c{i}": c["status"] for i, c in enumerate(checks)}), checks


def _sha256(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _download_url_for(record, http_requests):
    """The captured request whose URL ends in this artifact's filename."""
    name = str(record.get("name") or "")
    for request in reversed(http_requests or []):
        url = str(request.get("url") or "")
        if name and url.rstrip("/").endswith(name):
            return url
    return ""


def _verify_remote_bytes(path, url, name):
    if not url:
        return _check(f"Remote byte match for {name}", UNDECIDABLE,
                      "No captured request URL matches this artifact's "
                      "filename, so it cannot be re-fetched.")
    try:
        import requests
        head = requests.get(url, headers={"Range": "bytes=0-65535"}, timeout=30)
        head.raise_for_status()
        remote = head.content
        with open(path, "rb") as handle:
            local = handle.read(len(remote))
        ok = remote == local
        return _check(f"Remote byte match for {name}", PASS if ok else FAIL,
                      f"Re-fetched the first {len(remote)} bytes of {url} and "
                      f"they {'match' if ok else 'DO NOT match'} the artifact "
                      "on disk.")
    except Exception as exc:
        return _check(f"Remote byte match for {name}", UNDECIDABLE,
                      f"Re-fetch failed ({exc}); no conclusion drawn.")


def check_mechanism(http_requests, spec, mechanism=""):
    """Only decidable when the spec pins the scheme/host the mechanism
    implies. Left not-applicable otherwise -- guessing an access mechanism
    from a URL shape is exactly the kind of inference this module refuses to
    make silently."""
    rules = spec.get("mechanism") or {}
    expected_host = str(rules.get("host") or "").strip().lower()
    if not expected_host:
        return NA, []
    if not http_requests:
        return UNDECIDABLE, [_check("Access mechanism", UNDECIDABLE,
                                    "No HTTP requests were captured.")]
    hosts = {urlparse(str(r.get("url") or "")).netloc.lower()
             for r in http_requests} - {""}
    ok = any(expected_host in host for host in hosts)
    return (PASS if ok else FAIL), [_check(
        "Access mechanism host", PASS if ok else FAIL,
        f"Spec expects requests to '{expected_host}'; observed hosts: "
        f"{', '.join(sorted(hosts)) or 'none'}.")]


# --------------------------------------------------------------------------
# Top-level mechanical evaluation
# --------------------------------------------------------------------------

def evaluate_mechanical(task, execution_result, spec_text="", mechanism="",
                        artifact_root=None, allow_network=True):
    """Run every deterministic check the spec makes decidable.

    Returns a validation record: per-dimension three-valued verdicts, the
    supporting checks, and an overall verdict where ``undecidable`` is a
    first-class outcome that routes to human adjudication.
    """
    spec = parse_task_spec(spec_text)
    evidence_items = [item for item in (execution_result.get("output_evidence") or [])
                      if isinstance(item, dict)]
    http_requests = [r for r in (execution_result.get("http_requests") or [])
                     if isinstance(r, dict)]
    downloaded = [f for f in (execution_result.get("downloaded_files") or [])
                  if isinstance(f, dict)]

    dimensions, checks = {}, []
    for name, (status, found) in {
        "output_correctness": check_output(evidence_items, spec),
        "spatial_query_correctness": check_spatial(evidence_items, spec, allow_network),
        "temporal_query_correctness": check_temporal(evidence_items, http_requests, spec),
        "dataset_selection": check_dataset(evidence_items, http_requests, spec),
        "parameter_correctness": check_parameters(http_requests, spec),
        "mechanism_correctness": check_mechanism(http_requests, spec, mechanism),
        "provenance": check_provenance(downloaded, http_requests, spec, artifact_root),
    }.items():
        dimensions[name] = status
        checks.extend(found)

    verdict = roll_up(dimensions)
    undecided = [DIMENSION_LABELS[k] for k, v in dimensions.items()
                 if v == UNDECIDABLE]
    failed = [DIMENSION_LABELS[k] for k, v in dimensions.items() if v == FAIL]

    if not spec:
        summary = ("No pre-registered validation spec was authored for this "
                   "task, so nothing beyond artifact presence could be "
                   "checked mechanically. Add a spec on the Setup tab, or "
                   "adjudicate this run manually — it is reported as "
                   "undecidable rather than passed.")
    elif verdict == FAIL:
        summary = ("Mechanical validation FAILED on: " + ", ".join(failed) + "."
                   + (f" Undecidable and awaiting human review: "
                      f"{', '.join(undecided)}." if undecided else ""))
    elif verdict == UNDECIDABLE:
        summary = ("Mechanical validation could not settle: "
                   + ", ".join(undecided)
                   + ". Every other declared requirement passed. This run "
                     "needs human adjudication — it is deliberately not "
                     "reported as complete.")
    else:
        summary = ("Every requirement the pre-registered spec makes "
                   "decidable passed, checked against harness-captured URLs "
                   "and the artifact's own computed properties.")

    return build_record(
        method="mechanical", verdict=verdict, dimensions=dimensions,
        checks=checks, summary=summary,
        confidence=1.0 if verdict in (PASS, FAIL) else 0.0,
        spec_declared=bool(spec))


def build_record(method, verdict, dimensions, checks, summary, confidence=1.0,
                 rater="", elapsed_seconds=None, blinded=None,
                 spec_declared=False, handbook_gap=""):
    """One validation record, in a shape the existing semantic_validation
    renderer still understands (``task_completed`` stays a boolean) while
    carrying the three-valued ``verdict`` that actually drives routing."""
    verdict = normalize_verdict(verdict)
    dimensions = {name: normalize_verdict(dimensions.get(name, NA))
                  for name in DIMENSIONS}
    if not handbook_gap and verdict == FAIL:
        handbook_gap = summary
    record = {
        "method": method,
        "verdict": verdict,
        # Only an unambiguous pass counts as completed; "undecidable" must
        # never read as success anywhere downstream.
        "task_completed": verdict == PASS,
        "confidence": float(confidence),
        "summary": summary,
        "checks": list(checks or []),
        "spec_declared": bool(spec_declared),
        "failure_category": "handbook_deficiency" if verdict == FAIL else "",
        "should_refine_handbook": verdict == FAIL,
        "handbook_gap": handbook_gap,
    }
    record.update(dimensions)
    record["dimensions"] = dimensions
    if rater:
        record["rater"] = rater
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = float(elapsed_seconds)
    if blinded is not None:
        record["blinded"] = bool(blinded)
    return record


def manual_record(verdict_input):
    """Build a validation record from the human adjudication rubric.

    Unlike the previous manual path -- one global yes/no that pinned every
    correctness dimension to "not applicable", so a manually reviewed run
    contributed nothing to five of the experiment's eight metrics -- this
    keeps the reviewer's per-dimension verdicts and the evidence they cited
    for each.
    """
    verdict_input = verdict_input or {}
    dimensions = {}
    checks = []
    for name in DIMENSIONS:
        status = normalize_verdict(
            (verdict_input.get("dimensions") or {}).get(name, NA))
        dimensions[name] = status
        note = str((verdict_input.get("evidence") or {}).get(name, "")).strip()
        if status != NA or note:
            checks.append(_check(
                f"{DIMENSION_LABELS[name]} (human adjudication)", status,
                note or "No evidence was cited for this dimension."))

    overall = verdict_input.get("verdict")
    verdict = normalize_verdict(overall) if overall else roll_up(dimensions)
    summary = str(verdict_input.get("summary") or "").strip() or (
        f"Human adjudication: {verdict.replace('_', ' ')}.")
    return build_record(
        method="manual", verdict=verdict, dimensions=dimensions, checks=checks,
        summary=summary, confidence=1.0,
        rater=str(verdict_input.get("rater") or "").strip()[:120],
        elapsed_seconds=verdict_input.get("elapsed_seconds"),
        blinded=verdict_input.get("blinded"))


def llm_record(analysis):
    """Wrap the LLM judge's output in the same record shape so it can sit
    beside the other methods in the append-only log and be compared against
    the human gold standard. The judge has no ``undecidable`` vocabulary, so
    its per-dimension "unknown" maps to undecidable."""
    analysis = analysis or {}
    dimensions = {}
    for name in DIMENSIONS:
        raw = str(analysis.get(name) or NA).strip().lower()
        dimensions[name] = normalize_verdict(
            UNDECIDABLE if raw == "unknown" else raw)
    verdict = PASS if analysis.get("task_completed") else (
        FAIL if analysis.get("failure_category") else UNDECIDABLE)
    record = build_record(
        method="llm", verdict=verdict, dimensions=dimensions,
        checks=analysis.get("checks") or [],
        summary=str(analysis.get("summary") or ""),
        confidence=float(analysis.get("confidence") or 0.0),
        handbook_gap=str(analysis.get("handbook_gap") or ""))
    record["failure_category"] = str(analysis.get("failure_category") or "")
    record["should_refine_handbook"] = bool(analysis.get("should_refine_handbook"))
    return record


# Precedence when several methods have judged the same run. The human verdict
# is the gold standard, so it wins; the LLM judge is never the reported
# measurement when a deterministic or human result exists.
_METHOD_PRECEDENCE = {"manual": 3, "mechanical": 2, "metadata": 2, "llm": 1}


def primary_validation(validations):
    """Pick the record to report for a run, from the append-only log."""
    records = [v for v in (validations or []) if isinstance(v, dict)]
    if not records:
        return None
    return max(
        enumerate(records),
        key=lambda pair: (_METHOD_PRECEDENCE.get(pair[1].get("method"), 0),
                          pair[0]))[1]
