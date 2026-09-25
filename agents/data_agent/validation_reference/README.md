# Validation reference geometries

Reference extents for `[expect.spatial]` checks in
`agents/data_agent/validation.py`. Resolution is local-first, because a live
geocoder call inside a reported metric makes the experiment
non-reproducible — rate limits and silent result drift between runs.

Order of resolution:

1. **Any `.geojson` / `.gpkg` / `.shp` dropped in this directory.** Indexed by
   whatever name column it exposes (`name`, `NAME_EN`, `admin`, `shapeName`,
   `NAMELSAD`, `state_name`). This is the option that gives *real polygon*
   geometry, and therefore meaningful IoU and coverage fractions. Natural
   Earth `admin_0_countries` and `admin_1_states_provinces`, or a GADM /
   TIGER extract, cover most benchmark tasks.
2. **`reference_places.json`** — a cache written automatically the first time
   a place has to be geocoded.
3. **Nominatim**, once, as a last resort. The result is written to the cache
   with its source and retrieval timestamp.

**Commit `reference_places.json` alongside your results.** After the first
run the spatial measurement is frozen and offline, so the reported numbers
are reproducible by anyone with the repository.
