# Geospatial Data Source Onboarding

**Toward Self-Growing Data Retrieval in Autonomous GIS: An LLM-Based Framework for Geospatial Data Source Onboarding**
Temitope Akinboyewa, Zhenlong Li

GIS agents can now plan and run complex geospatial analyses, but they can only
retrieve data from sources someone has already integrated for them. Knowing that
a source holds the right data is not the same as knowing how to get it out:
retrieval depends on source-specific details such as endpoints, dataset
identifiers, authentication, query conventions, file-naming patterns and access
limits. Existing systems supply this knowledge in advance, for example as
manually authored handbooks that take an expert hours per source to write and
test.

This repository implements a framework for **geospatial data source
onboarding**: acquiring, verifying and retaining the operational knowledge an
agent needs to retrieve data from a source it has never used before. Given only
a source name and, optionally, a documentation URL, the framework does four
things:

1. It gathers the provider's documentation.
2. It synthesizes a structured handbook from it.
3. It verifies and refines that handbook against the live service.
4. It stores the handbook for reuse.

The agent's retrieval capability can therefore grow during operation instead of
being fixed at design time.

## Framework

The framework turns a minimal source descriptor into reusable operational
knowledge through five layers.

1. **Knowledge acquisition.** *Source resolution* confirms the provider,
   publishing organization, official website and documentation entry point from
   web evidence. *Documentation acquisition* gathers API references, developer
   guides and specifications and normalizes them to text. *Access specification
   extraction* records:
   - the retrieval mechanism and base URL
   - endpoints and parameters
   - authentication and registration requirements
   - response formats and usage constraints
   - the credentials the source needs
2. **Knowledge synthesis.** The acquisition context becomes a structured,
   source-specific handbook. It contains operational instructions (a runbook)
   and an executable retrieval example.
3. **Self-verification.** A *credential precheck* pauses for the user to supply
   any required keys. A *verification–revision loop* then executes the
   handbook's retrieval example against the live service. It revises the
   deficient instructions or code from the execution diagnostics and
   documentation until retrieval succeeds or a stopping condition is reached.
4. **Execution-grounded refinement.** An autonomous retrieval agent uses the
   handbook on a real task. Three nested loops handle failures:
   - repairing the retrieval code within a trial
   - retrying after an external failure
   - revising the handbook when a knowledge deficiency is identified

   Each failure is attributed before anything changes. The handbook is revised
   only when the failure traces back to the handbook itself, not to faulty
   generated code or a temporarily unavailable service. Revisions must keep
   behaviour that already works and must generalize beyond the task that
   exposed the problem.
5. **Persistence and reuse.** Verified handbooks are kept in a handbook library
   that is scanned during source selection:
   - A source that is already in the library is reused directly.
   - A source that is missing triggers onboarding, and the resulting handbook
     is added to the library.
   - A refinement replaces the stored handbook only after it has successfully
     delivered data, so an unsuccessful revision cannot displace a working one.

## Implementation

**Workflow.**
- **Acquisition.** An LLM (OpenAI GPT-5.2) finds information through the
  Responses API's hosted web-search tools and by fetching documentation
  directly. Each address has a 15-second timeout. HTML, plain text, JSON and
  XML responses are converted to text. Content is capped at 8,000 characters
  per document and 14,000 in total.
- **Synthesis.** The acquired content is combined with fixed authoring
  instructions and a reference handbook to produce the new handbook.
- **Self-verification.** The handbook's code example is run to retrieve a small
  data sample. Execution pauses at a user checkpoint when credentials are
  needed; they are supplied as environment variables. Verification stops after
  success, after six attempts, or after three consecutive failures with the
  same error.
- **Refinement.** A data retrieval agent uses the verified handbook on a real
  task. Failure attribution reads the generated code, the requests issued, the
  execution errors and the output checks.

**The handbook.** Each handbook is a structured record with nine fields:
`data_source_name`, `brief_description`, `runbook`, `code_example`, `website`,
`requires_key`, `key_name`, `caveats` and `key_signup_url`. The first four give
the retrieval agent the source identity, operational instructions and an
executable example. The rest support user guidance and credential collection.

**Handbook Studio.** Handbook Studio is the web interface for running
onboarding. It has three tabs:
- **Setup:** choose the source, the LLMs and the execution limits.
- **Generate & retrieve:** watch generation stage by stage, review and edit the
  handbook, run a retrieval task, and inspect diagnostics and preview outputs.
- **Session report:** runs, convergence and exports.

Each session is saved as a JSON record containing:
- handbook snapshots
- code attempts
- execution diagnostics and request traces
- output evidence
- model settings and token usage

**Running it.**

```bash
pip install -r requirements.txt
python WebUI/app.py        # then open http://localhost:4041/
```

Enter an OpenAI (or GIBD) API key under **Settings**. An Anthropic key is
needed only to use the Claude Agent SDK provider. See `.env.example` for
optional settings, including `MAPBOX_TOKEN` for the map previews.

## Experiments

### 1. Effect of generated handbooks on retrieval

The first experiment used 45 retrieval tasks across five data access
mechanisms, with three sources per mechanism. Each task was run twice: once
without a handbook and once with an automatically generated one. The tasks
covered product selection, spatial and temporal filtering, and attribute
selection. Outcomes were scored as *verified complete*, *partial output* or
*no usable output*.

| Access mechanism | Representative sources |
|---|---|
| REST API | NASA FIRMS, OpenStreetMap Overpass, CDC PLACES (Socrata SoQL) |
| STAC | Microsoft Planetary Computer, Element 84 Earth Search, USGS Landsat STAC |
| OGC | FEMA NFHL (WFS), USGS 3DEP elevation (WCS), an OGC API–Features service |
| HTTPS file download | WorldPop, Natural Earth, HydroSHEDS |
| ArcGIS FeatureServer | USGS Quaternary Faults and Folds, USGS PAD-US, NOAA CUSP |

| Condition | Verified complete | Partial output | No usable output |
|---|---|---|---|
| Without handbook | 28.9% | 22.2% | 48.9% |
| With generated handbook | **77.8%** | 17.8% | **4.4%** |

Verified completion by mechanism (without → with handbook):

| Mechanism | Without handbook | With handbook |
|---|---|---|
| REST | 44.4% | 100% |
| OGC | 33.3% | 88.9% |
| HTTPS file download | 0% | 77.8% |
| ArcGIS FeatureServer | 44.4% | 66.7% |
| STAC | 22.2% | 55.6% |

Handbooks helped with every mechanism. The largest gain was for file
repositories. There the difficulty is rarely forming a query; it is knowing the
provider's directory structure and file-naming conventions. Standards such as
STAC and OGC reduced but did not remove the need for source-specific knowledge,
such as collection IDs, asset keys, URL signing and service versions.

### 2. LLM-Find benchmark

The second experiment reran the 15 retrieval tasks from LLM-Find (Ning et al.,
2025). That benchmark used manually authored handbooks for seven sources. This
time the handbooks were generated automatically, and each one was reused across
its source's tasks.

| Data source | Generation tokens | Generation time | Tasks completed | Adaptation |
|---|---|---|---|---|
| OpenStreetMap | 56,716 | 259 s | 2/2 | 1 handbook refinement |
| Esri World Imagery | 100,724 | 217 s | 4/4 | None |
| Census TIGER/Line | 61,463 | 159 s | 2/2 | None |
| Census ACS API | 136,881 | 176 s | 2/2 | 1 handbook refinement |
| OpenTopography | 67,146 | 134 s | 2/2 | 1 external retry |
| OpenWeather | 60,782 | 142 s | 0/2 | Incorrect access-route guidance |
| NYT COVID-19 | 56,758 | 116 s | 1/1 | None |

13 of the 15 tasks succeeded, and 12 passed on the first retrieval run.
Generating a handbook took on average about 77,000 tokens and 2.9 minutes,
compared with the hours an expert needs to author one by hand.

### 3. End-to-end autonomous GIS

The third experiment integrated onboarding into an end-to-end autonomous GIS
workflow. The agent identified the data it needed and reused handbooks from the
library where it could. When a required source was missing, it generated a new
handbook. It then retrieved the data, built a geoprocessing workflow and
executed it. When repeated retrieval failures could not be fixed by repairing
the code, it re-onboarded the source. In one case this switched the source from
OpenStreetMap to Philadelphia's OpenDataPhilly FeatureServer.

| Spatial analysis task | Data acquisition | Spatial analysis |
|---|---|---|
| LA active fires by census tract | Successful | Successful |
| California earthquake activity by county | Successful | Successful |
| Philadelphia hospital accessibility | Successful (after re-onboarding) | Failed: wrong CRS for distance buffering |
| South Carolina obesity and population density | Successful after rerun | Successful |
| LA earthquake kernel density | Successful | Failed: density surface not saved as a spatial output |

The required data were acquired for all five tasks. Both downstream failures
were analysis errors, not retrieval errors. This separates *data-access
autonomy* from *analytical autonomy*.

## Limitations

- **Successful execution does not guarantee the right data.** A retrieval can
  succeed and still return the wrong product. For example, a HydroSHEDS handbook
  tied to flow direction returned flow direction when flow accumulation was
  requested. Task-aware validation of product, variables, coverage and
  resolution is future work.
- **Evaluation scope is limited.** The evaluation covers five access mechanisms
  and a limited set of providers. It does not include authenticated commercial
  platforms, cloud object stores, databases or asynchronous processing
  services.
- **Credentials stay with the user.** When a source needs an API key, the agent
  identifies which key is required and waits for the user to provide it.

## Citation

Akinboyewa, T., & Li, Z. *Toward Self-Growing Data Retrieval in Autonomous
GIS: An LLM-Based Framework for Geospatial Data Source Onboarding.* Manuscript
in preparation.
