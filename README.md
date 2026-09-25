# Geospatial Data Source Onboarding

Handbook Studio onboards a new geospatial data source in three steps. It
generates a *handbook* for the source from its documentation (endpoints,
parameters, auth and a working code example). It then tests the handbook by
having an agent retrieve real data for a task. Finally, it refines the
handbook from what the retrieval run revealed. Each session records the
handbook versions, the generated code, the outputs and the token cost of
producing them.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional; keys can also be entered in the UI
python WebUI/app.py
```

Open http://localhost:4041/, then enter an OpenAI (or GIBD) API key under
**Settings**. An Anthropic key is only needed to use the Claude Agent SDK
provider. Set `MAPBOX_TOKEN` in `.env` to use Mapbox basemaps in output
previews; without one they use OpenStreetMap tiles.

## Layout

| Path | What it does |
|---|---|
| `agents/data_agent/handbook_generator.py` | Drafts a handbook from a source's docs, and refines one from execution traces |
| `agents/data_agent/DataRetrieverAgent.py` | Runs a handbook: generates retrieval code, executes it, repairs on failure |
| `agents/data_agent/validation.py` | Checks retrieved outputs against a task's spatial/format spec |
| `agents/data_agent/user_data_sources.py` | Stores handbooks per user |
| `agents/data_agent/usage_ledger.py` | Token and cost accounting per session |
| `agents/data_agent/claude_agent_provider.py` | Claude Agent SDK provider |
| `WebUI/handbook_studio_runner.py` | The generate → test → refine pipeline, streamed as events |
| `WebUI/handbook_studio_store.py` | Session persistence |
| `WebUI/app.py` | Flask API and static file server |
| `WebUI/handbook_studio_mode.js` | The Studio UI |
| `WebUI/output_retention.py` | Disk retention for retrieved outputs on quota-limited hosts |
| `WebUI/studio_share_cli.py` | Make sessions public or private from the command line |

Sessions are saved under `Data_retrieval_skill_project/Handbook_Studio_Sessions/`,
and per-user handbooks under `agents/data_agent/DataRetriever_Handbooks/`.
Both are git-ignored.

## Tests

```bash
python -m unittest tests.test_handbook_studio tests.test_validation tests.test_studio_sharing tests.test_output_retention tests.test_token_accounting tests.test_control_prompt tests.test_missing_credentials tests.test_retrieval_repair_ladder
```
