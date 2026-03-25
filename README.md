# Book Recommender Agent

An end-to-end LLMOps project built on Databricks that produces an AI agent capable of answering complex natural language questions over a curated personal reading library.

_Course project — LLMOps on Databricks · Work in progress, updated weekly._

---

## What it does

The agent's knowledge base consists of Austin Tripp's personal book summaries — 28+ non-fiction and fiction titles spanning AI safety, philosophy, history, cognitive science, and decision-making — ingested as text, chunked, embedded, and indexed in a Mosaic AI Vector Search index. Users can ask questions like:

- _If I'm thinking about AI safety and human decision-making, which books overlap on that theme and what did the author conclude from each?_
- _What does Gödel Escher Bach say about consciousness, and how does that compare to Superintelligence?_
- _Which books does the author recommend most strongly, and why?_
- _I want to understand human tribalism — what did the author take away from The Righteous Mind?_

The agent uses retrieval-augmented generation (RAG) with tool calling, preserves author opinions verbatim, surfaces reading dates as context, and is fully traced and evaluated through MLflow. Deployment, promotion, and CI/CD are managed through Databricks Asset Bundles across three isolated environments.

---

## Why this matters

Personal reading lists are dense, interconnected, and hard to query. The same reader who absorbed _Sapiens_, _Guns Germs & Steel_, and _The Righteous Mind_ holds implicit cross-book insights — but surfacing them requires remembering which book said what. This agent replaces that mental overhead by grounding every answer in the actual summary text, with source attribution and author opinion intact.

---

## Tech stack

| Layer | Technology |
|---|---|
| Platform | Databricks (Serverless Compute v4) |
| Storage & Governance | Unity Catalog — dev, stg, prd |
| Deployment | Databricks Asset Bundles (DABs) |
| Local Development | Databricks Connect + VS Code Databricks extension |
| Package Manager | UV |
| Python Packaging | `src/` layout, `pyproject.toml`, built as `.whl` |
| Vector Search | Mosaic AI Vector Search |
| Model Serving | Mosaic AI Model Serving |
| Agent Memory | Lakebase (psycopg) |
| Experiment Tracking | MLflow (tracing, evaluation, prompt registry) |
| Logging | loguru |

---

## Unity Catalog structure

The same schema and volume structure is replicated across all three environments. Book summaries (`.md` or `.txt` files) must be uploaded to the volume before running the ingestion notebook.

```
{dev|stg|prd}
└── book_recommender                  ← schema
    └── summaries                     ← volume (book summary files)
        ├── godel_escher_bach.md
        ├── sapiens.md
        ├── superintelligence.md
        ├── deep_work.md
        ├── practical_ethics.md
        ├── guns_germs_steel.md
        ├── the_righteous_mind.md
        └── ...                       ← 28+ titles total
```

Environment promotion flow managed by DABs CI/CD (built in Week 6):

```
dev  ──►  stg  ──►  prd
```

---

## Project structure

```
llmops-databricks-course-iRahulPandey/
├── notebooks/
│   └── 1_book_summaries_ingestion.py   # Week 1: reads summaries → raw_documents Delta table
├── src/
│   └── book_recommender/               # Python package
│       ├── __init__.py
│       └── config.py                   # Pydantic config + env resolution
├── resources/
│   └── book_recommender_ingestion_job.yml  # DABs job definition
├── tests/
│   └── test_config.py                  # ProjectConfig unit tests (7 passing)
├── project_config.yml                  # Per-environment config (catalog, schema, endpoints)
├── databricks.yml                      # DABs bundle definition (dev / stg / prd targets)
├── pyproject.toml                      # Dependencies + build config
└── version.txt
```

---

## Weekly roadmap

| Week | Deliverable | Status |
|---|---|---|
| 1 | Environment setup · Book summary ingestion into Delta tables (`raw_documents`) | ✅ Done |
| 2 | Chunking · Embeddings · Vector Search index · Genie Space | ⬜ Planned |
| 3 | Agent definition · Tool calling · Memory with Lakebase | ⬜ Planned |
| 4 | MLflow tracing · Evaluation · Prompt optimisation | ⬜ Planned |
| 5 | Agent deployment · Monitoring and observability | ⬜ Planned |
| 6 | CI/CD pipeline via DABs · Promotion dev → stg → prd | ⬜ Planned |

---

## Setup

### Prerequisites

- Python 3.12
- [UV](https://docs.astral.sh/uv/getting-started/installation/) — `pip install uv`
- Databricks CLI v0.200+
- VS Code Databricks extension

### 1. Clone and install

```bash
git clone https://github.com/iRahulPandey/llmops-databricks-course-iRahulPandey.git
cd llmops-databricks-course-iRahulPandey

uv sync --extra dev
```

### 2. Authenticate with Databricks

```bash
databricks configure --host https://<your-workspace-url>
```

Or use the VS Code Databricks extension to sign in — it will configure `databricks-connect` automatically.

### 3. Configure local environment

Fill in the `warehouse_id` and `vector_search_endpoint` fields in `project_config.yml` and verify `databricks.yml` points to your workspace hosts for `stg` and `prd`.

### 4. Create Unity Catalog objects

```sql
-- Dev
CREATE CATALOG IF NOT EXISTS dev;
CREATE SCHEMA IF NOT EXISTS dev.book_recommender;
CREATE VOLUME IF NOT EXISTS dev.book_recommender.summaries;

-- Stg
CREATE CATALOG IF NOT EXISTS stg;
CREATE SCHEMA IF NOT EXISTS stg.book_recommender;
CREATE VOLUME IF NOT EXISTS stg.book_recommender.summaries;

-- Prd
CREATE CATALOG IF NOT EXISTS prd;
CREATE SCHEMA IF NOT EXISTS prd.book_recommender;
CREATE VOLUME IF NOT EXISTS prd.book_recommender.summaries;
```

Then upload your book summary files to each volume:

```
/Volumes/dev/book_recommender/summaries/
/Volumes/stg/book_recommender/summaries/
/Volumes/prd/book_recommender/summaries/
```

### 5. Deploy the bundle

```bash
# Deploy to dev (default)
databricks bundle deploy

# Deploy to a specific target
databricks bundle deploy --target stg
```

### 6. Run the ingestion job

```bash
# Run on dev (default)
databricks bundle run book_recommender_ingestion_job

# Run on a specific target
databricks bundle run book_recommender_ingestion_job --target stg
```

The run output and logs stream directly to your terminal. You can also monitor the run in the Databricks Jobs UI.

---

## Development

### Linting and formatting

Ruff handles both linting and formatting. It is included in the dev dependencies.

```bash
# Install dev dependencies (includes ruff)
uv sync --extra dev

# Lint and auto-fix
uv run ruff check . --fix

# Format
uv run ruff format .

# Both in one go
uv run ruff check . --fix && uv run ruff format .
```

### Pre-commit hooks

```bash
# One-time setup — installs the git hooks
uv run pre-commit install

# Run manually against all files
uv run pre-commit run --all-files
```

Once installed, hooks run automatically. If a hook fails or auto-fixes a file, the commit is blocked — `git add` the fixed files and retry.

### Running tests

```bash
uv sync --extra ci
uv run pytest
```

### Git conventions

| Convention | Format |
|---|---|
| Branch | `week{n}/{short-description}` — e.g. `week1/databricks-setup-and-ingestion` |
| PR title | `[Week N] Description` — e.g. `[Week 1] Book summary ingestion into Delta tables` |
| Direct commits to `main` | Never |

All changes go through a PR. `main` always reflects the latest stable weekly deliverable.

---

## Configuration

`project_config.yml` drives all environment-specific values. The active section is resolved at runtime from the `env` Databricks widget (set by the bundle target), falling back to `dev` for local development via `databricks-connect`.

```yaml
dev:
  catalog: dev
  schema_name: book_recommender
  volume: summaries
  llm_endpoint: databricks-llama-4-maverick
  embedding_endpoint: databricks-gte-large-en
  warehouse_id: ""         # fill in after first Databricks login
  vector_search_endpoint: ""   # fill in after creating VS endpoint
  ...

stg:
  catalog: stg
  ...

prd:
  catalog: prd
  ...
```

---

## Author

Rahul Pandey · [github.com/iRahulPandey](https://github.com/iRahulPandey)
