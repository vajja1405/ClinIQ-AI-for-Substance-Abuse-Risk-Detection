# ClinIQ — RAG-Powered AI for Substance Abuse Risk Detection

<p align="center">
  <img src="report/ClinIQ_LinkedIn.gif" alt="ClinIQ Demo" width="600"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Claude%20API-claude--haiku-orange?logo=anthropic&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-pgvector-336791?logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/NSF%20NRT-4th%20Place%202026-gold" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

> **4th Place — NSF NRT Research-A-Thon 2026 · UMKC · Challenge 1 Track A: AI Modeling and Reasoning**

ClinIQ is a production-architecture RAG + agentic AI system that:
1. Detects substance use disorder (SUD) risk signals in 52,184 anonymized patient drug reviews using three independent ML methods
2. Uncovers behavioral subpopulations and the 2008–2017 opioid crisis trajectory via UMAP + HDBSCAN clustering
3. Traces social signals into hospital billing gaps — quantifying missed ICD-10 CC/MCC codes and their $27.5M–$60M annual revenue impact

**Every fact in the knowledge base traces to a real government URL. Zero hardcoded medical knowledge.**

---

## Why This Project Stands Out

| Signal | What ClinIQ demonstrates |
|--------|--------------------------|
| **RAG at scale** | pgvector IVFFlat index over 384-dim sentence embeddings; all-MiniLM-L6-v2 + Claude Haiku for multi-step reasoning |
| **3-method ML comparison** | Rule-based vs embedding cosine vs LLM+RAG — full precision/recall/F1 evaluation with a 600-record stratified hold-out |
| **Unsupervised discovery** | UMAP dimensionality reduction + HDBSCAN density clustering reveals clinically meaningful patient subpopulations |
| **Full-stack delivery** | PostgreSQL schema (11 tables), Streamlit dashboard, reproducible Docker setup, CI via GitHub Actions |
| **Domain impact** | Bridges NLP outputs to hospital revenue cycle — a gap most academic ML projects never attempt |
| **Ethical AI** | Population-level only, anonymized data, full source auditability, human-in-the-loop design |

---

## Architecture

```
Patient Reviews (52,184)
        │
        ▼
┌───────────────────────────────────────────────────┐
│              Signal Detection Layer                │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  │
│  │  Rule-Based  │  │  Embedding   │  │LLM+RAG  │  │
│  │ F1=0.859     │  │ Recall=1.000 │  │Prec=0.88│  │
│  └──────────────┘  └──────────────┘  └─────────┘  │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│           Temporal + Behavioral Analysis           │
│  Year-over-year trends  │  UMAP + HDBSCAN clusters │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│           Clinical Bridge (RAG Agent)              │
│  ICD-10 gap detection  │  DRG revenue impact       │
│  pgvector retrieval    │  Claude Haiku reasoning    │
└───────────────────────────────────────────────────┘
        │
        ▼
     Streamlit Dashboard (5 panels, live RAG)
```

---

## Key Results

### Detection Performance (600-record stratified evaluation set)

| Method | Precision | Recall | F1 | Best Use Case |
|--------|-----------|--------|----|---------------|
| Rule-Based (ICD-10 vocab) | 0.867 | 0.850 | **0.859** | Operational deployment, auditability |
| Embedding (cosine ≥ 0.32) | 0.504 | **1.000** | 0.670 | Population screening (zero false negatives) |
| LLM + RAG (Claude Haiku) | **0.883** | 0.327 | 0.477 | Clinical audit (highest precision + reasoning) |

### Temporal Findings (2008–2017 opioid crisis arc)

- **3× volume surge**: SUD review volume from 2014→2016 tracks the CDC-documented fentanyl influx
- **18× distress escalation**: Patient distress proportion rose from 1.7% (2008) to 30.5% (2017)
- **Composition shift**: Opioid-specific proportion fell 40%→16% while total distress rose — the crisis diversified beyond opioids

### Clinical / Revenue Impact

- Missed ICD-10 MCC/CC codes (e.g. F11.23 opioid withdrawal = MCC, F10.230 alcohol withdrawal = MCC) cause DRG tier downgrades
- **$2,750–$6,000 per missed comorbidity** (CMS FY2024 DRG weights)
- **$27.5M–$60M annually** at a 10,000-admission hospital

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Anthropic Claude Haiku (via API) |
| **Embeddings** | `all-MiniLM-L6-v2` (384-dim, sentence-transformers) |
| **Vector DB** | PostgreSQL 16 + pgvector (IVFFlat index, cosine distance) |
| **Clustering** | UMAP + HDBSCAN |
| **Data** | pandas, scikit-learn, scipy |
| **Visualization** | Plotly, Streamlit |
| **Infrastructure** | Docker Compose, GitHub Actions CI |
| **Data Sources** | CDC API, CMS IPPS FY2024, ICD-10-CM, NIDA, Kaggle |

---

## Quick Start

### Option A — Docker (Recommended, zero configuration)

```bash
git clone https://github.com/rahulchowdary/cliniq.git
cd cliniq
cp .env.template .env          # add your ANTHROPIC_API_KEY
docker compose up --build      # starts PostgreSQL + pgvector + Streamlit app
```

Open `http://localhost:8501`

### Option B — Local

```bash
git clone https://github.com/rahulchowdary/cliniq.git
cd cliniq
make setup                     # creates venv + installs deps + copies .env
# Edit .env: set DATABASE_URL and ANTHROPIC_API_KEY
make db-docker                 # starts PostgreSQL with pgvector via Docker
make db-init                   # loads schema, data, and builds RAG index
make pipeline                  # runs full analysis (Task 1 + Task 2 + Agent)
make dashboard                 # launches Streamlit on http://localhost:8501
```

### Environment Variables

```
DATABASE_URL=postgresql://cliniq:cliniq@localhost:5432/cliniq
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Project Structure

```
cliniq/
├── data/
│   ├── load_reviews.py              # Load & classify 52,184 Kaggle drug reviews
│   └── load_public_health_data.py   # Download CDC, CMS, ICD-10 sources
│
├── db/
│   ├── schema.sql                   # 11-table PostgreSQL schema + pgvector
│   └── setup_db.py                  # Database initialization
│
├── agent/
│   ├── build_rag.py                 # Embed government docs into pgvector
│   ├── cliniq_agent.py              # Clinical gap detector (RAG + Claude)
│   └── explainability.py            # XAI justification generator
│
├── analysis/
│   ├── task1_signal_detection.py    # 3-method SUD detection + evaluation
│   └── task2_temporal_behavioral.py # Temporal trends + UMAP/HDBSCAN
│
├── streamlit_app/
│   └── app.py                       # Interactive 5-panel dashboard
│
├── tests/
│   └── test_detection.py            # Unit tests (no DB/API required)
│
├── raw_sources/                     # Auditable government documents
├── outputs/                         # Generated CSVs and JSON results
├── docker-compose.yml               # PostgreSQL + app in one command
├── Dockerfile
├── Makefile                         # make setup / run / test / dashboard
├── run_pipeline.py                  # Master orchestration script
└── requirements.txt                 # Pinned dependencies
```

---

## Database Schema

```sql
drug_reviews        -- 52,184 patient reviews with SUD labels + signal category
cdc_overdose        -- CDC mortality data by year/state/substance
rag_embeddings      -- pgvector knowledge base (384-dim, IVFFlat, cosine)
rag_source_registry -- Audit trail: every source URL + document registered
dim_diagnosis       -- ICD-10-CM codes with CC/MCC billing status
dim_patient         -- Synthetic patient demographics
dim_provider        -- Hospital department/specialty lookup
fact_claims         -- Synthetic clinical claims
ai_risk_findings    -- Detected coding gaps + revenue impact per claim
method_comparison   -- Task 1 precision/recall/F1 by method
temporal_analysis   -- Task 2 year-level SUD volume + distress metrics
```

---

## Data Sources

All knowledge base content sourced from real, publicly available government documents. Every `rag_embeddings` row carries `source_url` + `source_document`.

| Source | What it provides |
|--------|-----------------|
| Kaggle UCI Drug Reviews | 215k patient reviews, 2008–2017 |
| CDC Drug Overdose Surveillance API | Annual overdose death rates by state |
| ICD-10-CM 2024 (CDC FTP) | Official SUD codes F10–F19, T40, T43 with CC/MCC status |
| CMS IPPS FY2024 Final Rule | DRG relative weights and payment rates |
| CMS ICD-10-CM Coding Guidelines | Official Section I.C.5 substance use rules |
| NIDA Trends & Statistics | Population-level SUD prevalence data |

---

## Running Tests

```bash
make test
# or
pytest tests/ -v
```

Tests are self-contained — no database or API key needed.

---

## Ethical AI Principles

- **No individual identification** — all reviews are anonymized; clinical data is synthetic
- **Population-level only** — every analysis is aggregated across cohorts
- **Full auditability** — every knowledge base chunk traces to a government URL
- **Human-in-the-loop** — all AI findings require analyst review before action
- **Public data only** — CDC, CMS, ICD-10-CM; no proprietary clinical records

---

## Team

| Name | Role | Email |
|------|------|-------|
| Rahul Chowdary Vajja | RAG pipeline, clinical bridge, database, dashboard | rcvtk3@umsystem.edu |
| Bhavani Adula | Temporal analysis, behavioral clustering, ethics framework | barh3@umsystem.edu |

**Faculty Advisors:** Dr. Mostafizur Rahman · Dr. Yugyung Lee  
**Program:** NSF NRT — AI for Health Informatics, UMKC

---

## Citation

```bibtex
@misc{cliniq2026,
  title   = {ClinIQ: RAG-Powered AI for Substance Abuse Risk Detection from Social Signals},
  author  = {Vajja, Rahul Chowdary and Adula, Bhavani},
  year    = {2026},
  note    = {NSF NRT Research-A-Thon 2026, UMKC — 4th Place, Challenge 1 Track A},
}
```
