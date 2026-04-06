# ClinIQ — AI for Substance Abuse Risk Detection

**NSF NRT Research-A-Thon 2026 | UMKC | Challenge 1 Track A**

> ClinIQ is a RAG-powered agentic AI system that detects SUD risk signals from patient-written social text, uncovers behavioral patterns across the 2008–2017 opioid crisis arc, and traces those signals into clinical documentation gaps — built entirely on real published government sources with full auditability.

---

## What ClinIQ Does

| Layer | What it does |
|-------|-------------|
| **Social Signal Detection** | Classifies 52,184 anonymized patient drug reviews for SUD risk using three methods: rule-based, embedding-based, and LLM+RAG |
| **Temporal Analysis** | Tracks SUD signal volume, patient distress, and substance composition across 2008–2017 opioid crisis arc |
| **Behavioral Clustering** | UMAP + HDBSCAN reveals distinct SUD behavioral subpopulations in the patient population |
| **Clinical Bridge** | Identifies missed ICD-10 SUD codes in hospital billing; quantifies revenue and surveillance impact |
| **Interactive Dashboard** | Streamlit app with 5 panels — live RAG retrieval, charts, and physician query generation |

---

## Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 13+ with pgvector extension
- Anthropic API key

### Setup

```bash
# 1. Clone and enter directory
git clone <repo-url>
cd cliniq

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.template .env
# Edit .env: set DATABASE_URL and ANTHROPIC_API_KEY

# 5. Run full pipeline
python run_pipeline.py

# 6. Launch dashboard
streamlit run streamlit_app/app.py
```

### Environment Variables

```
DATABASE_URL=postgresql://username:password@localhost:5432/cliniq
ANTHROPIC_API_KEY=your_anthropic_api_key
```

---

## Project Structure

```
cliniq/
├── data/
│   ├── load_reviews.py              # Load & classify Kaggle drug reviews
│   ├── load_public_health_data.py   # Download CDC, CMS, ICD-10 data
│   ├── drugsComTest_raw.csv         # Kaggle test set (27 MB)
│   └── drugsComTrain_raw.csv        # Kaggle train set (82 MB)
│
├── db/
│   ├── schema.sql                   # PostgreSQL schema (11 tables + pgvector)
│   └── setup_db.py                  # Database initialization script
│
├── agent/
│   ├── build_rag.py                 # Embed government sources into pgvector
│   ├── cliniq_agent.py              # Clinical claim analyzer (RAG + Claude)
│   └── explainability.py            # XAI justification generator
│
├── analysis/
│   ├── task1_signal_detection.py    # Three-method SUD detection comparison
│   └── task2_temporal_behavioral.py # Temporal trends + UMAP/HDBSCAN clustering
│
├── streamlit_app/
│   └── app.py                       # Interactive 5-panel Streamlit dashboard
│
├── raw_sources/                     # Downloaded government documents (auditable)
│   ├── cdc_overdose_raw.json        # CDC Drug Overdose Surveillance API data
│   ├── sud_icd10_codes.csv          # ICD-10-CM SUD codes with CC/MCC status
│   ├── drg_weights_cleaned.csv      # CMS FY2024 DRG relative weights
│   ├── cms_coding_guidelines_fy2024.pdf  # CMS ICD-10-CM Guidelines PDF
│   └── nida_page.html               # NIDA Trends & Statistics page
│
├── outputs/                         # Generated analysis results
│   ├── method_comparison_results.csv
│   ├── temporal_trends.csv
│   ├── sud_clusters.csv
│   ├── substance_trends.csv
│   ├── narrative_shifts.json
│   └── nida_reference_stats.json
│
├── report/
│   └── ClinIQ_NSF_NRT_Report_2026.md   # 4-page submission report
│
├── run_pipeline.py                  # Master orchestration script
├── requirements.txt                 # Python dependencies
└── .env.template                    # Environment variable template
```

---

## Data Sources

All knowledge base content is sourced from real, publicly available government documents. Every embedding in `rag_embeddings` carries a `source_url` and `source_document` field — **zero hardcoded medical knowledge**.

| Source | URL | What it provides |
|--------|-----|-----------------|
| Kaggle UCI Drug Reviews | kaggle.com/datasets/jessicali9530 | 215k patient reviews, 2008–2017 |
| CDC Overdose Surveillance | data.cdc.gov/resource/95ax-ymtc.json | Annual overdose death rates by state |
| ICD-10-CM 2024 | ftp.cdc.gov/pub/Health_Statistics/NCHS | Official SUD diagnosis codes F10–F19, T40, T43 |
| CMS IPPS FY2024 | cms.gov (IPPS Final Rule) | DRG relative weights and payment rates |
| CMS Coding Guidelines | cms.gov/files/document | Official ICD-10-CM coding rules Section I.C.5 |
| NIDA Statistics | nida.nih.gov/research-topics/trends-statistics | Population-level SUD statistics |

---

## Pipeline Overview

```
run_pipeline.py orchestrates:

1. db/setup_db.py                      → Create 11 PostgreSQL tables
2. data/load_reviews.py                → Ingest 52,184 Kaggle reviews
3. data/load_public_health_data.py     → Download CDC/CMS/ICD-10 data
4. agent/build_rag.py                  → Build pgvector knowledge base
5. analysis/task1_signal_detection.py  → Compare 3 detection methods
6. analysis/task2_temporal_behavioral.py → Temporal analysis + clustering
7. agent/cliniq_agent.py               → Clinical gap detection
```

---

## Detection Methods

### Method 1: Rule-Based
- Keywords extracted at runtime from `dim_diagnosis` ICD-10 descriptions
- Augmented with patient-voice equivalents (slang, indirect references)
- **F1 = 0.859 | Precision = 0.867 | Recall = 0.850**
- Best for: balanced operational deployment, audit trails

### Method 2: Embedding-Based
- Cosine similarity against real patient SUD reviews in RAG knowledge base
- Threshold: 0.32 cosine similarity
- **F1 = 0.670 | Precision = 0.504 | Recall = 1.000**
- Best for: population screening (zero false negatives)

### Method 3: LLM + RAG (Claude)
- pgvector retrieval of top-3 social signals + top-2 ICD-10 codes per review
- Claude Haiku with retrieved context → structured JSON output
- **F1 = 0.477 | Precision = 0.883 | Recall = 0.327**
- Best for: clinical documentation audit (highest precision, explains reasoning)

---

## Database Schema

```sql
drug_reviews        -- 52,184 Kaggle reviews with SUD classification
cdc_overdose        -- CDC mortality data by year/state/substance
rag_embeddings      -- pgvector knowledge base (384-dim, IVFFlat indexed)
rag_source_registry -- Audit trail: every knowledge source registered
dim_diagnosis       -- ICD-10-CM codes with CC/MCC status
dim_patient         -- Synthetic patient demographics
dim_provider        -- Hospital department/specialty
fact_claims         -- Synthetic clinical claims
ai_risk_findings    -- Detected coding gaps + revenue impact
method_comparison   -- Task 1 evaluation results
temporal_analysis   -- Task 2 year-level SUD metrics
```

---

## Key Findings

- **Volume surge:** SUD review volume increases **3×** from 2014 to 2016, tracking the CDC-documented fentanyl influx
- **Distress escalation:** Patient distress proportion rises **18×** from 1.7% (2008) to 30.5% (2017)
- **Composition shift:** Opioid-specific proportion falls 40% → 16% while total distress rises — the crisis diversified beyond opioids
- **Surveillance gap:** Missed ICD-10 CC/MCC codes (F11.23 opioid withdrawal = MCC) cause DRG tier downgrades and remove patients from CDC monitoring
- **Revenue impact:** ~$2,750–$6,000 per missed SUD comorbidity; extrapolates to $27.5M–$60M annually at a 10k-admission hospital

---

## Ethical AI Principles

- **No individual identification:** All reviews are anonymized; all clinical data is synthetic
- **Population-level only:** Every analysis is at the population level
- **Full auditability:** Every knowledge base item traces to a government URL
- **Human-in-the-loop:** All findings require human analyst review before action
- **Public data only:** CDC, CMS, ICD-10-CM — no proprietary clinical data

---

## Requirements

```
pandas, numpy, psycopg2-binary, pgvector, sqlalchemy
anthropic, sentence-transformers, scikit-learn
hdbscan, umap-learn, streamlit, plotly
python-dotenv, requests, beautifulsoup4
scipy, pdfplumber, openpyxl, lxml, tqdm
```

---

## Citation

If you use ClinIQ in your research:

```
ClinIQ: RAG-Powered AI for Substance Abuse Risk Detection from Social Signals
NSF NRT Research-A-Thon 2026, UMKC
Challenge 1 Track A — AI Modeling and Reasoning
```

---

**NSF NRT Program | PI: Dr. Mostafizur Rahman | Co-PI: Dr. Yugyung Lee**
