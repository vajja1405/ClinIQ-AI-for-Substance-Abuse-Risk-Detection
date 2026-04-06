CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- PRIMARY LAYER: Social signal data from Kaggle
CREATE TABLE IF NOT EXISTS drug_reviews (
  review_id INT PRIMARY KEY,
  drug_name VARCHAR(200),
  condition VARCHAR(200),
  review_text TEXT,
  rating INT,
  review_date DATE,
  useful_count INT,
  is_sud_relevant BOOLEAN DEFAULT FALSE,
  signal_category VARCHAR(50),
  detected_signals JSONB,
  behavioral_cluster INT,
  umap_x FLOAT,
  umap_y FLOAT,
  risk_classification VARCHAR(20),
  confidence_score FLOAT,
  explanation JSONB,
  review_year INT
);

-- POPULATION CONTEXT
CREATE TABLE IF NOT EXISTS cdc_overdose (
  record_id SERIAL PRIMARY KEY,
  year INT,
  state VARCHAR(50),
  substance_type VARCHAR(100),
  death_count INT,
  rate_per_100k FLOAT
);

-- RAG KNOWLEDGE BASE
-- Every row in this table comes from a real downloaded
-- government document stored in raw_sources/
-- source_url field makes every embedding fully auditable
CREATE TABLE IF NOT EXISTS rag_embeddings (
  embedding_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  source_type VARCHAR(50),
  source_document VARCHAR(200),
  source_url TEXT,
  source_page_or_row VARCHAR(100),
  content TEXT,
  metadata JSONB,
  embedding vector(384)
);

-- RAG SOURCE REGISTRY
-- Tracks every document downloaded for the knowledge base
CREATE TABLE IF NOT EXISTS rag_source_registry (
  source_id SERIAL PRIMARY KEY,
  document_name VARCHAR(200),
  source_url TEXT,
  local_path VARCHAR(300),
  downloaded_at TIMESTAMP DEFAULT NOW(),
  chunk_count INT,
  description TEXT
);

-- DIMENSION TABLES
CREATE TABLE IF NOT EXISTS dim_patient (
  patient_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  age INT, gender VARCHAR(20), payer_type VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS dim_provider (
  provider_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  department VARCHAR(100), specialty VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS dim_diagnosis (
  icd10_code VARCHAR(20) PRIMARY KEY,
  description TEXT,
  cc_flag BOOLEAN DEFAULT FALSE,
  mcc_flag BOOLEAN DEFAULT FALSE,
  drg_weight FLOAT,
  substance_type VARCHAR(50),
  official_source VARCHAR(200)
);

-- FACT TABLE
CREATE TABLE IF NOT EXISTS fact_claims (
  claim_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  patient_id UUID REFERENCES dim_patient(patient_id),
  provider_id UUID REFERENCES dim_provider(provider_id),
  primary_dx_code VARCHAR(20)
    REFERENCES dim_diagnosis(icd10_code),
  drg_code VARCHAR(20),
  drg_weight_billed FLOAT,
  total_charges DECIMAL(12,2),
  expected_reimbursement DECIMAL(12,2),
  denial_risk_score FLOAT,
  clinical_note_text TEXT,
  claim_date DATE
);

-- AI FINDINGS
CREATE TABLE IF NOT EXISTS ai_risk_findings (
  finding_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  claim_id UUID REFERENCES fact_claims(claim_id),
  missed_codes JSONB,
  upcoding_flag BOOLEAN DEFAULT FALSE,
  doc_gap_summary TEXT,
  estimated_revenue_lift DECIMAL(12,2),
  confidence_score FLOAT,
  physician_query_letter TEXT,
  rag_evidence JSONB,
  public_health_impact TEXT,
  social_signal_connection TEXT,
  processed_at TIMESTAMP DEFAULT NOW()
);

-- ANALYSIS RESULTS
CREATE TABLE IF NOT EXISTS method_comparison (
  run_id SERIAL PRIMARY KEY,
  method VARCHAR(50),
  precision_score FLOAT,
  recall_score FLOAT,
  f1_score FLOAT,
  run_timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS temporal_analysis (
  year INT PRIMARY KEY,
  sud_review_count INT,
  avg_rating FLOAT,
  distress_proportion FLOAT,
  opioid_proportion FLOAT,
  cdc_overdose_rate FLOAT,
  centroid_distance_from_prev FLOAT,
  is_narrative_shift_year BOOLEAN DEFAULT FALSE,
  narrative_shift_summary TEXT
);
