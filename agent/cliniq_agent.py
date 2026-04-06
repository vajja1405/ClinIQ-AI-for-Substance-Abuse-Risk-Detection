import json
import os
import random
import sys
import uuid
from datetime import date, timedelta

import anthropic
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATABASE_URL = os.getenv('DATABASE_URL')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set in .env"); sys.exit(1)
if not ANTHROPIC_API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set in .env"); sys.exit(1)

client      = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
conn        = psycopg2.connect(DATABASE_URL)


# ─────────────────────────────────────────────────────────────────────────────
# RAG RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def get_rag_context(text):
    """Retrieve multi-source RAG context from pgvector knowledge base."""
    vec = embed_model.encode(text).tolist()

    with conn.cursor() as cur:
        # ICD-10 codes
        cur.execute("""
            SELECT source_type, source_document,
                   source_url, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM rag_embeddings
            WHERE source_type = 'icd10'
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        """, (vec, vec))
        icd10 = cur.fetchall()

        # Coding guidelines
        cur.execute("""
            SELECT source_type, source_document,
                   source_url, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM rag_embeddings
            WHERE source_type = 'guideline'
            ORDER BY embedding <=> %s::vector
            LIMIT 3
        """, (vec, vec))
        guidelines = cur.fetchall()

        # DRG weights
        cur.execute("""
            SELECT source_type, source_document,
                   source_url, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM rag_embeddings
            WHERE source_type = 'drg_weight'
            ORDER BY embedding <=> %s::vector
            LIMIT 3
        """, (vec, vec))
        drg = cur.fetchall()

        # Social signals
        cur.execute("""
            SELECT source_type, source_document,
                   source_url, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM rag_embeddings
            WHERE source_type = 'sud_signal'
            ORDER BY embedding <=> %s::vector
            LIMIT 3
        """, (vec, vec))
        signals = cur.fetchall()

    return {
        'icd10': icd10,
        'guidelines': guidelines,
        'drg_weights': drg,
        'social_signals': signals,
    }


def format_context(rag):
    """Format RAG results into readable text for Claude prompt."""
    def fmt(rows):
        return "\n".join([
            f"- {r[3][:200]} [Source: {r[1]}, Similarity: {r[5]:.2f}]"
            for r in rows
        ])
    return {
        'icd10_text':     fmt(rag['icd10']),
        'guideline_text': fmt(rag['guidelines']),
        'drg_text':       fmt(rag['drg_weights']),
        'signal_text':    fmt(rag['social_signals']),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REVENUE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_revenue(missed_codes, rag):
    """Estimate revenue lift from missed SUD codes using retrieved DRG weights."""
    total = 0.0
    for item in missed_codes:
        # A missed CC/MCC drops the MS-DRG to a lower severity tier.
        # For demonstration, we estimate a baseline $2,750 lift per gap.
        lift = 2750.00
        # If we have RAG DRG weights for the patient's condition, 
        # approximate the step-down penalty as 35% of the related DRG baseline.
        if rag.get('drg_weights'):
            meta = rag['drg_weights'][0][4] or {}
            weight = float(meta.get('weight', 0))
            if weight > 0:
                lift = weight * 5500 * 0.35
        total += lift
    return round(total, 2)


# ─────────────────────────────────────────────────────────────────────────────
# CLAIM ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_claim(claim_id):
    """Analyze a single claim for SUD coding gaps using RAG + Claude."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT f.claim_id, f.clinical_note_text,
                   f.primary_dx_code, d.description
            FROM fact_claims f
            JOIN dim_diagnosis d
              ON f.primary_dx_code = d.icd10_code
            WHERE f.claim_id = %s
        """, (claim_id,))
        claim = cur.fetchone()

    if not claim:
        return {"error": "claim not found"}

    note_text    = claim[1]
    current_code = claim[2]
    current_desc = claim[3]

    # retrieve RAG context
    rag = get_rag_context(note_text)
    ctx = format_context(rag)

    # Claude analysis
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system="""You are a senior clinical documentation \
integrity specialist. Identify ICD-10 coding gaps \
using ONLY the evidence retrieved from the knowledge \
base provided. Do not add medical knowledge beyond \
what is in the retrieved context. Every finding must \
cite a specific retrieved source.

Undercoded SUD diagnoses have two consequences:
1. Hospital receives lower Medicare reimbursement
2. Patient disappears from CDC overdose surveillance

Return only valid JSON.""",
        messages=[{"role": "user", "content": f"""
Review this clinical note for SUD coding gaps.

CLINICAL NOTE:
{note_text}

CURRENTLY CODED:
{current_code} — {current_desc}

RELEVANT ICD-10 CODES FROM KNOWLEDGE BASE:
{ctx['icd10_text']}

APPLICABLE CODING GUIDELINES FROM KNOWLEDGE BASE:
{ctx['guideline_text']}

DRG PAYMENT INFORMATION FROM KNOWLEDGE BASE:
{ctx['drg_text']}

SIMILAR PATIENT SOCIAL SIGNALS FROM KNOWLEDGE BASE:
{ctx['signal_text']}

Return ONLY:
{{
  "missed_codes": [{{
    "code": "F11.20",
    "description": "from retrieved ICD-10 content",
    "confidence": 0.94,
    "guideline_reference": "cite specific retrieved guideline",
    "supporting_text": "exact phrase from clinical note",
    "knowledge_base_source": "which retrieved document supports this"
  }}],
  "upcoding_flag": false,
  "doc_gap_summary": "plain English summary",
  "public_health_impact": "how missed code affects CDC surveillance",
  "social_signal_connection": "connection to retrieved social signals",
  "physician_query_letter": "Dear Dr. [Provider], [formal query citing retrieved guideline]",
  "revenue_basis": "cite retrieved DRG source for financial figure"
}}"""}]
    )

    # robust json decode
    import re
    try:
        result = json.loads(response.content[0].text)
    except json.JSONDecodeError:
        result = json.loads(
            re.sub(r'```json|```', '', response.content[0].text).strip()
        )
    revenue = calculate_revenue(result['missed_codes'], rag)

    # generate explainable AI justification
    from explainability import generate_explanation

    evidence_list = [
        {'source_type': r[0], 'source_document': r[1], 'content': r[3]}
        for r in rag['icd10'][:3] + rag['guidelines'][:2]
    ]
    explanation = generate_explanation(
        note_text,
        "GAP_DETECTED" if result['missed_codes'] else "OK",
        result['missed_codes'][0]['confidence']
            if result['missed_codes'] else 1.0,
        evidence_list,
        "RAG+Claude",
        "clinical_note",
    )

    # persist findings
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ai_risk_findings
                (claim_id, missed_codes, upcoding_flag,
                 doc_gap_summary, estimated_revenue_lift,
                 confidence_score, physician_query_letter,
                 rag_evidence, public_health_impact,
                 social_signal_connection)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            claim_id,
            json.dumps(result['missed_codes']),
            result['upcoding_flag'],
            result['doc_gap_summary'],
            revenue,
            result['missed_codes'][0]['confidence']
                if result['missed_codes'] else 0.0,
            result['physician_query_letter'],
            json.dumps({
                'icd10_retrieved':      [r[3][:100] for r in rag['icd10']],
                'guidelines_retrieved': [r[3][:100] for r in rag['guidelines']],
                'drg_retrieved':        [r[3][:100] for r in rag['drg_weights']],
                'signals_retrieved':    [r[3][:100] for r in rag['social_signals']],
                'explanation':          explanation,
                'revenue_basis':        result.get('revenue_basis', ''),
            }),
            result['public_health_impact'],
            result['social_signal_connection'],
        ))
    conn.commit()

    return {
        'claim_id':     claim_id,
        'missed_codes': result['missed_codes'],
        'revenue_lift': revenue,
        'explanation':  explanation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC CLAIM GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_claims(n=100):
    """Generate synthetic clinical claims for demonstration.

    All patients are fictional. No real data used.
    Synthea-style synthetic records for demonstration.
    """
    note_templates = [
        ("J18.9",
         "Patient admitted with community-acquired pneumonia. "
         "Temp 38.9C, WBC 14.2. Chest X-ray right lower lobe "
         "infiltrate. Social history: patient reports {sud_text}. "
         "Currently prescribed {med}. Pulmonology consulted. "
         "Day 2: {complication}. Discharge planning initiated."),
        ("I50.9",
         "Acute decompensated heart failure. BNP 1,847. "
         "3+ pitting edema bilateral lower extremities. "
         "Patient discloses {sud_text}. Receiving {med}. "
         "Cardiology and {consult} medicine consulted. "
         "Volume overloaded, diuresis initiated."),
        ("K70.30",
         "Alcoholic cirrhosis without ascites. LFTs elevated. "
         "Patient reports {sud_text}. "
         "Prescribed {med} for management. "
         "GI and hepatology consulted."),
    ]

    sud_scenarios = [
        ("opioid use disorder, currently on Suboxone 8mg daily "
         "for past 8 months",
         "buprenorphine/naloxone 8mg-2mg daily",
         "patient reports mild withdrawal symptoms, diaphoresis noted",
         "addiction"),
        ("alcohol use disorder, last drink 2 days ago, "
         "patient reports daily drinking prior to admission",
         "chlordiazepoxide per CIWA protocol",
         "CIWA score 12 at hour 8, tremors noted",
         "addiction"),
        ("opioid dependence, active withdrawal on admission day 2, "
         "diaphoresis, tachycardia HR 112, nausea",
         "clonidine 0.1mg for withdrawal symptom management",
         "withdrawal symptoms worsening overnight, methadone consult placed",
         "addiction"),
    ]

    departments = [
        "Internal Medicine", "Hospitalist",
        "Emergency Medicine", "Pulmonology",
    ]

    for i in range(n):
        template_dx, template_note = random.choice(note_templates)
        sud_text, med, complication, consult = random.choice(sud_scenarios)

        note = template_note.format(
            sud_text=sud_text, med=med,
            complication=complication, consult=consult,
        )

        pid   = str(uuid.uuid4())
        prid  = str(uuid.uuid4())
        cid   = str(uuid.uuid4())
        cdate = date(2024, 1, 1) + timedelta(days=random.randint(0, 364))

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO dim_patient
                    (patient_id, age, gender, payer_type)
                VALUES (%s, %s, %s, 'Medicare')
            """, (pid, random.randint(28, 75), random.choice(['M', 'F'])))

            cur.execute("""
                INSERT INTO dim_provider
                    (provider_id, department, specialty)
                VALUES (%s, %s, 'General')
            """, (prid, random.choice(departments)))

            # Ensure the primary diagnosis code is in dim_diagnosis to avoid FK constraint errors
            cur.execute("""
                INSERT INTO dim_diagnosis (icd10_code, description)
                VALUES (%s, 'Synthetic base diagnosis')
                ON CONFLICT (icd10_code) DO NOTHING
            """, (template_dx,))

            cur.execute("""
                INSERT INTO fact_claims
                    (claim_id, patient_id, provider_id,

                     primary_dx_code, drg_code,
                     total_charges, expected_reimbursement,
                     clinical_note_text, claim_date)
                VALUES (%s, %s, %s, %s, 'BASE_DRG',
                        %s, %s, %s, %s)
                ON CONFLICT (claim_id) DO NOTHING
            """, (
                cid, pid, prid, template_dx,
                random.uniform(9000, 28000),
                random.uniform(5000, 9000),
                note, cdate,
            ))
        conn.commit()

    print(f"Generated {n} synthetic claims. "
          f"All patients fictional — no real data used.")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(limit=50):
    """Process unanalyzed claims, generating synthetic data if needed."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT f.claim_id FROM fact_claims f
            LEFT JOIN ai_risk_findings a
              ON f.claim_id = a.claim_id
            WHERE a.claim_id IS NULL
            LIMIT %s
        """, (limit,))
        unprocessed = [str(r[0]) for r in cur.fetchall()]

    if not unprocessed:
        print("No claims found. Generating synthetic claims...")
        generate_synthetic_claims(n=100)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT claim_id FROM fact_claims LIMIT %s",
                (limit,))
            unprocessed = [str(r[0]) for r in cur.fetchall()]

    results = []
    for i, cid in enumerate(unprocessed):
        print(f"Processing {i + 1}/{len(unprocessed)}")
        try:
            r = analyze_claim(cid)
            results.append(r)
        except Exception as e:
            print(f"Error on {cid}: {e}")
            continue

    gaps = sum(1 for r in results if r.get('missed_codes'))
    lift = sum(r.get('revenue_lift', 0) for r in results)
    print(f"\nBatch complete: {len(results)} claims, "
          f"{gaps} gaps, ${lift:,.2f} total lift")
    return results


if __name__ == "__main__":
    run_batch(limit=50)
