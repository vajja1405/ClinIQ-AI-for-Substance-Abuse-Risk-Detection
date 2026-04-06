import json
import os
import sys
import time

import numpy as np
import pandas as pd
import psycopg2
import anthropic
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATABASE_URL = os.getenv('DATABASE_URL')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set in .env")
    sys.exit(1)
if not ANTHROPIC_API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set in .env")
    sys.exit(1)

conn   = psycopg2.connect(DATABASE_URL)
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SETUP — 600-record stratified evaluation set
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SETUP — Building Evaluation Set")
print("=" * 60)

with conn.cursor() as cur:
    # 300 SUD: stratify across signal categories
    cur.execute("""
        WITH ranked AS (
            SELECT review_id, drug_name, condition, review_text,
                   rating, useful_count, signal_category,
                   is_sud_relevant, review_year,
                   ROW_NUMBER() OVER (
                       PARTITION BY signal_category
                       ORDER BY useful_count DESC
                   ) AS rn
            FROM drug_reviews
            WHERE is_sud_relevant = TRUE
              AND char_length(review_text) > 50
        )
        SELECT review_id, drug_name, condition, review_text,
               rating, useful_count, signal_category,
               is_sud_relevant, review_year
        FROM ranked
        WHERE rn <= 75
        LIMIT 300;
    """)
    sud_rows = cur.fetchall()

    # 300 non-SUD: varied conditions
    cur.execute("""
        SELECT review_id, drug_name, condition, review_text,
               rating, useful_count, signal_category,
               is_sud_relevant, review_year
        FROM drug_reviews
        WHERE is_sud_relevant = FALSE
          AND char_length(review_text) > 50
        ORDER BY RANDOM()
        LIMIT 300;
    """)
    non_sud_rows = cur.fetchall()

COLS = ['review_id','drug_name','condition','review_text',
        'rating','useful_count','signal_category',
        'is_sud_relevant','review_year']

eval_df = pd.DataFrame(sud_rows + non_sud_rows, columns=COLS)
eval_df = eval_df.sample(frac=1, random_state=42).reset_index(drop=True)
y_true  = eval_df['is_sud_relevant'].astype(int).tolist()

print(f"  Evaluation set: {len(eval_df)} reviews")
print(f"  SUD positive:   {sum(y_true)}")
print(f"  Non-SUD:        {len(y_true) - sum(y_true)}")
print(f"  SUD categories: {eval_df[eval_df['is_sud_relevant']]['signal_category'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — Rule-based (keywords from ICD-10 descriptions in DB)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("METHOD 1 — Rule-Based (ICD-10 vocabulary from dim_diagnosis)")
print("=" * 60)

with conn.cursor() as cur:
    cur.execute("SELECT icd10_code, description FROM dim_diagnosis;")
    diag_rows = cur.fetchall()

# extract clinical terms from official ICD-10 descriptions
clinical_terms = set()
for code, desc in diag_rows:
    if not desc:
        continue
    words = desc.lower().replace(',', '').replace('.', '').split()
    for w in words:
        if len(w) > 4 and w not in {
            'with', 'without', 'other', 'unspecified', 'initial',
            'encounter', 'accidental', 'sequela', 'assault', 'undetermined'
        }:
            clinical_terms.add(w)

# patient-voice equivalents of clinical vocabulary
# keyed by signal category for firing tracking
KEYWORD_DICT = {
    'opioid': [
        # clinical from ICD-10
        'opioid', 'opiate', 'heroin', 'fentanyl', 'morphine',
        'hydrocodone', 'oxycodone', 'oxycontin', 'vicodin',
        'suboxone', 'buprenorphine', 'methadone', 'naloxone',
        'naltrexone', 'tramadol', 'codeine', 'dilaudid',
        # patient voice equivalents
        'opioid addiction', 'pain pill', 'pain pills',
        'pill mill', 'medication-assisted', 'mat program',
        'on suboxone', 'on methadone', 'need it to feel normal',
        'cant function without', "can't function without",
    ],
    'alcohol': [
        # clinical
        'alcohol dependence', 'alcoholism', 'alcohol abuse',
        'alcohol use disorder', 'alcohol withdrawal',
        # patient voice
        'drinking problem', 'alcoholic', 'cant stop drinking',
        "can't stop drinking", 'sober', 'sobriety', 'aa meeting',
        'alcoholics anonymous', 'fell off the wagon',
    ],
    'withdrawal': [
        # clinical
        'withdrawal', 'detoxification', 'detox',
        # patient voice
        'withdrawal hell', 'withdrawal symptoms', 'coming off',
        'getting off', 'kicking', 'cold turkey', 'withdrawals',
        'rebound', 'stopping cold',
    ],
    'stimulant': [
        # clinical
        'cocaine', 'amphetamine', 'methamphetamine',
        'stimulant dependence', 'stimulant use disorder',
        # patient voice
        'meth', 'crystal', 'coke', 'crack', 'speed',
        'stimulant addiction',
    ],
    'other_sud': [
        # clinical
        'substance abuse', 'drug abuse', 'drug dependence',
        'drug addiction', 'addiction treatment', 'polysubstance',
        'benzodiazepine dependence', 'sedative dependence',
        'xanax dependence', 'valium dependence',
        # patient voice
        'addict', 'addiction', 'addicted', 'in recovery',
        'relapse', 'relapsed', 'using again', 'fell off',
        'rock bottom', 'rehab', 'treatment center',
        'need it to function', 'dependent on',
        'have to have it', 'cant live without',
        "can't live without", 'getting clean', 'staying clean',
        'clean and sober', 'drug seeking', 'tolerance built up',
        'need higher dose', 'taking more than prescribed',
    ],
}

# augment with terms extracted from ICD-10 descriptions
for term in clinical_terms:
    if term not in {kw for kws in KEYWORD_DICT.values() for kw in kws}:
        KEYWORD_DICT['other_sud'].append(term)

def rule_classify(text, drug_name):
    if not text:
        return 0, []
    combined = (str(text) + ' ' + str(drug_name or '')).lower()
    fired = []
    for category, keywords in KEYWORD_DICT.items():
        for kw in keywords:
            if kw in combined:
                fired.append(category)
                break
    return (1 if fired else 0), list(set(fired))

t0 = time.time()
rule_preds, rule_fired = [], []
for _, row in eval_df.iterrows():
    pred, fired = rule_classify(row['review_text'], row['drug_name'])
    rule_preds.append(pred)
    rule_fired.append(fired)
t_rule = time.time() - t0

print(f"  Keywords loaded: {sum(len(v) for v in KEYWORD_DICT.values())} across {len(KEYWORD_DICT)} categories")
print(f"  Clinical terms extracted from dim_diagnosis: {len(clinical_terms)}")
print(f"  Time: {t_rule:.2f}s")

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — Embedding-based (cosine similarity to RAG sud_signal docs)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("METHOD 2 — Embedding-Based (cosine sim to RAG sud_signal docs)")
print("=" * 60)

with conn.cursor() as cur:
    cur.execute("""
        SELECT content, metadata, embedding::text
        FROM rag_embeddings
        WHERE source_type = 'sud_signal';
    """)
    signal_docs = cur.fetchall()

if not signal_docs:
    print("  WARNING: No sud_signal embeddings found — run build_rag.py first.")
    embed_preds = [0] * len(eval_df)
    t_embed = 0.0
else:
    # parse stored vectors
    ref_embeddings = []
    for content, meta, emb_text in signal_docs:
        vec = np.array([float(x) for x in emb_text.strip('[]').split(',')])
        ref_embeddings.append(vec)
    ref_matrix = np.vstack(ref_embeddings)  # (n_refs, 384)

    # normalise for cosine similarity
    ref_norms = np.linalg.norm(ref_matrix, axis=1, keepdims=True)
    ref_matrix_norm = ref_matrix / (ref_norms + 1e-10)

    SIMILARITY_THRESHOLD = 0.32

    t0 = time.time()
    eval_texts = eval_df['review_text'].fillna('').tolist()
    eval_embeddings = embed_model.encode(eval_texts, batch_size=64, show_progress_bar=False)
    eval_norms = np.linalg.norm(eval_embeddings, axis=1, keepdims=True)
    eval_embeddings_norm = eval_embeddings / (eval_norms + 1e-10)

    # cosine similarity matrix: (600, n_refs)
    sim_matrix = eval_embeddings_norm @ ref_matrix_norm.T
    max_sims   = sim_matrix.max(axis=1)
    embed_preds = (max_sims > SIMILARITY_THRESHOLD).astype(int).tolist()
    t_embed = time.time() - t0

    print(f"  Reference sud_signal docs: {len(signal_docs)}")
    print(f"  Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"  Mean max-similarity (SUD):     {max_sims[np.array(y_true)==1].mean():.3f}")
    print(f"  Mean max-similarity (non-SUD): {max_sims[np.array(y_true)==0].mean():.3f}")
    print(f"  Time: {t_embed:.2f}s")

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 — LLM + RAG (Claude with pgvector retrieval)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("METHOD 3 — LLM + RAG (Claude claude-haiku-4-5-20251001 + pgvector)")
print("=" * 60)

SYSTEM_PROMPT = """You are a public health analyst detecting substance use \
disorder risk signals in anonymized patient-written drug reviews for \
NSF-funded public health research. You have access to reference examples \
from real patient experiences and official ICD-10 clinical definitions.

This analysis supports population-level public health surveillance only. \
No individuals are identified."""

def retrieve_rag_context(review_text, n_signals=3, n_icd10=2):
    query_vec = embed_model.encode(review_text).tolist()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM rag_embeddings
            WHERE source_type = 'sud_signal'
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_vec, query_vec, n_signals))
        signal_docs = cur.fetchall()

        cur.execute("""
            SELECT content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM rag_embeddings
            WHERE source_type = 'icd10'
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_vec, query_vec, n_icd10))
        icd10_docs = cur.fetchall()

    signals_text = "\n".join(
        f"[similarity {sim:.2f}] {content[:300]}"
        for content, _, sim in signal_docs
    ) or "No reference signals found."

    icd10_text = "\n".join(
        f"[similarity {sim:.2f}] {content[:300]}"
        for content, _, sim in icd10_docs
    ) or "No ICD-10 references found."

    return signals_text, icd10_text

def llm_classify(review_text, drug_name, rating):
    signals_text, icd10_text = retrieve_rag_context(review_text)

    user_prompt = f"""Analyze this review for SUD risk signals.

SIMILAR PATIENT EXPERIENCES FROM KNOWLEDGE BASE:
{signals_text}

RELEVANT CLINICAL DEFINITIONS FROM ICD-10-CM:
{icd10_text}

REVIEW TO ANALYZE:
Drug: {drug_name or 'Unknown'}
Rating: {rating}/10
Text: {review_text[:600]}

Examples of nuanced language to detect:
- Indirect: "need it just to feel normal" = dependency
- Slang: "fell off the wagon" = relapse
- Emotional tone: "rock bottom desperate" = severe distress

Return ONLY JSON:
{{
  "classification": "SUD_RISK or NO_RISK",
  "confidence": 0.0,
  "direct_signals": [],
  "indirect_signals": [],
  "emotional_distress_score": 0,
  "relapse_indicator": false,
  "reasoning": "one sentence"
}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )
    raw = response.content[0].text.strip()
    # strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

t0 = time.time()
llm_preds, llm_results = [], []
BATCH_SIZE = 10

for i, (_, row) in enumerate(eval_df.iterrows()):
    if i % BATCH_SIZE == 0 and i > 0:
        elapsed = time.time() - t0
        remaining = (len(eval_df) - i) * (elapsed / i)
        print(f"  Processed {i}/{len(eval_df)} | elapsed {elapsed:.0f}s | ~{remaining:.0f}s remaining")
        time.sleep(1)

    try:
        result = llm_classify(
            str(row['review_text']),
            str(row['drug_name'] or ''),
            int(row['rating']) if pd.notna(row['rating']) else 5
        )
        pred = 1 if result.get('classification') == 'SUD_RISK' else 0
    except Exception as e:
        result = {
            'classification': 'NO_RISK', 'confidence': 0.0,
            'direct_signals': [], 'indirect_signals': [],
            'emotional_distress_score': 0, 'relapse_indicator': False,
            'reasoning': f'parse_error: {str(e)[:60]}'
        }
        pred = 0

    llm_preds.append(pred)
    llm_results.append(result)

t_llm = time.time() - t0
print(f"  Processed {len(eval_df)}/{len(eval_df)} | total time {t_llm:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

def compute_metrics(y_true, y_pred, name, elapsed):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    return {'method': name, 'precision': prec, 'recall': rec, 'f1': f1,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'elapsed': elapsed}

metrics = [
    compute_metrics(y_true, rule_preds,  'rule_based',   t_rule),
    compute_metrics(y_true, embed_preds, 'embedding',    t_embed),
    compute_metrics(y_true, llm_preds,   'llm_rag',      t_llm),
]

print(f"\n  {'Method':<14} {'Prec':>6} {'Rec':>6} {'F1':>6} "
      f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Time':>8}")
print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*6} "
      f"{'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*8}")
for m in metrics:
    print(f"  {m['method']:<14} {m['precision']:>6.3f} {m['recall']:>6.3f} "
          f"{m['f1']:>6.3f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
          f"{m['tn']:>5} {m['elapsed']:>7.1f}s")

# ── 3 examples per method per error type ──────────────────────────────────────
def get_examples(y_true, y_pred, label, n=3):
    """label: 'TP','FP','FN'"""
    cond = {
        'TP': lambda yt, yp: yt == 1 and yp == 1,
        'FP': lambda yt, yp: yt == 0 and yp == 1,
        'FN': lambda yt, yp: yt == 1 and yp == 0,
    }[label]
    idxs = [i for i,(yt,yp) in enumerate(zip(y_true,y_pred)) if cond(yt,yp)]
    return idxs[:n]

for method_name, preds in [
    ('RULE-BASED', rule_preds),
    ('EMBEDDING',  embed_preds),
    ('LLM+RAG',    llm_preds),
]:
    print(f"\n── {method_name} Examples ──")
    for label in ['TP', 'FP', 'FN']:
        idxs = get_examples(y_true, preds, label)
        if not idxs:
            continue
        print(f"  {label} ({len(idxs)}):")
        for idx in idxs:
            row = eval_df.iloc[idx]
            print(f"    Drug: {row['drug_name']} | Cat: {row['signal_category']} "
                  f"| Rating: {row['rating']}")
            print(f"    Text: {str(row['review_text'])[:130]}...")

# ─────────────────────────────────────────────────────────────────────────────
# WINNER ANALYSIS — what LLM caught that rule-based missed
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("WINNER ANALYSIS — LLM caught, Rule-Based missed")
print("=" * 60)

llm_only_hits = [
    i for i,(yt,rp,lp) in enumerate(zip(y_true, rule_preds, llm_preds))
    if yt == 1 and rp == 0 and lp == 1
]
print(f"\n  LLM found {len(llm_only_hits)} true positives that rules missed.\n")

for idx in llm_only_hits[:5]:
    row    = eval_df.iloc[idx]
    result = llm_results[idx]
    print(f"  Drug: {row['drug_name']} | Condition: {row['condition']}")
    print(f"  Review: {str(row['review_text'])[:200]}...")
    print(f"  LLM reasoning: {result.get('reasoning','')}")
    direct   = result.get('direct_signals', [])
    indirect = result.get('indirect_signals', [])
    if indirect:
        print(f"  Indirect signals detected: {indirect}")
    if result.get('relapse_indicator'):
        print(f"  Relapse indicator: TRUE")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# method_comparison CSV
metrics_df = pd.DataFrame(metrics)
csv_path = os.path.join(OUTPUTS_DIR, 'method_comparison_results.csv')
metrics_df.to_csv(csv_path, index=False)
print(f"  Saved metrics → {csv_path}")

# insert into method_comparison table
with conn.cursor() as cur:
    execute_values(cur, """
        INSERT INTO method_comparison
            (method, precision_score, recall_score, f1_score)
        VALUES %s;
    """, [(m['method'], m['precision'], m['recall'], m['f1'])
          for m in metrics])
conn.commit()
print(f"  Inserted {len(metrics)} rows into method_comparison.")

# update drug_reviews with LLM results for the 600 eval rows
update_count = 0
with conn.cursor() as cur:
    for i, (_, row) in enumerate(eval_df.iterrows()):
        result = llm_results[i]
        risk_class   = result.get('classification', 'NO_RISK')
        confidence   = float(result.get('confidence', 0.0))
        direct_sigs  = result.get('direct_signals', [])
        indirect_sigs= result.get('indirect_signals', [])
        detected     = {'direct': direct_sigs, 'indirect': indirect_sigs,
                        'relapse': result.get('relapse_indicator', False),
                        'distress_score': result.get('emotional_distress_score', 0)}
        explanation  = {'reasoning': result.get('reasoning', ''),
                        'method': 'llm_rag_claude-haiku-4-5-20251001',
                        'rag_sources': ['sud_signal', 'icd10']}

        cur.execute("""
            UPDATE drug_reviews
            SET risk_classification = %s,
                confidence_score    = %s,
                detected_signals    = %s,
                explanation         = %s
            WHERE review_id = %s;
        """, (
            risk_class, confidence,
            json.dumps(detected),
            json.dumps(explanation),
            int(row['review_id'])
        ))
        update_count += 1

conn.commit()
print(f"  Updated {update_count} drug_reviews rows with LLM classifications.")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
winner = max(metrics, key=lambda m: m['f1'])

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\n  Winner by F1: {winner['method'].upper()}  (F1={winner['f1']:.3f})")
print(f"\n  {'Method':<14} {'F1':>6}  {'Precision':>10}  {'Recall':>8}  {'Speed':>10}")
print(f"  {'-'*14} {'-'*6}  {'-'*10}  {'-'*8}  {'-'*10}")
for m in sorted(metrics, key=lambda x: x['f1'], reverse=True):
    speed = 'fast' if m['elapsed'] < 5 else ('medium' if m['elapsed'] < 60 else 'slow')
    print(f"  {m['method']:<14} {m['f1']:>6.3f}  {m['precision']:>10.3f}  "
          f"{m['recall']:>8.3f}  {m['elapsed']:>7.1f}s ({speed})")

print(f"\n  LLM unique catches (FN recovered): {len(llm_only_hits)}")

embed_only = [
    i for i,(yt,rp,ep,lp) in enumerate(zip(y_true,rule_preds,embed_preds,llm_preds))
    if yt==1 and rp==0 and ep==1 and lp==0
]
rule_only = [
    i for i,(yt,rp,ep,lp) in enumerate(zip(y_true,rule_preds,embed_preds,llm_preds))
    if yt==1 and rp==1 and ep==0 and lp==0
]
print(f"  Embedding unique catches:          {len(embed_only)}")
print(f"  Rule-based unique catches:         {len(rule_only)}")

all_agree_tp = [
    i for i,(yt,rp,ep,lp) in enumerate(zip(y_true,rule_preds,embed_preds,llm_preds))
    if yt==1 and rp==1 and ep==1 and lp==1
]
print(f"  All three agree (TP):              {len(all_agree_tp)}")

print(f"\n  Outputs saved:")
print(f"    {csv_path}")
print(f"    drug_reviews.risk_classification (600 rows updated)")
print(f"    method_comparison table (3 rows inserted)")

conn.close()
print("\nTask 1 complete.")
