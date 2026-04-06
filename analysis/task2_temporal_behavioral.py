import json
import os
import sys
import re
from collections import Counter

import numpy as np
import pandas as pd
import psycopg2
import anthropic
import hdbscan
import umap
from psycopg2.extras import execute_values
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATABASE_URL    = os.getenv('DATABASE_URL')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set in .env"); sys.exit(1)
if not ANTHROPIC_API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set in .env"); sys.exit(1)

conn        = psycopg2.connect(DATABASE_URL)
client      = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

STOPWORDS = {
    'i','me','my','we','our','you','your','it','its','the','a','an',
    'and','or','but','in','on','at','to','for','of','with','is','was',
    'are','were','be','been','being','have','has','had','do','did','does',
    'this','that','these','those','not','no','so','just','very','more',
    'also','than','then','when','if','as','by','from','up','out','about',
    'they','he','she','him','her','his','their','there','been','would',
    'could','should','will','can','may','might','must','after','before',
    'which','what','who','how','all','any','both','each','few','most',
    'other','same','such','into','through','during','again','further',
    'now','only','own','too','while','being','having','because','between',
    'am','got','get','got','take','took','tried','try','still','even',
    'over','back','think','feel','really','much','well','going','make',
    'made','see','use','used','give','given','time','one','two','like',
    'since','though','however','within','without','around','away','down',
    'every','some','never','always','often','already','again','however',
    'him','her','them','us','its','via','per','ie','eg','etc','day','days',
    'week','weeks','month','months','year','years','dose','mg','pill',
    'pills','medication','drug','drugs','medicine','doctor','told',
}

def top_terms(texts, n=25):
    words = []
    for t in texts:
        tokens = re.findall(r"[a-z']+", str(t).lower())
        words.extend([w for w in tokens if len(w) > 3 and w not in STOPWORDS])
    counts = Counter(words)
    return [w for w, _ in counts.most_common(n)]

def cosine_dist(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-10)
    b_n = b / (np.linalg.norm(b) + 1e-10)
    return float(1 - np.dot(a_n, b_n))

def claude_one_sentence(prompt_text):
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        messages=[{"role": "user", "content": prompt_text}]
    )
    return resp.content[0].text.strip()

# ═════════════════════════════════════════════════════════════════════════════
# PART A — TEMPORAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART A — Temporal Analysis (2008–2017)")
print("=" * 60)

# ── STEP A1 — year-over-year SUD signal volume ────────────────────────────────
print("\n── Step A1: Year-over-year SUD signal volume ──")

with conn.cursor() as cur:
    cur.execute("""
        SELECT
            review_year,
            COUNT(*)                                          AS total_sud,
            ROUND(AVG(rating)::numeric, 2)                    AS avg_rating,
            ROUND(
                SUM(CASE WHEN rating <= 3 THEN 1 ELSE 0 END)::numeric
                / NULLIF(COUNT(*),0), 4
            )                                                  AS distress_prop,
            ROUND(
                SUM(CASE WHEN signal_category = 'opioid' THEN 1 ELSE 0 END)::numeric
                / NULLIF(COUNT(*),0), 4
            )                                                  AS opioid_prop
        FROM drug_reviews
        WHERE is_sud_relevant = TRUE
          AND review_year BETWEEN 2008 AND 2017
          AND review_year IS NOT NULL
        GROUP BY review_year
        ORDER BY review_year;
    """)
    sud_year_rows = cur.fetchall()

    # national CDC rate per year (average across all records for that year)
    cur.execute("""
        SELECT year, AVG(rate_per_100k) AS avg_rate
        FROM cdc_overdose
        WHERE rate_per_100k IS NOT NULL
          AND year BETWEEN 2008 AND 2017
        GROUP BY year
        ORDER BY year;
    """)
    cdc_year_rows = cur.fetchall()

cdc_rate_map = {yr: float(rate) for yr, rate in cdc_year_rows}

year_records = []
for review_year, total_sud, avg_rating, distress_prop, opioid_prop in sud_year_rows:
    year_records.append({
        'year':              int(review_year),
        'sud_review_count':  int(total_sud),
        'avg_rating':        float(avg_rating or 0),
        'distress_proportion': float(distress_prop or 0),
        'opioid_proportion':   float(opioid_prop or 0),
        'cdc_overdose_rate':   cdc_rate_map.get(review_year),
    })

year_df = pd.DataFrame(year_records).sort_values('year').reset_index(drop=True)

print(f"\n  {'Year':<6} {'SUD Rev':>8} {'Avg Rtg':>8} "
      f"{'Distress%':>10} {'Opioid%':>9} {'CDC/100k':>9}")
print(f"  {'----':<6} {'-------':>8} {'-------':>8} "
      f"{'---------':>10} {'--------':>9} {'--------':>9}")
for _, r in year_df.iterrows():
    cdc_str = f"{r['cdc_overdose_rate']:.1f}" if r['cdc_overdose_rate'] else "N/A"
    print(f"  {int(r['year']):<6} {int(r['sud_review_count']):>8,} "
          f"{r['avg_rating']:>8.2f} "
          f"{r['distress_proportion']*100:>9.1f}% "
          f"{r['opioid_proportion']*100:>8.1f}% "
          f"{cdc_str:>9}")

year_df.to_csv(os.path.join(OUTPUTS_DIR, 'temporal_trends.csv'), index=False)
print(f"\n  Saved → outputs/temporal_trends.csv")

# ── STEP A2 — CDC Pearson correlation ─────────────────────────────────────────
print("\n── Step A2: CDC Pearson Correlation ──")

corr_df = year_df.dropna(subset=['cdc_overdose_rate']).copy()
if len(corr_df) >= 3:
    correlation, pvalue = pearsonr(
        corr_df['sud_review_count'],
        corr_df['cdc_overdose_rate']
    )
    print(f"\n  r  = {correlation:.4f}")
    print(f"  p  = {pvalue:.4f}")
    if pvalue < 0.05:
        direction = "positive" if correlation > 0 else "negative"
        strength  = "strong" if abs(correlation) > 0.7 else ("moderate" if abs(correlation) > 0.4 else "weak")
        print(f"  Interpretation: Statistically significant {strength} {direction} correlation.")
        print(f"  As overdose deaths rise, SUD review volume {'increases' if correlation>0 else 'decreases'}.")
    else:
        print(f"  Interpretation: No statistically significant correlation (p={pvalue:.3f}).")
else:
    correlation, pvalue = 0.0, 1.0
    print("  Insufficient overlapping years for correlation.")

# ── STEP A3 — Narrative evolution via centroid distance ───────────────────────
print("\n── Step A3: Narrative Evolution — Centroid Distance ──")

with conn.cursor() as cur:
    cur.execute("""
        SELECT review_year, review_id, review_text
        FROM drug_reviews
        WHERE is_sud_relevant = TRUE
          AND review_year BETWEEN 2008 AND 2017
          AND review_year IS NOT NULL
          AND char_length(review_text) > 30
        ORDER BY review_year;
    """)
    all_sud_rows = cur.fetchall()

# group by year
year_texts = {}
for yr, rid, txt in all_sud_rows:
    yr = int(yr)
    year_texts.setdefault(yr, []).append(str(txt))

# compute centroids for years with 10+ reviews
centroids = {}
year_term_lists = {}
for yr in sorted(year_texts):
    texts = year_texts[yr]
    if len(texts) < 10:
        continue
    embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=False)
    centroids[yr]       = embeddings.mean(axis=0)
    year_term_lists[yr] = top_terms(texts, n=40)

# consecutive centroid distances
narrative_shifts = {}
sorted_years = sorted(centroids.keys())
print(f"\n  {'Year':<6} {'Distance':>10}  Shift?")
print(f"  {'----':<6} {'---------':>10}  ------")
for i in range(1, len(sorted_years)):
    yr_prev = sorted_years[i-1]
    yr_curr = sorted_years[i]
    dist = cosine_dist(centroids[yr_prev], centroids[yr_curr])
    is_shift = dist > 0.15
    marker = " ← NARRATIVE SHIFT" if is_shift else ""
    print(f"  {yr_curr:<6} {dist:>10.4f}{marker}")
    if is_shift:
        narrative_shifts[yr_curr] = {'distance': dist, 'prev_year': yr_prev}

# generate Claude narrative for each shift year
narrative_json = {}
for yr, info in narrative_shifts.items():
    yr_prev  = info['prev_year']
    terms_curr = set(year_term_lists.get(yr, []))
    terms_prev = set(year_term_lists.get(yr_prev, []))
    increased  = sorted(terms_curr - terms_prev)[:15]
    decreased  = sorted(terms_prev - terms_curr)[:15]

    prompt = (
        f"You are a public health researcher.\n\n"
        f"These words significantly increased in patient drug reviews "
        f"in {yr} compared to {yr_prev}:\n{', '.join(increased)}\n\n"
        f"These words significantly decreased:\n{', '.join(decreased)}\n\n"
        f"In exactly one sentence, describe what this vocabulary shift suggests "
        f"about how patients were experiencing substance use disorder in {yr}."
    )
    try:
        summary = claude_one_sentence(prompt)
    except Exception as e:
        summary = f"[Claude error: {e}]"

    info['narrative_shift_summary'] = summary
    info['increased_terms']  = increased
    info['decreased_terms']  = decreased
    narrative_json[str(yr)]  = info
    print(f"\n  {yr} narrative: {summary}")

with open(os.path.join(OUTPUTS_DIR, 'narrative_shifts.json'), 'w') as f:
    json.dump(narrative_json, f, indent=2)
print(f"\n  Saved → outputs/narrative_shifts.json")

# ── STEP A4 — Substance trend lines ──────────────────────────────────────────
print("\n── Step A4: Substance Trend Lines ──")

with conn.cursor() as cur:
    cur.execute("""
        SELECT review_year, signal_category, COUNT(*) AS cnt
        FROM drug_reviews
        WHERE is_sud_relevant = TRUE
          AND review_year BETWEEN 2008 AND 2017
          AND review_year IS NOT NULL
        GROUP BY review_year, signal_category
        ORDER BY review_year, signal_category;
    """)
    trend_rows = cur.fetchall()

trend_df = pd.DataFrame(trend_rows, columns=['year','signal_category','count'])
trend_df.to_csv(os.path.join(OUTPUTS_DIR, 'substance_trends.csv'), index=False)

pivot = trend_df.pivot(index='year', columns='signal_category', values='count').fillna(0).astype(int)
print(f"\n  Substance trends by year:\n{pivot.to_string()}")
print(f"\n  Saved → outputs/substance_trends.csv")

# save to temporal_analysis table
with conn.cursor() as cur:
    cur.execute("DELETE FROM temporal_analysis WHERE year BETWEEN 2008 AND 2017;")
    rows_to_insert = []
    for _, r in year_df.iterrows():
        yr = int(r['year'])
        shift_info = narrative_shifts.get(yr, {})
        rows_to_insert.append((
            yr,
            int(r['sud_review_count']),
            float(r['avg_rating']),
            float(r['distress_proportion']),
            float(r['opioid_proportion']),
            r['cdc_overdose_rate'],
            shift_info.get('distance'),
            bool(shift_info),
            shift_info.get('narrative_shift_summary'),
        ))
    execute_values(cur, """
        INSERT INTO temporal_analysis
            (year, sud_review_count, avg_rating, distress_proportion,
             opioid_proportion, cdc_overdose_rate, centroid_distance_from_prev,
             is_narrative_shift_year, narrative_shift_summary)
        VALUES %s
        ON CONFLICT (year) DO UPDATE SET
            sud_review_count           = EXCLUDED.sud_review_count,
            avg_rating                 = EXCLUDED.avg_rating,
            distress_proportion        = EXCLUDED.distress_proportion,
            opioid_proportion          = EXCLUDED.opioid_proportion,
            cdc_overdose_rate          = EXCLUDED.cdc_overdose_rate,
            centroid_distance_from_prev = EXCLUDED.centroid_distance_from_prev,
            is_narrative_shift_year    = EXCLUDED.is_narrative_shift_year,
            narrative_shift_summary    = EXCLUDED.narrative_shift_summary;
    """, rows_to_insert)
conn.commit()
print(f"\n  Saved {len(rows_to_insert)} rows to temporal_analysis table.")

# ═════════════════════════════════════════════════════════════════════════════
# PART B — BEHAVIORAL CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B — Behavioral Clustering (UMAP + HDBSCAN)")
print("=" * 60)

# ── STEP B1 — embed all SUD reviews ──────────────────────────────────────────
print("\n── Step B1: Embed all SUD reviews ──")

with conn.cursor() as cur:
    cur.execute("""
        SELECT review_id, drug_name, condition, review_text,
               rating, signal_category, review_year, useful_count
        FROM drug_reviews
        WHERE is_sud_relevant = TRUE
          AND char_length(review_text) > 30
        ORDER BY review_id;
    """)
    sud_rows = cur.fetchall()

sud_df = pd.DataFrame(sud_rows,
    columns=['review_id','drug_name','condition','review_text',
             'rating','signal_category','review_year','useful_count'])

print(f"  Embedding {len(sud_df)} SUD reviews...")
texts      = sud_df['review_text'].fillna('').tolist()
embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=True)
print(f"  Embedding shape: {embeddings.shape}")

# ── STEP B2 — UMAP ────────────────────────────────────────────────────────────
print("\n── Step B2: UMAP Dimensionality Reduction ──")

reducer = umap.UMAP(
    n_neighbors=15, min_dist=0.1, n_components=2,
    random_state=42, metric='cosine'
)
umap_coords = reducer.fit_transform(embeddings)
sud_df['umap_x'] = umap_coords[:, 0]
sud_df['umap_y'] = umap_coords[:, 1]
print(f"  UMAP complete. x range: [{umap_coords[:,0].min():.2f}, {umap_coords[:,0].max():.2f}]  "
      f"y range: [{umap_coords[:,1].min():.2f}, {umap_coords[:,1].max():.2f}]")

# ── STEP B3 — HDBSCAN ─────────────────────────────────────────────────────────
print("\n── Step B3: HDBSCAN Clustering ──")

clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
cluster_labels = clusterer.fit_predict(umap_coords)
sud_df['behavioral_cluster'] = cluster_labels

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
noise_ct   = (cluster_labels == -1).sum()
print(f"  Clusters found: {n_clusters}")
print(f"  Noise points:   {noise_ct} ({noise_ct/len(sud_df)*100:.1f}%)")
for lbl in sorted(set(cluster_labels)):
    ct = (cluster_labels == lbl).sum()
    name = "noise" if lbl == -1 else f"cluster_{lbl}"
    print(f"    {name}: {ct} reviews")

# ── STEP B4 — Name each cluster via Claude ────────────────────────────────────
print("\n── Step B4: Cluster Naming via Claude ──")

cluster_names = {}
for lbl in sorted(set(cluster_labels)):
    if lbl == -1:
        cluster_names[-1] = "Noise / Unclassified"
        continue
    mask   = sud_df['behavioral_cluster'] == lbl
    subset = sud_df[mask]
    terms  = top_terms(subset['review_text'].tolist(), n=25)

    prompt = (
        f"These words appear most frequently in a group of anonymized "
        f"patient drug reviews:\n{', '.join(terms)}\n\n"
        f"Name this behavioral pattern in 3 words maximum. "
        f"The name should help a public health official understand "
        f"what these patients are experiencing. "
        f"Return only the name."
    )
    try:
        name = claude_one_sentence(prompt).strip().strip('"').strip("'")
    except Exception as e:
        name = f"Cluster {lbl}"

    cluster_names[lbl] = name
    print(f"  Cluster {lbl} ({len(subset)} reviews): \"{name}\"")
    print(f"    Top terms: {', '.join(terms[:12])}")

sud_df['cluster_name'] = sud_df['behavioral_cluster'].map(cluster_names)

# ── STEP B5 — update drug_reviews ─────────────────────────────────────────────
print("\n── Step B5: Updating drug_reviews ──")

with conn.cursor() as cur:
    rows = [
        (int(r['behavioral_cluster']), float(r['umap_x']),
         float(r['umap_y']), int(r['review_id']))
        for _, r in sud_df.iterrows()
    ]
    execute_values(cur, """
        UPDATE drug_reviews AS dr
        SET behavioral_cluster = v.cluster,
            umap_x             = v.ux,
            umap_y             = v.uy
        FROM (VALUES %s) AS v(cluster, ux, uy, rid)
        WHERE dr.review_id = v.rid;
    """, rows, template="(%s, %s, %s, %s)")
conn.commit()
print(f"  Updated {len(rows)} drug_reviews rows.")

# save cluster CSV
cluster_out = sud_df[['review_id','drug_name','condition','rating',
                       'signal_category','review_year','behavioral_cluster',
                       'cluster_name','umap_x','umap_y']].copy()
cluster_out.to_csv(os.path.join(OUTPUTS_DIR, 'sud_clusters.csv'), index=False)
print(f"  Saved → outputs/sud_clusters.csv")

# ── STEP B6 — Cluster evolution by year ──────────────────────────────────────
print("\n── Step B6: Cluster Evolution by Year ──")

evol = (
    sud_df[sud_df['behavioral_cluster'] >= 0]
    .groupby(['review_year','behavioral_cluster'])
    .size()
    .reset_index(name='count')
)
evol['cluster_name'] = evol['behavioral_cluster'].map(cluster_names)
evol.to_csv(os.path.join(OUTPUTS_DIR, 'cluster_evolution.csv'), index=False)

evol_pivot = evol.pivot(index='review_year', columns='cluster_name', values='count').fillna(0).astype(int)
print(f"\n  Cluster evolution by year:\n{evol_pivot.to_string()}")
print(f"\n  Saved → outputs/cluster_evolution.csv")

# ═════════════════════════════════════════════════════════════════════════════
# PART C — EARLY-WARNING INDICATOR
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART C — Early-Warning Indicator")
print("=" * 60)

def early_warning_analysis():
    evol_df = pd.read_csv(os.path.join(OUTPUTS_DIR, 'cluster_evolution.csv'))
    cdc_df  = pd.read_csv(os.path.join(OUTPUTS_DIR, 'temporal_trends.csv'))

    cdc_rate = cdc_df.set_index('year')['cdc_overdose_rate'].dropna().to_dict()

    results = []
    for lbl in sorted(set(evol_df['behavioral_cluster'].unique())):
        if lbl < 0:
            continue
        sub = evol_df[evol_df['behavioral_cluster'] == lbl].sort_values('review_year')
        if len(sub) < 4:
            continue
        sub = sub.copy()
        sub['growth_rate'] = sub['count'].pct_change().fillna(0)

        # align: cluster growth in Y, CDC rate in Y+1
        pairs = []
        for _, row in sub.iterrows():
            yr      = int(row['review_year'])
            growth  = row['growth_rate']
            cdc_nxt = cdc_rate.get(yr + 1)
            if cdc_nxt is not None and not np.isnan(growth) and not np.isinf(growth):
                pairs.append((growth, cdc_nxt))

        if len(pairs) < 3:
            continue
        growths, cdc_vals = zip(*pairs)
        try:
            r, p = pearsonr(growths, cdc_vals)
        except Exception:
            continue

        results.append({
            'cluster_label': int(lbl),
            'cluster_name':  cluster_names.get(lbl, f"Cluster {lbl}"),
            'correlation':   round(float(r), 4),
            'pvalue':        round(float(p), 4),
            'n_years':       len(pairs),
        })

    if not results:
        print("  Insufficient data for early-warning analysis.")
        return

    results_df = pd.DataFrame(results).sort_values('correlation', ascending=False)
    print(f"\n  {'Cluster':<35} {'r':>7}  {'p':>7}  {'n':>4}")
    print(f"  {'-'*35} {'-'*7}  {'-'*7}  {'-'*4}")
    for _, row in results_df.iterrows():
        sig = "*" if row['pvalue'] < 0.05 else ""
        print(f"  {row['cluster_name']:<35} {row['correlation']:>7.4f}  "
              f"{row['pvalue']:>7.4f}{sig}  {row['n_years']:>4}")

    best = results_df.iloc[0]
    best_lbl  = int(best['cluster_label'])
    best_name = best['cluster_name']
    best_r    = best['correlation']

    # get top terms from that cluster's actual review text
    best_mask  = sud_df['behavioral_cluster'] == best_lbl
    best_terms = top_terms(sud_df[best_mask]['review_text'].tolist(), n=15)

    prompt = (
        f"A cluster of patient drug reviews named '{best_name}' showed "
        f"growth rate correlation r={best_r:.2f} with CDC overdose mortality "
        f"rates one year later.\n\n"
        f"Top terms in this cluster from real patient reviews:\n"
        f"{', '.join(best_terms)}\n\n"
        f"Write one sentence explaining why growth in this type of patient "
        f"discussion might predict increased overdose mortality the following year. "
        f"Write for a public health official."
    )
    try:
        interpretation = claude_one_sentence(prompt)
    except Exception as e:
        interpretation = f"[Claude error: {e}]"

    print(f"\n  Best early-warning cluster: \"{best_name}\" (r={best_r:.4f})")
    print(f"  Top terms: {', '.join(best_terms[:10])}")
    print(f"  Interpretation: {interpretation}")

    findings = {
        'best_cluster':    best_name,
        'correlation':     float(best_r),
        'pvalue':          float(best['pvalue']),
        'top_terms':       best_terms,
        'interpretation':  interpretation,
        'all_clusters':    results_df.to_dict(orient='records'),
    }
    out_path = os.path.join(OUTPUTS_DIR, 'early_warning_findings.json')
    with open(out_path, 'w') as f:
        json.dump(findings, f, indent=2)
    print(f"\n  Saved → outputs/early_warning_findings.json")

early_warning_analysis()

# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 2 COMPLETE — Summary")
print("=" * 60)

print(f"\n  PART A — Temporal Analysis")
print(f"    Years analyzed:          2008–2017 ({len(year_df)} years)")
print(f"    Narrative shift years:   {sorted(narrative_shifts.keys()) or 'none detected'}")
print(f"    CDC correlation:         r={correlation:.4f}  p={pvalue:.4f}")

print(f"\n  PART B — Behavioral Clustering")
print(f"    SUD reviews clustered:   {len(sud_df):,}")
print(f"    Clusters found:          {n_clusters}")
print(f"    Noise / unclassified:    {noise_ct}")
for lbl, name in sorted(cluster_names.items()):
    if lbl == -1:
        continue
    ct = (sud_df['behavioral_cluster'] == lbl).sum()
    print(f"      Cluster {lbl}: \"{name}\" ({ct} reviews)")

print(f"\n  PART C — Early Warning")
ew_path = os.path.join(OUTPUTS_DIR, 'early_warning_findings.json')
if os.path.exists(ew_path):
    with open(ew_path) as f:
        ew = json.load(f)
    print(f"    Best predictor cluster:  \"{ew.get('best_cluster')}\"")
    print(f"    Lag-1 correlation:       r={ew.get('correlation',0):.4f}")

print(f"\n  Outputs saved:")
for fn in ['temporal_trends.csv','narrative_shifts.json','substance_trends.csv',
           'sud_clusters.csv','cluster_evolution.csv','early_warning_findings.json']:
    path = os.path.join(OUTPUTS_DIR, fn)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    print(f"    outputs/{fn}  ({size:,} bytes)")

print(f"\n  Tables updated:")
print(f"    temporal_analysis — {len(year_df)} rows")
print(f"    drug_reviews.behavioral_cluster / umap_x / umap_y — {len(sud_df):,} rows")

conn.close()
print("\nTask 2 complete.")
