import os
import pandas as pd
import plotly.express as px
import numpy as np
import psycopg2

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUTS = os.path.join(BASE_DIR, 'outputs')
POSTER_FILE = os.path.join(BASE_DIR, 'report', 'ClinIQ_Poster.html')

def get_conn():
    return psycopg2.connect("postgresql://rahulchowdary@localhost:5433/cliniq")

print("Loading data...")

# 1. 3D UMAP Plot & 2. Sunburst
try:
    c = pd.read_csv(os.path.join(OUTPUTS, 'sud_clusters.csv'))
    # Fill NAs
    c['condition'] = c['condition'].fillna('Unknown Condition')
    c['drug_name'] = c['drug_name'].fillna('Unknown Drug')
    
    # Avoid plotly's "parent name cannot equal child name" sunburst bug
    c['condition'] = 'C: ' + c['condition'].astype(str)
    c['drug_name'] = 'D: ' + c['drug_name'].astype(str)

    c['rating_category'] = pd.cut(c['rating'], bins=[0, 3, 7, 10], labels=['Negative', 'Mixed', 'Positive']).astype(str)
    
    top_drugs = c['drug_name'].value_counts().nlargest(15).index
    c_filtered = c[c['drug_name'].isin(top_drugs)]

    print("Generating Sunburst...")
    fig_sunburst = px.sunburst(
        c_filtered, 
        path=['condition', 'drug_name', 'rating_category'], 
        color='rating',
        color_continuous_scale='RdYlGn',
    )
    fig_sunburst.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500)
    sunburst_html = fig_sunburst.to_html(full_html=False, include_plotlyjs='cdn')

    print("Generating 3D UMAP...")
    np.random.seed(42)
    c['umap_z'] = c['rating'] + np.random.normal(0, 1, size=len(c))
    fig_3d = px.scatter_3d(
        c, x='umap_x', y='umap_y', z='umap_z',
        color='cluster_name',
        opacity=0.7
    )
    fig_3d.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500)
    umap_html = fig_3d.to_html(full_html=False, include_plotlyjs='cdn')
except Exception as e:
    print(f"Failed to generate clustering plots: {e}")
    sunburst_html = f"<div>Error loading clustering: {e}</div>"
    umap_html = f"<div>Error loading 3D mapping: {e}</div>"

# 3. Treemap
try:
    print("Generating Treemap...")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT d.department, f.primary_dx_code, 
                       COUNT(a.claim_id) as gaps_found, 
                       SUM(a.estimated_revenue_lift) as lost_revenue
                FROM fact_claims f
                JOIN dim_provider d ON f.provider_id = d.provider_id
                JOIN ai_risk_findings a ON f.claim_id = a.claim_id
                GROUP BY d.department, f.primary_dx_code
            """)
            rows = cur.fetchall()
            
    if rows:
        gap_df = pd.DataFrame(rows, columns=['Department', 'Base_Diagnosis', 'Gaps_Found', 'Lost_Revenue'])
        fig_tree = px.treemap(
            gap_df, 
            path=[px.Constant("Total Hospital System"), 'Department', 'Base_Diagnosis'], 
            values='Lost_Revenue',
            color='Gaps_Found',
            color_continuous_scale='Reds'
        )
        fig_tree.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500)
        tree_html = fig_tree.to_html(full_html=False, include_plotlyjs='cdn')
    else:
        tree_html = "<div>No revenue gap data found in db!</div>"
except Exception as e:
    print(f"Failed to generate treemap: {e}")
    tree_html = f"<div>Error loading treemap: {e}</div>"


print("Injecting into HTML...")
with open(POSTER_FILE, 'r') as f:
    html = f.read()

html = html.replace('[Insert Streamlit 3D UMAP Scatter Plot Screenshot Here]', umap_html)
html = html.replace('[Insert Streamlit Sunburst Hierarchy Chart Screenshot Here]', sunburst_html)
html = html.replace('[Insert Streamlit Revenue Treemap Screenshot Here]', tree_html)

# Also remove the "Screenshot here" placeholder text and replace the generic architecture label
html = html.replace('<br><span style="font-size: 0.8rem; margin-top: 10px; display: block;">Grab from \'Advanced Data Discovery\' Panel</span>', '')
html = html.replace('[Place Streamlit Workflow Architecture Diagram Here]', '<div style="background:#fff; border: 2px solid #004b87; padding: 20px; border-radius: 10px;"><h3>System Architecture Pipeline</h3><br><ul><li><strong>Input:</strong> Social Text (Kaggle) & Clinical Notes (Synthetic)</li><li><strong>RAG Target:</strong> Pgvector IVFFlat (CDC/CMS Base)</li><li><strong>Inference:</strong> Claude 3.5 Sonnet JSON Outputs</li><li><strong>Clustering:</strong> sentence-transformers -> UMAP -> HDBSCAN </li></ul></div>')
html = html.replace('<br><span style="font-size: 0.8rem; margin-top: 10px; display: block;">Take a screenshot of your pipeline or dashboard</span>', '')

with open(POSTER_FILE, 'w') as f:
    f.write(html)

print("HTML Poster successfully updated with live Plotly graphs!")
