import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUTS = os.path.join(BASE_DIR, 'outputs')
RADAR_FILE = os.path.join(BASE_DIR, 'report', 'ClinIQ_Method_Radar.png')
BAR_FILE = os.path.join(BASE_DIR, 'report', 'ClinIQ_Method_Bar.png')

PLOTLY_TEMPLATE = "plotly_dark"

print("Loading methodology data...")
path = os.path.join(OUTPUTS, 'method_comparison_results.csv')
mc = pd.read_csv(path)

# Normalize column names as done in app.py
rename = {}
for old, new in [('f1','f1_score'),('precision','precision_score'),('recall','recall_score')]:
    if old in mc.columns and new not in mc.columns:
        rename[old] = new
if rename:
    mc = mc.rename(columns=rename)
method_map = {'rule_based':'Rule-Based','embedding':'Embedding','llm_rag':'LLM + RAG'}
mc['method'] = mc['method'].map(lambda m: method_map.get(m, m))

# ==========================================
# 1. GENERATE RADAR CHART
# ==========================================
print("Generating Radar Chart...")
categories = ['F1 Score','Precision','Recall','Speed\n(inverse)','Explainability']
method_vals = {
    'Rule-Based': [
        mc[mc['method']=='Rule-Based']['f1_score'].iloc[0] if len(mc[mc['method']=='Rule-Based'])>0 else 0.86,
        mc[mc['method']=='Rule-Based']['precision_score'].iloc[0] if len(mc[mc['method']=='Rule-Based'])>0 else 0.87,
        mc[mc['method']=='Rule-Based']['recall_score'].iloc[0] if len(mc[mc['method']=='Rule-Based'])>0 else 0.85,
        0.99, 0.50,
    ],
    'Embedding': [
        mc[mc['method']=='Embedding']['f1_score'].iloc[0] if len(mc[mc['method']=='Embedding'])>0 else 0.67,
        mc[mc['method']=='Embedding']['precision_score'].iloc[0] if len(mc[mc['method']=='Embedding'])>0 else 0.50,
        mc[mc['method']=='Embedding']['recall_score'].iloc[0] if len(mc[mc['method']=='Embedding'])>0 else 1.0,
        0.95, 0.30,
    ],
    'LLM + RAG': [
        mc[mc['method']=='LLM + RAG']['f1_score'].iloc[0] if len(mc[mc['method']=='LLM + RAG'])>0 else 0.48,
        mc[mc['method']=='LLM + RAG']['precision_score'].iloc[0] if len(mc[mc['method']=='LLM + RAG'])>0 else 0.88,
        mc[mc['method']=='LLM + RAG']['recall_score'].iloc[0] if len(mc[mc['method']=='LLM + RAG'])>0 else 0.33,
        0.10, 0.98,
    ],
}
color_map = {'Rule-Based':'#e74c3c','Embedding':'#f39c12','LLM + RAG':'#27ae60'}
fill_map  = {
    'Rule-Based': 'rgba(231,76,60,0.18)',
    'Embedding':  'rgba(243,156,18,0.18)',
    'LLM + RAG':  'rgba(39,174,96,0.18)',
}
fig_radar = go.Figure()
for method, vals in method_vals.items():
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=categories + [categories[0]],
        name=method,
        line_color=color_map[method],
        fill='toself',
        fillcolor=fill_map[method],
    ))
fig_radar.update_layout(
    title=dict(text="AI Capability Matrix", font=dict(color='white', size=22)),
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1.05], color='white', gridcolor='#444'),
        angularaxis=dict(color='white', gridcolor='#444', tickfont=dict(size=16))
    ),
    template=PLOTLY_TEMPLATE, 
    height=600, width=800, 
    showlegend=True,
    legend=dict(font=dict(size=16)),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig_radar.write_image(RADAR_FILE, scale=4.0)
print(f"Radar chart saved: {RADAR_FILE}")


# ==========================================
# 2. GENERATE GROUPED BAR CHART
# ==========================================
print("Generating Grouped Bar Chart...")
metrics_long = []
for _, row in mc.iterrows():
    for metric, col_name in [('F1 Score','f1_score'),
                             ('Precision','precision_score'),
                             ('Recall','recall_score')]:
        metrics_long.append({
            'Method': row['method'],
            'Metric': metric,
            'Value':  round(row[col_name], 3),
        })
metrics_df = pd.DataFrame(metrics_long)

fig_grouped = px.bar(
    metrics_df, x='Metric', y='Value', color='Method', barmode='group',
    color_discrete_map={
        'Rule-Based': '#e74c3c',
        'Embedding':  '#f39c12',
        'LLM + RAG':  '#27ae60',
    },
    text='Value'
)
fig_grouped.update_traces(
    texttemplate='%{text:.2f}', 
    textposition='outside', 
    textfont=dict(size=16, color='white')
)
fig_grouped.update_layout(
    title=dict(text="Empirical Model Performance", font=dict(color='white', size=24)),
    template=PLOTLY_TEMPLATE, 
    height=600, width=1000,
    yaxis=dict(range=[0, 1.15], title=dict(text="Score", font=dict(size=18, color='white')), color='white', gridcolor='#444'),
    xaxis=dict(title=dict(text="", font=dict(size=18, color='white')), color='white', tickfont=dict(size=18)),
    legend_title=dict(text="Method", font=dict(size=18)),
    legend=dict(font=dict(size=16)),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig_grouped.write_image(BAR_FILE, scale=4.0)
print(f"Bar chart saved: {BAR_FILE}")

print("Both empirical methodology graphics successfully generated!")
