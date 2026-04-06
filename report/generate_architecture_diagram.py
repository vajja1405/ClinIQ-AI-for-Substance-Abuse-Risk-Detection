import os
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_FILE = os.path.join(BASE_DIR, 'report', 'ClinIQ_Architecture.png')

fig = go.Figure()

# Define blocks: (label, x, y, width, height, color)
blocks = [
    # Layer 1: Data Sources
    ("Kaggle Patient Reviews<br>(52k Texts)", 1, 3, 2.2, 0.9, '#000000'),
    ("CDC & CMS Guidelines<br>(ICD-10, DRG)", 1, 1, 2.2, 0.9, '#000000'),
    
    # Layer 2: Ingestion & VectorDB
    ("Sentence Transformers<br>(all-MiniLM-L6-v2)", 4, 2, 2.2, 0.9, '#000000'),
    ("PostgreSQL + pgvector<br>(Knowledge Base)", 6.5, 1, 2.2, 0.9, '#000000'),
    ("UMAP & HDBSCAN<br>(Semantic Cluster)", 6.5, 3, 2.2, 0.9, '#000000'),
    
    # Layer 3: Inference Pipeline
    ("Claude 3.5 Sonnet<br>RAG Orchestrator", 9.5, 2, 2.4, 1.2, '#000000'),
    
    # Layer 4: Clinical Bridge Outputs
    ("Clinical Notes Gap<br>Detection", 13, 3, 2.6, 0.9, '#000000'),
    ("DRG Revenue Recovery<br>Calculator", 13, 1, 2.6, 0.9, '#000000'),
]

for label, cx, cy, w, h, color in blocks:
    # Add rectangle
    fig.add_shape(
        type="rect",
        x0=cx - w/2, y0=cy - h/2,
        x1=cx + w/2, y1=cy + h/2,
        fillcolor=color,
        line=dict(color="white", width=2),
        layer="below"
    )
    # Add text
    fig.add_annotation(
        x=cx, y=cy,
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(color="white", size=15),
        align="center"
    )

# Define arrows (x0, y0, x1, y1)
arrows = [
    (2, 3, 3, 2.2),     # Kaggle -> Transformers
    (2, 1, 3, 1.8),     # CDC -> Transformers
    (5, 2, 5.5, 1.2),   # Transformers -> pgvector
    (5, 2, 5.5, 2.8),   # Transformers -> UMAP
    (7.5, 1, 8.25, 1.8),   # pgvector -> LLM
    (7.5, 3, 8.25, 2.2),   # UMAP -> LLM
    (10.75, 2, 11.75, 2.8), # LLM -> Clinical Gap
    (10.75, 2, 11.75, 1.2), # LLM -> Revenue Calc
]

for x0, y0, x1, y1 in arrows:
    fig.add_annotation(
        x=x1, y=y1,
        ax=x0, ay=y0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=2.5,
        arrowcolor="#95a5a6"
    )

# Formatting
fig.update_layout(
    title=dict(
        text="<b>System Architecture: ClinIQ Closed-Loop Retrieval-Augmented Generation (RAG) Pipeline</b>",
        font=dict(size=22, color="#333"),
        x=0.5
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=1400,
    height=600,
    xaxis=dict(visible=False, range=[0, 15]),
    yaxis=dict(visible=False, range=[0, 4]),
    margin=dict(t=80, b=40, l=40, r=40)
)

print("Exporting plot to PNG using kaleido...")
fig.write_image(OUT_FILE, scale=4.0)  # scale 4 enables crazy high-res for poster
print(f"Architecture diagram saved to {OUT_FILE}")
