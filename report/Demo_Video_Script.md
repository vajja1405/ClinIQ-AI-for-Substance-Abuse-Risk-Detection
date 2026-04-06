# ClinIQ - 3-Minute Video Demo Script
**Total estimated time:** 2 minutes 55 seconds (approx. 385 words)
**Pacing:** Confident, professional, steady (130 words per minute)

---

### [0:00 - 0:30] Introduction & Main Goal
**[Visual Cue: Start on the "Risk Signal Detection" tab.]**

**Speaker:**
"Welcome to ClinIQ. The opioid crisis is one of the deadliest epidemics of our generation, yet 90% of people with Substance Use Disorder receive zero treatment. 

Worse, when these patients visit hospitals for other acute reasons—like pneumonia—their underlying addiction is rarely documented in the clinical billing codes. This creates a massive dual gap: Hospitals lose millions in Medicare severity reimbursements, and the CDC loses critical public health surveillance data.

ClinIQ closes this gap. By leveraging RAG-powered agentic AI, we trace early-warning distress signals from social text directly into clinical documentation failures."

### [0:30 - 1:00] Innovation 1: Signal Detection
**[Visual Cue: Stay on 'Risk Signal Detection'. Scroll slowly down the Three-Method comparison charts.]**

**Speaker:**
"Our pipeline begins by analyzing over 50,000 anonymized patient reviews. Instead of a single model, we evaluate a three-tiered approach. 

Our Rule-Based NLP acts as a fast baseline, while Cosine Embeddings perfectly isolate high-recall screening. But our core innovation is the LLM-RAG agent. We've securely embedded official CMS billing guidelines and real patient experiences into a PostgreSQL vector database. Claude queries this knowledge base at inference, achieving 88% precision by successfully recognizing localized slang for withdrawal that traditional keyword parsers fundamentally misunderstand."

### [1:00 - 1:40] Innovation 2: Advanced Visual Analytics
**[Visual Cue: Click to the "Advanced Data Discovery" tab. Hover over the 3D UMAP and interact with the Sunburst chart.]**

**Speaker:**
"To make these high-dimensional insights actionable for hospital executives, ClinIQ features a suite of Tableau-grade analytics.

Here in our interactive 3D Behavioral Topological Map, we've clustered patient embeddings using UMAP and HDBSCAN, elevating extreme-distress negative sentiment along the Z-axis. Next to it, our Sunburst Matrix visualizes the intricate taxonomy between specific drug prescriptions and severe withdrawal experiences. 

This proves the crisis isn't monolithic—behaviors are diversifying into polysubstance abuse, and our AI maps it dynamically."

### [1:40 - 2:30] Innovation 3: The Clinical Bridge
**[Visual Cue: Click to the "Clinical Billing Impact" tab. Show the $7-8k revenue lift and the Treemap.]**

**Speaker:**
"So, how does this translate to the real world? 

We routed 105 synthetic hospital admission claims through our RAG agent. The AI successfully detected 46 uncoded instances of Substance Use Disorder. 

Because we integrated live CMS DRG severity weights into our database, the system automatically quantified this as nearly $8,000 in uncaptured, direct revenue. When extrapolated to an average urban hospital, automating this detection recovers approximately $27.5 Million dollars annually, while simultaneously correcting the CDC's structural undercount."

### [2:30 - 3:00] Live Demonstration & Conclusion
**[Visual Cue: Click to the "Live RAG Claim Analyzer" tab. Press 'Analyze Claim' and watch it generate the JSON.]**

**Speaker:**
"We can watch this happen live. Our clinician uploads an unstructured patient note. The agent queries the pgvector database in real-time, infers the missing ICD-10 dependency code, and instantly generates an evidence-backed physician query letter citing the exact CDC guideline constraint.

Zero patient data leaves the system. Zero AI hallucinations. By bridging population-level social signals with hospital revenue recovery, ClinIQ makes ending the opioid crisis financially sustainable. 

Thank you."
