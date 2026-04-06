import os
import sys
import psycopg2
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set in .env")
    sys.exit(1)

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.sql')

TABLES = [
    "drug_reviews",
    "cdc_overdose",
    "rag_embeddings",
    "rag_source_registry",
    "dim_patient",
    "dim_provider",
    "dim_diagnosis",
    "fact_claims",
    "ai_risk_findings",
    "method_comparison",
    "temporal_analysis",
]

def setup():
    print(f"Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()

    print("Executing schema.sql...\n")
    with open(SCHEMA_PATH, 'r') as f:
        sql = f.read()
    cur.execute(sql)

    print("Verifying tables:")
    for table in TABLES:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s);",
            (table,)
        )
        exists = cur.fetchone()[0]
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {table}")

    cur.close()
    conn.close()
    print("\nDatabase setup complete. All tables ready.")

if __name__ == "__main__":
    setup()
