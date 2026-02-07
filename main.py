from rag_pipeline import build_db, query_db
import os

# Build DB only first time
if not os.path.exists("faiss_index"):
    build_db()


while True:
    q = input("\nAsk something (type 'exit'): ")

    if q.lower() == "exit":
        break

    query_db(q)