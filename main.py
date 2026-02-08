from rag_pipeline import build_db, query_db

print("\n===== COREP RAG Assistant =====")
print("1️⃣ Build Database")
print("2️⃣ Ask Questions")

choice = input("Choose option: ")

if choice == "1":
    build_db()

elif choice == "2":

    while True:
        q = input("\nAsk something (type 'exit'): ")

        if q.lower() == "exit":
            break

        query_db(q)

else:
    print("Invalid choice")