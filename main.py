from rag_pipeline import build_db, query_db

def run():
    print("\n===== COREP RAG Assistant =====")
    print("1️⃣ Build Database")
    print("2️⃣ Ask Questions")

    choice = input("Choose option: ").strip()

    if choice == "1":
        build_db()

    elif choice == "2":
        while True:
            q = input("\nAsk something (type 'exit'): ").strip()

            if q.lower() == "exit":
                print("👋 Exiting...")
                break

            if q == "":
                continue

            query_db(q)

    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    run()