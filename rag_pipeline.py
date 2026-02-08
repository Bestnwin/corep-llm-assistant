import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

DB_PATH = "faiss_index"
DATA_PATH = "data"

# ---------- Text Cleaning ----------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    return text

# ---------- Load PDFs ----------
def load_docs():
    docs = []

    if not os.path.exists(DATA_PATH):
        print("❌ data/ folder not found")
        return docs

    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".pdf"):
            print(f"📄 Loading {file}")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            pages = loader.load()

            for p in pages:
                p.page_content = clean_text(p.page_content)

            docs.extend(pages)

    print("DOC COUNT:", len(docs))
    return docs

# ---------- Build Vector Database ----------
def build_db():
    documents = load_docs()

    if len(documents) == 0:
        print("⚠️ No PDFs loaded")
        return

    print("✂️ Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    print("📦 Building FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    print("✅ Database built and saved!")

# ---------- Query + Answer using FLAN-T5 ----------
def query_db(question):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    results = db.max_marginal_relevance_search(question, k=4, fetch_k=12)

    # Build context
    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""
    You are a helpful assistant. Use the context below to **write a clear, concise, human-readable answer** to the question. 
    Summarize the key points in full sentences. If the answer is not in the context, respond with "Not found in document".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """


    # ---------- FLAN-T5 Small ----------
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU
    )

    output = generator(prompt, max_new_tokens=128)[0]["generated_text"]

    print("\n🤖 Answer:\n")
    print(output.strip())

    print("\n📚 Sources:")
    for i, r in enumerate(results, 1):
        print(f"{i}. Page:", r.metadata.get("page"))


if __name__ == "__main__":
    print("===== COREP RAG Assistant =====")
    print("1️⃣ Build Database")
    print("2️⃣ Ask Questions")
    choice = input("Choose option: ")

    if choice == "1":
        build_db()
    elif choice == "2":
        while True:
            q = input("\nAsk something (type 'exit' to quit): ")
            if q.lower() == "exit":
                break
            query_db(q)