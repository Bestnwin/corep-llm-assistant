import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


DB_PATH = "faiss_index"
DATA_PATH = "data"


# ---------- Load PDFs ----------
def load_docs():
    docs = []

    if not os.path.exists(DATA_PATH):
        print("❌ data/ folder not found")
        return docs

    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".pdf"):
            print(f"Loading {file}")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())

    print("DOC COUNT:", len(docs))
    return docs


# ---------- Build Vector Database ----------
def build_db():

    documents = load_docs()

    if len(documents) == 0:
        print("\n⚠️ No PDFs loaded. Put COREP PDFs inside /data")
        return

    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(DB_PATH)

    print("✅ Database built and saved!")


# ---------- Query Database ----------
def query_db(question):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = db.similarity_search(question, k=3)

    print("\n🔎 Top Results:\n")

    for r in results:
        print(r.page_content)
        print("-" * 60)