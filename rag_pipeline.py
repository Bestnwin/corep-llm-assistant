import os
import re
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# ================= PATHS =================
DB_PATH = "faiss_index"
DATA_PATH = "data"


# ================= LOAD MODEL ONCE =================
print("🔄 Loading LLM once...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1
)


# ================= CLEAN TEXT =================
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ================= LOAD DOCS =================
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


# ================= BUILD VECTOR DB =================
def build_db():
    documents = load_docs()

    if not documents:
        print("⚠️ No PDFs loaded")
        return

    print("✂️ Splitting text...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,           # safer for T5
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    print("📦 Building FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)

    print("✅ Database built!")


# ================= TEMPLATE FORMAT =================
def format_template(structured):
    if "fields" not in structured:
        return "No structured data available"

    lines = ["\n📋 COREP Template Extract\n"]
    for k, v in structured["fields"].items():
        val = v if v else "⚠ Missing"
        lines.append(f"{k:<28} : {val}")

    return "\n".join(lines)


# ================= VALIDATION =================
def validate_fields(structured):
    warnings = []

    for k, v in structured.get("fields", {}).items():

        if v == "":
            warnings.append(f"{k} is missing")

        if "Capital" in k and v:
            if not any(ch.isdigit() for ch in v):
                warnings.append(f"{k} might not be numeric")

    return warnings


# ================= AUDIT =================
def audit_log(results):
    log = []
    for i, r in enumerate(results, 1):
        page = r.metadata.get("page")
        src = r.metadata.get("source", "PDF")
        log.append(f"Evidence {i} → {src} | Page {page}")
    return "\n".join(log)


# ================= QUERY =================
def query_db(question):

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    # 🔥 Smaller retrieval for token safety
    results = db.max_marginal_relevance_search(question, k=2, fetch_k=6)

    context = "\n\n".join([r.page_content for r in results])

    # HARD TOKEN LIMIT PROTECTION
    context = context[:1500]

    prompt = f"""
Extract structured COREP data.

Respond ONLY with JSON.
No text before or after.

Format:

{{
 "template":"Own Funds",
 "fields":{{
   "CET1 Capital":"",
   "AT1 Capital":"",
   "Tier2 Capital":"",
   "Risk Exposure Amount":""
 }}
}}

Context:
{context}

Question:
{question}
"""

    raw = generator(prompt, max_new_tokens=200)[0]["generated_text"]

    print("\n🧾 RAW MODEL OUTPUT:\n", raw)

    # ---------- Robust JSON Parse ----------
    try:
        json_text = re.search(r'\{.*\}', raw, re.DOTALL).group(0)
        structured = json.loads(json_text)
    except:
        print("\n❌ JSON parse failed")
        structured = {"template": "Parse Error", "fields": {}}

    # ---------- Display ----------
    print(format_template(structured))

    # ---------- Validation ----------
    warnings = validate_fields(structured)
    print("\n🔎 Validation Results")

    if warnings:
        for w in warnings:
            print("⚠", w)
    else:
        print("✅ No issues detected")

    # ---------- Audit ----------
    print("\n📚 Audit Log")
    print(audit_log(results))