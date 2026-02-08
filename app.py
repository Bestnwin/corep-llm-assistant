import os
import re
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------- Paths ----------
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
        return docs
    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            pages = loader.load()
            for p in pages:
                p.page_content = clean_text(p.page_content)
            docs.extend(pages)
    return docs

# ---------- Build Vector DB ----------
def build_db():
    documents = load_docs()
    if len(documents) == 0:
        return "❌ No PDFs found in the 'data/' folder."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    return f"✅ Database built with {len(chunks)} chunks!"

# ---------- Query DB ----------
def query_db(question):
    if not os.path.exists(DB_PATH):
        return "❌ Database not found. Please build it first.", ""

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    results = db.max_marginal_relevance_search(question, k=4, fetch_k=12)
    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""
You are a helpful assistant. Use the context below to write a clear, concise, human-readable answer.
If the answer is not in the context, respond with "Not found in document".

Context:
{context}

Question:
{question}

Answer:
"""

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )

    output = generator(prompt, max_new_tokens=256)[0]["generated_text"]
    sources = [f"Page {r.metadata.get('page')}" for r in results]

    return output.strip(), ", ".join(sources)

# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style="text-align:center; color:#4B0082;">🧠 COREP RAG Assistant</h1>
        <p style="text-align:center;">Upload PDFs to the <code>data/</code> folder, build the database, then ask questions.</p>
        """
    )

    with gr.Group():
        gr.Markdown("### 1️⃣ Build / Update Database")
        with gr.Row():
            build_btn = gr.Button("Build Database", variant="primary")
            build_output = gr.Textbox(label="Build Status", interactive=False, lines=2, placeholder="Status messages will appear here...")

    with gr.Group():
        gr.Markdown("### 2️⃣ Ask Questions")
        question_input = gr.Textbox(label="Type your question here", lines=2, placeholder="E.g., What is the EBA Risk Reduction Package?")
        ask_btn = gr.Button("Get Answer", variant="primary")

        with gr.Row():
            answer_output = gr.Textbox(label="Answer", lines=8, interactive=False, placeholder="The assistant's answer will appear here...", show_label=True)
            sources_output = gr.Textbox(label="Source Pages", lines=4, interactive=False, placeholder="Pages used for answer...", show_label=True)

    # ---------- Button Actions ----------
    build_btn.click(fn=build_db, inputs=None, outputs=build_output)
    ask_btn.click(fn=query_db, inputs=question_input, outputs=[answer_output, sources_output])

# ---------- Launch ----------
demo.launch(server_name="0.0.0.0", server_port=7860)