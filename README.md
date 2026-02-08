# 🧠 COREP RAG Assistant

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/) 
[![Gradio](https://img.shields.io/badge/Gradio-6.5.1-orange?logo=gradio)](https://gradio.app/)
[![Transformers](https://img.shields.io/badge/Transformers-4.57.6-purple?logo=huggingface)](https://huggingface.co/docs/transformers/index)

- A **RAG (Retrieval-Augmented Generation)** assistant for analyzing PDFs and answering regulatory questions (like EBA/PRA COREP reports).  

- Upload PDFs to the `data/` folder, build a vector database using **FAISS**, and ask questions using **FLAN-T5** and **HuggingFace embeddings**.  

---

## 📁 Project Structure
```bash
corep-assistant/
│
├─ data/ # Place your PDFs here
├─ docs/ # Optional documentation files
├─ faiss_index/ # FAISS database will be saved here
│ ├─ index.faiss
│ └─ index.pkl
├─ vector_db/ # Optional, for future vector storage
│
├─ main.py # Command-line interface for building DB & querying
├─ app.py # Gradio web interface
├─ rag_pipeline.py # Core functions: load docs, build DB, query DB
├─ requirements.txt # Python dependencies
├─ README.md # This file
└─ .env # Optional environment variables
```

---

## ⚡ Features

- Load PDFs from `data/` and clean text automatically
- Split text into chunks for embedding
- Build a **FAISS vector database**
- Query your PDFs using **natural language questions**
- Answers generated using **FLAN-T5-small**
- Web interface via **Gradio** for easy use

---

## 🛠 Installation

1. Clone this repository:

```bash
git clone <your-repo-url>
cd corep-assistant
```
## Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```
## Install dependencies:
```bash
pip install -r requirements.txt
```
## 💻 Running from Command Line

### Build database:
```bash
python main.py
```
Choose 1 to build the FAISS database from PDFs in data/.
Choose 2 and type your question.
Type exit to quit.

## 🌐 Running the Gradio Web App
1.Make sure the FAISS database is built.
2.Run:
```bash
python app.py
```
3.Open your browser at:
```bash
http://localhost:7860
```

4. Features:
- Build / update the database
- Type questions and get answers along with source pages

## 📄 Usage Notes

- Place all PDFs in the data/ folder before building the database.

- The database will be saved to faiss_index/.

- If no answer is found in the documents, the assistant will return "Not found in document".

- You can increase chunk_size in rag_pipeline.py for larger text chunks if needed.

## ⚙️ Dependencies
- Key Python libraries:

- transformers

- torch

- faiss-cpu

- langchain, langchain-huggingface, langchain-community

- gradio

- PyPDF2 or pypdf

See full requirements.txt for all dependencies

## 📌 Tips

- For faster performance, use a GPU if available by changing device=-1 to device=0 in the pipeline.

- Regularly update requirements.txt with:

```bash
pip freeze > requirements.txt
```
## 🙋‍♂️ Author
Akshit Sharma|akshit6299@outlook.com|Bestnwin


