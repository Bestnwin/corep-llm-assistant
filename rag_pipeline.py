from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


DATA_PATH = "data/sample.txt"
DB_PATH = "vectordb"


def build_db():

    loader = TextLoader(DATA_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory=DB_PATH
    )

    vectordb.persist()


def query_db(q):

    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=OpenAIEmbeddings()
    )

    results = vectordb.similarity_search(q, k=2)

    return results