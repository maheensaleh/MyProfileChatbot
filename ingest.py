from pathlib import Path
import argparse
import sys
import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import os, streamlit as st
from dotenv import load_dotenv
load_dotenv()  # still works locally

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
HF_API_TOKEN =  st.secrets.get("HUGGING_FACE_API_TOKEN", os.getenv("HUGGING_FACE_API_TOKEN"))

EMBED_MODEL_NAME = st.secrets.get("HUGGING_FACE_EMBEDDING_MODEL", os.getenv("HUGGING_FACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
LLM_MODEL_NAME = st.secrets.get("LLM_MODEL", os.getenv("LLM_MODEL", "gemini-1.5-flash"))

ROOT_DIR = Path(__file__).parent
INDEX_DIR = Path(f"{ROOT_DIR}/data_index")  
DATA_DIR = Path(f"{ROOT_DIR}/data")  


def load_documents(data_dir: Path):
    docs = []
    for path in data_dir.rglob("*"):
        if path.is_dir():
            continue
        try:
            if path.suffix.lower() in [".txt", ".md"]:
                docs.extend(TextLoader(str(path), encoding="utf-8").load())
            elif path.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(path)).load())
        except Exception as e:
            print(f"[skip] {path.name}: {e}", file=sys.stderr)
    if not docs:
        raise RuntimeError(f"No documents found in {data_dir}. Put .txt/.md/.pdf files there.")
    return docs

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def main():
    parser = argparse.ArgumentParser(description="Ingest documents and build FAISS index.")
    args = parser.parse_args()



    print(f"Loading documents from {DATA_DIR}")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents. Building index…")

    vs = build_vectorstore(docs)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))

    # Persist embedding model name for safety
    (INDEX_DIR / "embeddings_model.txt").write_text(EMBED_MODEL_NAME, encoding="utf-8")

    print(f"Index saved to {INDEX_DIR.resolve()}")

if __name__ == "__main__":
    main()
