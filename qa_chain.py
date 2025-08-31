import argparse
import textwrap
from pathlib import Path
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
# from langchain_community.llms import HuggingFaceHub 

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

load_dotenv()

EMBED_MODEL_NAME = os.getenv("HUGGING_FACE_EMBEDDING_MODEL")
LLM_MODEL_NAME = os.getenv("HUGGING_FACE_LLM_MODEL")
HF_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
ROOT_DIR = Path(__file__).parent
INDEX_DIR = Path(f"{ROOT_DIR}/data_index")  
PROMPT_TMPL = """You are a helpful chatbot that answers questions about the candidate's profile for recruiters. You will act as the candidate when answering the questions.
Use ONLY the provided context. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

def load_retriever(index_dir: Path, k: int = 4):
    # Ensure we use the same embedding model that was used during ingest
    embed_model_name_path = index_dir / "embeddings_model.txt"
    if not embed_model_name_path.exists():
        raise RuntimeError(f"Missing {embed_model_name_path}. Re-run ingest.py.")
    embed_model_name = embed_model_name_path.read_text(encoding="utf-8").strip()

    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": k})

def build_chain(retriever, model_name: str = LLM_MODEL_NAME):
    # Local HF pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
    )
    llm = HuggingFacePipeline(pipeline=gen)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TMPL,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa

def main():
    parser = argparse.ArgumentParser(description="Run recruiter Q/A over a saved FAISS index.")
    args = parser.parse_args()

    retriever = load_retriever(INDEX_DIR)
    chain = build_chain(retriever)

    print("\My Profile Chatbot ready. Ask about me.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        try:
            res = chain.invoke({"query": q})
            answer = res["result"] if isinstance(res, dict) else str(res)
        except Exception as e:
            answer = f"[error] {e}"

        print("\nMaheen:", textwrap.fill(answer, width=100))
        print()

if __name__ == "__main__":
    main()
