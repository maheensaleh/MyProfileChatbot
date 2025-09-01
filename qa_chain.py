import argparse
import textwrap
from pathlib import Path
import os
from dotenv import load_dotenv
from qa_prompts import PROMPT_TMPL

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

EMBED_MODEL_NAME = os.getenv("HUGGING_FACE_EMBEDDING_MODEL")
LLM_MODEL_NAME = os.getenv("HUGGING_FACE_LLM_MODEL")
HF_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
ROOT_DIR = Path(__file__).parent
INDEX_DIR = Path(f"{ROOT_DIR}/data_index")  



def load_retriever(index_dir: Path, k: int = 4):
    # Ensure we use the same embedding model that was used during ingest
    embed_model_name_path = index_dir / "embeddings_model.txt"
    if not embed_model_name_path.exists():
        raise RuntimeError(f"Missing {embed_model_name_path}. Re-run ingest.py.")
    embed_model_name = embed_model_name_path.read_text(encoding="utf-8").strip()

    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": k})



def build_chain_gemini(retriever):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in your .env to use the Gemini inference endpoint.")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

    # Uses Google Generative AI (Gemini) hosted inference endpoint
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.1,
        max_output_tokens=512,
        convert_system_message_to_human=True,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TMPL,
    )

    # map_reduce keeps per-call size manageable and robust
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

    chain = build_chain_gemini(retriever)

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
