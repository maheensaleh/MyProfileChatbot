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
from langchain_community.llms import HuggingFaceHub 

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


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def build_chain_qwen(retriever, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    # Qwen2.5 is a causal LM (decoder-only), not seq2seq.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure padding token exists (use EOS as pad for causal models if missing)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name)

    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,            # deterministic for QA
        truncation=True,            # avoid context overruns
        return_full_text=False,     # only the generated answer
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    llm = HuggingFacePipeline(pipeline=gen)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TMPL,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                 # keep as in your snippet
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def build_chain_gemma(retriever, model_name: str = "google/gemma-2-2b-it"):
    # Gemma 2 is a causal LM (decoder-only)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name)

    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,            # deterministic for QA
        truncation=True,            # avoid context overruns
        return_full_text=False,     # only generated continuation
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    llm = HuggingFacePipeline(pipeline=gen)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TMPL,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                 # keep your current behavior
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa

from langchain_google_genai import ChatGoogleGenerativeAI

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
    # chain = build_chain(retriever)
    # chain = build_chain_qwen(retriever)
    # chain = build_chain_gemma(retriever)
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
