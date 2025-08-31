from pathlib import Path
import os
import textwrap

# LangChain (HF + community)
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceHubEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFaceHub

# Hugging Face transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

DATA_DIR = Path("/Users/maheensaleh/Documents/myprojects/MyProfileChatbot/data")  

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
            print(f"[skip] {path.name}: {e}")
    if not docs:
        raise RuntimeError(f"No documents found in {data_dir}. Put .txt/.md/.pdf files there.")
    return docs

def build_retriever(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    # HF sentence-transformers embeddings (local)
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)

#    # Embeddings via Hugging Face Inference API (no local model)
#     embed_model = "sentence-transformers/all-MiniLM-L6-v2"
#     embeddings = HuggingFaceHubEmbeddings(
#         repo_id=embed_model,
#         # Batch size helps when indexing many chunks (tune if needed)
#         task="feature-extraction",
#     )

    vs = FAISS.from_documents(chunks, embeddings)
    return vs.as_retriever(search_kwargs={"k": 4})


PROMPT_TMPL = """You are a helpful chatbot that answers questions about the candidate's profile for recruiters.
Use ONLY the provided context. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

def build_chain(retriever, model_name="google/flan-t5-base", llm_repo_id="mistralai/Mistral-7B-Instruct-v0.3"):
    # Local HF pipeline (CPU-friendly model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
    )
    llm = HuggingFacePipeline(pipeline=gen)


    # # Text-generation via Hugging Face Inference API
    # llm = HuggingFaceHub(
    #     repo_id=llm_repo_id,
    #     task="text-generation",
    #     model_kwargs={
    #         "max_new_tokens": 512,
    #         "temperature": 0.1,
    #         "return_full_text": False,
    #     },
    # )
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TMPL,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    return qa


def main():

    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        print("Please set HUGGINGFACEHUB_API_TOKEN environment variable.")
        return

    print("Loading documents from", DATA_DIR)
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents. Building index…")
    retriever = build_retriever(docs)
    print('Retriever built successfully')
    # exit()
    chain = build_chain(retriever)

    print("\nRecruiter Chatbot ready. Ask about the candidate's profile.")
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

        print("\nAssistant:", textwrap.fill(answer, width=100))
        print()

if __name__ == "__main__":
    main()
