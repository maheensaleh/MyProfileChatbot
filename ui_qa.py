import os
import subprocess
from pathlib import Path
from typing import List
import streamlit as st
from qa_prompts import PROMPT_TMPL

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from huggingface_hub import InferenceClient

from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBED_MODEL_NAME = os.getenv("HUGGING_FACE_EMBEDDING_MODEL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL")

ROOT_DIR = Path(__file__).parent
INDEX_DIR = Path(f"{ROOT_DIR}/data_index")  



class HFAPIEmbeddings(Embeddings):
    def __init__(self, repo_id: str, token: str | None = None, timeout: float = 120.0):
        self.client = InferenceClient(model=repo_id, token=token, timeout=timeout)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.feature_extraction(texts)

    def embed_query(self, text: str) -> List[float]:
        vec = self.client.feature_extraction(text)
        return vec[0] if (isinstance(vec, list) and vec and isinstance(vec[0], list)) else vec



def build_chain_gemini(retriever, _llm_repo, _max_new, _temp, _show_sources):
    if not GOOGLE_API_KEY:
        raise RuntimeError("Set GOOGLE_API_KEY in your .env to use the Gemini inference endpoint.")

    # Uses Google Generative AI (Gemini) hosted inference endpoint
    llm = ChatGoogleGenerativeAI(
        model=_llm_repo,
        api_key=GOOGLE_API_KEY,
        temperature=_temp,
        max_output_tokens=_max_new,
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
        return_source_documents=_show_sources,
    )
    return qa



# ========================= Streamlit UI =========================
st.set_page_config(page_title="Recruiter Profile Chatbot", page_icon="💬", layout="centered")
st.title("💬 Recruiter Profile Chatbot")
st.caption("RAG over your profile docs using FAISS + Hugging Face Inference API")

# Sidebar settings
st.sidebar.header("Settings")
hf_token = HF_API_TOKEN
if not hf_token:
    st.sidebar.warning("HUGGINGFACEHUB_API_TOKEN is not set. Set it in your shell before running the app.")

# store_dir = st.sidebar.text_input("FAISS store path", value=INDEX_DIR)

# llm_repo_id = st.sidebar.text_input("LLM repo (HF)", value=LLM_MODEL_NAME)
# embed_repo_id = st.sidebar.text_input("Embedding model (HF)", value=EMBED_MODEL_NAME)

# Display model names as text (read-only)
st.sidebar.markdown(f"**Embedding Model:** `{EMBED_MODEL_NAME}`")
st.sidebar.markdown(f"**Chat Model:** `{LLM_MODEL_NAME}`")


# k = st.sidebar.number_input("Top-k retrieved chunks", min_value=1, max_value=20, value=4, step=1)
k = 4
# max_new_tokens = st.sidebar.number_input("Max new tokens", min_value=64, max_value=2048, value=512, step=64)
max_new_tokens = 512
# temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
temperature = 0.1
# show_sources = st.sidebar.checkbox("Show sources", value=False)
show_sources = False



# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []  # list of (user, assistant, sources)

# Load vector store & chain lazily, cache across reruns
@st.cache_resource(show_spinner=True)
def _load_chain(_store_dir: str, _embed_repo: str, _llm_repo: str, _k: int, _max_new: int, _temp: float, _show_sources: bool):
    if not Path(_store_dir).exists():
        raise FileNotFoundError(f"FAISS store not found at '{_store_dir}'. Run ingest.py first.")
    embeddings = HFAPIEmbeddings(repo_id=_embed_repo, token=hf_token)
    vs = FAISS.load_local(
        _store_dir,
        embeddings,
        allow_dangerous_deserialization=True,  # required by newer LC versions
    )
    retriever = vs.as_retriever(search_kwargs={"k": 4}) # hardcoded, change later
    chain = build_chain_gemini(retriever, _llm_repo, _max_new, _temp, _show_sources)
    return chain

# Try to prepare chain
chain = None
load_error = None
with st.spinner("Preparing retriever & LLM…"):
    try:
        chain = _load_chain(INDEX_DIR, EMBED_MODEL_NAME, LLM_MODEL_NAME, k, max_new_tokens, temperature, show_sources)
    except Exception as e:
        load_error = str(e)

if load_error:
    st.error(load_error)
    st.stop()

# Chat UI
user_input = st.text_input("Ask about the candidate’s profile:", value="", placeholder="e.g., What are their key projects?")
ask = st.button("Ask")

def render_sources(docs):
    if not docs:
        return
    st.markdown("**Sources**")
    st.error(docs)
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        label = f"{Path(src).name}" + (f" (page {page+1})" if isinstance(page, int) else "")
        with st.expander(f"{i}. {label}"):
            st.write(d.page_content[:1500] + ("…" if len(d.page_content) > 1500 else ""))

if ask and user_input.strip():
    with st.spinner("Thinking…"):
        try:
            res = chain.invoke({"query": user_input.strip()})
            if isinstance(res, dict):
                answer = res.get("result", "")
                sources = res.get("source_documents", []) if show_sources else []
            else:
                answer, sources = str(res), []
        except Exception as e:
            answer, sources = f"[error] {e}", []

    st.session_state.history.append((user_input.strip(), answer, sources))

# Display history
# print(st.session_state.history)
for q, a, srcs in st.session_state.history:
    # print('q a')
    # print(q)
    # print(a)
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")
    if show_sources:
        render_sources(srcs)
    st.markdown("---")

# Footer
st.caption("Tip: If you change model/settings in the sidebar, the app reloads the chain automatically.")
