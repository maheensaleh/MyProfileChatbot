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
