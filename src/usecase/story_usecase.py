# src/usecase/story_usecase.py
import threading

from src.configuration.config import Config
from src.llm.model import initialize_llm
from src.rag.chain import setup_rag_chain
from src.rag.retriever import load_and_tokenize_documents, setup_vectorstore
from src.thread.ThreadedGenerator import ThreadedGenerator
from src.utils.file_utils import save_generated_text


def run_query(rag_chain, query):
    result = rag_chain.invoke(query)
    return result


def init():
    docs = load_and_tokenize_documents(Config.PDF_DIRECTORY)
    vectorstore = setup_vectorstore(docs)
    generator = ThreadedGenerator()
    model = initialize_llm(generator)
    rag_chain = setup_rag_chain(vectorstore, model)
    return {
        "generator": generator,
        "rag_chain": rag_chain
    }


def get_story(state, query):
    print("Stated generating ....")
    result = run_query(state["rag_chain"], query)
    name = "story"
    save_generated_text(result, name)
    print("Completed")
    print(result)


def generate_story():
    data = init()
    prompt = Config.get_prompt()
    print(prompt)
    get_story(data, prompt)
