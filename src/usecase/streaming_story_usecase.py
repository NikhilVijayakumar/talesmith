# src/usecase/streaming_story_usecase.py
import threading

from src.configuration.config import Config
from src.llm.model import initialize_llm
from src.rag.chain import setup_rag_chain
from src.rag.retriever import load_and_tokenize_documents, setup_vectorstore
from src.thread.ThreadedGenerator import ThreadedGenerator


def run_query(rag_chain, query):
    result = rag_chain.invoke(query)
    return result


def start_threaded_query(rag_chain, query):
    generator = ThreadedGenerator()
    threading.Thread(target=lambda: run_query(rag_chain, query)).start()
    return generator


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


def get_streaming_story(state, query):
    start_threaded_query(state["rag_chain"], query)
    for token in state["generator"]:
        if token is None:
            break
        print(token, end='')


def generate_streaming_story():
    data = init()
    prompt = Config.get_prompt()
    print(prompt)
    get_streaming_story(data, prompt)
