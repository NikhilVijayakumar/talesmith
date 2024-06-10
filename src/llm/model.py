#src/llm/model.py
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from src.thread.ChainStreamHandler import ChainStreamHandler
from src.configuration.config import Config

def streaming_llm_init(generator):
    return LlamaCpp(
        model_path=Config.MODEL_PATH,
        temperature=0.9,
        max_tokens=0,
        top_p=3,
        streaming=True,
        callback_manager=CallbackManager([ChainStreamHandler(generator)]),
        n_ctx=Config.CONTEXT_WINDOW,
        verbose=True,
    )


def llm_init():
    return LlamaCpp(
        model_path=Config.MODEL_PATH,
        temperature=0.9,
        max_tokens=0,
        n_ctx=Config.CONTEXT_WINDOW,
        verbose=True,
    )



