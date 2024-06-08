from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.outputs import LLMResult


class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)

