from langchain_community.llms import LlamaCpp
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import threading
import queue
from langchain_core.outputs import LLMResult
from src.config import Config

def load_and_tokenize_documents(directory, chunk_size=512, chunk_overlap=16):
    loader = PyPDFDirectoryLoader(directory)
    all_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    docs = text_splitter.split_documents(all_docs)
    return docs


def create_embedding_function(model_name="BAAI/bge-large-en-v1.5"):
    encode_kwargs = {'normalize_embeddings': True}
    embedding_function = HuggingFaceBgeEmbeddings(
        model_name=model_name, encode_kwargs=encode_kwargs
    )
    return embedding_function

def create_vectorstore(docs, embedding_function, persist_directory="./chroma_db"):
    db = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)
    db.persist()
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def initialize_llm(model_url, generator):
    model = LlamaCpp(
        model_path=model_url,
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        streaming=True,
        callback_manager=CallbackManager([ChainStreamHandler(generator)]),
        n_ctx=8096,
        verbose=True,
    )
    return model


def create_retriever(vectorstore, k=5):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def create_rag_chain(retriever, model):
    template = """<human>: Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':
    ### CONTEXT
    {context}
    ### QUESTION
    Question: {question}
    \n
    <bot>:
    """
    prompt = ChatPromptTemplate.from_template(template)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)



class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Run when LLM ends running."""
        print("\n\ngeneration concluded")
        self._queue.put(None)

def load_and_preprocess_documents():
    return load_and_tokenize_documents("E:/Python/talesmith/data/pdfs/")


def setup_vectorstore(docs):
    embedding_function = create_embedding_function()
    return create_vectorstore(docs, embedding_function)


def setup_llm(generator):
    return initialize_llm(Config.MODEL_PATH, generator)


def setup_rag_chain(vectorstore, model):
    retriever = create_retriever(vectorstore)
    return create_rag_chain(retriever, model)


def run_query(rag_chain, query):
    result = rag_chain.invoke(query)
    return result

def start_threaded_query(rag_chain, query):
    generator = ThreadedGenerator()
    threading.Thread(target=lambda: run_query(rag_chain, query)).start()
    return generator


def init():
  docs = load_and_preprocess_documents()
  vectorstore = setup_vectorstore(docs)
  generator = ThreadedGenerator()
  model = setup_llm(generator)
  rag_chain = setup_rag_chain(vectorstore, model)
  return {
        "generator": generator,
        "rag_chain": rag_chain
    }

def generate_story(state,query):
  start_threaded_query(state["rag_chain"], query)
  for token in state["generator"]:
    if token is None:
      break
    print(token, end='',flush=True)


state = init()

query = "i was alone with my hot couin"
generate_story(state,query)