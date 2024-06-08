#src/rag/retriever.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.configuration.config import Config
import os

def load_and_tokenize_documents(directory, chunk_size=512, chunk_overlap=16):
    if not os.path.exists(Config.CHROMA_DIRECTORY):
        loader = PyPDFDirectoryLoader(directory)
        all_docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        docs = text_splitter.split_documents(all_docs)
        return docs
    else:
        return None

def create_embedding_function():
    encode_kwargs = {'normalize_embeddings': True}
    embedding_function = HuggingFaceBgeEmbeddings(
        model_name=Config.EMBEDDING_MODEL_NAME, encode_kwargs=encode_kwargs
    )
    return embedding_function

def create_vectorstore(docs, embedding_function):
    if not os.path.exists(Config.CHROMA_DIRECTORY):
        # Create the Chroma database from the documents and the embedding function
        db = Chroma.from_documents(docs, embedding_function, persist_directory=Config.CHROMA_DIRECTORY)
        # Persist the database
        db.persist()
    return Chroma(persist_directory=Config.CHROMA_DIRECTORY, embedding_function=embedding_function)

def setup_vectorstore(docs):
    embedding_function = create_embedding_function()
    return create_vectorstore(docs, embedding_function)

def create_retriever(vectorstore, k=5):
    return vectorstore.as_retriever(search_kwargs={"k": k})
