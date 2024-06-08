# src/rag_test_impl.py
from langchain_community.llms import LlamaCpp
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import threading
from src.thread.ChainStreamHandler import ChainStreamHandler
from src.thread.ThreadedGenerator import ThreadedGenerator
from src.configuration.config import Config
import datetime
import re


def load_and_tokenize_documents(directory, chunk_size=512, chunk_overlap=16, persist_directory=Config.CHROMA_DIRECTORY):
    if not os.path.exists(persist_directory):
        loader = PyPDFDirectoryLoader(directory)
        all_docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        docs = text_splitter.split_documents(all_docs)
        return docs
    else:
        return None


def create_embedding_function(model_name=Config.EMBEDDING_MODEL_NAME):
    encode_kwargs = {'normalize_embeddings': True}
    embedding_function = HuggingFaceBgeEmbeddings(
        model_name=model_name, encode_kwargs=encode_kwargs
    )
    return embedding_function


def create_vectorstore(docs, embedding_function, persist_directory=Config.CHROMA_DIRECTORY):
    if not os.path.exists(persist_directory):
        # Create the Chroma database from the documents and the embedding function
        db = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)
        # Persist the database
        db.persist()
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_function)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def initialize_llm(model_url, generator):
    model = LlamaCpp(
        model_path=model_url,
        temperature=0.9,
        max_tokens=2000,
        top_p=3,
        streaming=True,
        callback_manager=CallbackManager([ChainStreamHandler(generator)]),
        n_ctx=8096,
        verbose=True,
    )
    return model


def create_retriever(vectorstore, k=5):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def create_rag_chain(retriever, model):
    template = Config.get_template()
    prompt = ChatPromptTemplate.from_template(template)
    return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )


def load_and_preprocess_documents():
    return load_and_tokenize_documents(Config.PDF_DIRECTORY)


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


def generate_streaming_story(state, query):
    start_threaded_query(state["rag_chain"], query)
    for token in state["generator"]:
        if token is None:
            break
        print(token, end='')


def clean_text(generated_content):
    """
    Clean the text by removing URLs, dates, names, and emails.

    :param generated_content: The text to be cleaned.
    :return: Cleaned text.
    """
    # Regex patterns to match sensitive information
    url_pattern = r'https?://\S+|www\.\S+'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b'
    name_pattern = r'\b[A-Z][a-z]*\s[A-Z][a-z]*\b'  # Simple pattern for First Last names

    # Remove or replace sensitive information
    generated_content = re.sub(url_pattern, '[URL]', generated_content)
    generated_content = re.sub(email_pattern, '[EMAIL]', generated_content)
    generated_content = re.sub(date_pattern, '[DATE]', generated_content)
    generated_content = re.sub(name_pattern, '[NAME]', generated_content)

    return generated_content


def save_generated_text(generated_text, optional_name=None):
    """
    Save generated text to a file with an optional name and a timestamp in the /content/story folder.

    :param generated_text: The text to be saved.
    :param optional_name: An optional name to include in the filename.
    """
    # Clean the generated text
    cleaned_text = clean_text(generated_text)

    # Define the folder path
    folder_path = 'E:/Python/talesmith/data/story/'

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the filename
    if optional_name:
        file_name = f"{optional_name}_{timestamp}.txt"
    else:
        file_name = f"generated_text_{timestamp}.txt"

    # Complete file path
    file_path = os.path.join(folder_path, file_name)

    # Write the generated text to a file
    with open(file_path, 'w') as file:
        file.write(cleaned_text)

    print(f"Text has been saved to {file_path}")


def generate_story(state, query):
    print("Stated generating ....")
    result = run_query(state["rag_chain"], query)
    name = "story"
    save_generated_text(result, name)
    print("Completed")
    print(result)


state = init()
prompt = Config.get_prompt()
print(prompt)
generate_story(state, prompt)

