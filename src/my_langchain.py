import glob
import logging
import os
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import LlamaCpp
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def load_documents(file_path_pattern):
    try:
        loader = TextLoader(file_path_pattern)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise


def split_documents(documents, chunk_size, chunk_overlap):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(texts)} chunks")
    return texts


def create_vector_store(texts):
    try:
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        db = Chroma.from_documents(texts, embeddings)
        logger.info("Created vector store")
        return db
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise


def create_qa_chain(_db, _prompt):
    try:
        llm = LlamaCpp(
            model_path=Config.MODEL_PATH,
            context_window=Config.CONTEXT_WINDOW,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=0.6,
            n_ctx=8096,
            model_kwargs={}
        )
        retriever = db.as_retriever(search_type='mmr')
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        logger.info("Created QA chain")
        return qa
    except Exception as e:
        logger.error(f"Failed to create QA chain: {e}")
        raise


# Initialize components
pdf_directory = Config.PDF_DIRECTORY
pdf_files = glob.glob(os.path.join(pdf_directory, '*.txt'))

if not pdf_files:
    raise FileNotFoundError(f"No .txt files found in {pdf_directory}")

documents = []
for file in pdf_files:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        document = Document(page_content=text)
        documents.append(document)

texts = split_documents(documents, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
db = create_vector_store(texts)


# Main function to run queries
def run_query(prompt):
    try:
        qa = create_qa_chain(db, prompt)
        result = qa.run(prompt)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return str(e)


if __name__ == "__main__":
    prompt = "i Was with my cousin"
    result = run_query(prompt)
    print(result)
