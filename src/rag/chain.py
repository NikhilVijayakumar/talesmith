#src/rag/chain.py
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.configuration.config import Config
from src.rag.retriever import create_retriever

def create_rag_chain(retriever, model):
    template = Config.get_template()
    prompt = ChatPromptTemplate.from_template(template)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_chain(vectorstore, model):
    retriever = create_retriever(vectorstore)
    return create_rag_chain(retriever, model)
