# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    MODEL_PATH = os.getenv('MODEL_PATH')
    EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
    PDF_DIRECTORY = os.getenv('PDF_DIRECTORY')
    CONTEXT_WINDOW = int(os.getenv('CONTEXT_WINDOW', 8096))  # Default to 8096 if not set
    MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 200))  # Default to 200 if not set
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 512))  # Default to 512 if not set
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))  # Default to 50 if not set
    PROMPT_FILE_PATH = os.getenv('PROMPT_FILE_PATH')
    TEMPLATE_FILE_PATH = os.getenv('TEMPLATE_FILE_PATH')
    CHROMA_DIRECTORY = os.getenv('CHROMA_DIRECTORY')

    @classmethod
    def get_prompt(cls):
        if cls.PROMPT_FILE_PATH and os.path.exists(cls.PROMPT_FILE_PATH):
            with open(cls.PROMPT_FILE_PATH, 'r') as file:
                return file.read()
        return "Generate a story"  # Default prompt if the file is not found

    @classmethod
    def get_template(cls):
        if cls.TEMPLATE_FILE_PATH and os.path.exists(cls.TEMPLATE_FILE_PATH):
            with open(cls.TEMPLATE_FILE_PATH, 'r') as file:
                return file.read()
        return """<human>: Continue the story based on the following context. If you cannot continue the story with the context, please respond with 'I don't know':
            ### CONTEXT
            {context}
            ### STORY CONTINUATION
            \n
            <bot>:
            """

