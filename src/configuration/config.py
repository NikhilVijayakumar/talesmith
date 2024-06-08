# src/configuration.py
import os
from dotenv import load_dotenv
from src.configuration.environment import Environment








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
    def load_environment(cls, env: Environment):
        """
        Load environment variables from the corresponding .env file.

        Args:
            env (str): The environment to load, must be a string.
        """
        env_file = f'E:/Python/talesmith/.env.{env.value}'
        load_dotenv()
        load_dotenv(env_file, verbose=True)
        if os.path.exists(env_file):
            load_dotenv(env_file)
        else:
            raise ValueError(f"Environment file {env_file} not found")

        # Reload all environment values
        cls.MODEL_PATH = os.getenv('MODEL_PATH')
        cls.EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
        cls.PDF_DIRECTORY = os.getenv('PDF_DIRECTORY')
        cls.CONTEXT_WINDOW = int(os.getenv('CONTEXT_WINDOW', 8096))  # Default to 8096 if not set
        cls.MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 200))  # Default to 200 if not set
        cls.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 512))  # Default to 512 if not set
        cls.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))  # Default to 50 if not set
        cls.PROMPT_FILE_PATH = os.getenv('PROMPT_FILE_PATH')
        cls.TEMPLATE_FILE_PATH = os.getenv('TEMPLATE_FILE_PATH')
        cls.CHROMA_DIRECTORY = os.getenv('CHROMA_DIRECTORY')

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
                return file.read().strip()
        return """<human>: Continue the story based on the following context. If you cannot continue the story with the context, please respond with 'I don't know':
            ### CONTEXT
            {context}
            ### STORY CONTINUATION
            \n
            <bot>:
            """

    @classmethod
    def print_config(cls):
        """
        Print all configuration values.
        """
        print("Configuration Values:")
        print(f"MODEL_PATH: {cls.MODEL_PATH}")
        print(f"EMBEDDING_MODEL_NAME: {cls.EMBEDDING_MODEL_NAME}")
        print(f"PDF_DIRECTORY: {cls.PDF_DIRECTORY}")
        print(f"CONTEXT_WINDOW: {cls.CONTEXT_WINDOW}")
        print(f"MAX_NEW_TOKENS: {cls.MAX_NEW_TOKENS}")
        print(f"CHUNK_SIZE: {cls.CHUNK_SIZE}")
        print(f"CHUNK_OVERLAP: {cls.CHUNK_OVERLAP}")
        print(f"PROMPT_FILE_PATH: {cls.PROMPT_FILE_PATH}")
        print(f"TEMPLATE_FILE_PATH: {cls.TEMPLATE_FILE_PATH}")
        print(f"CHROMA_DIRECTORY: {cls.CHROMA_DIRECTORY}")

