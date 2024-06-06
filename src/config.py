# src/config.py
class Config:
    MODEL_PATH = 'E:/AI/LM Studio/Models/TheDrummer/Llama-3SOME-8B-v1-BETA-GGUF/Llama-3some-8B-v1-rc1-Q8_0.gguf'
    EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    PDF_DIRECTORY = 'E:/Python/talesmith/data/pdfs/'
    CONTEXT_WINDOW = 8096
    MAX_NEW_TOKENS = 200
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
