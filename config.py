from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "ca_knowledge"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
