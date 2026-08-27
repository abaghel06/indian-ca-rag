from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_anthropic import AnthropicEmbeddings
from langchain.schema import Document

from config import settings

# --- Embeddings & Vector Store ---

def get_vectorstore() -> Chroma:
    embeddings = AnthropicEmbeddings(
        model="voyage-large-2",
        anthropic_api_key=settings.anthropic_api_key,
    )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


# --- PDF Ingestion ---

def ingest_pdf(file_path: str) -> int:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = get_splitter()
    chunks = splitter.split_documents(pages)
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return len(chunks)


# --- Web Ingestion ---

def scrape_url(url: str) -> Document:
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove nav/footer/script noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return Document(page_content=text, metadata={"source": url})


def ingest_urls(urls: List[str]) -> int:
    splitter = get_splitter()
    docs = []
    for url in urls:
        doc = scrape_url(url)
        docs.extend(splitter.split_documents([doc]))

    vs = get_vectorstore()
    vs.add_documents(docs)
    return len(docs)
