from langchain.schema import Document
from typing import List

from ingest import get_vectorstore
from config import settings


def retrieve(query: str) -> List[Document]:
    vs = get_vectorstore()
    return vs.similarity_search(query, k=settings.retrieval_k)
