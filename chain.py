from langchain_anthropic import ChatAnthropic
from langchain.schema import Document
from typing import List

from config import settings

CA_SYSTEM_PROMPT = """You are a qualified Indian Chartered Accountant (CA) with expertise in:
- Indian Income Tax Act, 1961
- GST (Goods and Services Tax)
- Companies Act, 2013
- ICAI standards and guidelines
- TDS/TCS provisions
- Advance tax, capital gains, deductions (80C, 80D, etc.)

Rules you must follow:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. If the context does not contain enough information, say: "I don't have sufficient information in my knowledge base to answer this accurately. Please consult an ICAI-registered CA."
3. Always cite the source (section number, act, or document) when possible.
4. Use Indian financial terminology and INR (₹) for amounts.
5. Never give a definitive legal opinion — add a disclaimer when appropriate.
"""

def build_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer(query: str, docs: List[Document]) -> dict:
    if not docs:
        return {
            "answer": "I don't have relevant information in my knowledge base for this query.",
            "sources": [],
        }

    context = build_context(docs)
    sources = list({doc.metadata.get("source", "Unknown") for doc in docs})

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
        max_tokens=1024,
    )

    messages = [
        {"role": "system", "content": CA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": sources,
    }
