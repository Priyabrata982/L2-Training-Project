"""RAG pipeline over bank policy documents (LangChain + FAISS/Ollama + TF-IDF fallback)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import PrivateAttr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import settings
from .schemas import PolicyClause


def _extract_policy_id(text: str) -> str:
    match = re.search(r"Policy ID:\s*([A-Z0-9-]+)", text)
    return match.group(1) if match else ""


def load_policy_documents(kb_dir: Optional[Path] = None) -> list[Document]:
    """Load and section-chunk markdown policy documents from the knowledge base."""
    directory = kb_dir or settings.knowledge_base_dir
    if not directory.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {directory}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    chunks: list[Document] = []
    for md_path in sorted(directory.glob("*.md")):
        raw_text = md_path.read_text(encoding="utf-8")
        policy_id = _extract_policy_id(raw_text)

        # Split by top-level sections first to preserve section metadata
        sections = re.split(r"(?=^##\s+)", raw_text, flags=re.MULTILINE)
        for sec in sections:
            sec_stripped = sec.strip()
            if not sec_stripped:
                continue
            first_line = sec_stripped.splitlines()[0].lstrip("#").strip()
            sub_docs = splitter.create_documents(
                texts=[sec_stripped],
                metadatas=[
                    {
                        "source": md_path.name,
                        "section": first_line,
                        "policy_id": policy_id,
                    }
                ],
            )
            chunks.extend(sub_docs)

    return chunks


class TfidfPolicyRetriever(BaseRetriever):
    """Offline LangChain-compatible retriever using TF-IDF cosine similarity."""

    documents: list[Document]
    k: int = 4
    _vectorizer: TfidfVectorizer = PrivateAttr()
    _matrix: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        corpus = [
            f"{doc.metadata.get('section', '')} {doc.page_content}"
            for doc in self.documents
        ]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(corpus)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        results = self.search_with_scores(query, k=self.k)
        return [doc for doc, _ in results]

    def search_with_scores(
        self, query: str, k: int = 4, source_filter: Optional[str] = None
    ) -> list[tuple[Document, float]]:
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()

        scored_docs: list[tuple[Document, float]] = []
        for idx, doc in enumerate(self.documents):
            if source_filter and source_filter not in doc.metadata.get("source", ""):
                continue
            scored_docs.append((doc, float(scores[idx])))

        scored_docs.sort(key=lambda item: item[1], reverse=True)
        return scored_docs[:k]


class PolicyKnowledgeBase:
    """Manages policy document indexing and retrieval via FAISS+Ollama or TF-IDF fallback."""

    def __init__(self, kb_dir: Optional[Path] = None, prefer_ollama: Optional[bool] = None):
        self.kb_dir = kb_dir or settings.knowledge_base_dir
        self.documents = load_policy_documents(self.kb_dir)
        self.tfidf_retriever = TfidfPolicyRetriever(documents=self.documents, k=4)
        self.vector_store = None
        self.backend_name = "tfidf_local"

        should_try_ollama = (
            (not settings.use_fallback_llm) if prefer_ollama is None else prefer_ollama
        )
        if should_try_ollama:
            self._try_init_faiss_ollama()

    def _try_init_faiss_ollama(self) -> None:
        try:
            from .llm_provider import is_ollama_available

            if not is_ollama_available():
                self.vector_store = None
                self.backend_name = "tfidf_local"
                return

            from langchain_community.vectorstores import FAISS
            from langchain_ollama import OllamaEmbeddings

            embeddings = OllamaEmbeddings(
                base_url=settings.ollama_base_url,
                model=settings.ollama_embed_model,
            )
            # Probe single embedding to confirm Ollama embed model is available
            embeddings.embed_query("loan eligibility check")
            self.vector_store = FAISS.from_documents(self.documents, embeddings)
            self.backend_name = f"faiss_ollama ({settings.ollama_embed_model})"
        except Exception:
            self.vector_store = None
            self.backend_name = "tfidf_local"

    def retrieve(
        self,
        query: str,
        k: int = 4,
        source_filter: Optional[str] = None,
    ) -> list[PolicyClause]:
        """Retrieve top-k relevant policy clauses for a query."""
        if self.vector_store is not None:
            try:
                raw_results = self.vector_store.similarity_search_with_score(
                    query, k=k * 2
                )
                filtered: list[tuple[Document, float]] = []
                for doc, dist in raw_results:
                    if source_filter and source_filter not in doc.metadata.get("source", ""):
                        continue
                    # Convert L2 distance to bounded similarity score
                    similarity = round(1.0 / (1.0 + float(dist)), 4)
                    filtered.append((doc, similarity))
                if filtered:
                    return [
                        PolicyClause(
                            source=doc.metadata.get("source", ""),
                            section=doc.metadata.get("section", ""),
                            policy_id=doc.metadata.get("policy_id", ""),
                            content=doc.page_content,
                            score=score,
                        )
                        for doc, score in filtered[:k]
                    ]
            except Exception:
                pass

        tfidf_results = self.tfidf_retriever.search_with_scores(
            query, k=k, source_filter=source_filter
        )
        return [
            PolicyClause(
                source=doc.metadata.get("source", ""),
                section=doc.metadata.get("section", ""),
                policy_id=doc.metadata.get("policy_id", ""),
                content=doc.page_content,
                score=round(score, 4),
            )
            for doc, score in tfidf_results
        ]


_kb_instance: Optional[PolicyKnowledgeBase] = None


def get_knowledge_base() -> PolicyKnowledgeBase:
    """Return cached PolicyKnowledgeBase singleton."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = PolicyKnowledgeBase()
    return _kb_instance
