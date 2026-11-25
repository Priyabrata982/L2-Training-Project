"""Tests for the RAG policy knowledge base."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loan_assistant.rag import PolicyKnowledgeBase


def test_knowledge_base_loads_all_policies():
    kb = PolicyKnowledgeBase(prefer_ollama=False)
    assert len(kb.documents) >= 6
    sources = {doc.metadata["source"] for doc in kb.documents}
    assert "01_loan_eligibility_criteria.md" in sources
    assert "02_required_documents_policy.md" in sources
    assert "03_interest_rates_and_emi_schedule.md" in sources


def test_retrieve_eligibility_clauses():
    kb = PolicyKnowledgeBase(prefer_ollama=False)
    clauses = kb.retrieve("minimum credit score and maximum DTI ratio", k=3)
    assert len(clauses) > 0
    combined = " ".join(c.content.lower() for c in clauses)
    assert "credit score" in combined or "dti" in combined


def test_retrieve_required_documents_clauses():
    kb = PolicyKnowledgeBase(prefer_ollama=False)
    clauses = kb.retrieve(
        "required documents for home loan property valuation",
        k=3,
        source_filter="02_required_documents",
    )
    assert len(clauses) > 0
    assert all("02_required_documents" in c.source for c in clauses)


def test_retrieve_interest_rate_and_emi_clauses():
    kb = PolicyKnowledgeBase(prefer_ollama=False)
    clauses = kb.retrieve(
        "EMI calculation formula interest rate personal loan tenure",
        k=3,
        source_filter="03_interest_rates",
    )
    assert len(clauses) > 0
    combined = " ".join(c.content.lower() for c in clauses)
    assert "emi" in combined or "interest rate" in combined
