"""Interactive CLI runner for the Loan Application Risk Assistant."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loan_assistant.graph import LoanReviewService
from loan_assistant.llm_provider import is_ollama_available


def sample_applications() -> dict[str, dict[str, Any]]:
    return {
        "1": {
            "application_id": "APP-CLI-001",
            "full_name": "Aarav Sharma",
            "age": 32,
            "residency_status": "resident",
            "employment_type": "salaried",
            "employment_months": 48,
            "monthly_income": 120000.0,
            "existing_monthly_debt": 10000.0,
            "credit_score": 780,
            "loan_type": "personal",
            "loan_amount": 500000.0,
            "tenure_months": 36,
            "credit_notes": "Good repayment history.",
            "documents": [
                {"doc_type": "identity_proof", "reference_id": "ID-101", "status": "verified"},
                {"doc_type": "tax_id", "reference_id": "TAX-102", "status": "verified"},
                {"doc_type": "address_proof", "reference_id": "ADDR-103", "status": "verified"},
                {"doc_type": "bank_statements", "reference_id": "BANK-104", "status": "verified"},
                {"doc_type": "salary_slips", "reference_id": "PAY-105", "status": "verified"},
                {"doc_type": "employment_letter", "reference_id": "EMP-106", "status": "verified"},
                {"doc_type": "form16_or_itr", "reference_id": "F16-107", "status": "verified"},
            ],
        },
        "2": {
            "application_id": "APP-CLI-002",
            "full_name": "Priya Nair",
            "age": 35,
            "residency_status": "resident",
            "employment_type": "salaried",
            "employment_months": 60,
            "monthly_income": 180000.0,
            "existing_monthly_debt": 15000.0,
            "credit_score": 760,
            "loan_type": "home",
            "loan_amount": 4000000.0,
            "tenure_months": 240,
            "collateral_value": 5500000.0,
            "credit_notes": "Missing valuation_report initially.",
            "documents": [
                {"doc_type": "identity_proof", "reference_id": "ID-201", "status": "verified"},
                {"doc_type": "tax_id", "reference_id": "TAX-202", "status": "verified"},
                {"doc_type": "address_proof", "reference_id": "ADDR-203", "status": "verified"},
                {"doc_type": "bank_statements", "reference_id": "BANK-204", "status": "verified"},
                {"doc_type": "salary_slips", "reference_id": "PAY-205", "status": "verified"},
                {"doc_type": "employment_letter", "reference_id": "EMP-206", "status": "verified"},
                {"doc_type": "form16_or_itr", "reference_id": "F16-207", "status": "verified"},
                {"doc_type": "property_documents", "reference_id": "PROP-208", "status": "verified"},
            ],
        },
        "3": {
            "application_id": "APP-CLI-003",
            "full_name": "Rohan Verma",
            "age": 29,
            "residency_status": "resident",
            "employment_type": "salaried",
            "employment_months": 8,
            "monthly_income": 45000.0,
            "existing_monthly_debt": 20000.0,
            "credit_score": 610,
            "loan_type": "personal",
            "loan_amount": 600000.0,
            "tenure_months": 36,
            "credit_notes": "Recent late credit card payments.",
            "documents": [
                {"doc_type": "identity_proof", "reference_id": "ID-301", "status": "verified"},
                {"doc_type": "tax_id", "reference_id": "TAX-302", "status": "verified"},
                {"doc_type": "address_proof", "reference_id": "ADDR-303", "status": "verified"},
                {"doc_type": "bank_statements", "reference_id": "BANK-304", "status": "verified"},
                {"doc_type": "salary_slips", "reference_id": "PAY-305", "status": "verified"},
                {"doc_type": "employment_letter", "reference_id": "EMP-306", "status": "verified"},
                {"doc_type": "form16_or_itr", "reference_id": "F16-307", "status": "verified"},
            ],
        },
    }


def main() -> None:
    print("=== Loan Application Risk Assistant (CLI) ===")
    print(f"Ollama online: {is_ollama_available()}")
    service = LoanReviewService()
    print(f"RAG backend: {service.kb.backend_name}\n")

    print("Select a sample application:")
    print("  1. Prime Personal Loan (Eligible -> Human Approval)")
    print("  2. Home Loan with Missing Document (Document Resubmission HITL)")
    print("  3. High-Risk Ineligible Personal Loan (Auto-Rejection)")
    choice = input("Enter choice [1/2/3] (default 1): ").strip() or "1"

    apps = sample_applications()
    app = apps.get(choice, apps["1"])
    result = service.start_review(app)

    while result["status"] == "INTERRUPTED":
        intr = result["interrupt"]
        itype = intr.get("interrupt_type")
        print(f"\n[INTERRUPT] {itype}: {intr.get('message')}")

        if itype == "document_resubmission":
            missing = intr.get("missing_documents", [])
            print("Missing/Invalid Documents:")
            for m in missing:
                print(f"  - {m}")
            action = input("Resubmit corrected documents? [y/n] (default y): ").strip().lower()
            if action == "n":
                result = service.resubmit_documents(result["thread_id"], [], action="cancel")
            else:
                docs = []
                for m in missing:
                    key = m.split(":")[0].strip()
                    ref = input(f"  Reference ID for {key} (default RESUB-{key.upper()}): ").strip()
                    docs.append(
                        {
                            "doc_type": key,
                            "reference_id": ref or f"RESUB-{key.upper()}",
                            "status": "verified",
                        }
                    )
                result = service.resubmit_documents(result["thread_id"], docs, action="resubmit")

        elif itype == "human_approval":
            print(f"Summary: {intr.get('summary')}")
            print(f"Recommendation: {intr.get('recommendation')} (confidence={intr.get('confidence')})")
            dec = input("Officer Decision [approve/reject] (default approve): ").strip().lower() or "approve"
            reviewer = input("Officer Name (default Loan Officer): ").strip() or "Loan Officer"
            comments = input("Comments: ").strip()
            result = service.submit_human_decision(
                result["thread_id"],
                decision="approve" if dec.startswith("a") else "reject",
                reviewer=reviewer,
                comments=comments,
            )

    print("\n=== Final Structured Output ===")
    print(json.dumps(result["structured_output"], indent=2))


if __name__ == "__main__":
    main()
