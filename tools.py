"""LangChain tools and deterministic policy calculators for the Loan Risk Agent."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .rag import get_knowledge_base
from .schemas import DocumentReference, DocumentStatus, EmploymentType, LoanType

# Interest rate slabs matching knowledge_base/03_interest_rates_and_emi_schedule.md
INTEREST_RATE_SLABS: dict[str, dict[str, float]] = {
    "personal": {"prime": 10.75, "standard": 12.50, "subprime": 15.25},
    "home": {"prime": 8.40, "standard": 8.90, "subprime": 9.65},
    "auto": {"prime": 8.85, "standard": 9.50, "subprime": 10.75},
    "education": {"prime": 9.15, "standard": 9.85, "subprime": 10.90},
}

TENURE_LIMITS: dict[str, tuple[int, int]] = {
    "personal": (12, 60),
    "home": (60, 360),
    "auto": (12, 84),
    "education": (12, 180),
}

PROCESSING_FEE_PCT: dict[str, float] = {
    "personal": 1.50,
    "home": 0.50,
    "auto": 1.00,
    "education": 0.50,
}

LOAN_AMOUNT_LIMITS: dict[str, tuple[float, float]] = {
    "personal": (50_000.0, 2_500_000.0),
    "home": (500_000.0, 50_000_000.0),
    "auto": (100_000.0, 5_000_000.0),
    "education": (100_000.0, 10_000_000.0),
}

MIN_ANNUAL_INCOME: dict[str, float] = {
    "personal": 300_000.0,
    "home": 450_000.0,
    "auto": 300_000.0,
    "education": 350_000.0,
}

DOCUMENT_LABELS: dict[str, str] = {
    "identity_proof": "Identity Proof (Passport / National ID / Aadhaar / DL)",
    "tax_id": "Tax / PAN Card",
    "address_proof": "Address Proof (Utility Bill / Rental Agreement)",
    "bank_statements": "Last 6 Months Bank Statements",
    "salary_slips": "Last 3 Months Salary Slips",
    "employment_letter": "Employment Verification Letter",
    "form16_or_itr": "Latest Form-16 or ITR",
    "itr_2_years": "Last 2 Years Income Tax Returns",
    "business_proof": "Business Registration / GST Certificate",
    "audited_financials": "Audited P&L and Balance Sheet",
    "property_documents": "Property Sale Agreement / Title Deed",
    "valuation_report": "Bank-Empanelled Property Valuation Report",
    "vehicle_invoice": "Authorized Dealer Vehicle Proforma Invoice",
    "admission_letter": "University Admission Offer & Fee Schedule",
    "co_applicant_income": "Co-Applicant Income Proof",
}


def get_required_document_keys(loan_type: str, employment_type: str) -> list[str]:
    """Return canonical required document keys per POL-DOCS-2026."""
    required = ["identity_proof", "tax_id", "address_proof", "bank_statements"]

    if employment_type == EmploymentType.SALARIED.value:
        required.extend(["salary_slips", "employment_letter", "form16_or_itr"])
    else:
        required.extend(["itr_2_years", "business_proof", "audited_financials"])

    if loan_type == LoanType.HOME.value:
        required.extend(["property_documents", "valuation_report"])
    elif loan_type == LoanType.AUTO.value:
        required.append("vehicle_invoice")
    elif loan_type == LoanType.EDUCATION.value:
        required.extend(["admission_letter", "co_applicant_income"])

    return required


def evaluate_documents(
    loan_type: str,
    employment_type: str,
    submitted_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify submitted document references against required bank policy documents."""
    required_keys = get_required_document_keys(loan_type, employment_type)
    doc_map: dict[str, DocumentReference] = {}

    for item in submitted_documents:
        ref = item if isinstance(item, DocumentReference) else DocumentReference(**item)
        doc_map[ref.doc_type.strip().lower()] = ref

    missing_docs: list[str] = []
    invalid_issues: list[str] = []
    valid_statuses = {DocumentStatus.VERIFIED, DocumentStatus.SUBMITTED}

    for key in required_keys:
        label = DOCUMENT_LABELS.get(key, key)
        if key not in doc_map:
            missing_docs.append(f"{key}: {label} (not provided)")
            continue

        doc_ref = doc_map[key]
        if doc_ref.status == DocumentStatus.MISSING:
            missing_docs.append(f"{key}: {label} (marked missing)")
        elif doc_ref.status not in valid_statuses:
            invalid_issues.append(
                f"{key}: {label} has invalid status '{doc_ref.status.value}'"
            )
        elif not doc_ref.reference_id.strip():
            invalid_issues.append(
                f"{key}: {label} is missing a document reference ID"
            )

    all_issues = missing_docs + invalid_issues
    return {
        "required_keys": required_keys,
        "missing_documents": all_issues,
        "document_issues": invalid_issues,
        "documents_correct": len(all_issues) == 0,
    }


def compute_emi_details(
    loan_type: str,
    credit_score: int,
    loan_amount: float,
    tenure_months: int,
) -> dict[str, Any]:
    """Compute annual interest rate, monthly EMI, processing fee, and tenure validity."""
    lt = loan_type.lower()
    slabs = INTEREST_RATE_SLABS.get(lt, INTEREST_RATE_SLABS["personal"])

    if credit_score >= 750:
        tier = "prime"
    elif credit_score >= 700:
        tier = "standard"
    elif credit_score >= 650:
        tier = "subprime"
    else:
        tier = "ineligible"

    annual_rate = slabs.get(tier, slabs["subprime"])
    min_t, max_t = TENURE_LIMITS.get(lt, (12, 60))
    tenure_valid = min_t <= tenure_months <= max_t

    monthly_rate = annual_rate / (12.0 * 100.0)
    if monthly_rate == 0:
        emi = loan_amount / tenure_months
    else:
        factor = (1.0 + monthly_rate) ** tenure_months
        emi = loan_amount * monthly_rate * factor / (factor - 1.0)

    total_payment = emi * tenure_months
    total_interest = total_payment - loan_amount
    fee_pct = PROCESSING_FEE_PCT.get(lt, 1.0)
    processing_fee = loan_amount * (fee_pct / 100.0)

    return {
        "credit_tier": tier,
        "annual_interest_rate": round(annual_rate, 2),
        "tenure_months": tenure_months,
        "allowed_tenure_range": [min_t, max_t],
        "tenure_valid": tenure_valid,
        "monthly_emi": round(emi, 2),
        "total_interest": round(total_interest, 2),
        "total_payment": round(total_payment, 2),
        "processing_fee": round(processing_fee, 2),
    }


@tool
def calculate_emi_and_rate(
    loan_type: str, credit_score: int, loan_amount: float, tenure_months: int
) -> dict[str, Any]:
    """Calculate interest rate slab, monthly EMI, total interest, and processing fee."""
    return compute_emi_details(loan_type, credit_score, loan_amount, tenure_months)


@tool
def calculate_dti_ratio(
    monthly_income: float, existing_monthly_debt: float, proposed_emi: float
) -> dict[str, Any]:
    """Calculate Debt-to-Income (DTI) ratio and classify affordability band."""
    if monthly_income <= 0:
        return {"dti_ratio": 1.0, "status": "INVALID_INCOME"}
    total_debt = existing_monthly_debt + proposed_emi
    dti = round(total_debt / monthly_income, 4)
    if dti > 0.50:
        band = "EXCEEDS_MAX_50_PCT"
    elif dti > 0.40:
        band = "CAUTION_40_TO_50_PCT"
    else:
        band = "HEALTHY_UNDER_40_PCT"
    return {
        "total_monthly_debt": round(total_debt, 2),
        "monthly_income": round(monthly_income, 2),
        "dti_ratio": dti,
        "dti_percentage": round(dti * 100.0, 2),
        "band": band,
    }


@tool
def retrieve_policy_clauses(query: str) -> list[dict[str, Any]]:
    """Retrieve relevant bank policy clauses from the RAG knowledge base."""
    kb = get_knowledge_base()
    return [clause.model_dump() for clause in kb.retrieve(query, k=4)]
