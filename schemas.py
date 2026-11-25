"""Pydantic schemas and LangGraph state definitions."""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Optional, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class LoanType(str, Enum):
    PERSONAL = "personal"
    HOME = "home"
    AUTO = "auto"
    EDUCATION = "education"


class EmploymentType(str, Enum):
    SALARIED = "salaried"
    SELF_EMPLOYED = "self_employed"


class DocumentStatus(str, Enum):
    VERIFIED = "verified"
    SUBMITTED = "submitted"
    MISSING = "missing"
    EXPIRED = "expired"
    UNREADABLE = "unreadable"
    UNVERIFIED = "unverified"


class DocumentReference(BaseModel):
    """Reference to an applicant-submitted supporting document."""

    doc_type: str = Field(..., min_length=2, description="Canonical document type key")
    reference_id: str = Field(default="", description="Document reference or file ID")
    status: DocumentStatus = Field(default=DocumentStatus.SUBMITTED)
    notes: str = Field(default="", description="Optional verification notes")


class ApplicantData(BaseModel):
    """Structured loan application payload submitted for risk review."""

    application_id: str = Field(default_factory=lambda: f"APP-{uuid4().hex[:8].upper()}")
    full_name: str = Field(..., min_length=2, max_length=120)
    age: int = Field(..., ge=18, le=100)
    residency_status: Literal["resident", "permanent_resident", "non_resident"] = "resident"
    employment_type: EmploymentType = EmploymentType.SALARIED
    employment_months: int = Field(..., ge=0, le=600)
    monthly_income: float = Field(..., gt=0)
    existing_monthly_debt: float = Field(default=0.0, ge=0)
    credit_score: int = Field(..., ge=300, le=900)
    loan_type: LoanType = LoanType.PERSONAL
    loan_amount: float = Field(..., gt=0)
    tenure_months: int = Field(..., ge=1, le=480)
    collateral_value: Optional[float] = Field(default=None, ge=0)
    credit_notes: str = Field(default="", max_length=2000)
    documents: list[DocumentReference] = Field(default_factory=list)

    @field_validator("full_name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Applicant full_name cannot be empty.")
        return cleaned


class PolicyClause(BaseModel):
    """Retrieved policy clause from the RAG knowledge base."""

    source: str
    section: str
    policy_id: str = ""
    content: str
    score: float = 0.0


class RiskIndicator(BaseModel):
    """Specific risk factor identified during evaluation."""

    code: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    description: str
    policy_reference: str = ""


class LLMEligibilityAssessment(BaseModel):
    """Structured LLM output schema for eligibility & risk evaluation."""

    summary: str = Field(..., min_length=10)
    risk_indicators: list[RiskIndicator] = Field(default_factory=list)
    recommendation: Literal[
        "APPROVE_RECOMMENDED",
        "MANUAL_REVIEW",
        "REJECT_RECOMMENDED",
    ]
    recommendation_rationale: str = Field(..., min_length=10)
    confidence: float = Field(..., ge=0.0, le=1.0)


class StructuredRiskOutput(BaseModel):
    """Final validated output schema produced by the Loan Application Risk Assistant."""

    application_id: str
    applicant_name: str
    loan_type: str
    summary: str
    missing_documents: list[str] = Field(default_factory=list)
    risk_indicators: list[RiskIndicator] = Field(default_factory=list)
    recommendation: Literal[
        "APPROVE_RECOMMENDED",
        "MANUAL_REVIEW",
        "REJECT_RECOMMENDED",
        "RESUBMIT_DOCUMENTS",
        "INVALID_INPUT",
    ]
    recommendation_rationale: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)
    low_confidence_warning: bool = False
    human_decision_required: bool = True
    final_status: Literal[
        "APPROVED_BY_HUMAN",
        "REJECTED_BY_HUMAN",
        "REJECTED_INELIGIBLE",
        "PENDING_DOCUMENTS",
        "PENDING_HUMAN_REVIEW",
        "INVALID_APPLICATION",
    ]
    human_decision: Optional[str] = None
    human_reviewer: Optional[str] = None
    human_comments: Optional[str] = None
    emi_estimate: Optional[float] = None
    interest_rate: Optional[float] = None
    dti_ratio: Optional[float] = None
    retrieved_policies: list[PolicyClause] = Field(default_factory=list)
    guardrail_notes: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class GraphState(TypedDict, total=False):
    """Shared state passed across LangGraph nodes."""

    thread_id: str
    application: dict[str, Any]
    validation_passed: bool
    validation_errors: list[str]
    retrieved_doc_policies: list[dict[str, Any]]
    retrieved_elig_policies: list[dict[str, Any]]
    documents_correct: bool
    missing_documents: list[str]
    document_issues: list[str]
    resubmit_count: int
    loan_eligible: bool
    eligibility_reasons: list[str]
    risk_indicators: list[dict[str, Any]]
    emi_details: dict[str, Any]
    ai_assessment: dict[str, Any]
    confidence: float
    human_decision: Optional[str]
    human_reviewer: Optional[str]
    human_comments: Optional[str]
    final_status: str
    structured_output: dict[str, Any]
    guardrail_notes: list[str]
    execution_trace: Annotated[list[str], operator.add]
    error_message: Optional[str]
