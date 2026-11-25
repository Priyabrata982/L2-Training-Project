"""Input and Output Guardrails for the Loan Application Risk Assistant."""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from .config import settings
from .schemas import ApplicantData, PolicyClause, RiskIndicator, StructuredRiskOutput

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"override\s+(policy|rules|guardrail)", re.IGNORECASE),
    re.compile(r"force\s+approv(e|al)", re.IGNORECASE),
    re.compile(r"bypass\s+(check|review|human)", re.IGNORECASE),
]

# PII patterns to redact from stored/displayed narrative summaries
PII_PATTERNS = [
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[REDACTED_TAX_ID]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[REDACTED_GOV_ID]"),
]


def redact_pii(text: str) -> str:
    """Redact common national ID / tax ID patterns from free-text output."""
    redacted = text
    for pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def run_input_guardrails(
    raw_application: dict[str, Any],
) -> tuple[bool, dict[str, Any], list[str], list[str]]:
    """Validate applicant payload and scan for prompt injection or invalid inputs.

    Returns (is_valid, sanitized_application_dict, validation_errors, guardrail_notes).
    """
    errors: list[str] = []
    notes: list[str] = []

    if not isinstance(raw_application, dict):
        return False, {}, ["Application payload must be a JSON object."], ["Blocked non-dict payload."]

    try:
        applicant = ApplicantData(**raw_application)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", []))
            errors.append(f"Invalid field '{loc}': {err.get('msg')}")
        notes.append("Input Guardrail: Rejected invalid applicant schema.")
        return False, raw_application, errors, notes

    # Prompt injection screening on free-text fields
    text_fields = [applicant.full_name, applicant.credit_notes] + [
        doc.notes for doc in applicant.documents
    ]
    combined_text = "\n".join(t for t in text_fields if t)
    for pattern in INJECTION_PATTERNS:
        if pattern.search(combined_text):
            errors.append(
                f"Security guardrail triggered: prohibited instruction pattern detected ({pattern.pattern})."
            )
            notes.append("Input Guardrail: Blocked potential prompt injection in application text.")

    # Business sanity bounds
    if applicant.existing_monthly_debt > applicant.monthly_income * 3:
        errors.append(
            "Existing monthly debt exceeds 300% of monthly income; data appears erroneous."
        )
    if applicant.employment_months > (applicant.age - 14) * 12:
        errors.append(
            "Employment tenure exceeds plausible working years for applicant age."
        )

    if errors:
        return False, applicant.model_dump(mode="json"), errors, notes

    notes.append("Input Guardrail: Applicant schema and safety checks passed.")
    return True, applicant.model_dump(mode="json"), [], notes


def _enforce_advisory_language(text: str) -> tuple[str, bool]:
    """Ensure AI text never claims autonomous final approval or disbursement."""
    forbidden = [
        (re.compile(r"\bloan is finally approved\b", re.IGNORECASE), "loan is recommended for officer review"),
        (re.compile(r"\bautonomously approved\b", re.IGNORECASE), "recommended for human approval"),
        (re.compile(r"\bdisbursement authorized\b", re.IGNORECASE), "pending human officer authorization"),
    ]
    modified = False
    updated = text
    for pattern, replacement in forbidden:
        if pattern.search(updated):
            updated = pattern.sub(replacement, updated)
            modified = True
    return updated, modified


def run_output_guardrails(state: dict[str, Any]) -> tuple[StructuredRiskOutput, list[str]]:
    """Validate final structured output, enforce HITL governance, and handle low confidence."""
    app = state.get("application", {})
    notes = list(state.get("guardrail_notes", []))
    validation_passed = bool(state.get("validation_passed", False))
    ai = state.get("ai_assessment") or {}
    emi = state.get("emi_details") or {}
    missing_docs = list(state.get("missing_documents", []))
    human_decision = state.get("human_decision")
    final_status = state.get("final_status", "PENDING_HUMAN_REVIEW")

    raw_policies = state.get("retrieved_doc_policies", []) + state.get(
        "retrieved_elig_policies", []
    )
    deduped_policies: list[PolicyClause] = []
    seen = set()
    for p in raw_policies:
        key = (p.get("source", ""), p.get("section", ""))
        if key not in seen:
            seen.add(key)
            deduped_policies.append(PolicyClause(**p))

    if not validation_passed:
        errors = state.get("validation_errors", ["Invalid applicant input."])
        output = StructuredRiskOutput(
            application_id=str(app.get("application_id", "UNKNOWN")),
            applicant_name=str(app.get("full_name", "Unknown")),
            loan_type=str(app.get("loan_type", "unknown")),
            summary=redact_pii(
                "Application could not be processed due to input validation errors: "
                + "; ".join(errors)
            ),
            missing_documents=[],
            risk_indicators=[
                RiskIndicator(
                    code="INPUT_VALIDATION_FAILED",
                    severity="HIGH",
                    description="; ".join(errors),
                    policy_reference="Input Guardrails",
                )
            ],
            recommendation="INVALID_INPUT",
            recommendation_rationale="Fix applicant data errors and resubmit.",
            confidence=0.0,
            low_confidence_warning=True,
            human_decision_required=True,
            final_status="INVALID_APPLICATION",
            retrieved_policies=[],
            guardrail_notes=notes,
        )
        return output, notes

    confidence = float(ai.get("confidence", state.get("confidence", 0.5)))
    low_confidence = confidence < settings.confidence_threshold

    recommendation = ai.get("recommendation", "MANUAL_REVIEW")
    if final_status == "REJECTED_INELIGIBLE":
        recommendation = "REJECT_RECOMMENDED"

    if low_confidence:
        notes.append(
            f"Output Guardrail: Low confidence ({confidence:.2f} < {settings.confidence_threshold:.2f}); "
            "escalated for mandatory senior loan officer review."
        )
        if recommendation == "APPROVE_RECOMMENDED":
            recommendation = "MANUAL_REVIEW"

    summary = redact_pii(
        ai.get("summary")
        or f"Application {app.get('application_id', '')} reviewed for {app.get('full_name', 'Applicant')}."
    )
    rationale = redact_pii(
        ai.get("recommendation_rationale")
        or "Advisory assessment generated from bank policy rules; human officer retains final authority."
    )

    summary, sum_mod = _enforce_advisory_language(summary)
    rationale, rat_mod = _enforce_advisory_language(rationale)
    if sum_mod or rat_mod:
        notes.append(
            "Output Guardrail: Rewrote phrasing to preserve human-in-the-loop final decision authority."
        )

    raw_risks = ai.get("risk_indicators") or state.get("risk_indicators") or []
    risk_models = [
        r if isinstance(r, RiskIndicator) else RiskIndicator(**r) for r in raw_risks
    ]
    if low_confidence and not any(r.code == "LOW_CONFIDENCE" for r in risk_models):
        risk_models.append(
            RiskIndicator(
                code="LOW_CONFIDENCE",
                severity="MEDIUM",
                description=(
                    f"AI assessment confidence ({confidence:.2f}) is below threshold "
                    f"({settings.confidence_threshold:.2f})."
                ),
                policy_reference="Output Guardrails",
            )
        )

    # Confirm human-in-the-loop governance status
    if human_decision in ("approve", "reject"):
        notes.append(
            f"Output Guardrail: Verified human officer decision ('{human_decision}') by "
            f"{state.get('human_reviewer') or 'Loan Officer'}."
        )
    else:
        notes.append(
            "Output Guardrail: Confirmed no autonomous final lending approval was issued."
        )

    output = StructuredRiskOutput(
        application_id=str(app.get("application_id", "UNKNOWN")),
        applicant_name=str(app.get("full_name", "Applicant")),
        loan_type=str(app.get("loan_type", "personal")),
        summary=summary,
        missing_documents=missing_docs,
        risk_indicators=risk_models,
        recommendation=recommendation,
        recommendation_rationale=rationale,
        confidence=round(confidence, 2),
        low_confidence_warning=low_confidence,
        human_decision_required=True,
        final_status=final_status,
        human_decision=human_decision,
        human_reviewer=state.get("human_reviewer"),
        human_comments=state.get("human_comments"),
        emi_estimate=emi.get("monthly_emi"),
        interest_rate=emi.get("annual_interest_rate"),
        dti_ratio=emi.get("dti_ratio"),
        retrieved_policies=deduped_policies,
        guardrail_notes=notes,
    )
    return output, notes
