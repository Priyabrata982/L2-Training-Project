"""Local LLM provider (Ollama) with deterministic policy-grounded fallback."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .config import settings
from .schemas import LLMEligibilityAssessment, RiskIndicator


SYSTEM_PROMPT = """You are an internal Bank Loan Application Risk Assistant.
Your role is strictly ADVISORY: summarize applicant details, highlight policy-grounded
risk indicators, and provide an explainable recommendation for a human loan officer.

STRICT RULES:
1. NEVER state or imply that you have issued a final, binding loan approval or rejection.
   Always frame recommendations as advisory for the reviewing loan officer.
2. Base all eligibility and risk findings strictly on the retrieved bank policy clauses
   and computed EMI/DTI metrics provided in the prompt.
3. If data is borderline, incomplete, or ambiguous, lower your confidence score (< 0.65)
   and recommend MANUAL_REVIEW.
4. Return ONLY valid JSON matching the requested schema.
"""


def is_ollama_available() -> bool:
    """Check whether the local Ollama server is reachable."""
    if settings.use_fallback_llm:
        return False
    try:
        req = urllib.request.Request(
            f"{settings.ollama_base_url.rstrip('/')}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def _deterministic_assessment(
    application: dict[str, Any],
    emi_details: dict[str, Any],
    dti_ratio: float,
    eligible: bool,
    eligibility_reasons: list[str],
    risk_indicators: list[dict[str, Any]],
    retrieved_policies: list[dict[str, Any]],
) -> LLMEligibilityAssessment:
    """Deterministic policy-grounded analyzer used when Ollama is offline or unavailable."""
    name = application.get("full_name", "Applicant")
    loan_type = str(application.get("loan_type", "personal")).upper()
    amount = float(application.get("loan_amount", 0))
    tenure = int(application.get("tenure_months", 0))
    income = float(application.get("monthly_income", 0))
    score = int(application.get("credit_score", 0))
    emi = float(emi_details.get("monthly_emi", 0))
    rate = float(emi_details.get("annual_interest_rate", 0))
    notes = (application.get("credit_notes") or "").lower()

    parsed_risks = [RiskIndicator(**r) for r in risk_indicators]

    # Confidence calculation grounded in clarity of policy match & risk ambiguity
    confidence = 0.88
    if any(r.severity == "HIGH" for r in parsed_risks):
        confidence -= 0.14
    if any(r.severity == "MEDIUM" for r in parsed_risks):
        confidence -= 0.08
    if any(word in notes for word in ("dispute", "late", "restructur", "unclear", "gap")):
        confidence -= 0.15
    if 650 <= score <= 699:
        confidence -= 0.08
    confidence = round(max(0.35, min(0.96, confidence)), 2)

    if not eligible:
        recommendation = "REJECT_RECOMMENDED"
        rationale = (
            "Advisory recommendation: Application does not satisfy mandatory bank "
            f"eligibility policy ({'; '.join(eligibility_reasons)}). "
            "Requires loan officer confirmation for final disposition."
        )
    elif confidence < settings.confidence_threshold or any(
        r.severity in ("HIGH", "CRITICAL") for r in parsed_risks
    ):
        recommendation = "MANUAL_REVIEW"
        rationale = (
            "Advisory recommendation: Applicant meets baseline criteria, but risk "
            "indicators or borderline confidence warrant detailed human officer review "
            "before any lending decision."
        )
    else:
        recommendation = "APPROVE_RECOMMENDED"
        rationale = (
            f"Advisory recommendation: Applicant satisfies age, income, credit score ({score}), "
            f"and DTI ({dti_ratio:.1%}) rules under retrieved policy clauses. "
            "Submitted for final human loan officer approval."
        )

    policy_refs = ", ".join(
        sorted({p.get("policy_id") or p.get("source", "") for p in retrieved_policies if p})
    )
    summary = (
        f"{name} applied for a {loan_type} loan of {amount:,.2f} over {tenure} months "
        f"(proposed EMI {emi:,.2f} at {rate:.2f}% p.a.). Monthly income is {income:,.2f}, "
        f"credit score is {score}, and post-loan DTI is {dti_ratio:.1%}. "
        f"Evaluated against policies: {policy_refs or 'POL-ELIG-2026, POL-RATES-2026'}."
    )

    return LLMEligibilityAssessment(
        summary=summary,
        risk_indicators=parsed_risks,
        recommendation=recommendation,
        recommendation_rationale=rationale,
        confidence=confidence,
    )


def generate_eligibility_assessment(
    application: dict[str, Any],
    emi_details: dict[str, Any],
    dti_ratio: float,
    eligible: bool,
    eligibility_reasons: list[str],
    risk_indicators: list[dict[str, Any]],
    retrieved_policies: list[dict[str, Any]],
) -> tuple[LLMEligibilityAssessment, str]:
    """Generate structured eligibility assessment via Ollama, with deterministic fallback.

    Returns (assessment, llm_backend_used).
    """
    if is_ollama_available():
        try:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.1,
            )

            policy_context = "\n\n".join(
                f"[{p.get('source')} | {p.get('section')}]\n{p.get('content')}"
                for p in retrieved_policies
            )

            user_prompt = (
                "Evaluate this loan application strictly using the policy context and computed metrics.\n\n"
                f"APPLICANT DATA:\n{json.dumps(application, indent=2)}\n\n"
                f"COMPUTED EMI & RATE:\n{json.dumps(emi_details, indent=2)}\n\n"
                f"COMPUTED DTI RATIO: {dti_ratio:.4f}\n"
                f"RULE ENGINE ELIGIBLE: {eligible}\n"
                f"ELIGIBILITY REASONS: {json.dumps(eligibility_reasons)}\n"
                f"DETECTED RISK INDICATORS: {json.dumps(risk_indicators, indent=2)}\n\n"
                f"RETRIEVED POLICY CLAUSES:\n{policy_context}\n\n"
                "Return JSON with keys: summary, risk_indicators (list of {code, severity, "
                "description, policy_reference}), recommendation (APPROVE_RECOMMENDED | "
                "MANUAL_REVIEW | REJECT_RECOMMENDED), recommendation_rationale, confidence (0.0-1.0)."
            )

            structured_llm = llm.with_structured_output(LLMEligibilityAssessment)
            result = structured_llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            if isinstance(result, LLMEligibilityAssessment):
                return result, f"ollama:{settings.ollama_model}"
        except Exception:
            # Fall back gracefully to deterministic policy-grounded analyzer
            pass

    fallback = _deterministic_assessment(
        application=application,
        emi_details=emi_details,
        dti_ratio=dti_ratio,
        eligible=eligible,
        eligibility_reasons=eligibility_reasons,
        risk_indicators=risk_indicators,
        retrieved_policies=retrieved_policies,
    )
    return fallback, "deterministic_fallback"
