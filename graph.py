"""LangGraph Loan Approval Agentic Application workflow."""

from __future__ import annotations

from typing import Any, Literal, Optional
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .config import settings
from .guardrails import run_input_guardrails, run_output_guardrails
from .llm_provider import generate_eligibility_assessment
from .rag import PolicyKnowledgeBase, get_knowledge_base
from .schemas import GraphState
from .storage import ReviewHistoryStore
from .tools import (
    LOAN_AMOUNT_LIMITS,
    MIN_ANNUAL_INCOME,
    compute_emi_details,
    evaluate_documents,
)


def _evaluate_policy_eligibility(
    application: dict[str, Any], emi_details: dict[str, Any]
) -> tuple[bool, list[str], list[dict[str, Any]], float]:
    """Evaluate deterministic eligibility rules and build risk indicators."""
    age = int(application.get("age", 0))
    residency = application.get("residency_status", "resident")
    emp_type = application.get("employment_type", "salaried")
    emp_months = int(application.get("employment_months", 0))
    monthly_income = float(application.get("monthly_income", 0))
    annual_income = monthly_income * 12.0
    existing_debt = float(application.get("existing_monthly_debt", 0))
    credit_score = int(application.get("credit_score", 0))
    loan_type = str(application.get("loan_type", "personal")).lower()
    loan_amount = float(application.get("loan_amount", 0))
    tenure_months = int(application.get("tenure_months", 12))
    collateral_value = application.get("collateral_value")
    credit_notes = (application.get("credit_notes") or "").lower()

    proposed_emi = float(emi_details.get("monthly_emi", 0))
    dti_ratio = round((existing_debt + proposed_emi) / monthly_income, 4)

    ineligible_reasons: list[str] = []
    risks: list[dict[str, Any]] = []

    # 1. Age rules (POL-ELIG-2026 Section 1)
    maturity_age = age + (tenure_months / 12.0)
    max_maturity_age = 65 if emp_type == "salaried" else 70
    if age < 21:
        ineligible_reasons.append(f"Applicant age {age} is below minimum 21 years.")
        risks.append(
            {
                "code": "UNDERAGE_APPLICANT",
                "severity": "CRITICAL",
                "description": f"Applicant age ({age}) is under minimum 21 years.",
                "policy_reference": "POL-ELIG-2026 §1",
            }
        )
    elif maturity_age > max_maturity_age:
        ineligible_reasons.append(
            f"Age at loan maturity ({maturity_age:.1f}) exceeds {max_maturity_age} years."
        )
        risks.append(
            {
                "code": "MATURITY_AGE_EXCEEDED",
                "severity": "HIGH",
                "description": f"Age at loan maturity ({maturity_age:.1f}) exceeds policy cap ({max_maturity_age}).",
                "policy_reference": "POL-ELIG-2026 §1",
            }
        )

    # 2. Residency
    if residency == "non_resident":
        ineligible_reasons.append("Non-resident applicants are not eligible under retail policy.")
        risks.append(
            {
                "code": "RESIDENCY_INELIGIBLE",
                "severity": "CRITICAL",
                "description": "Applicant residency status is non_resident.",
                "policy_reference": "POL-ELIG-2026 §1",
            }
        )

    # 3. Employment stability
    min_emp_months = 12 if emp_type == "salaried" else 24
    if emp_months < min_emp_months:
        ineligible_reasons.append(
            f"Employment tenure ({emp_months} months) is below minimum {min_emp_months} months for {emp_type}."
        )
        risks.append(
            {
                "code": "SHORT_EMPLOYMENT_TENURE",
                "severity": "HIGH",
                "description": f"Employment continuity {emp_months} months < required {min_emp_months} months.",
                "policy_reference": "POL-ELIG-2026 §1",
            }
        )

    # 4. Credit score rules (POL-ELIG-2026 Section 2)
    if credit_score < 650:
        ineligible_reasons.append(
            f"Credit score ({credit_score}) is below minimum 650 threshold."
        )
        risks.append(
            {
                "code": "CREDIT_SCORE_BELOW_MINIMUM",
                "severity": "CRITICAL",
                "description": f"Credit score {credit_score} < 650 minimum threshold.",
                "policy_reference": "POL-ELIG-2026 §2",
            }
        )
    elif credit_score < 700:
        risks.append(
            {
                "code": "SUBPRIME_CREDIT_SCORE",
                "severity": "HIGH",
                "description": f"Credit score {credit_score} is in sub-prime caution band (650-699).",
                "policy_reference": "POL-ELIG-2026 §2",
            }
        )
    elif credit_score < 750:
        risks.append(
            {
                "code": "STANDARD_CREDIT_TIER",
                "severity": "LOW",
                "description": f"Credit score {credit_score} falls in standard tier (700-749).",
                "policy_reference": "POL-ELIG-2026 §2",
            }
        )

    # 5. Income & DTI rules (POL-ELIG-2026 Section 3)
    min_annual = MIN_ANNUAL_INCOME.get(loan_type, 300_000.0)
    if annual_income < min_annual:
        ineligible_reasons.append(
            f"Annual income ({annual_income:,.0f}) is below minimum {min_annual:,.0f} for {loan_type} loan."
        )
        risks.append(
            {
                "code": "LOW_INCOME",
                "severity": "HIGH",
                "description": f"Annual income {annual_income:,.0f} < required {min_annual:,.0f}.",
                "policy_reference": "POL-ELIG-2026 §3",
            }
        )

    max_dti = 0.40 if credit_score < 700 else settings.max_dti_ratio
    if dti_ratio > max_dti:
        ineligible_reasons.append(
            f"Post-loan DTI ({dti_ratio:.1%}) exceeds maximum allowed {max_dti:.0%}."
        )
        risks.append(
            {
                "code": "HIGH_DTI_RATIO",
                "severity": "CRITICAL",
                "description": f"DTI {dti_ratio:.1%} exceeds policy cap of {max_dti:.0%}.",
                "policy_reference": "POL-ELIG-2026 §3",
            }
        )
    elif dti_ratio > 0.40:
        risks.append(
            {
                "code": "CAUTION_DTI_ZONE",
                "severity": "MEDIUM",
                "description": f"DTI {dti_ratio:.1%} is in caution band (40%-50%).",
                "policy_reference": "POL-ELIG-2026 §3",
            }
        )

    # 6. Loan amount, tenure, and LTV limits (POL-ELIG-2026 §4, POL-RATES-2026 §2)
    min_amt, max_amt = LOAN_AMOUNT_LIMITS.get(loan_type, (50_000.0, 2_500_000.0))
    if loan_amount < min_amt or loan_amount > max_amt:
        ineligible_reasons.append(
            f"Loan amount {loan_amount:,.0f} is outside allowed range [{min_amt:,.0f}, {max_amt:,.0f}]."
        )
        risks.append(
            {
                "code": "LOAN_AMOUNT_OUT_OF_BOUNDS",
                "severity": "HIGH",
                "description": f"Requested amount {loan_amount:,.0f} outside policy bounds.",
                "policy_reference": "POL-ELIG-2026 §4",
            }
        )

    if loan_type == "personal" and loan_amount > monthly_income * 20:
        ineligible_reasons.append(
            "Personal loan amount exceeds 20x monthly net income cap."
        )
        risks.append(
            {
                "code": "PERSONAL_LOAN_INCOME_MULTIPLE_EXCEEDED",
                "severity": "HIGH",
                "description": "Personal loan exceeds 20x monthly income cap.",
                "policy_reference": "POL-ELIG-2026 §4",
            }
        )

    if not emi_details.get("tenure_valid", True):
        t_range = emi_details.get("allowed_tenure_range", [12, 60])
        ineligible_reasons.append(
            f"Requested tenure ({tenure_months} months) outside allowed range {t_range[0]}-{t_range[1]} months."
        )
        risks.append(
            {
                "code": "INVALID_TENURE",
                "severity": "HIGH",
                "description": f"Tenure {tenure_months} months violates policy limits {t_range}.",
                "policy_reference": "POL-RATES-2026 §2",
            }
        )

    if loan_type in ("home", "auto") and collateral_value is not None and collateral_value > 0:
        ltv = loan_amount / float(collateral_value)
        max_ltv = 0.80 if loan_type == "home" else 0.85
        if ltv > max_ltv:
            ineligible_reasons.append(
                f"Loan-to-Value ({ltv:.1%}) exceeds maximum {max_ltv:.0%} for {loan_type} loan."
            )
            risks.append(
                {
                    "code": "HIGH_LTV",
                    "severity": "HIGH",
                    "description": f"LTV {ltv:.1%} > maximum {max_ltv:.0%}.",
                    "policy_reference": "POL-ELIG-2026 §4",
                }
            )

    # 7. Credit notes risk signals
    if any(kw in credit_notes for kw in ("late", "delinquen", "default", "write-off", "dispute")):
        risks.append(
            {
                "code": "CREDIT_HISTORY_FLAG",
                "severity": "MEDIUM",
                "description": "Credit notes mention late payment, delinquency, or dispute requiring officer inspection.",
                "policy_reference": "Credit Notes Review",
            }
        )

    eligible = len(ineligible_reasons) == 0
    return eligible, ineligible_reasons, risks, dti_ratio


def build_loan_workflow(
    kb: Optional[PolicyKnowledgeBase] = None,
    store: Optional[ReviewHistoryStore] = None,
    checkpointer: Optional[MemorySaver] = None,
):
    """Build and compile the LangGraph Loan Approval Agentic Application."""
    knowledge_base = kb or get_knowledge_base()
    history_store = store or ReviewHistoryStore()
    memory = checkpointer or MemorySaver()

    def input_guardrails_node(state: GraphState) -> dict[str, Any]:
        raw_app = state.get("application", {})
        is_valid, sanitized, errors, notes = run_input_guardrails(raw_app)
        return {
            "application": sanitized,
            "validation_passed": is_valid,
            "validation_errors": errors,
            "guardrail_notes": notes,
            "resubmit_count": state.get("resubmit_count", 0),
            "execution_trace": [
                f"Input Guardrails -> {'PASSED' if is_valid else 'FAILED'}"
            ],
        }

    def route_after_input_guardrails(
        state: GraphState,
    ) -> Literal["LoanApplicationDocumentsChecker_Node", "Output Guardrails"]:
        if state.get("validation_passed"):
            return "LoanApplicationDocumentsChecker_Node"
        return "Output Guardrails"

    def docs_checker_node(state: GraphState) -> dict[str, Any]:
        app = state["application"]
        loan_type = str(app.get("loan_type", "personal"))
        emp_type = str(app.get("employment_type", "salaried"))

        query = (
            f"Required documents for {loan_type} loan {emp_type} applicant "
            "identity tax bank statements"
        )
        clauses = knowledge_base.retrieve(
            query, k=3, source_filter="02_required_documents"
        )
        doc_eval = evaluate_documents(
            loan_type=loan_type,
            employment_type=emp_type,
            submitted_documents=app.get("documents", []),
        )

        status_label = (
            "COMPLETE"
            if doc_eval["documents_correct"]
            else f"MISSING/INVALID ({len(doc_eval['missing_documents'])})"
        )
        return {
            "retrieved_doc_policies": [c.model_dump() for c in clauses],
            "documents_correct": doc_eval["documents_correct"],
            "missing_documents": doc_eval["missing_documents"],
            "document_issues": doc_eval["document_issues"],
            "execution_trace": [
                f"LoanApplicationDocumentsChecker_Node -> {status_label}"
            ],
        }

    def route_documents_correct(
        state: GraphState,
    ) -> Literal["CheckLoanEligibility_Node", "Request corrected documents"]:
        if state.get("documents_correct"):
            return "CheckLoanEligibility_Node"
        return "Request corrected documents"

    def request_corrected_documents_node(state: GraphState) -> dict[str, Any]:
        resubmit_count = int(state.get("resubmit_count", 0))
        missing = state.get("missing_documents", [])
        app = state.get("application", {})

        if resubmit_count >= settings.max_resubmit_attempts:
            return {
                "final_status": "PENDING_DOCUMENTS",
                "ai_assessment": {
                    "summary": (
                        f"Application {app.get('application_id')} paused: maximum document "
                        f"resubmission attempts ({settings.max_resubmit_attempts}) reached."
                    ),
                    "risk_indicators": [
                        {
                            "code": "MISSING_DOCUMENTS_MAX_RETRIES",
                            "severity": "HIGH",
                            "description": f"Missing/invalid documents: {', '.join(missing)}",
                            "policy_reference": "POL-DOCS-2026 §4",
                        }
                    ],
                    "recommendation": "RESUBMIT_DOCUMENTS",
                    "recommendation_rationale": "Applicant must provide all mandatory documents before eligibility evaluation.",
                    "confidence": 0.90,
                },
                "execution_trace": [
                    "Request corrected documents -> MAX RETRIES REACHED (PENDING_DOCUMENTS)"
                ],
            }

        # Human-in-the-loop pause for document resubmission
        resume_payload = interrupt(
            {
                "interrupt_type": "document_resubmission",
                "application_id": app.get("application_id"),
                "applicant_name": app.get("full_name"),
                "missing_documents": missing,
                "resubmit_count": resubmit_count,
                "max_resubmit_attempts": settings.max_resubmit_attempts,
                "message": "Mandatory documents are missing or invalid. Resubmit corrected documents to continue.",
            }
        )

        action = "resubmit"
        new_docs: list[dict[str, Any]] = []
        if isinstance(resume_payload, dict):
            action = str(resume_payload.get("action", "resubmit")).lower()
            new_docs = resume_payload.get("documents", []) or []

        if action == "cancel":
            return {
                "final_status": "PENDING_DOCUMENTS",
                "ai_assessment": {
                    "summary": f"Application {app.get('application_id')} held pending missing documents.",
                    "risk_indicators": [
                        {
                            "code": "MISSING_DOCUMENTS",
                            "severity": "HIGH",
                            "description": f"Missing/invalid documents: {', '.join(missing)}",
                            "policy_reference": "POL-DOCS-2026 §4",
                        }
                    ],
                    "recommendation": "RESUBMIT_DOCUMENTS",
                    "recommendation_rationale": "Review cancelled/paused by officer pending document collection.",
                    "confidence": 0.90,
                },
                "execution_trace": [
                    "Request corrected documents -> HELD (PENDING_DOCUMENTS)"
                ],
            }

        # Merge updated documents into application state by doc_type
        merged_map: dict[str, dict[str, Any]] = {}
        for doc in app.get("documents", []):
            key = str(doc.get("doc_type", "")).strip().lower()
            if key:
                merged_map[key] = doc
        for doc in new_docs:
            key = str(doc.get("doc_type", "")).strip().lower()
            if key:
                merged_map[key] = doc

        updated_app = dict(app)
        updated_app["documents"] = list(merged_map.values())

        return {
            "application": updated_app,
            "resubmit_count": resubmit_count + 1,
            "final_status": "",
            "execution_trace": [
                f"Request corrected documents -> Resubmit #{resubmit_count + 1}"
            ],
        }

    def route_after_document_request(
        state: GraphState,
    ) -> Literal["LoanApplicationDocumentsChecker_Node", "Output Guardrails"]:
        if state.get("final_status") == "PENDING_DOCUMENTS":
            return "Output Guardrails"
        return "LoanApplicationDocumentsChecker_Node"

    def check_loan_eligibility_node(state: GraphState) -> dict[str, Any]:
        app = state["application"]
        loan_type = str(app.get("loan_type", "personal"))
        credit_score = int(app.get("credit_score", 0))
        loan_amount = float(app.get("loan_amount", 0))
        tenure_months = int(app.get("tenure_months", 12))

        # Retrieve relevant eligibility & rate clauses via RAG BEFORE generating assessment
        elig_clauses = knowledge_base.retrieve(
            f"{loan_type} loan eligibility age credit score {credit_score} DTI income limits",
            k=3,
            source_filter="01_loan_eligibility",
        )
        rate_clauses = knowledge_base.retrieve(
            f"{loan_type} loan interest rate tenure {tenure_months} months EMI calculation",
            k=2,
            source_filter="03_interest_rates",
        )
        retrieved_policies = [c.model_dump() for c in (elig_clauses + rate_clauses)]

        emi_details = compute_emi_details(
            loan_type=loan_type,
            credit_score=credit_score,
            loan_amount=loan_amount,
            tenure_months=tenure_months,
        )

        eligible, reasons, risks, dti_ratio = _evaluate_policy_eligibility(
            app, emi_details
        )
        emi_details["dti_ratio"] = dti_ratio

        assessment, llm_backend = generate_eligibility_assessment(
            application=app,
            emi_details=emi_details,
            dti_ratio=dti_ratio,
            eligible=eligible,
            eligibility_reasons=reasons,
            risk_indicators=risks,
            retrieved_policies=retrieved_policies,
        )

        return {
            "retrieved_elig_policies": retrieved_policies,
            "emi_details": emi_details,
            "loan_eligible": eligible,
            "eligibility_reasons": reasons,
            "risk_indicators": risks,
            "ai_assessment": assessment.model_dump(mode="json"),
            "confidence": assessment.confidence,
            "execution_trace": [
                f"CheckLoanEligibility_Node ({llm_backend}) -> "
                f"{'ELIGIBLE' if eligible else 'INELIGIBLE'} (confidence={assessment.confidence:.2f})"
            ],
        }

    def route_loan_eligible(
        state: GraphState,
    ) -> Literal["Human Approval Node", "Loan rejection"]:
        if state.get("loan_eligible"):
            return "Human Approval Node"
        return "Loan rejection"

    def loan_rejection_node(state: GraphState) -> dict[str, Any]:
        return {
            "final_status": "REJECTED_INELIGIBLE",
            "execution_trace": ["Loan rejection -> REJECTED_INELIGIBLE"],
        }

    def human_approval_node(state: GraphState) -> dict[str, Any]:
        app = state["application"]
        ai = state.get("ai_assessment", {})
        emi = state.get("emi_details", {})

        decision_payload = interrupt(
            {
                "interrupt_type": "human_approval",
                "application_id": app.get("application_id"),
                "applicant_name": app.get("full_name"),
                "loan_type": app.get("loan_type"),
                "loan_amount": app.get("loan_amount"),
                "summary": ai.get("summary", ""),
                "recommendation": ai.get("recommendation", "MANUAL_REVIEW"),
                "recommendation_rationale": ai.get("recommendation_rationale", ""),
                "confidence": ai.get("confidence", state.get("confidence", 0.5)),
                "risk_indicators": ai.get("risk_indicators", state.get("risk_indicators", [])),
                "emi_details": emi,
                "retrieved_policies": state.get("retrieved_elig_policies", []),
                "message": "Loan application passed automated eligibility checks. Human Loan Officer approval or rejection is required.",
            }
        )

        decision = "reject"
        reviewer = "Loan Officer"
        comments = ""
        if isinstance(decision_payload, dict):
            decision = str(decision_payload.get("decision", "reject")).strip().lower()
            reviewer = str(decision_payload.get("reviewer") or "Loan Officer").strip()
            comments = str(decision_payload.get("comments") or "").strip()

        normalized = "approve" if decision in ("approve", "approved", "yes") else "reject"
        return {
            "human_decision": normalized,
            "human_reviewer": reviewer,
            "human_comments": comments,
            "execution_trace": [
                f"Human Approval Node -> {normalized.upper()} by {reviewer}"
            ],
        }

    def route_human_approves(
        state: GraphState,
    ) -> Literal["Loan approved", "Loan rejected by human"]:
        if state.get("human_decision") == "approve":
            return "Loan approved"
        return "Loan rejected by human"

    def loan_approved_node(state: GraphState) -> dict[str, Any]:
        return {
            "final_status": "APPROVED_BY_HUMAN",
            "execution_trace": ["Loan approved -> APPROVED_BY_HUMAN"],
        }

    def loan_rejected_by_human_node(state: GraphState) -> dict[str, Any]:
        return {
            "final_status": "REJECTED_BY_HUMAN",
            "execution_trace": ["Loan rejected by human -> REJECTED_BY_HUMAN"],
        }

    def output_guardrails_node(state: GraphState) -> dict[str, Any]:
        structured_output, updated_notes = run_output_guardrails(state)
        out_dict = structured_output.model_dump(mode="json")
        trace = list(state.get("execution_trace", [])) + [
            f"Output Guardrails -> {structured_output.final_status} "
            f"(recommendation={structured_output.recommendation}, confidence={structured_output.confidence:.2f})"
        ]

        thread_id = state.get("thread_id") or structured_output.application_id
        history_store.save_review(
            thread_id=thread_id,
            application=state.get("application", {}),
            structured_output=out_dict,
            execution_trace=trace,
        )

        return {
            "structured_output": out_dict,
            "guardrail_notes": updated_notes,
            "final_status": structured_output.final_status,
            "execution_trace": [
                f"Output Guardrails -> {structured_output.final_status}"
            ],
        }

    builder = StateGraph(GraphState)

    # Nodes matching the architecture diagram
    builder.add_node("Input Guardrails", input_guardrails_node)
    builder.add_node("LoanApplicationDocumentsChecker_Node", docs_checker_node)
    builder.add_node("Request corrected documents", request_corrected_documents_node)
    builder.add_node("CheckLoanEligibility_Node", check_loan_eligibility_node)
    builder.add_node("Loan rejection", loan_rejection_node)
    builder.add_node("Human Approval Node", human_approval_node)
    builder.add_node("Loan approved", loan_approved_node)
    builder.add_node("Loan rejected by human", loan_rejected_by_human_node)
    builder.add_node("Output Guardrails", output_guardrails_node)

    # Edges matching the architecture diagram
    builder.add_edge(START, "Input Guardrails")
    builder.add_conditional_edges("Input Guardrails", route_after_input_guardrails)
    builder.add_conditional_edges(
        "LoanApplicationDocumentsChecker_Node", route_documents_correct
    )
    builder.add_conditional_edges(
        "Request corrected documents", route_after_document_request
    )
    builder.add_conditional_edges("CheckLoanEligibility_Node", route_loan_eligible)
    builder.add_edge("Loan rejection", "Output Guardrails")
    builder.add_conditional_edges("Human Approval Node", route_human_approves)
    builder.add_edge("Loan approved", "Output Guardrails")
    builder.add_edge("Loan rejected by human", "Output Guardrails")
    builder.add_edge("Output Guardrails", END)

    return builder.compile(checkpointer=memory)


class LoanReviewService:
    """High-level facade for running and resuming LangGraph loan application reviews."""

    def __init__(
        self,
        kb: Optional[PolicyKnowledgeBase] = None,
        store: Optional[ReviewHistoryStore] = None,
    ):
        self.kb = kb or get_knowledge_base()
        self.store = store or ReviewHistoryStore()
        self.checkpointer = MemorySaver()
        self.graph = build_loan_workflow(
            kb=self.kb,
            store=self.store,
            checkpointer=self.checkpointer,
        )

    def _format_response(self, thread_id: str, result: dict[str, Any]) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        interrupts: list[Any] = []
        if snapshot.tasks:
            for task in snapshot.tasks:
                if getattr(task, "interrupts", None):
                    interrupts.extend(item.value for item in task.interrupts)

        if interrupts:
            return {
                "status": "INTERRUPTED",
                "thread_id": thread_id,
                "interrupt": interrupts[0],
                "state": snapshot.values,
            }

        return {
            "status": "COMPLETED",
            "thread_id": thread_id,
            "structured_output": result.get("structured_output", {}),
            "state": result,
        }

    def start_review(
        self, application_data: dict[str, Any], thread_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Start a new loan review workflow."""
        tid = thread_id or f"thread-{uuid4().hex[:10]}"
        config = {"configurable": {"thread_id": tid}}
        initial_state: GraphState = {
            "thread_id": tid,
            "application": application_data,
            "execution_trace": [],
        }
        result = self.graph.invoke(initial_state, config=config)
        return self._format_response(tid, result)

    def resubmit_documents(
        self, thread_id: str, documents: list[dict[str, Any]], action: str = "resubmit"
    ) -> dict[str, Any]:
        """Resume a paused review after providing corrected/missing documents."""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(
            Command(resume={"action": action, "documents": documents}),
            config=config,
        )
        return self._format_response(thread_id, result)

    def submit_human_decision(
        self,
        thread_id: str,
        decision: Literal["approve", "reject"],
        reviewer: str = "Loan Officer",
        comments: str = "",
    ) -> dict[str, Any]:
        """Resume a paused review at Human Approval Node with officer's decision."""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(
            Command(
                resume={
                    "decision": decision,
                    "reviewer": reviewer,
                    "comments": comments,
                }
            ),
            config=config,
        )
        return self._format_response(thread_id, result)
