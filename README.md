# Loan Application Risk Assistant

An AI-assisted loan application review service built with **Python**, **LangGraph**, **LangChain**, **RAG**, **Guardrails**, **Human-in-the-Loop (HITL)**, and **Local LLM (Ollama)**.

It helps loan officers review applications faster by summarizing applicant details, checking required documents against bank policy, retrieving relevant eligibility & interest-rate clauses, calculating EMI/DTI metrics, identifying risk factors, and generating an explainable structured recommendation—while keeping final lending decisions strictly under human control.

---

## Architecture & LangGraph Workflow

The agent workflow in [`src/loan_assistant/graph.py`](src/loan_assistant/graph.py) implements the full state machine:

```
START
  │
  ▼
[Input Guardrails]  ──(invalid data / prompt injection)──► [Output Guardrails] ──► END
  │ (valid)
  ▼
[LoanApplicationDocumentsChecker_Node]  ◄─────────────────────┐
  │                                                           │ (Resubmit)
  ▼                                                           │
<Documents correct?> ──(No)──► [Request corrected documents] ─┘  (HITL interrupt)
  │ (Yes)
  ▼
[CheckLoanEligibility_Node]  (RAG + EMI/DTI tools + Local LLM)
  │
  ▼
<Loan eligible?>
  ├──(No)──► [Loan rejection] ────────────────────────────────┐
  │                                                           │
  └──(Yes)─► [Human Approval Node]  (HITL interrupt)          │
                │                                             │
                ▼                                             │
             <Human approves?>                                │
                ├──(No)──► [Loan rejected by human] ──────────┤
                │                                             │
                └──(Yes)─► [Loan approved] ───────────────────┤
                                                              ▼
                                                     [Output Guardrails]
                                                              │
                                                              ▼
                                                             END
```

---

## RAG Pipeline & Prompt Design

1. **Knowledge Base (`knowledge_base/`)**:
   - `01_loan_eligibility_criteria.md`: Age, credit score, income, employment tenure, and DTI thresholds across Personal, Home, Auto, and Education loans.
   - `02_required_documents_policy.md`: Mandatory KYC, income, bank statement, employment, and collateral documents by loan type and employment type.
   - `03_interest_rates_and_emi_schedule.md`: Base interest rate slabs by credit score tier, tenure bounds, processing fees, and EMI formula.
2. **Indexing & Retrieval (`src/loan_assistant/rag.py`)**:
   - Markdown documents are split into section-aware chunks with metadata (`source`, `section`, `doc_id`).
   - Uses `OllamaEmbeddings` (`nomic-embed-text`) with a `FAISS` vector store when Ollama is available, and automatically falls back to a local TF-IDF/cosine retriever so the pipeline works seamlessly even offline.
   - Policy clauses are retrieved **before** document verification and eligibility evaluation, and cited in the final structured output.
3. **Structured Response Schema (`src/loan_assistant/schemas.py`)**:
   - `summary`: Concise applicant & financial summary.
   - `missing_documents`: List of missing or invalid documents per policy.
   - `risk_indicators`: Identified risk factors with severity and policy references.
   - `recommendation`: Explainable advisory recommendation (`APPROVE_RECOMMENDED`, `MANUAL_REVIEW`, `REJECT_RECOMMENDED`, `RESUBMIT_DOCUMENTS`) — never an autonomous final decision.
   - `confidence`: Score in `[0.0, 1.0]` with low-confidence warning flag when `< 0.65`.

---

## Guardrails & Error Handling

- **Input Guardrails**: Validates applicant age, income, credit score, loan amount, tenure, and document references; detects suspicious prompt-injection patterns in free-text notes.
- **Output Guardrails**:
  - Enforces the Pydantic structured output schema.
  - Ensures AI recommendations never claim autonomous final approval/rejection without human officer sign-off.
  - Flags low-confidence responses (`confidence < 0.65`) for mandatory senior officer review.
  - Redacts sensitive identifiers (e.g., full PAN/SSN/Aadhaar patterns) from logged summaries.

---

## Quick Start

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Local LLM (Ollama)

Install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Copy `.env.example` to `.env` if you want to customize model names or thresholds.
*(If Ollama is not running, the application automatically runs in deterministic offline fallback mode so you can still test the full LangGraph, RAG, Guardrails, HITL, and UI flows.)*

### 3. Run the Reviewer UI (Streamlit)

```bash
streamlit run app.py
```

### 4. Run via CLI

```bash
python main.py
```

### 5. Run Tests

```bash
pytest -v
```
