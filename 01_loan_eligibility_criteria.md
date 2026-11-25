# Bank Loan Eligibility Criteria Policy (Policy ID: POL-ELIG-2026)

## 1. General Applicant Eligibility
- **Minimum Age**: 21 years at the time of application.
- **Maximum Age**: 65 years at loan maturity for salaried applicants; 70 years for self-employed applicants.
- **Residency**: Must be a resident citizen or permanent resident with valid KYC verification.
- **Employment Stability**:
  - Salaried applicants: Minimum 12 months of continuous employment (at least 6 months in current organization).
  - Self-employed / Business owners: Minimum 24 months of documented business continuity with audited financials or ITR.

## 2. Credit Score Requirements
- **Minimum Credit Score (CIBIL / Equivalent)**:
  - **750 and above**: Prime tier; eligible for lowest interest rate slab and fast-track review.
  - **700 – 749**: Standard tier; eligible subject to income and DTI checks.
  - **650 – 699**: Sub-prime / Caution tier; high risk indicator; requires manual officer review, stricter DTI cap (<= 40%), and possible collateral or co-applicant.
  - **Below 650**: Ineligible under standard retail lending guidelines; must be marked ineligible unless explicitly escalated under secured exception policy.

## 3. Income & Debt-to-Income (DTI) Ratio
- **Minimum Monthly Net Income**:
  - Personal Loan: $2,000 (or equivalent local currency unit: 25,000 INR/month; system normalizes to application currency units). Minimum annual income: 300,000.
  - Home Loan: Minimum annual income 450,000.
  - Auto Loan: Minimum annual income 300,000.
  - Education Loan: Minimum co-applicant annual income 350,000.
- **Debt-to-Income (DTI) Ratio**:
  - Calculated as `(Existing Monthly Debt Obligations + Proposed Loan EMI) / Gross Monthly Income`.
  - **Maximum Allowed DTI**: 50% (0.50) for prime/standard applicants (credit score >= 700).
  - **Caution DTI Zone**: 40% – 50% triggers a moderate risk indicator.
  - **Hard Cap for Sub-prime (650–699)**: Maximum allowed DTI is 40% (0.40).
  - **DTI > 50%**: Applicant is ineligible due to excessive debt burden.

## 4. Loan Amount Limits
- **Personal Loan**: 50,000 to 2,500,000 (max 20x monthly net income).
- **Home Loan**: 500,000 to 50,000,000 (Loan-to-Value <= 80%).
- **Auto Loan**: 100,000 to 5,000,000 (Loan-to-Value <= 85%).
- **Education Loan**: 100,000 to 10,000,000.

## 5. Human-in-the-Loop Governance Rule
- Automated AI assistants are strictly advisory. An AI system must **never** issue a binding final loan approval or disbursement instruction without explicit human loan officer sign-off.
