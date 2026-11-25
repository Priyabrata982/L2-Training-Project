# Loan Interest Rates & EMI Schedule Policy (Policy ID: POL-RATES-2026)

## 1. Annual Interest Rates by Loan Type & Credit Score Tier

### Personal Loan (Tenure: 12 to 60 months)
- Credit Score >= 750: **10.75% p.a.**
- Credit Score 700 – 749: **12.50% p.a.**
- Credit Score 650 – 699: **15.25% p.a.**

### Home Loan (Tenure: 60 to 360 months)
- Credit Score >= 750: **8.40% p.a.**
- Credit Score 700 – 749: **8.90% p.a.**
- Credit Score 650 – 699: **9.65% p.a.**

### Auto Loan (Tenure: 12 to 84 months)
- Credit Score >= 750: **8.85% p.a.**
- Credit Score 700 – 749: **9.50% p.a.**
- Credit Score 650 – 699: **10.75% p.a.**

### Education Loan (Tenure: 12 to 180 months)
- Credit Score >= 750: **9.15% p.a.**
- Credit Score 700 – 749: **9.85% p.a.**
- Credit Score 650 – 699: **10.90% p.a.**

## 2. Tenure Rules & Processing Fees
- **Personal Loan**: Allowed tenure 12–60 months; Processing fee 1.50% of loan amount.
- **Home Loan**: Allowed tenure 60–360 months; Processing fee 0.50% of loan amount.
- **Auto Loan**: Allowed tenure 12–84 months; Processing fee 1.00% of loan amount.
- **Education Loan**: Allowed tenure 12–180 months; Processing fee 0.50% of loan amount.
- Applications requesting a tenure outside the allowed range for that loan type are flagged as invalid/ineligible.

## 3. EMI Calculation Standard
The monthly Equated Monthly Installment (EMI) is computed using the standard reducing-balance formula:

`EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)`

Where:
- `P` = Principal loan amount
- `r` = Monthly interest rate = `Annual Interest Rate / (12 * 100)`
- `n` = Loan tenure in months

### Reference EMI Examples (per 100,000 principal)
- **Home Loan @ 8.40% p.a.**:
  - 120 months (10 yrs): 1,234.52 / month
  - 240 months (20 yrs): 861.50 / month
  - 360 months (30 yrs): 761.84 / month
- **Personal Loan @ 10.75% p.a.**:
  - 12 months (1 yr): 8,826.55 / month
  - 36 months (3 yrs): 3,262.03 / month
  - 60 months (5 yrs): 2,161.80 / month
- **Auto Loan @ 8.85% p.a.**:
  - 36 months (3 yrs): 3,172.88 / month
  - 60 months (5 yrs): 2,068.54 / month
