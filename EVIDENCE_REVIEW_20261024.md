# CPAF 2.0 provider evidence review — 2026-10-24

## Scope and review method

The working evidence register is
`cpaf_outputs/evidence_reviews/evidence_register_reviewed.csv`. The original
`cpaf_outputs/evidence_register_template.csv` remains an empty template. Each of
the 21 existing provider/dimension combinations was checked against the named
official provider page or help/terms page. The `access_date` records the date
the pages were opened and checked in this review. Legal context was checked
against the notified Digital Personal Data Protection Act commencement
notification and the notified Rules.

The reviewer field explicitly says **CPAF research review (AI-assisted)**.
Seventeen entries are scored documentary assessments of what a provider
publicly states. The 1–5 score uses the repository's existing rubric and
formula; evidence strength (0–1) is a qualitative source reliability/currency
weight, not a probability that the control works. Primary official sources are
given higher source strength than the older Paytm privacy statement. No
operational, application, security, deletion, or independent audit tests were
performed. The scores therefore do not establish technical implementation or
provider-wide privacy performance.

## Rows left unscored

Four rows remain unreviewed and have blank reviewer and score fields:

| Provider | Dimension | Reason |
|---|---|---|
| Paytm | Anonymisation & Re-identification | Paytm Token Gateway materials concern card tokenisation for merchant/payment-gateway processing. They do not establish anonymisation or re-identification resistance for the consumer Paytm app or its transaction histories. |
| Paytm | DPDP Alignment | The provider policy cites the IT Act and other applicable Indian law but does not establish operational DPDP conformity. The policy states it was last updated 10 July 2024. |
| Google Pay | DPDP Alignment | The India terms discuss applicable laws but do not establish operational DPDP conformity. |
| PhonePe | DPDP Alignment | The current privacy policy references IT Rules/SPDI and Aadhaar law but does not establish operational DPDP conformity. |

The official MeitY commencement notification of 13 November 2025 schedules
major Act duties 18 months after publication, and the Rules also phase key
requirements over 18 months. As of the register's 24 October 2026 access date,
that transition period had not elapsed. The policy language and the transition
schedule are not evidence that any provider is compliant or non-compliant.
These rows require human legal/compliance review and further provider-specific
operational evidence. Absence of public disclosure is not treated as proof that
a control is absent.

## Official sources consulted

- Paytm Privacy Policy, including data collection, purposes, sharing,
  safeguards, retention, consent and contact sections; last updated 10 July
  2024: <https://paytm.com/company/privacy-policy>.
- Paytm Token Gateway product material, used only to document why merchant
  gateway tokenisation cannot support a consumer-app anonymisation score:
  <https://business.paytm.com/token-gateway>.
- Google Pay Terms of Service for India Residents, last updated 31 July 2026:
  <https://pay.google.com/intl/en_in/about/terms/>.
- Google Pay privacy settings and security help:
  <https://support.google.com/pay/india/answer/10270219?hl=en-IN>.
- Google Pay data export/deletion help:
  <https://support.google.com/googlepay/answer/9015738?hl=en-IN>.
- Google Pay card-tokenisation help:
  <https://support.google.com/pay/india/answer/16526545?hl=en>.
- PhonePe Privacy Policy, updated 15 May 2026:
  <https://www.phonepe.com/privacy-policy/>.
- PhonePe card payment and tokenisation terms:
  <https://www.phonepe.com/terms-conditions/debit-credit/>.
- MeitY notification specifying phased commencement of the DPDP Act:
  <https://www.meity.gov.in/static/uploads/2025/11/c56ceae6c383460ca69577428d36828b.pdf>.
- MeitY notified Digital Personal Data Protection Rules, 2025:
  <https://www.meity.gov.in/static/uploads/2025/11/53450e6e5dc0bfa85ebd78686cadad39.pdf>.

## Reproduction and validation

The 21-row working register can be rebuilt from the unchanged template and
review findings using:

```powershell
$py = 'C:\Users\harsh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py scripts\build_evidence_review_20261024.py
```

If the working register already exists, replacement requires explicit
confirmation and creates a timestamped backup:

```powershell
& $py scripts\build_evidence_review_20261024.py --confirm-overwrite
```

The offline regression test checks all combinations, URL syntax, numeric
ranges, score eligibility, unsupported rows, display formatting and saved-CSV
reload. A passing URL syntax test is not a continuing guarantee that a remote
page remains available or unchanged.
