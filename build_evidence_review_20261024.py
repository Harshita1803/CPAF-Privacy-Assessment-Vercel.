"""Build the 2026-10-24 source-reviewed evidence CSV from the unchanged template.

Claims summarize official public disclosures accessed on the recorded date; they
do not independently verify a provider's technical implementation. Four rows are
left unscored where the available official material does not support a responsible
in-scope score. Use --confirm-overwrite to replace an existing working review;
the safe writer creates a timestamped backup before replacement.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from cpaf.evidence import read_evidence_csv, save_evidence_csv
from cpaf_advanced import APPS, DIMENSIONS


ACCESS_DATE = "2026-10-24"
REVIEWER = "CPAF research review (AI-assisted)"
OFFICIAL_PAYTM_POLICY = "https://paytm.com/company/privacy-policy"
OFFICIAL_PHONEPE_POLICY = "https://www.phonepe.com/privacy-policy/"
OFFICIAL_GOOGLE_TERMS = "https://pay.google.com/intl/en_in/about/terms/"
OFFICIAL_GOOGLE_PRIVACY_HELP = "https://support.google.com/pay/india/answer/10270219?hl=en-IN"
OFFICIAL_GOOGLE_DELETE_HELP = "https://support.google.com/googlepay/answer/9015738?hl=en-IN"
OFFICIAL_GOOGLE_CARD_HELP = "https://support.google.com/pay/india/answer/16526545?hl=en"
OFFICIAL_PHONEPE_CARD_TERMS = "https://www.phonepe.com/terms-conditions/debit-credit/"
OFFICIAL_PAYTM_TOKEN_GATEWAY = "https://business.paytm.com/token-gateway"
DPDP_ACT_COMMENCEMENT = "https://www.meity.gov.in/static/uploads/2025/11/c56ceae6c383460ca69577428d36828b.pdf"
DPDP_RULES = "https://www.meity.gov.in/static/uploads/2025/11/53450e6e5dc0bfa85ebd78686cadad39.pdf"


def finding(app, dimension, claim, title, url, source_type, evidence_note,
            rationale, score, strength, notes):
    return {
        "app": app, "dimension": dimension, "claim": claim,
        "source_title": title, "source_url": url, "access_date": ACCESS_DATE,
        "source_type": source_type, "evidence_note": evidence_note,
        "evidence_strength": strength, "reviewer": REVIEWER if score is not None else "",
        "review_status": "reviewed" if score is not None else "unreviewed",
        "score_rationale": rationale,
        "raw_score": score, "notes": notes,
    }


UNAUDITED = "Score is a documentary assessment of a provider-published statement, not independent verification of technical or operational implementation."
GAP = "No raw score: the material accessed does not provide sufficient in-scope evidence to rate this dimension. Lack of disclosure is not proof that a control is absent. No evidence or score has been invented."
RESEARCHER_LIMIT = "AI-assisted source review; a human reviewer should confirm the source snapshot and judgment before formal submission."


FINDINGS = [
    finding("Paytm", "Data Minimisation",
        "Paytm's privacy statement says it collects personal data only as necessary for requirements, while listing a broad set of identity, financial, device, communication and in-app permission data.",
        "Paytm Privacy Policy", OFFICIAL_PAYTM_POLICY, "Official provider privacy policy",
        "Sections 3–4: necessity statement and examples including transaction/financial data, location, SIM/device identifiers, contacts/call logs, SMS notifications, camera, microphone, files, Wi-Fi SSID and NFC.",
        "3 — The policy states a necessity limit, but the collection catalogue and service purposes are broad and non-exhaustive. This is mixed disclosure, not a measured collection audit.", 3, 0.75,
        f"{UNAUDITED} {RESEARCHER_LIMIT} Policy page states last updated 10 July 2024; recency is weaker than the 2026 source pages."),
    finding("Paytm", "Consent & Transparency",
        "Paytm describes consent through use of its products/services and says users can manage/review marketing choices; the policy also lists categories and processing purposes.",
        "Paytm Privacy Policy", OFFICIAL_PAYTM_POLICY, "Official provider privacy policy",
        "Sections 1, 4 and 11: consent language, data/purpose disclosure, marketing-choice promise, and change-notification description.",
        "3 — The policy provides substantial categories and purposes, but describes consent in connection with using the service and does not establish that every in-app consent is granular or freely withdrawable.", 3, 0.75,
        f"{UNAUDITED} Marketing choice statements are policy disclosures; the actual app consent flow was not tested. {RESEARCHER_LIMIT}"),
    finding("Paytm", "Third-Party Sharing",
        "Paytm says it does not sell personal data and describes need-to-know sharing with third parties, affiliates and subsidiaries, including payment gateway processors and service providers.",
        "Paytm Privacy Policy", OFFICIAL_PAYTM_POLICY, "Official provider privacy policy",
        "Sections 7–8: stated no-sale promise, sharing examples/recipient categories, and data-centre/contracted-server statements.",
        "3 — The policy states a no-sale commitment and need-to-know framing, while permitting sharing for broad service, product and operational purposes. Recipient handling is not independently audited.", 3, 0.75,
        f"Do not interpret the no-sale statement as no third-party sharing. {UNAUDITED} {RESEARCHER_LIMIT}"),
    finding("Paytm", "Security Controls",
        "Paytm's policy describes encryption and access controls, PCI-standard encryption for card data, an optional app lock, and additional authentication when using the account on another device.",
        "Paytm Privacy Policy", OFFICIAL_PAYTM_POLICY, "Official provider privacy policy",
        "Section 9: stated server-layer encryption/access control, card-data PCI encryption, app lock and additional authentication/OTP; policy disclaims absolute protection.",
        "4 — Several specific safeguards and user-facing controls are disclosed. The rating reflects the specificity of the policy statement, not audited deployment or effectiveness.", 4, 0.75,
        f"{UNAUDITED} The provider expressly notes protection is not absolute. {RESEARCHER_LIMIT}"),
    finding("Paytm", "Anonymisation & Re-identification",
        "", "Paytm Token Gateway", OFFICIAL_PAYTM_TOKEN_GATEWAY, "Official Paytm payment-gateway documentation",
        "The merchant Token Gateway page describes replacing card numbers with virtual tokens and keeping the token-to-card mapping in a card-network or issuing-bank vault. This material concerns merchant/payment-gateway processing, not a demonstrated anonymisation or re-identification control in the consumer Paytm app.",
        GAP, None, None,
        "Unsupported for the consumer-app dimension: payment-card tokenization in the merchant gateway is not anonymisation of Paytm app transaction histories and does not establish re-identification resistance. Human review required."),
    finding("Paytm", "Data Lifecycle & User Rights",
        "Paytm says users can delete their account and describes purpose-based retention/erasure with legal, safety, investigation and rights-protection exceptions.",
        "Paytm Privacy Policy", OFFICIAL_PAYTM_POLICY, "Official provider privacy policy",
        "Section 10: account deletion route, purpose/consent-based retention, and listed exceptions; section 12 gives a privacy/grievance contact.",
        "4 — The policy describes account deletion and a purpose-linked retention rule, with exceptions disclosed. Actual deletion timing and downstream deletion were not tested.", 4, 0.75,
        f"{UNAUDITED} Legal and safety exceptions remain. {RESEARCHER_LIMIT}"),
    finding("Paytm", "DPDP Alignment",
        "", "Paytm Privacy Policy", OFFICIAL_PAYTM_POLICY, "Official provider privacy policy",
        "The published policy identifies the IT Act, 2000 and other applicable Indian laws and shows a last-updated date of 10 July 2024; this document does not provide evidence of operational conformity with the DPDP Act and Rules.",
        GAP, None, None,
        f"Unsupported as a compliance score. MeitY's 13 November 2025 commencement notification phases the Act's principal duties in 18 months; the Rules also phase key duties in 18 months. Legal references or an older privacy policy do not establish provider compliance. Official schedule: {DPDP_ACT_COMMENCEMENT} ; {DPDP_RULES}. Human legal/compliance review required."),

    finding("Google Pay", "Data Minimisation",
        "Google Pay's India terms list identity and payment-instrument information and additional loan-application information when relevant; its privacy help says core transaction data remain saved when optional in-app personalisation is off.",
        "Google Pay Terms of Service for India Residents", OFFICIAL_GOOGLE_TERMS, "Official provider terms",
        "Terms sections 4 and 27: registration/payment data, additional loan data and purpose-specific processing; official privacy help describes optional personalisation while core service data remain stored.",
        "3 — Service-specific information is described, and optional personalisation is separable, but broad payment and loan data categories are listed and core transactions remain saved. No independent data-collection audit is available here.", 3, 0.85,
        f"Scope is Google Pay India, not all Google products. {UNAUDITED} Privacy-setting behavior is described in official help and was not tested in the app. {RESEARCHER_LIMIT}"),
    finding("Google Pay", "Consent & Transparency",
        "Google documents an optional in-app personalisation control and says loan-application data is shared with lending third parties after explicit consent; core transaction data remain saved for service purposes when personalisation is off.",
        "How Google Pay Helps Keep Your Data Private", OFFICIAL_GOOGLE_PRIVACY_HELP, "Official Google Pay help documentation",
        "Sections on Personalisation within Google Pay and data management; India terms section 27 describes explicit consent for lender sharing.",
        "4 — Specific user choice over personalisation and explicit consent for a distinct loan-sharing flow are described. The choice does not stop core transaction storage, so the rating is limited to disclosed controls.", 4, 0.85,
        f"{UNAUDITED} This does not verify that all consent flows are understandable or equally easy to withdraw. {RESEARCHER_LIMIT}"),
    finding("Google Pay", "Third-Party Sharing",
        "Google's India terms disclose sharing UPI payment information with merchants, banks, third-party providers and service providers for payment operations/settlement; lender application data is shared one-time after explicit consent.",
        "Google Pay Terms of Service for India Residents", OFFICIAL_GOOGLE_TERMS, "Official provider terms",
        "Section 27: recipient categories and stated operational purposes; separate loan-data section says lender sharing is one-time and consent-based; UPI transaction data is excluded from Google Ads personalisation.",
        "3 — The policy makes recipient categories and some purposes explicit, including a limit on ad personalisation, while allowing broad service sharing. It does not evidence downstream handling by every recipient.", 3, 0.85,
        f"The terms distinguish Google Ads personalization from other service uses. {UNAUDITED} {RESEARCHER_LIMIT}"),
    finding("Google Pay", "Security Controls",
        "Google Pay help says the app requires a phone screen lock for each use and that Google Account security is used to help block unauthorised access to stored Google Pay data.",
        "How Google Pay Helps Keep Your Data Private", OFFICIAL_GOOGLE_PRIVACY_HELP, "Official Google Pay help documentation",
        "How we protect your data: required device screen lock and Google Account security protections. The India terms state generally that personal data is secured in compliance with applicable law.",
        "3 — The source specifies user-facing access protection and account-security safeguards but provides limited Google Pay-specific technical detail for storage, key management, audits or independent assurance.", 3, 0.85,
        f"{UNAUDITED} Provider help text does not constitute a technical security audit. {RESEARCHER_LIMIT}"),
    finding("Google Pay", "Anonymisation & Re-identification",
        "Google Pay help says saved payment cards use device-specific tokens replacing the card number, expiry date and CVV; this card-payment tokenization does not establish anonymisation of UPI transaction records.",
        "Fix Problems with Credit or Debit Card Payments on Google Pay", OFFICIAL_GOOGLE_CARD_HELP, "Official Google Pay help documentation",
        "The card troubleshooting section describes device-specific tokens as substitute values and says cards must be tokenized again on a new device.",
        "3 — A specific card-data substitution control is described, but the source does not describe token unlinkability, transaction-history anonymisation, or a re-identification risk assessment.", 3, 0.85,
        f"Limited to supported card use cases; do not generalise to UPI transaction histories or claim irreversibility. {UNAUDITED} {RESEARCHER_LIMIT}"),
    finding("Google Pay", "Data Lifecycle & User Rights",
        "Google Pay help describes data export, deletion of specific activity and service data, and activity auto-delete choices of 3, 18 or 36 months; it also states some data may be retained for service, business or legal needs.",
        "Find, Export or Delete Google Pay and Google Wallet Info", OFFICIAL_GOOGLE_DELETE_HELP, "Official Google Pay help documentation",
        "Data export, deletion, service deletion and auto-delete instructions; deletion caveats state some data may remain for essential services/business/legal requirements.",
        "4 — The help provides concrete access/export and deletion controls and configurable activity retention. It clearly limits deletion scope and identifies legally retained data, so not all payment records are covered by one setting.", 4, 0.85,
        f"The auto-delete control applies to My Activity, not necessarily every transaction record or partner-held data. {UNAUDITED} {RESEARCHER_LIMIT}"),
    finding("Google Pay", "DPDP Alignment",
        "", "Google Pay Terms of Service for India Residents", OFFICIAL_GOOGLE_TERMS, "Official provider terms",
        "The current India terms describe personal-data processing under applicable laws but do not provide evidence of operational conformity with the DPDP Act and Rules; the terms were updated 31 July 2026.",
        GAP, None, None,
        f"Unsupported as a compliance score. MeitY's 13 November 2025 commencement notification phases the Act's principal duties in 18 months; the Rules also phase key duties in 18 months. Applicable-law wording is not proof of compliance. Official schedule: {DPDP_ACT_COMMENCEMENT} ; {DPDP_RULES}. Human legal/compliance review required."),

    finding("PhonePe", "Data Minimisation",
        "PhonePe's policy says it collects information relevant and absolutely necessary for requested services but lists broad identity, KYC, financial, transaction, device, contact and optional permission data across its platform and partners.",
        "PhonePe Privacy Policy", OFFICIAL_PHONEPE_POLICY, "Official provider privacy policy",
        "Information Collection and Purpose and Use sections: necessity wording and non-exhaustive categories including contact, PAN/Aadhaar/KYC, financial/transaction history, device identifiers, SMS, and camera/microphone/location where applicable.",
        "3 — A relevance/necessity limit is disclosed, but the list spans multiple products and partner services and is expressly non-exhaustive. No data-flow or collection audit was performed.", 3, 0.85,
        f"The policy applies to PhonePe entities/subsidiaries and platform services, not solely UPI. {UNAUDITED} {RESEARCHER_LIMIT}"),
    finding("PhonePe", "Consent & Transparency",
        "PhonePe says using the platform or providing information signifies consent to policy processing; it separately describes marketing opt-outs and a route to revoke consent for Aadhaar e-KYC storage.",
        "PhonePe Privacy Policy", OFFICIAL_PHONEPE_POLICY, "Official provider privacy policy",
        "Your Consent; Choice/Opt-out; Personal Information Access/Rectification and Consent sections. The policy says revoking Aadhaar e-KYC storage consent can affect related services.",
        "3 — The policy identifies consent, an opt-out for non-essential communications, and an e-KYC consent-revocation route, while general service consent is framed through use/provision of information and may affect service access.", 3, 0.85,
        f"Policy wording does not verify actual in-app consent UX or granular controls for every purpose. {UNAUDITED} {RESEARCHER_LIMIT}"),
    finding("PhonePe", "Third-Party Sharing",
        "PhonePe lists sharing with business partners, service providers, sellers, merchants, financial institutions, authorities and other entities for service, KYC, payment, lending, security and legal purposes.",
        "PhonePe Privacy Policy", OFFICIAL_PHONEPE_POLICY, "Official provider privacy policy",
        "Information Sharing and Disclosures: recipient categories and need/purpose examples; policy says third-party processing follows recipient policies and describes protections as applicable and to the extent possible.",
        "3 — Recipients and many purposes are disclosed, and the policy makes a qualified protection statement, but the categories and purposes are broad and downstream practices vary by recipient.", 3, 0.85,
        f"This is a statement about permitted disclosures, not evidence of a particular transfer or recipient's controls. {UNAUDITED} {RESEARCHER_LIMIT}"),
    finding("PhonePe", "Security Controls",
        "PhonePe's policy describes encryption/controls for data in transit and at rest, internal/external reviews, firewall-protected databases, restricted password access, app lock and additional device authentication.",
        "PhonePe Privacy Policy", OFFICIAL_PHONEPE_POLICY, "Official provider privacy policy",
        "Reasonable Security Practices section also identifies administrative, technical and physical measures and acknowledges that no security system is impenetrable.",
        "4 — The policy gives multiple specific technical and user-facing safeguards, including data-in-motion/rest controls and access restriction. These are provider statements, not independently tested controls.", 4, 0.85,
        f"{UNAUDITED} No audit report or implementation evidence was examined. {RESEARCHER_LIMIT}"),
    finding("PhonePe", "Anonymisation & Re-identification",
        "PhonePe's card-payment terms describe optional tokenisation that replaces actual card details with a device-specific token and say actual card data are not stored except as permitted by law; the terms do not establish anonymisation of UPI records.",
        "Credit Card / Debit Card Payment Services – Terms and Conditions", OFFICIAL_PHONEPE_CARD_TERMS, "Official provider card-payment terms",
        "Section 5: token definition, user-initiated tokenisation, token service provider, actual-card-data storage statement, token management/detokenisation and token sharing for card processing.",
        "3 — This is a concrete, limited card-tokenisation disclosure, but tokens remain usable through token service providers and the source does not claim or establish irreversible anonymisation or re-identification resistance for transaction data.", 3, 0.85,
        f"Scope is card payment tokenisation; it is not evidence that UPI transaction data are anonymised. {UNAUDITED} {RESEARCHER_LIMIT}"),
    finding("PhonePe", "Data Lifecycle & User Rights",
        "PhonePe says data are retained no longer than necessary for collection purposes subject to stated exceptions, and describes access/review, correction, Aadhaar e-KYC consent revocation and account/personal-information deletion requests.",
        "PhonePe Privacy Policy", OFFICIAL_PHONEPE_POLICY, "Official provider privacy policy",
        "Storage and Retention; Personal Information Access/Rectification and Consent: purpose-linked retention, deletion after retention period, access/correction/delete route and exceptions for fraud, law and proceedings.",
        "4 — The policy discloses user request routes and a purpose-linked retention rule with explicit exceptions. Actual request completion/deletion across third parties was not verified.", 4, 0.85,
        f"{UNAUDITED} Exceptions and partner-side handling remain. {RESEARCHER_LIMIT}"),
    finding("PhonePe", "DPDP Alignment",
        "", "PhonePe Privacy Policy", OFFICIAL_PHONEPE_POLICY, "Official provider privacy policy",
        "The policy was updated 15 May 2026 and identifies the IT Act/SPDI Rules and Aadhaar law; the accessed text does not provide evidence of operational conformity with the DPDP Act and Rules.",
        GAP, None, None,
        f"Unsupported as a compliance score. MeitY's 13 November 2025 commencement notification phases the Act's principal duties in 18 months; the Rules also phase key duties in 18 months. A policy's legal references do not establish implementation. Official schedule: {DPDP_ACT_COMMENCEMENT} ; {DPDP_RULES}. Human legal/compliance review required."),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-overwrite", action="store_true",
                        help="confirm replacement of an existing reviewed CSV; a timestamped backup is created")
    args = parser.parse_args()

    template = PROJECT_ROOT / "cpaf_outputs" / "evidence_register_template.csv"
    destination = PROJECT_ROOT / "cpaf_outputs" / "evidence_reviews" / "evidence_register_reviewed.csv"
    frame = read_evidence_csv(template)
    expected = {(app, dimension) for app in APPS for dimension in DIMENSIONS}
    actual = list(zip(frame["app"], frame["dimension"]))
    if len(actual) != 21 or set(actual) != expected or len(set(actual)) != len(actual):
        raise ValueError("The original template does not contain each expected provider/dimension exactly once")
    if len(FINDINGS) != 21 or {(r["app"], r["dimension"]) for r in FINDINGS} != expected:
        raise ValueError("The research findings do not cover exactly the 21 expected register rows")

    # The template's completely empty columns are inferred by pandas as float64;
    # use object for edited CSV fields so truthful text/blank values can be placed.
    for column in frame.columns:
        frame[column] = frame[column].astype(object)
    findings = {(r["app"], r["dimension"]): r for r in FINDINGS}
    for row_index, row in frame.iterrows():
        record = findings[(row["app"], row["dimension"])]
        for column, value in record.items():
            frame.at[row_index, column] = "" if value is None else value

    existed = destination.exists()
    saved = save_evidence_csv(frame, destination, confirm_overwrite=args.confirm_overwrite)
    print(f"Saved reviewed evidence: {saved}")
    print(f"Rows: {len(frame)}; scored: {sum(r['raw_score'] is not None for r in FINDINGS)}; unscored: {sum(r['raw_score'] is None for r in FINDINGS)}")
    print(f"Previous working review existed: {existed}; safe writer created a timestamped backup when replacing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
