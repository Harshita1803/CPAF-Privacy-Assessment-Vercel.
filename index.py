from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from cpaf.core import generate_synthetic, generalize, suppress_k, privacy_diagnostics, validate_dataset

app = FastAPI(title="CPAF 2.0 API", version="1.0.0")

class RunRequest(BaseModel):
    n: int = Field(default=5000, ge=100, le=50000)
    seed: int = Field(default=42, ge=0, le=1000000)
    k: int = Field(default=5, ge=1, le=1000)
    age_level: int = Field(default=1, ge=0, le=2)
    amount_level: int = Field(default=1, ge=0, le=2)
    location_level: int = Field(default=1, ge=0, le=2)
    time_level: int = Field(default=1, ge=0, le=2)
    l_threshold: int = Field(default=2, ge=1, le=20)
    t_threshold: float = Field(default=0.2, ge=0, le=1)
    operation: Literal["raw", "generalize", "suppress"] = "suppress"

@app.get("/api/health")
def health():
    return {"status":"ok", "project":"CPAF 2.0", "data_policy":"synthetic-only"}

@app.post("/api/experiment")
def experiment(req: RunRequest):
    try:
        raw = generate_synthetic(req.n, req.seed)
        quality = validate_dataset(raw)
        generalized = generalize(raw, req.age_level, req.amount_level,
                                 req.location_level, req.time_level)
        quasi = ["age_band", "location", "merchant_type", "amount_band", "time_band"]
        if req.operation == "raw":
            result = raw.copy()
            transform_diag = {"operation":"raw", "rows_before":len(raw), "rows_after":len(raw)}
        elif req.operation == "generalize":
            result = generalized
            transform_diag = {"operation":"generalize", "rows_before":len(raw), "rows_after":len(generalized)}
        else:
            result, transform_diag = suppress_k(generalized, quasi, req.k)
            transform_diag = dict(transform_diag)
            transform_diag["operation"] = "generalize + k-suppression"
            transform_diag["rows_before"] = len(raw)
            transform_diag["rows_after"] = len(result)
        diagnostics = privacy_diagnostics(result, quasi, categorical_t=req.t_threshold,
                                          l_threshold=req.l_threshold) if len(result) else {}
        preview = result.head(100).copy()
        preview = preview.astype(object).where(pd.notna(preview), None)
        return {
            "synthetic_only": True,
            "notice": "Artificial data only. These results do not measure Paytm, Google Pay, or PhonePe practices and are not a formal differential-privacy guarantee.",
            "quality": quality,
            "transformation": transform_diag,
            "diagnostics": diagnostics,
            "preview_columns": list(preview.columns),
            "preview_rows": preview.to_dict(orient="records")
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

class EvidenceItem(BaseModel):
    app: str
    dimension: str
    raw_score: float = Field(ge=1, le=5)
    evidence_strength: float = Field(ge=0, le=1)
    source_url: str
    access_date: str
    claim: str
    reviewer: str
    score_rationale: str
    review_status: str = "reviewed"

class EvidenceRequest(BaseModel):
    items: list[EvidenceItem] = Field(min_length=1, max_length=500)

@app.post("/api/evidence-summary")
def evidence_summary(req: EvidenceRequest):
    dims = ["Data Minimisation", "Consent & Transparency", "Third-Party Sharing",
            "Security Controls", "Anonymisation & Re-identification",
            "Data Lifecycle & User Rights", "DPDP Alignment"]
    weights = {"Data Minimisation":1.2,"Consent & Transparency":1.3,
               "Third-Party Sharing":1.2,"Security Controls":1.2,
               "Anonymisation & Re-identification":1.0,
               "Data Lifecycle & User Rights":1.1,"DPDP Alignment":1.0}
    apps = ["Paytm","Google Pay","PhonePe"]
    accepted = []
    rejected = []
    for item in req.items:
        d = item.model_dump()
        from urllib.parse import urlparse
        u = urlparse(d["source_url"])
        missing = []
        if u.scheme not in ("http","https") or not u.netloc: missing.append("valid HTTP(S) source URL")
        for field in ("claim","access_date","reviewer","score_rationale"):
            if not d[field].strip(): missing.append(field)
        if d["review_status"].strip().lower() != "reviewed": missing.append("review_status=reviewed")
        if d["app"] not in apps: missing.append("app must be Paytm, Google Pay, or PhonePe")
        if d["dimension"] not in dims: missing.append("valid privacy dimension")
        if missing: rejected.append({"app":d["app"],"dimension":d["dimension"],"issues":missing})
        else: accepted.append(d)
    summaries=[]
    for appname in apps:
        scores=[]; covered=0
        for dim in dims:
            rows=[r for r in accepted if r["app"]==appname and r["dimension"]==dim]
            if not rows: continue
            sw=sum(r["evidence_strength"] for r in rows)
            val=sum((3+(r["raw_score"]-3)*r["evidence_strength"])*r["evidence_strength"] for r in rows)/sw if sw else 3.0
            scores.append((val,weights[dim])); covered+=1
        weighted=sum(s*w for s,w in scores)/sum(w for s,w in scores) if scores else None
        summaries.append({"app":appname,"weighted_score":round(weighted,3) if weighted is not None else None,
                          "dimensions_covered":covered,"coverage_pct":round(100*covered/len(dims),1),
                          "status":"evidence-based provisional score" if scores else "insufficient eligible evidence"})
    return {"summaries":summaries,"eligible_rows":len(accepted),"rejected_rows":rejected,
            "notice":"Reviewer-entered public evidence only. This is not an independent technical audit or a legal-compliance finding. Missing evidence is unknown, not proof of non-compliance."}
