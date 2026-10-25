# Evidence review workflow

The dashboard's Payment-app evidence scoring section defaults to
`cpaf_outputs/evidence_register_template.csv`. It can also load the saved review
or a user-selected CSV. CSV input must contain the evidence-register columns;
unknown additional columns are retained. Duplicate or missing required headers,
missing files, and malformed CSV input are reported instead of silently repaired.

Edits are session-local until **Save reviewed evidence** is used. The separate
working destination is
`cpaf_outputs/evidence_reviews/evidence_register_reviewed.csv`. The original
template and experiment output directories are not save targets. Replacing the
working review requires confirmation and creates a timestamped backup in the
same directory. Reload actions warn before discarding dirty editor contents.

Scoring eligibility is shared between the dashboard status diagnostics and
`compute_app_metrics`: a row must be marked `reviewed`, have a valid HTTP(S)
source, access date, claim, reviewer, rationale, and numeric score from 1 to 5.
The app does not set review status or fill missing values. Passing these field
checks does not prove the source claim is true; human source review remains
necessary. The seven provider dimensions are unchanged and separate from all
synthetic-data experiments.

Automated tests use temporary directories beneath the test folder and do not
write to the project evidence register or historical experiment outputs.
