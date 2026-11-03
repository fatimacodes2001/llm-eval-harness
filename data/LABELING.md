# How the labels are made

The file in `data/golden/` is **synthetic**. It exists so the harness runs end to end
without a model or an API key. Every row says so in `annotation.source`. The protocol
below is what a real golden set for this task follows; the schema already has the
fields to record it.

## Sampling

Stratify over the things that make the task harder, not over what is convenient:
detection count, presence of borderline attribute values (an overhang of 0.19 vs 0.21),
imagery date, region. Record the stratum on the row (`difficulty` here; add more tags
if you need them).

## Annotation

- Three annotators per property, working independently from the same detections plus
  the source imagery. They never see model output.
- Disagreements go to a fourth, senior annotator for adjudication. The adjudicated
  value is stored; `annotation.agreement` keeps the pre-adjudication agreement so
  low-agreement rows can be sliced out or weighted down later.
- `annotation.source` is `human` for unanimous rows and `human_adjudicated` for rows
  that needed a tie-break.
- Every hazard flag must cite the detection ids that justify it. If an annotator cannot
  point at a detection, the flag is not recorded. That rule is what makes the
  hallucination metric well defined.

## Splits

`dev` rows feed few-shot examples and prompt iteration. `test` rows are only ever read
by the runner. Nothing from `test` is pasted into a prompt, a fixture, or a bug report
that someone might later copy into a prompt.

## Versioning

The golden file is immutable once a run has referenced it. Fixes go into a new file
(`hazards_v2.jsonl`) and the run manifest records which one was used.
