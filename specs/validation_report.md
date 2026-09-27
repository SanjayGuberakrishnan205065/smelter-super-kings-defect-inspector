# Validation Report -- Smelter Super Kings

This report records the local-capstone acceptance checks for the image-track project.

## Synthetic Data Declaration

- Result: `PASS` / `FAIL`
- Evidence:
  - `data/images/SYNTHETIC_DATASET_NOTICE.txt`
  - `data/images/metadata.csv`
  - `README.md`

## Acceptance Criteria Summary

| Acceptance criterion | Result | Evidence |
|---|---|---|
| Problem brief exists and reflects the Smelter image-track framing | PASS / FAIL | `specs/problem_brief.md` |
| Capstone spec exists and defines the image-track solution | PASS / FAIL | `specs/capstone_spec.md` |
| Validation report completed | PASS / FAIL | `specs/validation_report.md` |
| Data is clearly marked `SYNTHETIC` | PASS / FAIL | notice file, metadata, README |
| Generated dataset is structurally valid | PASS / FAIL | `python -m pytest tests/test_data.py` |
| Train/test metadata splits exist | PASS / FAIL | `split/train_metadata.csv`, `split/test_metadata.csv` |
| Baseline model artifact exists | PASS / FAIL | `models/baseline_model.joblib` |
| Main model artifact exists | PASS / FAIL | `models/main_model.pt` |
| Baseline metrics saved | PASS / FAIL | `metrics/baseline_metrics.json` |
| Main metrics saved | PASS / FAIL | `metrics/main_metrics.json` |
| Comparison artifact saved | PASS / FAIL | `metrics/comparison.json`, `models/selected_model.json` |
| Macro F1 is reported for both models | PASS / FAIL | metrics JSON files |
| Stain recall is reported for both models | PASS / FAIL | metrics JSON files |
| Stain F1 is reported for both models | PASS / FAIL | metrics JSON files |
| Local inference works on a generated image | PASS / FAIL | `src/predict.py` / `src/score.py` output |
| Local app runs in `MODE=local` | PASS / FAIL | Streamlit app launch |
| App handles a bad upload without raw traceback | PASS / FAIL | manual UI check |
| Automated tests pass locally | PASS / FAIL | `python -m pytest` |

## Metrics Review

### Baseline

- Macro F1:
- Stain recall:
- Stain F1:
- Notes:

### Main

- Macro F1:
- Stain recall:
- Stain F1:
- Notes:

### Selection Decision

- Recommended model:
- Reason:
- Is the result acceptable for the business goal of reliable stain identification?

## Manual Checks

| Manual check | Result | Evidence |
|---|---|---|
| Upload a known stain image in the app | PASS / FAIL | screenshot / notes |
| Upload a wrong file type in the app | PASS / FAIL | screenshot / notes |
| Low-confidence warning appears when applicable | PASS / FAIL | screenshot / notes |
| The app clearly states the model/data are `SYNTHETIC` | PASS / FAIL | screenshot / notes |

## Open Issues / Follow-Up

- Issue 1:
- Issue 2:
- Issue 3:
