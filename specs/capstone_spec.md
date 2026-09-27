# Capstone Spec -- Smelter Super Kings

## 1. Project Goal

Build an image-track capstone project that uses the inspection image and product type available at first inspection to classify each aluminium product as `acceptable`, `scratch`, `dent`, `stain`, or `other`.

The project goal is not only to achieve strong overall multiclass performance, but specifically to help the quality team identify `stain` reliably and consistently, separate it from other surface defects, and trigger faster investigation of upstream causes.

## 2. Business Context

- KPI: reduce surface-related rejection rate.
- Current business issue: overall surface rejection increased from `2.3%` to `3.6%`.
- Priority business signal: `stain` defects increased from `0.4%` to `1.4%`, which is the largest category shift in the evidence provided.
- Business user: quality team at first inspection.
- Intended action: identify likely stain cases faster and more consistently, separate stain from other visible surface issues, and escalate investigation of upstream causes earlier.

This capstone follows the **Image (Surface Defect)** track. The model is a first-inspection decision support tool, not a root-cause engine. Root-cause investigation remains a downstream business activity informed by better stain identification.

## 3. Scope and Constraints

- Follow the image track only.
- Use only inputs known at prediction time:
  - inspection image
  - product type
- Treat `stain` as the priority defect throughout model design, evaluation, testing, and reporting.
- Use `macro F1` as the primary overall model metric.
- Track `stain recall` and `stain F1` explicitly as priority metrics.
- All class data used in this capstone must be clearly marked `SYNTHETIC`.
- Do not touch `azureml/` or `azure-pipelines.yml` if they are present later. They are out of scope for this local spec and local build phase.

## 4. Classes and Their Meaning

- `acceptable`: product surface appears acceptable at first inspection and should not be flagged as a defect case.
- `scratch`: visible scratch-like surface marking that should be separated from stain and other defect types.
- `dent`: visible dent or indentation-type surface defect.
- `stain`: visible stain or discoloration-type defect. This is the priority class because it is the main driver in the rejection-rate increase described in the business evidence.
- `other`: visible surface defect that does not fit the above named categories but is still considered non-acceptable.

Class definitions should remain operational and inspection-oriented. The classes are intended for first-inspection categorization, not detailed metallurgy or root-cause labeling.

## 5. Synthetic Data Assumptions

- Data used in class is `SYNTHETIC`.
- Synthetic images must be labeled clearly as synthetic in the codebase and documentation.
- Synthetic data should simulate plausible visual differences among `acceptable`, `scratch`, `dent`, `stain`, and `other`.
- Synthetic data should include realistic overlap and ambiguity so the task is not trivially easy.
- Product type must be available as a structured input associated with each synthetic image.
- The synthetic dataset should support all five target classes and avoid extreme under-representation of `stain`.
- Synthetic generation should reflect the business priority by ensuring enough `stain` examples exist to measure `stain recall` and `stain F1` meaningfully.
- Synthetic data is a classroom proxy only; no claim should be made that it represents plant-validated production performance.

## 6. Baseline Model Approach

The project must include a simple non-deep-learning baseline so the deep learning model earns its complexity.

Baseline approach:

- Convert images to a simple fixed representation suitable for classical ML.
- Use a lightweight baseline classifier, such as logistic regression on resized grayscale pixel features.
- Include product type in a simple encoded form if supported by the pipeline design.
- Train and evaluate the baseline on the same synthetic task as the deep learning model.
- Record baseline performance in a comparison artifact.

The baseline is required for decision discipline. It is acceptable if the baseline performs worse, but the result must be reported honestly.

## 7. Deep Learning Model Approach

The primary model should be a transfer-learning image classifier suitable for the image track.

Deep learning approach:

- Use a CNN-based transfer-learning model, such as MobileNetV2 or an equivalent lightweight image backbone.
- Fine-tune the classifier head for the five Smelter classes.
- Support multiclass prediction across `acceptable`, `scratch`, `dent`, `stain`, and `other`.
- Incorporate product type either through metadata handling in the pipeline or through a documented simplification if the first implementation uses image-only inference.
- Prefer choices that improve minority and priority-class performance, especially for `stain`.
- Save the selected trained model artifact for local inference.

The deep learning model should be the main candidate for shipping if it demonstrates a better balance of overall macro F1 and priority-class performance than the baseline.

## 8. Evaluation Metrics

Primary overall metric:

- `macro F1`

Priority class metrics:

- `stain recall`
- `stain F1`

Supporting metrics:

- overall accuracy
- per-class precision, recall, and F1
- confusion matrix

Evaluation principles:

- Macro F1 is the main overall metric because the problem is multiclass and class balance may differ.
- `stain recall` matters because missing stain cases weakens the business objective.
- `stain F1` matters because stain performance should remain useful without excessive mislabeling.
- Confusion patterns between `stain` and other defect classes should be visible in the evaluation output.

## 9. Acceptance Criteria

The project is considered ready for local review only if all items below are satisfied:

- `specs/problem_brief.md` exists and matches the agreed Smelter image-track framing.
- `specs/capstone_spec.md` exists and defines the image-track solution, metrics, tests, and artifacts.
- Data is clearly marked `SYNTHETIC` in the repository and user-facing documentation.
- A baseline model is trained and evaluated.
- A deep learning model is trained and evaluated on the same task.
- A comparison artifact states baseline performance, deep learning performance, and which model should be shipped locally.
- Evaluation output includes `macro F1`, `stain recall`, and `stain F1`.
- The chosen model meets the quality gate:
  - macro F1 is reported and passes the agreed threshold for the implementation phase
  - stain recall is reported explicitly
  - stain F1 is reported explicitly
- A confusion matrix is produced for review.
- Local inference works on at least one sample image.
- The local app runs and presents predictions without raw tracebacks on common bad inputs.
- Automated tests pass locally.
- No edits are made to `azureml/` or `azure-pipelines.yml` during the local implementation covered by this spec.

Note: the exact numeric thresholds should be locked during implementation, but the reporting of `macro F1`, `stain recall`, and `stain F1` is mandatory in all cases.

## 10. Local Test Requirements

Local test requirements must cover data, inference, and user-facing behavior.

Required local tests:

- data test:
  - verify the synthetic image dataset exists
  - verify expected classes are present
  - verify files are readable image files
- evaluation test:
  - verify metric output includes `macro F1`, `stain recall`, and `stain F1`
- inference test:
  - verify the selected local model loads successfully
  - verify one sample image can be scored end-to-end
- app behavior test:
  - verify the app handles invalid file types gracefully
  - verify the app handles corrupted or empty image inputs gracefully
  - verify the app does not expose raw Python tracebacks to the user

`pytest` should be runnable from the project root and should pass before the project is considered ready for review.

## 11. App Requirements

The local app should be a simple first-inspection support tool for the quality team.

Required app behavior:

- built for local use in Streamlit
- title clearly tied to the Smelter defect-inspection use case
- allow image upload for prediction
- display the uploaded image back to the user
- show predicted class and confidence
- show a simple visualization of class scores or probabilities
- communicate uncertainty clearly when confidence is low
- provide user-friendly validation and error messages for unsupported or broken files
- operate in local mode using the locally saved model artifact
- clearly indicate that the data and demo context are `SYNTHETIC`

The app should support the business goal of distinguishing `stain` reliably, not just maximizing a generic demo experience.

## 12. Output Artifacts

The implementation driven by this spec should produce the following artifacts:

- `specs/problem_brief.md`
- `specs/capstone_spec.md`
- synthetic image data under `data/images/`
- baseline and deep learning training/evaluation code under `src/`
- selected local model artifact under `models/`
- evaluation metrics artifact, including macro F1 and stain metrics
- comparison artifact showing baseline vs deep learning decision
- local inference utility for sample requests
- local Streamlit app under `app/`
- automated tests under `tests/`
- `specs/validation_report.md`
- `README.md`
- `requirements.txt`
- `.gitignore`

## 13. Definition of Done for Spec Review

This spec is review-ready when it is clear that:

- the project is on the image track
- `stain` is the priority defect
- `macro F1` is the main overall metric
- `stain recall` and `stain F1` are explicitly required
- `SYNTHETIC` data labeling is mandatory
- Azure deployment files remain untouched in this phase
- the implementation path from data to model to app to local validation is fully defined
