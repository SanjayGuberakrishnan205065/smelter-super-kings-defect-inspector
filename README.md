# Smelter Super Kings Image-Track Capstone

This repository contains the local image-track capstone for the Smelter Super Kings team.

The project classifies aluminium inspection images into five classes:

- `acceptable`
- `scratch`
- `dent`
- `stain`
- `other`

## Scope

- Task: multiclass surface-defect image classification
- Priority defect: `stain`
- Main overall metric: `macro F1`
- Priority metrics: `stain recall`, `stain F1`
- Data used in class: `SYNTHETIC`
- Track: `Image (Surface Defect)`

## Important Note

Everything in `data/images/`, along with the local model artifacts trained from it, is **SYNTHETIC classroom data**. Nothing in this repo should be described as plant-validated production performance.

## Repo Layout

- `specs/`: problem brief, capstone spec, and validation report
- `src/`: data prep, baseline, training, evaluation, inference, and scoring
- `app/`: local Streamlit review app
- `tests/`: local readiness and validation tests
- `data/images/`: synthetic image data
- `models/`: trained local model artifacts
- `metrics/`: evaluation outputs
- `split/`: generated train/test metadata splits

## Local Setup

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Generate the SYNTHETIC Dataset

```powershell
python src/make_synthetic_images.py --out data/images --per_class 40 --seed 42
python src/prep.py --data_dir data/images --min_images_per_class 40 --train_out split/train_metadata.csv --test_out split/test_metadata.csv --test_ratio 0.2 --seed 42
```

What this creates:

- synthetic image files under `data/images/`
- `data/images/metadata.csv`
- `data/images/SYNTHETIC_DATASET_NOTICE.txt`
- `split/train_metadata.csv`
- `split/test_metadata.csv`

## Train the Models

Baseline model:

```powershell
python src/baseline.py --data_dir data/images --train_metadata split/train_metadata.csv --model_dir models --image_width 32 --image_height 32 --seed 42
```

Main tuned image model:

```powershell
python src/train.py --data_dir data/images --train_metadata split/train_metadata.csv --model_dir models --metrics_dir metrics --epochs 6 --batch_size 16 --learning_rate 0.001 --seed 42 --pretrained auto --freeze_features --priority_loss_boost 2.5 --priority_sample_boost 2.0 --selection_macro_weight 1.0 --selection_priority_f1_weight 0.75 --selection_priority_recall_weight 0.5
```

## Evaluate and Compare Models

```powershell
python src/evaluate.py --data_dir data/images --test_metadata split/test_metadata.csv --model_dir models --metrics_dir metrics --model_name all --min_macro_f1 0.60 --min_stain_recall 0.50 --min_stain_f1 0.50
```

Key outputs:

- `metrics/baseline_metrics.json`
- `metrics/main_metrics.json`
- `metrics/comparison.json`
- `models/selected_model.json`

## Run Local Inference

Create a request JSON from a generated sample:

```powershell
python src/make_request.py --data_dir data/images --sample_label stain --sample_index 0 --out requests/stain_request.json
```

Score the request:

```powershell
python src/score.py --request_json requests/stain_request.json --model_dir models --model_name auto
```

Run direct single-image prediction:

```powershell
python src/predict.py --image_path data/images/stain/stain_000.png --product_type coil --model_dir models --model_name auto
```

## Launch the Demo App

```powershell
$env:MODE="local"
python -m streamlit run app/app.py
```

The app:

- loads the selected local artifact from `models/`
- accepts image upload
- previews the uploaded image
- returns predicted label, confidence, and per-class scores
- warns on low-confidence predictions
- clearly states the capstone data/model are `SYNTHETIC`

## Run Tests

Run the local readiness suite after data generation, training, and evaluation:

```powershell
python -m pytest
```

The tests currently check:

- dataset structure and `SYNTHETIC` labeling
- split artifacts
- model artifacts
- evaluation artifacts
- selected-model summary
- one successful local inference call
- one structured error response for invalid inference input

## Notes

- Keep all demo and training data clearly marked `SYNTHETIC`.
- Do not modify `azureml/` or `azure-pipelines.yml` during this local capstone phase if they appear later.
