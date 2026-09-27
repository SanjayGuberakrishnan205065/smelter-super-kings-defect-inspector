"""Streamlit MVP for the Smelter Super Kings defect-classification demo."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
import sys

from PIL import Image, UnidentifiedImageError
import streamlit as st

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.score import init, run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
PRIORITY_CLASS = "stain"
LOW_CONFIDENCE_THRESHOLD = 0.60
SUPPORTED_FILE_TYPES = {"png", "jpg", "jpeg"}


@st.cache_resource
def load_local_runtime(model_dir: str) -> bool:
    """Initialize the local scoring runtime once per app session."""
    init(model_dir=model_dir, model_name="auto")
    return True


def set_page_config() -> None:
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Smelter Super Kings Defect Inspector",
        page_icon="",
        layout="centered",
    )


def render_header() -> None:
    """Render the app title and business framing."""
    st.title("Smelter Super Kings Defect Inspector")
    st.caption("Image-track capstone MVP for first-inspection surface-defect classification.")
    st.write(
        "This demo helps the quality team classify each uploaded aluminium surface image as "
        "`acceptable`, `scratch`, `dent`, `stain`, or `other`, with extra attention on identifying "
        f"`{PRIORITY_CLASS}` reliably."
    )
    st.info(
        "Capstone note: both the dataset and the trained model artifacts are SYNTHETIC and are intended "
        "for classroom demonstration only."
    )


def render_model_status() -> None:
    """Show local-mode expectations and selected artifact behavior."""
    mode = os.getenv("MODE", "local").strip().lower()
    if mode != "local":
        st.warning(
            f"MODE is set to `{mode}`. This MVP currently supports only `MODE=local`, so local model "
            "artifacts from `models/` will be used."
        )
    else:
        st.caption("Running in `MODE=local` and loading the selected model artifact from `models/`.")


def validate_uploaded_file(uploaded_file) -> tuple[Image.Image | None, str | None]:
    """Validate and decode an uploaded image file."""
    if uploaded_file is None:
        return None, None

    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_FILE_TYPES:
        return None, (
            "Unsupported file type. Please upload a PNG or JPG/JPEG image so the app can run "
            "surface-defect classification."
        )

    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image, None
    except UnidentifiedImageError:
        return None, "The uploaded file does not appear to be a valid image. Please try a different file."
    except Exception:
        return None, "The upload could not be read cleanly. Please try another image file."


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image into base64 PNG bytes for local scoring."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_request_payload(image: Image.Image, product_type: str) -> dict:
    """Build a deployment-friendly local scoring request payload."""
    return {
        "inputs": [
            {
                "image_base64": image_to_base64(image),
                "product_type": product_type,
            }
        ]
    }


def render_probability_display(scores: dict[str, float]) -> None:
    """Render a simple probability panel without extra plotting dependencies."""
    st.subheader("Class Probabilities")
    for class_name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        label = class_name
        if class_name == PRIORITY_CLASS:
            label = f"{class_name} (priority)"
        st.write(f"**{label}**: {score:.1%}")
        st.progress(min(max(float(score), 0.0), 1.0))


def render_prediction(response: dict) -> None:
    """Render prediction results from the scoring response."""
    prediction = response["prediction"]
    label = prediction["label"]
    confidence = float(prediction["confidence"])
    scores = prediction["scores"]

    st.subheader("Prediction")
    if label == PRIORITY_CLASS:
        st.success(f"Predicted label: **{label}**")
    else:
        st.write(f"Predicted label: **{label}**")
    st.write(f"Confidence: **{confidence:.1%}**")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            "Low-confidence result. Treat this as a review signal rather than a final decision, especially "
            "if stain identification matters for follow-up investigation."
        )

    priority_score = float(scores.get(PRIORITY_CLASS, 0.0))
    st.caption(f"Priority class `{PRIORITY_CLASS}` score: {priority_score:.1%}")

    render_probability_display(scores)

    warnings = response.get("warnings") or []
    for warning in warnings:
        st.warning(warning)


def render_error(response: dict) -> None:
    """Render a user-friendly inference error."""
    error = response.get("error", {})
    message = error.get("message", "An unexpected scoring error occurred.")
    st.error(message)


def main() -> None:
    """Render the local Streamlit MVP."""
    set_page_config()
    render_header()
    render_model_status()

    try:
        load_local_runtime(str(MODEL_DIR))
    except Exception as exc:
        st.error(
            "The local model runtime could not be initialized. Make sure the selected model artifacts exist "
            f"under `models/`. Details: {exc}"
        )
        st.stop()

    with st.form("prediction_form"):
        uploaded_file = st.file_uploader(
            "Upload an inspection image",
            type=sorted(SUPPORTED_FILE_TYPES),
            help="PNG or JPG/JPEG images only.",
        )
        product_type = st.selectbox(
            "Product type",
            options=["coil", "sheet", "plate", "strip"],
            index=0,
            help="This is used as a known-at-prediction-time input for local scoring.",
        )
        submitted = st.form_submit_button("Run prediction")

    if not submitted:
        st.caption("Upload an image and click **Run prediction** to test the current local model.")
        st.stop()

    image, validation_error = validate_uploaded_file(uploaded_file)
    if validation_error:
        st.error(validation_error)
        st.stop()

    assert image is not None
    st.subheader("Image Preview")
    st.image(image, caption="Uploaded inspection image", use_container_width=True)

    payload = build_request_payload(image=image, product_type=product_type)
    response = run(payload, model_dir=str(MODEL_DIR), model_name="auto")

    if response.get("status") != "ok":
        render_error(response)
        st.stop()

    render_prediction(response)


if __name__ == "__main__":
    main()
