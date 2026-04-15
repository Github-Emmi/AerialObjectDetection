"""Detection mode UI — upload an image, run YOLOv8m, show bounding boxes.

Supports two inference modes:
  • Remote (GPU): Sends image to Kaggle GPU via Ngrok tunnel (fast)
  • Local (CPU):  Runs YOLOv8m inference locally (fallback)
"""

from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DETECTION_WEIGHTS = PROJECT_ROOT / "models" / "detection" / "best.pt"


@st.cache_resource
def load_detector():
    """Load YOLOv8m with cached weights (local inference)."""
    from ultralytics import YOLO
    model = YOLO(str(DETECTION_WEIGHTS))
    return model


def _detect_remote(image, conf_threshold):
    """Detect via remote Kaggle GPU API."""
    from api_client import remote_detect, decode_annotated_image
    resp = remote_detect(image, conf_threshold)
    if resp is None:
        return None
    # Convert API response to the format the UI expects
    annotated = decode_annotated_image(resp["annotated_image_b64"])
    return {
        "annotated": annotated,
        "detections": resp["detections"],
        "count": resp["count"],
        "device": resp.get("device", "cuda"),
        "inference_ms": resp.get("inference_ms"),
    }


def _detect_local(image, conf_threshold):
    """Detect via local YOLOv8m CPU inference."""
    model = load_detector()
    results = model.predict(image, conf=conf_threshold, verbose=False)
    result = results[0]

    annotated = Image.fromarray(result.plot()[..., ::-1])  # BGR → RGB

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class": result.names[cls_id],
            "confidence": round(conf, 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    return {
        "annotated": annotated,
        "detections": detections,
        "count": len(detections),
        "device": "cpu",
        "inference_ms": None,
    }


def render_detector():
    st.header("🎯 Bird vs Drone — Object Detection")
    st.write("Upload an aerial image to detect and localize birds and drones.")

    # Check remote API availability
    from api_client import get_api_url, is_api_available
    api_url = get_api_url()
    use_remote = False
    if api_url:
        use_remote = is_api_available()
        if use_remote:
            st.success("⚡ GPU Inference Server connected — fast mode enabled")
        else:
            st.warning("🔄 GPU server unreachable — using local CPU inference (slower)")

    col_upload, col_result = st.columns(2)

    with col_upload:
        conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    col_upload.image(image, caption="Uploaded image", width="stretch")

    with st.spinner("Detecting objects..."):
        if use_remote:
            result = _detect_remote(image, conf_threshold)
            if result is None:
                st.warning("Remote server failed — falling back to local inference")
                result = _detect_local(image, conf_threshold)
        else:
            result = _detect_local(image, conf_threshold)

    with col_result:
        st.subheader("Detections")
        st.image(result["annotated"], caption="Annotated result", width="stretch")

        if result.get("inference_ms"):
            st.caption(f"⚡ {result['inference_ms']}ms on {result['device']}")

        if result["count"] == 0:
            st.info("No objects detected at this confidence threshold.")
        else:
            rows = []
            for d in result["detections"]:
                rows.append({
                    "Class": d["class"],
                    "Confidence": f"{d['confidence']:.2%}",
                    "x1": d["bbox"][0], "y1": d["bbox"][1],
                    "x2": d["bbox"][2], "y2": d["bbox"][3],
                })
            st.dataframe(rows, width="stretch")
