"""Detection mode UI — upload an image, run YOLOv8m, show bounding boxes."""

from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DETECTION_WEIGHTS = PROJECT_ROOT / "models" / "detection" / "best.pt"


@st.cache_resource
def load_detector():
    """Load YOLOv8m with cached weights."""
    from ultralytics import YOLO
    model = YOLO(str(DETECTION_WEIGHTS))
    return model


def render_detector():
    st.header("🎯 Bird vs Drone — Object Detection")
    st.write("Upload an aerial image to detect and localize birds and drones.")

    col_upload, col_result = st.columns(2)

    with col_upload:
        conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    col_upload.image(image, caption="Uploaded image", width="stretch")

    model = load_detector()
    results = model.predict(image, conf=conf_threshold, verbose=False)
    result = results[0]

    annotated = Image.fromarray(result.plot()[..., ::-1])  # BGR → RGB

    with col_result:
        st.subheader("Detections")
        st.image(annotated, caption="Annotated result", width="stretch")

        boxes = result.boxes
        if len(boxes) == 0:
            st.info("No objects detected at this confidence threshold.")
        else:
            rows = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = result.names[cls_id]
                rows.append({
                    "Class": label,
                    "Confidence": f"{conf:.2%}",
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                })
            st.dataframe(rows, width="stretch")
