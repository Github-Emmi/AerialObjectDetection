"""Classification mode UI — upload an image, pick a model, get a prediction."""

from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CLASS_NAMES = ["Bird", "Drone"]

MODEL_OPTIONS = {
    "MobileNetV2 (recommended)": ("mobilenet_v2", "mobilenet_v2"),
    "EfficientNet-B0": ("efficientnet_b0", "efficientnet_b0"),
    "ResNet50": ("resnet50", "resnet50"),
    "Custom CNN": ("custom_cnn", "custom_cnn"),
}

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_classifier(backbone: str):
    """Load a classification model with cached weights."""
    if backbone == "custom_cnn":
        from src.models.custom_cnn import AerialCNN
        model = AerialCNN(num_classes=2)
    else:
        from src.models.transfer_learning import create_transfer_model
        model = create_transfer_model(backbone=backbone, num_classes=2, freeze_ratio=0.0)

    weights_dir = PROJECT_ROOT / "models" / "classification" / backbone
    weights_path = weights_dir / "best_model.pth"
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def render_classifier():
    st.header("🐦 Bird vs Drone — Classification")
    st.write("Upload an aerial image and select a classifier model.")

    col_upload, col_result = st.columns(2)

    with col_upload:
        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    col_upload.image(image, caption="Uploaded image", width="stretch")

    backbone, weight_key = MODEL_OPTIONS[model_label]
    model = load_classifier(weight_key)

    tensor = PREPROCESS(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    pred_idx = probs.argmax().item()
    pred_label = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item()

    with col_result:
        st.subheader("Prediction")
        st.metric(label="Class", value=pred_label)
        st.metric(label="Confidence", value=f"{confidence:.1%}")

        st.markdown("**Class probabilities**")
        chart_data = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(chart_data)
