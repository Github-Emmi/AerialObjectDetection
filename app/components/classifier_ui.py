"""Classification mode UI — upload an image, pick a model, get a prediction.

Supports two inference modes:
  • Remote (GPU): Sends image to Kaggle GPU via Ngrok tunnel (fast)
  • Local (CPU):  Runs PyTorch inference locally (fallback)
"""

from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CLASS_NAMES = ["Bird", "Drone"]

MODEL_OPTIONS = {
    "MobileNetV2 (recommended)": ("mobilenet_v2", "mobilenet_v2"),
    "EfficientNet-B0": ("efficientnet_b0", "efficientnet_b0"),
    "ResNet50": ("resnet50", "resnet50"),
    "Custom CNN": ("custom_cnn", "custom_cnn"),
}


# ── Local inference helpers (lazy-loaded) ───────────────────────
def _get_preprocess():
    """Lazy-load torchvision transforms only when needed for local inference."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@st.cache_resource
def load_classifier(backbone: str):
    """Load a classification model with cached weights (local inference)."""
    import torch
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


def _classify_remote(image, backbone):
    """Classify via remote Kaggle GPU API."""
    from api_client import remote_classify
    return remote_classify(image, backbone)


def _classify_local(image, backbone):
    """Classify via local PyTorch CPU inference."""
    import torch
    model = load_classifier(backbone)
    tensor = _get_preprocess()(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
    pred_idx = probs.argmax().item()
    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "backbone": backbone,
        "device": "cpu",
        "inference_ms": None,
    }


def render_classifier():
    st.header("🐦 Bird vs Drone — Classification")
    st.write("Upload an aerial image and select a classifier model.")

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
    # If no API URL configured at all, run silently in local mode

    col_upload, col_result = st.columns(2)

    with col_upload:
        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    col_upload.image(image, caption="Uploaded image", width="stretch")

    backbone, weight_key = MODEL_OPTIONS[model_label]

    with st.spinner("Classifying..."):
        if use_remote:
            result = _classify_remote(image, weight_key)
            if result is None:
                st.warning("Remote server failed — falling back to local inference")
                result = _classify_local(image, weight_key)
        else:
            result = _classify_local(image, weight_key)

    with col_result:
        st.subheader("Prediction")
        st.metric(label="Class", value=result["prediction"])
        st.metric(label="Confidence", value=f"{result['confidence']:.1%}")

        if result.get("inference_ms"):
            st.caption(f"⚡ {result['inference_ms']}ms on {result['device']}")

        st.markdown("**Class probabilities**")
        st.bar_chart(result["probabilities"])
