"""API client for the remote Kaggle GPU inference server.

When INFERENCE_API_URL is set (env var or Streamlit secrets), inference
requests are forwarded to the Kaggle GPU via Ngrok tunnel.  When the
remote server is unreachable, the client reports unavailability so the
UI can fall back to local inference or show an appropriate message.
"""

import io
import os
import base64
from typing import Optional

import requests
import streamlit as st
from PIL import Image


def get_api_url() -> Optional[str]:
    """Return the remote API URL, or None if not configured."""
    # 1. Check environment variable
    url = os.environ.get("INFERENCE_API_URL", "").strip()
    if url:
        return url.rstrip("/")
    # 2. Check Streamlit secrets
    try:
        url = st.secrets.get("INFERENCE_API_URL", "").strip()
        if url:
            return url.rstrip("/")
    except Exception:
        pass
    return None


def is_api_available() -> bool:
    """Check if the remote inference server is reachable."""
    url = get_api_url()
    if not url:
        return False
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def remote_classify(image: Image.Image, backbone: str = "mobilenet_v2") -> Optional[dict]:
    """Send an image to the remote server for classification.

    Returns dict with keys: prediction, confidence, probabilities,
    backbone, device, inference_ms.  Returns None on failure.
    """
    url = get_api_url()
    if not url:
        return None
    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        r = requests.post(
            f"{url}/classify",
            files={"file": ("image.jpg", buf, "image/jpeg")},
            params={"backbone": backbone},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def remote_detect(image: Image.Image, confidence: float = 0.25) -> Optional[dict]:
    """Send an image to the remote server for detection.

    Returns dict with keys: detections, count, annotated_image_b64,
    confidence_threshold, device, inference_ms.  Returns None on failure.
    """
    url = get_api_url()
    if not url:
        return None
    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        r = requests.post(
            f"{url}/detect",
            files={"file": ("image.jpg", buf, "image/jpeg")},
            params={"confidence": confidence},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def decode_annotated_image(b64_string: str) -> Image.Image:
    """Decode a base64-encoded JPEG image from the API response."""
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes))
