import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


# =========================================================
# Streamlit configuration
# =========================================================
st.set_page_config(
    page_title="ML CSC Final Project",
    page_icon="🚦",
    layout="wide",
)

st.title("ML CSC Final Project")
st.write(
    "Upload a traffic sign image. ResNet34 classifies the sign, "
    "and YOLOv8 detects traffic signs with bounding boxes."
)


# =========================================================
# Google Drive model IDs
# These are the IDs from the links you provided.
# =========================================================
RESNET_FILE_ID = "1Y8Z5WslwgcNiDL_znPE6bet14ZmP92F_"
RESNET_FILENAME = "best_resnet_gtsrb.pth"

YOLO_FILE_ID = "1cLBmPcbHnJfc5gigszh6NfKlq1SjmhLf"
YOLO_FILENAME = "best.pt"

# SHA-256 copied from your PowerShell screenshot:
# (Get-FileHash "D:\ml final project\best.pt" -Algorithm SHA256).Hash
EXPECTED_YOLO_SHA256 = (
    "B12DD88AD523E1AC20B53855D8FEF048E228E23DEB5CBE020E969E5A60C21EDB"
)

MODEL_VERSION = "v5"

BASE_DIR = Path(__file__).resolve().parent
CNN_MODEL_PATH = BASE_DIR / RESNET_FILENAME
YOLO_MODEL_PATH = BASE_DIR / YOLO_FILENAME
DATA_DIR = BASE_DIR / "archive (3)"

IMAGE_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50",
    "Speed limit 60", "Speed limit 70", "Speed limit 80",
    "End of speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing",
    "No passing for vehicles over 3.5 tonnes",
    "Right-of-way at next intersection", "Priority road",
    "Yield", "Stop", "No vehicles",
    "Vehicles over 3.5 tonnes prohibited", "No entry",
    "General caution", "Dangerous curve left",
    "Dangerous curve right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing",
    "End of no passing for vehicles over 3.5 tonnes"
]


# =========================================================
# Utility functions
# =========================================================
def sha256_file(file_path: Path) -> str:
    """Calculate the SHA-256 hash without loading the whole file into RAM."""
    digest = hashlib.sha256()
    with file_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def download_drive_file(file_id: str, destination: Path, label: str) -> None:
    """Download one public Google Drive file to a known path."""
    import gdown

    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = Path(str(destination) + ".version")

    already_downloaded = (
        destination.is_file()
        and destination.stat().st_size > 100_000
        and marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == MODEL_VERSION
    )

    if already_downloaded:
        return

    # Remove stale files so a failed download cannot leave an old model
    # that looks like the newly requested version.
    if destination.exists():
        destination.unlink()
    if marker.exists():
        marker.unlink()

    with st.spinner(f"Downloading {label} model from Google Drive..."):
        downloaded_path = gdown.download(
            id=file_id,
            output=str(destination),
            quiet=False,
        )

    if (
        not downloaded_path
        or not destination.is_file()
        or destination.stat().st_size <= 100_000
    ):
        if destination.exists():
            destination.unlink()
        raise RuntimeError(
            f"Failed to download {label} weights. Check that the Google "
            "Drive file is shared as 'Anyone with the link — Viewer', and "
            "that the file ID is correct."
        )

    marker.write_text(MODEL_VERSION, encoding="utf-8")


# =========================================================
# Download and validate weights
# =========================================================
@st.cache_resource
def download_models():
    download_drive_file(
        RESNET_FILE_ID,
        CNN_MODEL_PATH,
        "ResNet34 CNN",
    )
    download_drive_file(
        YOLO_FILE_ID,
        YOLO_MODEL_PATH,
        "YOLOv8",
    )

    # Do not run a different checkpoint if the Drive download does not
    # exactly match the best.pt hash from the user's local computer.
    actual_yolo_hash = sha256_file(YOLO_MODEL_PATH)
    if actual_yolo_hash != EXPECTED_YOLO_SHA256.upper():
        raise RuntimeError(
            "The downloaded YOLO best.pt does NOT match your local trained "
            "best.pt. The app has stopped rather than silently running a "
            "different model.\n\n"
            f"Expected SHA-256: {EXPECTED_YOLO_SHA256}\n"
            f"Downloaded SHA-256: {actual_yolo_hash}\n\n"
            "Check the Google Drive file ID and confirm that the file at "
            "that ID is the exact best.pt from your local training folder."
        )

    return {
        "cnn_path": str(CNN_MODEL_PATH),
        "yolo_path": str(YOLO_MODEL_PATH),
        "yolo_sha256": actual_yolo_hash,
    }


# =========================================================
# Load models
# =========================================================
@st.cache_resource
def load_cnn():
    if not CNN_MODEL_PATH.is_file():
        raise FileNotFoundError(f"CNN weights not found: {CNN_MODEL_PATH}")

    model = models.resnet34(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES)),
    )

    state = torch.load(
        CNN_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource
def load_yolo():
    if not YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_MODEL_PATH}")

    model = YOLO(str(YOLO_MODEL_PATH))
    return model


# =========================================================
# Ground-truth lookup
# =========================================================
@st.cache_data
def load_ground_truth_lookup():
    lookup = {}

    for csv_name in ["Test.csv", "Train.csv"]:
        csv_path = DATA_DIR / csv_name
        if not csv_path.is_file():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if "Path" not in df.columns or "ClassId" not in df.columns:
            continue

        for _, row in df.iterrows():
            filename = Path(str(row["Path"])).name.lower()
            try:
                lookup[filename] = int(row["ClassId"])
            except (TypeError, ValueError):
                continue

    return lookup


# =========================================================
# Download and load models with clear errors
# =========================================================
try:
    model_info = download_models()
except Exception as exc:
    st.error("Could not download or verify the model weights.")
    st.exception(exc)
    st.stop()

st.sidebar.header("Model verification")
st.sidebar.success("Downloaded YOLO checkpoint matches the local SHA-256.")
st.sidebar.write("YOLO file:", YOLO_FILENAME)
st.sidebar.code(model_info["yolo_sha256"])
st.sidebar.write("YOLO weights size:", f"{YOLO_MODEL_PATH.stat().st_size:,} bytes")

try:
    cnn_model = load_cnn()
except Exception as exc:
    st.error("Could not load the ResNet34 CNN model.")
    st.exception(exc)
    cnn_model = None

try:
    yolo_model = load_yolo()
except Exception as exc:
    st.error("Could not load the YOLOv8 model.")
    st.exception(exc)
    yolo_model = None

if yolo_model is not None:
    st.sidebar.write("YOLO checkpoint class names:")
    st.sidebar.json(yolo_model.names)


# =========================================================
# ResNet preprocessing
# =========================================================
cnn_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# =========================================================
# Image upload
# =========================================================
uploaded_file = st.file_uploader(
    "Choose a traffic sign image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        ).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be opened as an image.")
        st.stop()

    st.sidebar.write("Image dimensions:", image.size)
    st.sidebar.write("Image mode:", image.mode)
    st.sidebar.write("Uploaded bytes:", f"{uploaded_file.size:,}")

    st.subheader("Uploaded Image")
    st.image(image, caption=uploaded_file.name, use_container_width=True)

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------
    ground_truth = load_ground_truth_lookup()
    file_key = uploaded_file.name.lower()
    real_id = None
    real_answer = "Not specified"

    if file_key in ground_truth:
        real_id = ground_truth[file_key]
        if 0 <= real_id < len(CLASS_NAMES):
            real_answer = f"{CLASS_NAMES[real_id]} (Class ID: {real_id})"
    else:
        selected_real = st.selectbox(
            "Real answer (optional; used only for comparison)",
            ["Not specified"] + [
                f"{i}: {name}" for i, name in enumerate(CLASS_NAMES)
            ],
        )
        if selected_real != "Not specified":
            real_id = int(selected_real.split(":")[0])
            real_answer = f"{CLASS_NAMES[real_id]} (Class ID: {real_id})"

    st.write(f"**Real answer:** {real_answer}")
    st.divider()

    # -----------------------------------------------------
    # ResNet34 classification
    # -----------------------------------------------------
    st.header("CNN Classification — ResNet34")

    if cnn_model is None:
        st.error("CNN model is unavailable. See the error above.")
    else:
        try:
            tensor = cnn_transform(image).unsqueeze(0)

            with torch.inference_mode():
                logits = cnn_model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
                top_scores, top_indices = torch.topk(
                    probabilities,
                    k=min(5, len(CLASS_NAMES)),
                )

            predicted_id = int(top_indices[0].item())
            predicted_name = CLASS_NAMES[predicted_id]
            confidence = float(top_scores[0].item()) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted class", predicted_name)
            c2.metric("Class ID", predicted_id)
            c3.metric("Confidence", f"{confidence:.2f}%")

            if real_id is not None:
                st.write(
                    "**Comparison with real answer:** "
                    + ("Correct" if predicted_id == real_id else "Incorrect")
                )

            st.write("**Top 5 CNN predictions**")
            top_df = pd.DataFrame({
                "Traffic sign": [
                    CLASS_NAMES[int(i)] for i in top_indices
                ],
                "Class ID": [int(i) for i in top_indices],
                "Probability (%)": [
                    round(float(score) * 100, 2)
                    for score in top_scores
                ],
            })
            st.dataframe(
                top_df,
                hide_index=True,
                use_container_width=True,
            )

        except Exception as exc:
            st.error("CNN prediction failed.")
            st.exception(exc)

    # -----------------------------------------------------
    # YOLOv8 object detection
    # -----------------------------------------------------
    st.divider()
    st.header("Object Detection — YOLOv8")

    threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.01,
        max_value=0.90,
        value=0.25,
        step=0.01,
    )

    if yolo_model is None:
        st.error("YOLO model is unavailable. See the error above.")
    else:
        try:
            with st.spinner("Running YOLOv8 detection..."):
                # Ultralytics accepts an RGB PIL image directly.
                result = yolo_model.predict(
                    source=image,
                    conf=threshold,
                    iou=0.45,
                    imgsz=640,
                    device="cpu",
                    verbose=False,
                )[0]

            boxes = result.boxes
            count = 0 if boxes is None else len(boxes)

            # plot() returns a BGR NumPy array.
            annotated_bgr = result.plot()
            annotated_rgb = annotated_bgr[:, :, ::-1].copy()

            image_col, results_col = st.columns([1.3, 1])

            with image_col:
                st.image(
                    annotated_rgb,
                    caption="YOLOv8 detections",
                    use_container_width=True,
                )

            with results_col:
                st.metric("Objects detected", count)

                if count == 0:
                    st.info(
                        "No objects detected at this confidence threshold. "
                        "Try a lower threshold or a clearer image."
                    )
                else:
                    names = result.names
                    rows = []

                    for box in boxes:
                        class_id = int(box.cls.item())
                        confidence_value = float(box.conf.item()) * 100

                        if isinstance(names, dict):
                            class_name = names.get(class_id, str(class_id))
                        else:
                            class_name = (
                                names[class_id]
                                if 0 <= class_id < len(names)
                                else str(class_id)
                            )

                        xyxy = box.xyxy[0].cpu().tolist()
                        rows.append({
                            "Detected class": class_name,
                            "Class ID": class_id,
                            "Confidence (%)": round(confidence_value, 2),
                            "X1": round(xyxy[0], 1),
                            "Y1": round(xyxy[1], 1),
                            "X2": round(xyxy[2], 1),
                            "Y2": round(xyxy[3], 1),
                        })

                    st.dataframe(
                        pd.DataFrame(rows),
                        hide_index=True,
                        use_container_width=True,
                    )

        except Exception as exc:
            st.error("YOLOv8 prediction failed.")
            st.exception(exc)
