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


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------
RESNET_FILE_ID = "1Y8Z5WslwgcNiDL_znPE6bet14ZmP92F_"
RESNET_PATH = "best_resnet_gtsrb.pth"

YOLO_FILE_ID = "1cLBmPcbHnJfc5gigszh6NfKlq1SjmhLf"
YOLO_PATH = "best.pt"

# Increment this value whenever you replace either model in Drive.
MODEL_VERSION = "v3"

BASE_DIR = Path(__file__).resolve().parent
CNN_MODEL_PATH = BASE_DIR / RESNET_PATH
YOLO_MODEL_PATH = BASE_DIR / YOLO_PATH
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


# ---------------------------------------------------------
# Model downloads
# ---------------------------------------------------------
def fetch_model(file_id: str, path: Path, label: str) -> None:
    """Download model weights if missing or marked as an old version."""
    import gdown

    marker = Path(str(path) + ".version")
    valid_existing_file = path.is_file() and path.stat().st_size >= 100_000
    current_version = (
        marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == MODEL_VERSION
    )

    if valid_existing_file and current_version:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale/invalid weights and marker before downloading.
    if path.exists():
        path.unlink()
    if marker.exists():
        marker.unlink()

    with st.spinner(f"Downloading {label} model..."):
        downloaded_path = gdown.download(
            id=file_id,
            output=str(path),
            quiet=False,
        )

    if (
        not downloaded_path
        or not path.is_file()
        or path.stat().st_size < 100_000
    ):
        raise RuntimeError(
            f"Could not download valid {label} weights. "
            "Check the Google Drive sharing permissions and file ID."
        )

    marker.write_text(MODEL_VERSION, encoding="utf-8")


@st.cache_resource
def download_models():
    fetch_model(RESNET_FILE_ID, CNN_MODEL_PATH, "ResNet34")
    fetch_model(YOLO_FILE_ID, YOLO_MODEL_PATH, "YOLOv8")
    return True


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
@st.cache_resource
def load_cnn():
    if not CNN_MODEL_PATH.is_file() or CNN_MODEL_PATH.stat().st_size < 100_000:
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
    if not YOLO_MODEL_PATH.is_file() or YOLO_MODEL_PATH.stat().st_size < 100_000:
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_MODEL_PATH}")

    model = YOLO(str(YOLO_MODEL_PATH))
    return model


# ---------------------------------------------------------
# Ground-truth lookup
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Load weights with useful error messages
# ---------------------------------------------------------
try:
    download_models()
except Exception as exc:
    st.error("Could not download the model weights.")
    st.exception(exc)
    st.stop()

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


# ---------------------------------------------------------
# Image preprocessing for ResNet34
# ---------------------------------------------------------
cnn_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------------------------------------------------
# Upload image
# ---------------------------------------------------------
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

    st.sidebar.header("Deployment diagnostics")
    st.sidebar.write("Image dimensions:", image.size)
    st.sidebar.write("Image mode:", image.mode)
    st.sidebar.write("Upload size:", f"{uploaded_file.size:,} bytes")
    st.sidebar.write("PyTorch version:", torch.__version__)
    st.sidebar.write("CUDA available:", torch.cuda.is_available())

    st.subheader("Uploaded Image")
    st.image(image, caption=uploaded_file.name, use_container_width=True)

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------
    ground_truth = load_ground_truth_lookup()
    file_key = uploaded_file.name.lower()

    real_answer = "Not specified"
    real_id = None

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
                is_correct = predicted_id == real_id
                st.write(
                    "**Comparison with real answer:** "
                    + ("Correct" if is_correct else "Incorrect")
                )

            st.write("**Top 5 CNN predictions**")
            top_df = pd.DataFrame({
                "Traffic sign": [
                    CLASS_NAMES[int(i)] for i in top_indices
                ],
                "Class ID": [
                    int(i) for i in top_indices
                ],
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
    # YOLOv8 detection
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
                # Use the same RGB PIL image for every prediction.
                # Ultralytics accepts PIL images in RGB format.
                raw_result = yolo_model.predict(
                    source=image,
                    conf=0.001,
                    iou=0.45,
                    imgsz=640,
                    device="cpu",
                    verbose=False,
                )[0]

                result = yolo_model.predict(
                    source=image,
                    conf=threshold,
                    iou=0.45,
                    imgsz=640,
                    device="cpu",
                    verbose=False,
                )[0]

            raw_boxes = raw_result.boxes
            boxes = result.boxes

            raw_count = 0 if raw_boxes is None else len(raw_boxes)
            final_count = 0 if boxes is None else len(boxes)

            # Diagnostic information helps compare the public deployment
            # against the local environment.
            st.sidebar.subheader("YOLO diagnostics")
            st.sidebar.write("Weights file:", YOLO_MODEL_PATH.name)
            st.sidebar.write(
                "Weights size:",
                f"{YOLO_MODEL_PATH.stat().st_size:,} bytes",
            )
            st.sidebar.write("YOLO class names:", yolo_model.names)
            st.sidebar.write("Raw boxes at confidence 0.001:", raw_count)
            st.sidebar.write("Boxes at selected threshold:", final_count)

            if raw_boxes is not None and raw_count > 0:
                st.sidebar.write(
                    "Raw confidence scores:",
                    [
                        round(float(x), 5)
                        for x in raw_boxes.conf.cpu().tolist()
                    ],
                )
                st.sidebar.write(
                    "Raw class IDs:",
                    [
                        int(x)
                        for x in raw_boxes.cls.cpu().tolist()
                    ],
                )
            else:
                st.warning(
                    "YOLO returned no boxes even at confidence 0.001. "
                    "Check that the deployed best.pt is the same trained "
                    "checkpoint used locally, and that it was trained to "
                    "detect signs in full images."
                )

            # Ultralytics result.plot() returns a BGR NumPy image.
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
                st.metric("Objects detected", final_count)

                if final_count == 0:
                    st.info(
                        "No objects detected at this confidence threshold."
                    )
                else:
                    names = result.names
                    rows = []

                    for box in boxes:
                        class_id = int(box.cls.item())
                        confidence_value = float(box.conf.item()) * 100

                        if isinstance(names, dict):
                            class_name = names.get(
                                class_id, str(class_id)
                            )
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
