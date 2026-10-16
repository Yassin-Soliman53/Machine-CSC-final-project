import io
from pathlib import Path
import os
import cv2
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

RESNET_FILE_ID = "1Y8Z5WslwgcNiDL_znPE6bet14ZmP92F_"
RESNET_PATH = "best_resnet_gtsrb.pth"

YOLO_FILE_ID = "1cLBmPcbHnJfc5gigszh6NfKlq1SjmhLf"
YOLO_PATH = "best.pt"

# غيّر الرقم ده كل ما تحدّث الموديلات في Drive عشان يتنزلوا من جديد
MODEL_VERSION = "v2"


def _fetch_model(file_id, path, label):
    import gdown
    marker = path + ".version"
    up_to_date = (
        os.path.exists(path)
        and os.path.getsize(path) >= 100000
        and os.path.exists(marker)
        and open(marker).read().strip() == MODEL_VERSION
    )
    if up_to_date:
        return

    if os.path.exists(path):
        os.remove(path)

    with st.spinner(f"Downloading {label} model..."):
        gdown.download(id=file_id, output=path, quiet=False)

    if not os.path.exists(path) or os.path.getsize(path) < 100000:
        raise RuntimeError(f"Failed to download {label} model from Google Drive.")

    with open(marker, "w") as f:
        f.write(MODEL_VERSION)


@st.cache_resource
def download_models():
    _fetch_model(RESNET_FILE_ID, RESNET_PATH, "ResNet")
    _fetch_model(YOLO_FILE_ID, YOLO_PATH, "YOLO")

download_models()
import hashlib
st.sidebar.write("YOLO:", os.path.getsize(YOLO_PATH), hashlib.md5(open(YOLO_PATH, "rb").read()).hexdigest())
import sys, importlib.metadata as m
st.sidebar.write(sys.version)
for p in ["ultralytics", "torch", "torchvision", "opencv-python-headless", "opencv-python", "numpy", "pillow"]:
    try:
        st.sidebar.write(p, m.version(p))
    except Exception:
        st.sidebar.write(p, "not installed")
st.sidebar.write(image.size, image.mode, uploaded_file.size)
BASE_DIR = Path(__file__).resolve().parent
CNN_MODEL_PATH = BASE_DIR / RESNET_PATH
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

st.set_page_config(
    page_title="ML CSC Final Project",
    layout="wide"
)

st.title("ML CSC Final Project")
st.write("Upload a traffic sign image. ResNet34 CNN classifies the image, and YOLOv8 detects traffic signs with bounding boxes.")

@st.cache_data
def load_ground_truth_lookup():
    lookup = {}
    for csv_name in ["Test.csv", "Train.csv"]:
        csv_path = DATA_DIR / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "Path" in df.columns and "ClassId" in df.columns:
                for _, row in df.iterrows():
                    filename = Path(str(row["Path"])).name.lower()
                    lookup[filename] = int(row["ClassId"])
    return lookup

@st.cache_resource
def load_cnn():
    if not CNN_MODEL_PATH.exists() or CNN_MODEL_PATH.stat().st_size < 100000:
        return None

    model = models.resnet34(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    )

    state = torch.load(
        CNN_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_resource
def load_yolo():
    weights_path = BASE_DIR / YOLO_PATH
    if weights_path.exists() and weights_path.is_file() and weights_path.stat().st_size > 100000:
        return YOLO(str(weights_path))
    return None

cnn_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

uploaded_file = st.file_uploader(
    "Choose a traffic sign image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be opened as an image.")
        st.stop()

    st.subheader("Uploaded Image")
    st.image(image, caption=uploaded_file.name, use_container_width=True)
    ground_truth = load_ground_truth_lookup()
    file_key = uploaded_file.name.lower()

    if file_key in ground_truth:
        real_id = ground_truth[file_key]
        real_answer = f"{CLASS_NAMES[real_id]} (Class ID: {real_id})"
    else:
        selected_real = st.selectbox(
            "Real answer (optional; used only for comparison)",
            ["Not specified"] + [
                f"{i}: {name}" for i, name in enumerate(CLASS_NAMES)
            ],
        )
        if selected_real == "Not specified":
            real_answer = "Not specified"
        else:
            real_id = int(selected_real.split(":")[0])
            real_answer = f"{CLASS_NAMES[real_id]} (Class ID: {real_id})"

    st.write(f"**Real answer:** {real_answer}")
    st.divider()
    st.header("CNN Classification — ResNet34")

    cnn_model = load_cnn()
    if cnn_model is None:
        st.error("CNN weights not found.")
    else:
        tensor = cnn_transform(image).unsqueeze(0)

        with torch.inference_mode():
            logits = cnn_model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_scores, top_indices = torch.topk(
                probabilities, k=min(5, len(CLASS_NAMES))
            )

        predicted_id = int(top_indices[0].item())
        predicted_name = CLASS_NAMES[predicted_id]
        confidence = float(top_scores[0].item()) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted class", predicted_name)
        c2.metric("Class ID", predicted_id)
        c3.metric("Confidence", f"{confidence:.2f}%")

        if real_answer != "Not specified":
            is_correct = real_answer.startswith(predicted_name + " (")
            st.write(f"**Comparison with real answer:** {'Correct' if is_correct else 'Incorrect'}")

        st.write("**Top 5 CNN predictions**")
        top_df = pd.DataFrame({
            "Traffic sign": [CLASS_NAMES[int(i)] for i in top_indices],
            "Class ID": [int(i) for i in top_indices],
            "Probability (%)": [round(float(s) * 100, 2) for s in top_scores],
        })
        st.dataframe(top_df, hide_index=True, use_container_width=True)

    st.header("Object Detection — YOLOv8")

    threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    yolo_model = load_yolo()
    if yolo_model is None:
        st.error("YOLO weights (best.pt) not found.")
    else:
        with st.spinner("Detecting objects..."):
            result = yolo_model.predict(
                source=image,
                conf=threshold,
                iou=0.45,
                imgsz=640,
                device="cpu",
                verbose=False,
            )[0]

        annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
        boxes = result.boxes

        image_col, results_col = st.columns([1.3, 1])
        with image_col:
            st.image(
                annotated,
                caption="YOLOv8 detections",
                use_container_width=True,
            )

        with results_col:
            count = 0 if boxes is None else len(boxes)
            st.metric("Objects detected", count)

            if count == 0:
                st.info("No objects detected at this confidence threshold.")
            else:
                rows = []
                for box in boxes:
                    class_id = int(box.cls.item())
                    confidence_value = float(box.conf.item()) * 100
                    names = result.names
                    if isinstance(names, dict):
                        class_name = names.get(class_id, str(class_id))
                    else:
                        class_name = names[class_id]

                    rows.append({
                        "Detected class": class_name,
                        "Class ID": class_id,
                        "Confidence (%)": round(confidence_value, 2),
                    })

                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True,
                    use_container_width=True,
                )
