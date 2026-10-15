import io
from pathlib import Path
import os
import gdown
import cv2
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

FILE_ID = "1mxUO_r_Lub4MRvAyFWZ_OGbd8-N0DHjJ"
MODEL_PATH = "best_resnet_gtsrb.pth"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("جاري تحميل نموذج ResNet34 الذكي من السحابة... برجاء الانتظار..."):
            url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

download_model()

BASE_DIR = Path(__file__).resolve().parent
CNN_MODEL_PATH = BASE_DIR / "best_resnet_gtsrb.pth"
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

st.title("ML CSC Final Project - Traffic Sign Recognition")
st.write("Upload a traffic sign image. ResNet34 CNN classifies the image, and YOLOv8 detects traffic signs with bounding boxes.")

@st.cache_resource
def load_cnn():
    if not CNN_MODEL_PATH.exists():
        return None
    model = models.resnet34(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    )
    state = torch.load(CNN_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def find_yolo_weights():
    candidates = [
        BASE_DIR / "yolov8n.pt",
        BASE_DIR / "best.pt",
        BASE_DIR / "yolo_runs" / "gtsrb" / "weights" / "best.pt",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None

@st.cache_resource
def load_yolo():
    weights_path = find_yolo_weights()
    if weights_path is None:
        return None
    return YOLO(str(weights_path))

cnn_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

uploaded_file = st.file_uploader("Choose a traffic sign image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be opened as an image.")
        st.stop()

    st.subheader("Uploaded Image")
    st.image(image, caption=uploaded_file.name, use_container_width=True)
    
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
            top_scores, top_indices = torch.topk(probabilities, k=min(5, len(CLASS_NAMES)))

        predicted_id = int(top_indices[0].item())
        predicted_name = CLASS_NAMES[predicted_id]
        confidence = float(top_scores[0].item()) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted class", predicted_name)
        c2.metric("Class ID", predicted_id)
        c3.metric("Confidence", f"{confidence:.2f}%")

        st.write("**Top 5 CNN predictions**")
        top_df = pd.DataFrame({
            "Traffic sign": [CLASS_NAMES[int(i)] for i in top_indices],
            "Class ID": [int(i) for i in top_indices],
            "Probability (%)": [round(float(s) * 100, 2) for s in top_scores],
        })
        st.dataframe(top_df, hide_index=True, use_container_width=True)

    st.header("Object Detection — YOLOv8")

    threshold = st.slider("Detection confidence threshold", 0.10, 0.90, 0.25, 0.05)
    yolo_model = load_yolo()
    
    if yolo_model is None:
        st.error("YOLO weights not found. Please ensure yolov8n.pt or best.pt is uploaded to GitHub.")
    else:
        with st.spinner("Detecting objects..."):
            result = yolo_model.predict(source=image, conf=threshold, device="cpu", verbose=False)[0]

        annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
        boxes = result.boxes

        image_col, results_col = st.columns([1.3, 1])
        with image_col:
            st.image(annotated, caption="YOLOv8 detections", use_container_width=True)

        with results_col:
            count = 0 if boxes is None else len(boxes)
            st.metric("Objects detected", count)
            if count > 0:
                rows = []
                for box in boxes:
                    class_id = int(box.cls.item())
                    confidence_value = float(box.conf.item()) * 100
                    names = result.names
                    class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                    rows.append({
                        "Detected class": class_name,
                        "Class ID": class_id,
                        "Confidence (%)": round(confidence_value, 2),
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
