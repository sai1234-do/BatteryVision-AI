
# ============================================================
# BatteryVision AI — Production Backend
# ============================================================

import os
import io
import base64
import tempfile
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import models, transforms

from PIL import Image

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


# ============================================================
# Production paths
# ============================================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

MODELS_PATH = os.path.join(
    BASE_PATH,
    "model"
)

FINETUNED_MODEL_PATH = os.path.join(
    MODELS_PATH,
    "battery_resnet18_finetuned.pth"
)

REPORTS_PATH = os.path.join(
    BASE_PATH,
    "reports"
)

os.makedirs(REPORTS_PATH, exist_ok=True)


# ============================================================
# Production device
# ============================================================

device = torch.device("cpu")

print("BatteryVision AI starting...")
print("Device:", device)

# ============================================
# 03 — DATASET + FIXED SPLIT
# CELL 3 — Class Names
# ============================================

class_names = [
    "Tin_bad",
    "Tin_good",
    "Tin_medium",
    "Titanium_bad",
    "Titanium_good",
    "Zinc_Bad",
    "Zinc_Good"
]

NUM_CLASSES = len(class_names)

print("Number of classes:", NUM_CLASSES)
print("Classes:")

for index, class_name in enumerate(class_names):
    print(index, "->", class_name)

# ============================================
# 09 — GRAD-CAM
# CELL 1 — Imports + Target Layer
# ============================================

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

# Instantiate ResNet18 model structure
resnet = models.resnet18(weights=None)
num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

# Load the fine-tuned model checkpoint
if os.path.exists(FINETUNED_MODEL_PATH):
    checkpoint = torch.load(FINETUNED_MODEL_PATH, map_location=device, weights_only=False)
    # Extract the actual weights from 'model_state_dict'
    resnet.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded fine-tuned ResNet18 checkpoint successfully.")
else:
    print(f"Warning: Checkpoint not found at {FINETUNED_MODEL_PATH}")

resnet = resnet.to(device)
resnet.eval()

# Grad-CAM target layer
target_layer = resnet.layer4[-1]

print("Grad-CAM target layer:")
print(target_layer)

# ============================================
# 09 — GRAD-CAM
# CELL 2 — Grad-CAM Function
# ============================================

def generate_gradcam(
    model,
    image_tensor,
    target_layer,
    device
):

    model.eval()

    # Store activations and gradients
    activations = []
    gradients = []

    # Forward hook
    def forward_hook(module, input, output):
        activations.append(output)

    # Backward hook
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    # Prepare image
    image = image_tensor.unsqueeze(0).to(device)

    # Forward pass
    output = model(image)

    # Predicted class
    predicted_class = output.argmax(dim=1).item()

    # Score of predicted class
    score = output[0, predicted_class]

    # Clear gradients
    model.zero_grad()

    # Backpropagate
    score.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Get stored values
    activation = activations[0]
    gradient = gradients[0]

    # Average gradient over height and width
    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    # Weighted combination of feature maps
    cam = (
        weights * activation
    ).sum(
        dim=1,
        keepdim=True
    )

    # ReLU
    cam = F.relu(cam)

    # Resize heatmap to image size
    cam = F.interpolate(
        cam,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    # Remove unnecessary dimensions
    cam = (
        cam.squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    # Normalize between 0 and 1
    cam = cam - cam.min()

    if cam.max() > 0:
        cam = cam / cam.max()

    return cam, predicted_class

# ============================================
# 10 — INFERENCE PIPELINE
# CELL 1 — Configuration
# ============================================

INFERENCE_IMAGE_SIZE = 224

print("Inference configuration ready ✅")
print("Image size:", INFERENCE_IMAGE_SIZE)
print("Model:", "Fine-tuned ResNet18")
print("Classes:", class_names)

# ============================================
# CELL 2 — Image Preprocessing
# ============================================

def preprocess_image(image):
    """
    Convert a PIL image into the tensor format
    expected by the fine-tuned ResNet18.
    """

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INFERENCE_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image.convert("RGB"))

# ============================================
# CELL 3 — Prediction
# ============================================

def predict_image(image):

    model_input = preprocess_image(image)
    model_input = model_input.unsqueeze(0).to(device)

    with torch.inference_mode():

        output = resnet(model_input)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = probabilities.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    return {
        "class_name": class_names[predicted_class],
        "class_index": predicted_class,
        "confidence": confidence
    }

# ============================================
# CELL 5 — Complete Image Analysis
# ============================================

def analyze_image(
    image,
    generate_explanation=True
):

    prediction = predict_image(image)

    result = {
        "prediction": prediction["class_name"],
        "confidence": prediction["confidence"],
        "model": "Fine-tuned ResNet18"
    }

    if generate_explanation:

        image_tensor = preprocess_image(image)

        cam, predicted_class = generate_gradcam(
            model=resnet,
            image_tensor=image_tensor,
            target_layer=target_layer,
            device=device
        )

        result["gradcam"] = cam
        result["predicted_class_index"] = predicted_class

    else:

        result["gradcam"] = None

    return result

# ============================================
# 11 — REPORT GENERATOR
# CELL 1 — Configuration
# ============================================

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from datetime import datetime

os.makedirs(REPORTS_PATH, exist_ok=True)

print("Report generator configuration ready ✅")
print("Report directory:", REPORTS_PATH)

# ============================================
# CELL 2 — Report Generator
# ============================================

def generate_battery_report(
    image,
    result,
    filename=None
):

    document = Document()

    # ----------------------------------------
    # Title
    # ----------------------------------------

    title = document.add_heading(
        "BatteryVision AI",
        level=0
    )

    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = document.add_paragraph(
        "AI-Assisted Battery Surface Inspection Report"
    )

    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ----------------------------------------
    # Date
    # ----------------------------------------

    document.add_paragraph(
        f"Inspection Date: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    document.add_paragraph("")

    # ----------------------------------------
    # Inspection Summary
    # ----------------------------------------

    document.add_heading(
        "Inspection Summary",
        level=1
    )

    prediction = result["prediction"]
    confidence = result["confidence"]

    document.add_paragraph(
        f"Prediction: {prediction}"
    )

    document.add_paragraph(
        f"Confidence: {confidence * 100:.2f}%"
    )

    document.add_paragraph(
        "Model: Fine-tuned ResNet18"
    )

    # ----------------------------------------
    # Confidence Interpretation
    # ----------------------------------------

    if confidence >= 0.80:
        confidence_note = "High confidence"
    elif confidence >= 0.60:
        confidence_note = "Moderate confidence"
    else:
        confidence_note = "Low confidence"

    document.add_paragraph(
        f"Confidence Level: {confidence_note}"
    )

    # ----------------------------------------
    # Original Image
    # ----------------------------------------

    document.add_heading(
        "Inspected Image",
        level=1
    )

    image_stream = io.BytesIO()

    image.convert("RGB").save(
        image_stream,
        format="JPEG"
    )

    image_stream.seek(0)

    document.add_picture(
        image_stream,
        width=Inches(5.5)
    )

    # ----------------------------------------
    # Grad-CAM
    # ----------------------------------------

    if result.get("gradcam") is not None:

        document.add_heading(
            "Grad-CAM Explanation",
            level=1
        )

        cam = result["gradcam"]

        # Create visualization
        image_display = np.array(
            image.convert("RGB").resize(
                (224, 224)
            )
        ) / 255.0

        plt.figure(
            figsize=(6, 6)
        )

        plt.imshow(image_display)

        plt.imshow(
            cam,
            cmap="jet",
            alpha=0.45
        )

        plt.axis("off")

        # Save visualization temporarily
        cam_buffer = io.BytesIO()

        plt.savefig(
            cam_buffer,
            format="PNG",
            bbox_inches="tight"
        )

        plt.close()

        cam_buffer.seek(0)

        document.add_picture(
            cam_buffer,
            width=Inches(5.5)
        )

        document.add_paragraph(
            "The Grad-CAM visualization highlights "
            "image regions that contributed strongly "
            "to the model's prediction."
        )

    # ----------------------------------------
    # Processing Pipeline
    # ----------------------------------------

    document.add_heading(
        "Processing Pipeline",
        level=1
    )

    pipeline = [
        "Input battery image",
        "Image preprocessing",
        "Fine-tuned ResNet18 inference",
        "Prediction and confidence calculation",
        "Grad-CAM explanation"
    ]

    for step in pipeline:
        document.add_paragraph(
            step,
            style="List Bullet"
        )

    # ----------------------------------------
    # Model Information
    # ----------------------------------------

    document.add_heading(
        "Model Information",
        level=1
    )

    document.add_paragraph(
        "Architecture: ResNet18"
    )

    document.add_paragraph(
        "Training strategy: Transfer learning + fine-tuning"
    )

    document.add_paragraph(
        "Number of classes: 7"
    )

    # ----------------------------------------
    # Performance
    # ----------------------------------------

    document.add_heading(
        "Model Performance",
        level=1
    )

    table = document.add_table(
        rows=1,
        cols=4
    )

    table.style = "Table Grid"

    header = table.rows[0].cells

    header[0].text = "Model"
    header[1].text = "Accuracy"
    header[2].text = "Macro F1"
    header[3].text = "Weighted F1"

    performance_data = [
        (
            "Custom CNN",
            "77.42%",
            "71.88%",
            "79.66%"
        ),
        (
            "ResNet18 Frozen",
            "83.87%",
            "77.29%",
            "83.96%"
        ),
        (
            "ResNet18 Fine-tuned",
            "87.10%",
            "77.76%",
            "85.92%"
        )
    ]

    for row_data in performance_data:

        cells = table.add_row().cells

        for i, value in enumerate(row_data):
            cells[i].text = value

    # ----------------------------------------
    # Limitations
    # ----------------------------------------

    document.add_heading(
        "Limitations",
        level=1
    )

    document.add_paragraph(
        "The model is a closed-set classifier trained "
        "on seven battery surface classes. Predictions "
        "on images outside these classes may still "
        "produce high-confidence results."
    )

    document.add_paragraph(
        "Grad-CAM provides an interpretation of "
        "important activation regions but does not "
        "prove causal reasoning."
    )

    # ----------------------------------------
    # Conclusion
    # ----------------------------------------

    document.add_heading(
        "Conclusion",
        level=1
    )

    document.add_paragraph(
        "BatteryVision AI successfully analyzes battery "
        "surface images using a fine-tuned ResNet18 "
        "model and provides both classification and "
        "visual explanation through Grad-CAM."
    )

    # ----------------------------------------
    # Save
    # ----------------------------------------

    if filename is None:
        filename = "battery_inspection_report"

    filename = os.path.splitext(filename)[0]

    report_path = os.path.join(
        REPORTS_PATH,
        f"{filename}_report.docx"
    )

    document.save(report_path)

    return report_path

# ============================================
# 12 — FASTAPI BACKEND
# CELL 1 — Imports
# ============================================

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse

from PIL import Image

import io
import base64

print("FastAPI imports ready ✅")

# ============================================
# CELL 2 — FastAPI Application
# ============================================

app = FastAPI(
    title="BatteryVision AI API",
    description="AI-Assisted Battery Surface Inspection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

print("FastAPI application created ✅")

# ============================================
# CELL 3 — Health Endpoint
# ============================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "Fine-tuned ResNet18",
        "device": str(device),
        "num_classes": len(class_names)
    }

print("Health endpoint registered ✅")

# ============================================
# CELL 4 — Image Validation
# ============================================

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

def validate_image_file(file):

    if file.content_type not in ALLOWED_IMAGE_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPEG, PNG, or WEBP."
            )
        )

# ============================================
# CELL 5 — Grad-CAM Encoding
# ============================================

def gradcam_to_base64(
    image,
    cam
):

    image_display = np.array(
        image.convert("RGB").resize(
            (224, 224)
        )
    ) / 255.0

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(image_display)

    plt.imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    plt.axis("off")

    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format="PNG",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    buffer.seek(0)

    encoded = base64.b64encode(
        buffer.read()
    ).decode("utf-8")

    return encoded

# ============================================
# CELL 6 — Prediction Endpoint
# ============================================

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    explain: bool = True
):

    validate_image_file(file)

    contents = await file.read()

    try:

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    result = analyze_image(
        image,
        generate_explanation=explain
    )

    confidence = result["confidence"]

    if confidence >= 0.80:
        confidence_level = "High"
    elif confidence >= 0.60:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    response = {
        "filename": file.filename,
        "prediction": result["prediction"],
        "confidence": confidence,
        "confidence_percent": round(
            confidence * 100,
            2
        ),
        "confidence_level": confidence_level,
        "model": result["model"],
        "explanation_generated": (
            result["gradcam"] is not None
        )
    }

    if result["gradcam"] is not None:

        response["gradcam"] = gradcam_to_base64(
            image,
            result["gradcam"]
        )

    return response

# ============================================
# CELL 7 — Report Endpoint
# ============================================

@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...)
):

    validate_image_file(file)

    contents = await file.read()

    try:

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    result = analyze_image(
        image,
        generate_explanation=True
    )

    report_path = generate_battery_report(
        image,
        result,
        file.filename
    )

    return FileResponse(
        path=report_path,
        filename=os.path.basename(report_path),
        media_type=(
            "application/vnd.openxmlformats-"
            "officedocument.wordprocessingml.document"
        )
    )