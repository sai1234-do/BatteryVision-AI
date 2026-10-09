# ============================================================
# BatteryVision AI — Production Backend
# Memory-Optimized for Render Free (512 MB)
# ============================================================

import os
import io
import base64
import threading
from datetime import datetime

import numpy as np

from PIL import Image, ImageOps

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from docx import Document
from docx.shared import Inches
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
# Safety / Resource Limits
# ============================================================

# Maximum uploaded file size.
MAX_UPLOAD_SIZE = 8 * 1024 * 1024  # 8 MB

# Maximum decoded image dimensions.
# This protects the server from huge phone/camera images.
MAX_IMAGE_DIMENSION = 1600

# Pillow protection against extremely large images.
Image.MAX_IMAGE_PIXELS = 20_000_000


# ============================================================
# Production device
# ============================================================

device = "cpu"

print("BatteryVision AI starting...")
print("Device:", device)


# ============================================================
# Class Names
# ============================================================

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

for index, class_name in enumerate(class_names):
    print(index, "->", class_name)


# ============================================================
# Lazy Model State
# ============================================================

resnet = None
target_layer = None

# Prevent two simultaneous requests from creating
# multiple large PyTorch computation graphs.
inference_lock = threading.Lock()


# ============================================================
# Lazy Model Loader
# ============================================================

def load_model():

    global resnet
    global target_layer
    global device

    if resnet is not None:
        return

    print("Loading fine-tuned ResNet18...")

    import torch
    from torchvision import models

    # Reduce CPU thread overhead on the tiny Render instance.
    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Safe if PyTorch has already initialized
        # its inter-op thread pool.
        pass

    device = torch.device("cpu")

    # Build ResNet18 without pretrained weights.
    model = models.resnet18(weights=None)

    num_ftrs = model.fc.in_features

    model.fc = torch.nn.Linear(
        num_ftrs,
        NUM_CLASSES
    )

    if not os.path.exists(FINETUNED_MODEL_PATH):

        raise RuntimeError(
            f"Checkpoint not found at "
            f"{FINETUNED_MODEL_PATH}"
        )

    # Load only the checkpoint data required for inference.
    checkpoint = torch.load(
        FINETUNED_MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Release checkpoint reference immediately.
    del checkpoint

    model = model.to(device)

    model.eval()

    resnet = model

    # Grad-CAM target layer.
    target_layer = resnet.layer4[-1]

    print(
        "Loaded fine-tuned ResNet18 successfully."
    )


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="BatteryVision AI API",
    description="AI-Assisted Battery Surface Inspection API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

# This application does not use browser credentials/cookies.
# Therefore wildcard CORS is safe for this public demo API
# and prevents deployment-origin problems.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

print("FastAPI application created.")


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "Fine-tuned ResNet18",
        "device": str(device),
        "num_classes": NUM_CLASSES,
        "model_loaded": resnet is not None
    }


print("Health endpoint registered.")


# ============================================================
# Image Validation
# ============================================================

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


async def read_uploaded_image(file):

    validate_image_file(file)

    contents = await file.read(
        MAX_UPLOAD_SIZE + 1
    )

    if len(contents) > MAX_UPLOAD_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "Image is too large. "
                "Maximum allowed size is 8 MB."
            )
        )

    try:

        image = Image.open(
            io.BytesIO(contents)
        )

        # Force actual image decoding while
        # the uploaded buffer is still available.
        image.load()

        image = image.convert("RGB")

        # Reduce huge camera images before inference.
        if (
            image.width > MAX_IMAGE_DIMENSION
            or image.height > MAX_IMAGE_DIMENSION
        ):

            image.thumbnail(
                (
                    MAX_IMAGE_DIMENSION,
                    MAX_IMAGE_DIMENSION
                ),
                Image.Resampling.LANCZOS
            )

        return image

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    finally:

        # Release the raw upload buffer.
        del contents


# ============================================================
# Inference Preprocessing
# ============================================================

INFERENCE_IMAGE_SIZE = 224
IMAGE_SIZE = INFERENCE_IMAGE_SIZE

print("Inference configuration ready.")
print("Image size:", INFERENCE_IMAGE_SIZE)
print("Model:", "Fine-tuned ResNet18")
print("Classes:", class_names)


def preprocess_image(image):

    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(
            256,
            antialias=True
        ),
        transforms.CenterCrop(
            INFERENCE_IMAGE_SIZE
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    tensor = transform(
        image
    )

    return tensor


# ============================================================
# Grad-CAM
# ============================================================

def generate_gradcam(
    model,
    image_tensor,
    target_layer,
    device
):

    import torch
    import torch.nn.functional as F

    activations = []
    gradients = []

    def forward_hook(
        module,
        module_input,
        output
    ):

        # Detach immediately.
        # We only need the activation values,
        # not their computation graph.
        activations.append(
            output.detach()
        )

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        gradients.append(
            grad_output[0].detach()
        )

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        image_batch = image_tensor.unsqueeze(
            0
        ).to(device)

        # Gradients are required for Grad-CAM,
        # so this section intentionally does NOT use
        # inference_mode().
        output = model(
            image_batch
        )

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

        score = output[
            0,
            predicted_class
        ]

        model.zero_grad(
            set_to_none=True
        )

        score.backward()

        if not activations or not gradients:

            raise RuntimeError(
                "Grad-CAM hooks did not capture "
                "required tensors."
            )

        activation = activations[0]

        gradient = gradients[0]

        # Global average pooling of gradients.
        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activation
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=(
                INFERENCE_IMAGE_SIZE,
                INFERENCE_IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = (
            cam.squeeze()
            .cpu()
            .numpy()
        )

        cam -= cam.min()

        cam_max = cam.max()

        if cam_max > 0:

            cam /= cam_max

        return (
            cam,
            predicted_class,
            confidence
        )

    finally:

        forward_handle.remove()
        backward_handle.remove()

        # Release temporary tensors.
        activations.clear()
        gradients.clear()

        model.zero_grad(
            set_to_none=True
        )


# ============================================================
# Lightweight Heatmap Generation
# ============================================================

def create_gradcam_image(original_image, cam):

    import numpy as np

    original = original_image.convert("RGB")
    original = original.resize((IMAGE_SIZE, IMAGE_SIZE))

    cam_array = np.asarray(cam, dtype=np.float32)
    cam_array = np.squeeze(cam_array)

    if cam_array.ndim != 2:
        raise ValueError(
            "Grad-CAM heatmap must be a 2D array. "
            "Received shape: "
            f"{cam_array.shape}"
        )

    cam_array = np.clip(cam_array, 0, 1)

    heatmap = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8
    )

    heatmap[:, :, 0] = np.clip(
        255 * (2 * cam_array - 0.5),
        0,
        255
    ).astype(np.uint8)

    heatmap[:, :, 1] = np.clip(
        255 * (2 * cam_array),
        0,
        255
    ).astype(np.uint8)

    heatmap[:, :, 2] = np.clip(
        255 * (1 - 2 * cam_array),
        0,
        255
    ).astype(np.uint8)

    heatmap_image = Image.fromarray(
        heatmap,
        mode="RGB"
    )

    overlay = Image.blend(
        original,
        heatmap_image,
        alpha=0.45
    )

    return overlay


def create_gradcam_overlay(original_image, cam):

    overlay = create_gradcam_image(
        original_image,
        cam
    )

    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer.getvalue()

# ============================================================
# Complete Image Analysis
# ============================================================

def analyze_image(
    image,
    generate_explanation=True
):

    load_model()

    image_tensor = preprocess_image(
        image
    )

    with inference_lock:

        if generate_explanation:

            # ONE forward pass provides:
            # prediction + confidence + Grad-CAM.
            cam, predicted_class, confidence = (
                generate_gradcam(
                    model=resnet,
                    image_tensor=image_tensor,
                    target_layer=target_layer,
                    device=device
                )
            )

            gradcam_png = create_gradcam_overlay(
                image,
                cam
            )

        else:

            import torch

            model_input = image_tensor.unsqueeze(
                0
            ).to(device)

            with torch.inference_mode():

                output = resnet(
                    model_input
                )

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

            gradcam_png = None

    return {
        "prediction": class_names[
            predicted_class
        ],
        "confidence": confidence,
        "model": "Fine-tuned ResNet18",
        "gradcam_png": gradcam_png,
        "predicted_class_index": predicted_class
    }


# ============================================================
# Report Generator
# ============================================================

print("Report generator configuration ready.")
print("Report directory:", REPORTS_PATH)


def generate_battery_report(
    image,
    result,
    filename=None
):

    document = Document()

    # ========================================================
    # Title
    # ========================================================

    title = document.add_heading(
        "BatteryVision AI",
        level=0
    )

    title.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    subtitle = document.add_paragraph(
        "AI-Assisted Battery Surface Inspection Report"
    )

    subtitle.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    # ========================================================
    # Date
    # ========================================================

    document.add_paragraph(
        f"Inspection Date: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    document.add_paragraph("")

    # ========================================================
    # Inspection Summary
    # ========================================================

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

    if confidence >= 0.80:

        confidence_note = "High confidence"

    elif confidence >= 0.60:

        confidence_note = "Moderate confidence"

    else:

        confidence_note = "Low confidence"

    document.add_paragraph(
        f"Confidence Level: {confidence_note}"
    )

    # ========================================================
    # Original Image
    # ========================================================

    document.add_heading(
        "Inspected Image",
        level=1
    )

    image_stream = io.BytesIO()

    # Use a controlled-size JPEG for the report.
    report_image = image.convert(
        "RGB"
    ).copy()

    report_image.thumbnail(
        (
            1400,
            1400
        ),
        Image.Resampling.LANCZOS
    )

    report_image.save(
        image_stream,
        format="JPEG",
        quality=82,
        optimize=True
    )

    image_stream.seek(0)

    document.add_picture(
        image_stream,
        width=Inches(5.5)
    )

    # ========================================================
    # Grad-CAM
    # ========================================================

    if result.get("gradcam_png") is not None:

        document.add_heading(
            "Grad-CAM Explanation",
            level=1
        )

        cam_buffer = io.BytesIO(
            result["gradcam_png"]
        )

        document.add_picture(
            cam_buffer,
            width=Inches(5.5)
        )

        document.add_paragraph(
            "The Grad-CAM visualization highlights "
            "image regions that contributed strongly "
            "to the model's prediction."
        )

    # ========================================================
    # Processing Pipeline
    # ========================================================

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

    # ========================================================
    # Model Information
    # ========================================================

    document.add_heading(
        "Model Information",
        level=1
    )

    document.add_paragraph(
        "Architecture: ResNet18"
    )

    document.add_paragraph(
        "Training strategy: "
        "Transfer learning + fine-tuning"
    )

    document.add_paragraph(
        "Number of classes: 7"
    )

    # ========================================================
    # Performance
    # ========================================================

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

        for index, value in enumerate(
            row_data
        ):

            cells[index].text = value

    # ========================================================
    # Limitations
    # ========================================================

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

    # ========================================================
    # Conclusion
    # ========================================================

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

    # ========================================================
    # Save
    # ========================================================

    if filename is None:

        filename = "battery_inspection_report"

    filename = os.path.splitext(
        filename
    )[0]

    # Remove potentially problematic characters.
    safe_filename = "".join(
        character
        if character.isalnum()
        or character in (
            "_",
            "-",
            " "
        )
        else "_"
        for character in filename
    )

    report_path = os.path.join(
        REPORTS_PATH,
        f"{safe_filename}_report.docx"
    )

    document.save(
        report_path
    )

    return report_path


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    explain: bool = True
):

    image = await read_uploaded_image(
        file
    )

    try:

        result = analyze_image(
            image,
            generate_explanation=explain
        )

    except Exception as error:

        print(
            "Prediction error:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "AI inference failed. "
                "Check backend logs."
            )
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
            result["gradcam_png"] is not None
        )
    }

    if result["gradcam_png"] is not None:

        response["gradcam"] = (
            base64.b64encode(
                result["gradcam_png"]
            ).decode(
                "utf-8"
            )
        )

    return response


# ============================================================
# Report Endpoint
# ============================================================

@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...)
):

    image = await read_uploaded_image(
        file
    )

    try:

        result = analyze_image(
            image,
            generate_explanation=True
        )

        report_path = generate_battery_report(
            image,
            result,
            file.filename
        )

    except Exception as error:

        print(
            "Report generation error:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Inspection report generation failed. "
                "Check backend logs."
            )
        )

    return FileResponse(
        path=report_path,
        filename=os.path.basename(
            report_path
        ),
        media_type=(
            "application/vnd.openxmlformats-"
            "officedocument.wordprocessingml.document"
        )
    )