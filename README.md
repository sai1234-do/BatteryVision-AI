# 🔋 BatteryVision AI

### AI-Assisted Battery Surface Inspection System

> **Computer Vision • Transfer Learning • Explainable AI • FastAPI • React**

BatteryVision AI is an AI-assisted computer vision system for analyzing battery surface images using a **fine-tuned ResNet18 model**.

The system classifies battery images into **7 supported material/condition classes**, provides prediction confidence, generates **Grad-CAM visual explanations**, and produces a downloadable **AI inspection report** through a production-style FastAPI backend and React inspection interface.

---

## 🚀 What BatteryVision Does

```text
Battery Image
      │
      ▼
Image Preprocessing
      │
      ▼
Fine-tuned ResNet18
      │
      ├──────────────► Prediction
      │
      ├──────────────► Confidence
      │
      └──────────────► Grad-CAM Explanation
                              │
                              ▼
                     Inspection Evidence
                              │
                              ▼
                       DOCX Report
```

### Core capabilities

* 🖼️ Battery image upload
* 🧠 7-class image classification
* 🎯 Prediction confidence
* 🔥 Grad-CAM visual explanations
* ⚡ FastAPI inference backend
* 💻 React inspection interface
* 📄 Automated DOCX inspection reports
* 🌐 Public API access through Cloudflare Tunnel
* 📊 Model experimentation and comparison
* 🔍 Failure analysis and model limitation analysis

---

# 🎯 Problem Statement

Visual inspection of battery surfaces can involve identifying material types and surface conditions from images.

BatteryVision explores how deep learning and explainable computer vision can assist this process by:

1. Learning visual patterns from battery surface images.
2. Classifying images into predefined material/condition categories.
3. Providing confidence information alongside predictions.
4. Showing regions associated with the model's prediction using Grad-CAM.
5. Generating an inspection report containing the prediction and visual evidence.

The project is designed as both a **computer vision learning laboratory** and a practical AI portfolio system.

---

# 🧩 Supported Classification Space

BatteryVision currently operates as a **closed-set seven-class image classifier**.

The model can classify an input into one of the following trained classes:

| Material    | Condition | Model Label     |
| ----------- | --------- | --------------- |
| 🔩 Tin      | Bad       | `Tin_bad`       |
| 🔩 Tin      | Good      | `Tin_good`      |
| 🔩 Tin      | Medium    | `Tin_medium`    |
| ⚙️ Titanium | Bad       | `Titanium_bad`  |
| ⚙️ Titanium | Good      | `Titanium_good` |
| 🧪 Zinc     | Bad       | `Zinc_Bad`      |
| 🧪 Zinc     | Good      | `Zinc_Good`     |

### Closed-set behavior

The current classifier does **not** contain an `Unknown`, `Other`, or `Non-battery` class.

Therefore:

```text
Unknown / Unrelated Image
          │
          ▼
   Fine-tuned ResNet18
          │
          ▼
One of the 7 trained classes
```

This behavior was explicitly tested with unrelated images.

A future version could incorporate **out-of-distribution detection** or a rejection mechanism to identify inputs outside the supported classification space.

---

# 🏗️ System Architecture

```text
┌─────────────────────────────────────────────┐
│                React Frontend               │
│                                             │
│  Upload • Preview • Analyze • Results       │
│  Grad-CAM • Report Generation               │
└──────────────────────┬──────────────────────┘
                       │ HTTP
                       ▼
┌─────────────────────────────────────────────┐
│                 FastAPI API                 │
│                                             │
│  GET  /health                               │
│  POST /predict                              │
│  POST /report                               │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│            Image Preprocessing              │
│                                             │
│ Resize → Crop → Tensor → Normalize          │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│          Fine-tuned ResNet18                │
│                                             │
│  Image Features → 7-Class Prediction        │
└───────────────┬─────────────────┬───────────┘
                │                 │
                ▼                 ▼
        ┌──────────────┐  ┌────────────────┐
        │ Prediction   │  │    Grad-CAM    │
        │ + Confidence │  │ Visualization  │
        └──────┬───────┘  └───────┬────────┘
               │                  │
               └─────────┬────────┘
                         ▼
              ┌─────────────────────┐
              │ Inspection Evidence │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   DOCX Report       │
              │ Prediction + GradCAM│
              └─────────────────────┘
```

---

# 🧠 Model Development

Rather than immediately using a pretrained network, BatteryVision was developed through multiple model stages.

### 1️⃣ Custom CNN

A custom convolutional neural network was implemented as the initial baseline.

**Test performance:**

* Accuracy: **77.42%**
* Macro F1: **71.88%**
* Weighted F1: **79.66%**

---

### 2️⃣ ResNet18 — Frozen Backbone

A pretrained ResNet18 was introduced using transfer learning.

The convolutional backbone was frozen and the final classification layer was adapted for the seven target classes.

**Test performance:**

* Accuracy: **83.87%**
* Macro F1: **77.29%**
* Weighted F1: **83.96%**

---

### 3️⃣ ResNet18 — Fine-Tuning

The final model fine-tuned `layer4` and the classification head while retaining pretrained representations from the earlier layers.

**Final test performance:**

* Accuracy: **87.10%**
* Macro F1: **77.76%**
* Weighted F1: **85.92%**

The fine-tuned ResNet18 became the model used by the deployed application.

---

# 📊 Model Comparison

| Model                     |   Accuracy |   Macro F1 | Weighted F1 |
| ------------------------- | ---------: | ---------: | ----------: |
| Custom CNN                |     77.42% |     71.88% |      79.66% |
| ResNet18 — Frozen         |     83.87% |     77.29% |      83.96% |
| **ResNet18 — Fine-tuned** | **87.10%** | **77.76%** |  **85.92%** |

### Improvement through experimentation

```text
Custom CNN
   │
   │ +6.45 percentage points
   ▼
ResNet18 Frozen
   │
   │ +3.23 percentage points
   ▼
ResNet18 Fine-tuned
```

This progression demonstrates the impact of **transfer learning followed by targeted fine-tuning** on the dataset.

---

# 🔥 Explainable AI — Grad-CAM

A classification result alone does not show *where* the model was looking.

BatteryVision therefore integrates **Grad-CAM** to visualize image regions associated with the model's prediction.

```text
Original Image
      │
      ▼
ResNet18
      │
      ▼
Target Prediction
      │
      ▼
Gradient + Feature Activations
      │
      ▼
Grad-CAM
      │
      ▼
Attention Heatmap
```

The Grad-CAM implementation targets the final convolutional feature layer:

```text
ResNet18
   └── layer4[-1]
```

The resulting heatmap is resized and overlaid on the original battery image.

### Important interpretation

Grad-CAM identifies regions associated with the model's prediction.

It **does not establish causal reasoning** or prove that the highlighted region is the sole reason for the prediction.

---

# 🔍 Failure Analysis

Model evaluation went beyond overall accuracy.

The final ResNet18 model produced:

* **124 test images**
* **16 misclassified images**
* **14/16 errors involved Tin**
* **9/16 errors were `Tin_good → Tin_bad`**
* Only **1/10 Tin_good** test images were correctly classified

Grad-CAM analysis showed that the model generally focused on localized battery-surface regions rather than obvious background or border regions.

However, the difficult Tin cases demonstrate that surface-focused attention does not necessarily mean the model has learned the desired fine-grained distinction.

This analysis helped identify an important area for future improvement: **better discrimination between visually similar Tin conditions**.

---

# 🗂️ Dataset

The dataset contains **989 RGB JPG images** distributed across seven classes.

| Class         |  Images |
| ------------- | ------: |
| Tin_bad       |     105 |
| Tin_good      |      76 |
| Tin_medium    |      60 |
| Titanium_bad  |     110 |
| Titanium_good |     182 |
| Zinc_Bad      |     356 |
| Zinc_Good     |     100 |
| **Total**     | **989** |

### Dataset characteristics

* Total images: **989**
* Image format: **RGB JPG**
* Landscape images: **797**
* Portrait images: **192**

A fixed train/validation/test split was created and preserved throughout the project:

| Split      |  Images |
| ---------- | ------: |
| Training   |     712 |
| Validation |     153 |
| Test       |     124 |
| **Total**  | **989** |

---

# ⚙️ Image Pipeline

### Training

```text
Original Image
      ↓
Resize
      ↓
Random Crop
      ↓
Horizontal Flip
      ↓
Random Rotation
      ↓
Tensor Conversion
      ↓
ImageNet Normalization
```

### Validation / Testing / Inference

```text
Original Image
      ↓
Resize
      ↓
Center Crop
      ↓
Tensor Conversion
      ↓
ImageNet Normalization
      ↓
ResNet18
```

This separation prevents random augmentation from being applied during evaluation and inference.

---

# 🔬 Inference Pipeline

The production inference flow is:

```text
Upload Image
     ↓
Validate File Type
     ↓
Read Image
     ↓
Convert to RGB
     ↓
Preprocess
     ↓
Fine-tuned ResNet18
     ↓
Softmax Probabilities
     ↓
Predicted Class
     ↓
Confidence
     ↓
Grad-CAM
     ↓
API Response
```

Supported upload formats:

```text
JPG
PNG
WEBP
```

Unsupported file types are rejected by the API.

---

# 🌐 FastAPI Backend

BatteryVision exposes the model through a FastAPI service.

### Health Check

```http
GET /health
```

Returns system and model status.

### Prediction

```http
POST /predict
```

Accepts an image and returns:

* Filename
* Predicted class
* Confidence
* Confidence percentage
* Confidence level
* Model name
* Grad-CAM information

### Inspection Report

```http
POST /report
```

Generates a downloadable DOCX inspection report containing the model's inspection information and visual evidence.

---

# 💻 React Inspection Interface

The frontend provides a dedicated inspection workstation with:

* Drag-and-drop image upload
* Image preview
* AI prediction display
* Confidence indicator
* Grad-CAM visualization
* Inspection pipeline display
* Error handling
* Inspection report generation

The interface is designed around an inspection workflow rather than a generic AI chatbot/dashboard.

---

# 📄 Automated Inspection Reports

BatteryVision can generate a DOCX inspection report containing:

* Inspection prediction
* Confidence
* Original image
* Grad-CAM visualization
* Model information
* Processing pipeline
* Performance information
* Limitations
* Conclusion

This converts the model output into a reusable inspection artifact.

---

# 🧪 Testing

The application was tested across:

### Classification

* Tin images
* Titanium images
* Zinc images
* Good / bad / medium conditions where applicable

### File validation

* JPG
* PNG
* WEBP
* Unsupported file types

### System integration

```text
React
  ↓
Cloudflare Tunnel
  ↓
FastAPI
  ↓
ResNet18
  ↓
Grad-CAM
  ↓
React Result
```

### Out-of-distribution behavior

Unrelated images were also tested.

The model returned one of the seven trained classes, confirming the current **closed-set classifier limitation**.

---

# 🛠️ Technology Stack

### Machine Learning

* Python
* PyTorch
* Torchvision
* ResNet18
* Transfer Learning
* Fine-Tuning
* Grad-CAM

### Data Processing

* NumPy
* PIL
* Image transforms

### Backend

* FastAPI
* Uvicorn
* Python Multipart

### Frontend

* React
* JavaScript
* CSS

### Deployment / Connectivity

* Google Colab
* Google Drive
* Cloudflare Tunnel

### Reporting

* Python-docx

---

# 📁 Project Structure

```text
BatteryVision/
│
├── dataset/
│   ├── Tin/
│   ├── Titanium/
│   └── Zinc/
│
├── models/
│   ├── battery_resnet18_frozen.pth
│   └── battery_resnet18_finetuned.pth
│
├── results/
│   ├── resnet_results.json
│   └── resnet_finetuned_results.json
│
├── reports/
│
├── split.json
│
├── BatteryVision_Master.ipynb
│
└── README.md
```

---

# 📈 Key Results

The final BatteryVision model achieved:

> ## **87.10% Test Accuracy**

with:

> **77.76% Macro F1**

and:

> **85.92% Weighted F1**

The project demonstrates an end-to-end computer vision workflow:

```text
Dataset
   ↓
Image Pipeline
   ↓
CNN Baseline
   ↓
Transfer Learning
   ↓
Fine-Tuning
   ↓
Evaluation
   ↓
Failure Analysis
   ↓
Grad-CAM
   ↓
FastAPI
   ↓
React
   ↓
Inspection Report
```

---

# ⚠️ Limitations

BatteryVision is a research/portfolio prototype and has several known limitations.

### 1. Closed-set classification

The model always selects one of the seven trained classes and does not currently perform OOD detection.

### 2. Class imbalance

The dataset contains substantially different numbers of images across classes, which affects class-wise performance.

### 3. Tin classification difficulty

The majority of observed test errors involved Tin, particularly `Tin_good` and `Tin_bad`.

### 4. Grad-CAM interpretation

Grad-CAM highlights regions associated with a prediction but does not establish causal reasoning.

### 5. Dataset scope

Performance is based on the available dataset and may not generalize to different cameras, lighting conditions, battery designs, environments, or industrial datasets.

---

# 🔮 Future Improvements

Potential next steps include:

* [ ] Out-of-distribution detection
* [ ] Unknown / rejection class
* [ ] Larger and more balanced dataset
* [ ] Improved Tin classification
* [ ] Additional battery materials and conditions
* [ ] More robust augmentation
* [ ] Calibration of confidence scores
* [ ] Production deployment with persistent infrastructure
* [ ] Model monitoring
* [ ] Automated experiment tracking
* [ ] Larger-scale industrial validation

---

# 🎓 What I Learned

BatteryVision was built as a hands-on exploration of modern computer vision engineering.

Key concepts covered:

* Image representation
* RGB images
* Tensor conversion
* Image resizing and cropping
* Normalization
* Data augmentation
* CNN architecture
* Convolutional feature extraction
* Transfer learning
* ResNet residual connections
* Freezing and fine-tuning
* Cross-entropy classification
* Model evaluation
* Precision / Recall / F1
* Class imbalance
* Failure analysis
* Grad-CAM
* PyTorch inference
* FastAPI model serving
* React integration
* API-based deployment
* Automated report generation

---

# 👨‍💻 Project Philosophy

BatteryVision was developed with a simple principle:

> **Don't just train a model. Understand it, evaluate it, explain it, and build a usable system around it.**

The project therefore goes beyond a notebook-level classifier and connects:

**Computer Vision → Explainability → API → Frontend → Inspection Evidence**

---

# 📌 Status

### 🟢 Project Status: Functional Portfolio Prototype

| Component                     | Status         |
| ----------------------------- | -------------- |
| Dataset pipeline              | ✅ Completed     |
| Custom CNN baseline           | ✅ Completed    |
| ResNet18 transfer learning    | ✅ Completed    |
| ResNet18 fine-tuning          | ✅ Completed    |
| Model evaluation              | ✅ Completed     |
| Failure analysis              | ✅ Completed     |
| Grad-CAM                      | ✅ Completed     |
| Inference pipeline            | ✅ Completed     |
| FastAPI backend               | ✅ Completed     |
| React frontend                | ✅ Completed     |
| Cloudflare connectivity       | ✅ Tested       |
| DOCX report generation        | ✅ Completed     |
| Final portfolio documentation |  ✅ Completed|

---

## ⭐ Final Model

**Fine-tuned ResNet18**

**Test Accuracy:** `87.10%`

**Macro F1:** `77.76%`

**Weighted F1:** `85.92%`

---

> **BatteryVision AI — From battery image to explainable AI-assisted inspection.**
