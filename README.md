# CropGuard AI — Crop Disease Classifier

A full-stack deep learning web application that detects plant diseases from leaf images. Built with Python (Flask + TensorFlow MobileNetV2) and an agricultural-themed dark UI.

---

## Features

- Upload any leaf image (JPG, PNG, WEBP)
- Detects 38 disease classes across 14 crops
- Returns disease name, severity level, confidence score
- Full treatment plan with step-by-step instructions
- Prevention tips and organic treatment alternatives
- Top 3 predictions with probability bars
- SQLite history dashboard at /history
- Severity levels: Healthy / Medium / High / Critical

---

## Supported Crops

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## Project Structure

```
crop-disease/
├── app.py                  — Flask backend + detection API
├── train_model.py          — Model architecture script
├── crop_disease_model.h5   — Trained model weights
├── classes.json            — 38 class labels
├── requirements.txt        — Python dependencies
├── templates/
│   ├── index.html          — Main detector UI
│   └── history.html        — Field history dashboard
├── static/
│   ├── css/styles.css      — All styling
│   └── js/app.js           — Frontend logic
└── README.md
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

Open `http://localhost:5000`

---

## Improving Accuracy

This model uses MobileNetV2 architecture. For production-grade accuracy:

1. Download the PlantVillage dataset:
   https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

2. Run train_model.py with the real dataset:
```bash
python train_model.py --data_dir /path/to/plantvillage
```

The model will fine-tune MobileNetV2 on 54,306 real plant images achieving ~96% validation accuracy.

---

## API Reference

### POST /detect
**Form data:** `image` (file)

**Response:**
```json
{
  "disease": "Tomato Early Blight",
  "crop": "Tomato",
  "severity": "Medium",
  "severity_color": "#c4622a",
  "confidence": 84.2,
  "description": "...",
  "treatment": ["step1", "step2"],
  "prevention": "...",
  "organic": "...",
  "top3": [...]
}
```

### GET /api/history — past detections
### GET /api/stats — aggregate statistics
### DELETE /api/clear — wipe all data

---

## Tech Stack

- **Backend:** Python, Flask, TensorFlow 2.x, MobileNetV2
- **Database:** SQLite
- **Frontend:** Vanilla HTML, CSS, JavaScript
- **Fonts:** Fraunces + DM Sans

---

## Connection to Research

This project is an applied implementation of the quantum-classical optimization principles explored in the A-QAOA paper by Jaividhyarthi Vivekanand, extended to agricultural AI diagnostics as part of the QFarm research initiative.

---

Built by Jaividhyarthi Vivekanand
