# 🚨 Helmet Violation Detection System using Vision Transformer (ViT) and YOLO

This project implements a **real-time helmet violation detection** pipeline using YOLO (v8/v10), Vision Transformer (ViT), and OCR for license plate recognition. It includes an automated fine messaging system via Twilio.

---

## 🧭 Overview

✔️ Detects helmet usage in traffic surveillance videos  
✔️ Identifies and classifies: Helmet, Non-Helmet, Bike, License Plate  
✔️ OCR extraction of license plate numbers using EasyOCR  
✔️ Automated SMS fine notification using Twilio  
✔️ Supports live webcam and video file inputs via Streamlit app

---

## 🎯 Features

- 🔎 **YOLOv8 & YOLOv10**: Real-time object detection
- 🤖 **Vision Transformer (ViT)** and **ResNet18**: Image classification
- 📝 **EasyOCR**: License plate reading
- 📲 **Twilio API**: Automated SMS sending to violators
- 📊 **Evaluation**: Confusion Matrix, Precision, Recall, F1-Score
- 🌐 **Streamlit App**: User-friendly interface for live/demo usage

---

## 📦 Project Structure

📂 src/
📂 notebooks/
📂 data/ # Sample images or data config
📂 results/ # Confusion matrix, graphs
📂 app/ # Streamlit UI code
requirements.txt
README.md

yaml
Copy
Edit

---

## 🚀 Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/Helmet-Violation-Detection-ViT.git
cd Helmet-Violation-Detection-ViT
2️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the detection pipeline
bash
Copy
Edit
python src/main_pipeline.py
4️⃣ Launch the Streamlit app
bash
Copy
Edit
streamlit run app/app.py
📊 Example Results
Confusion Matrix (ViT):

Precision, Recall, F1-score displayed in evaluation report

Sample OCR reading:

TN68AJ0661 ➜ "+919659339243"

🧪 Dataset
Custom YOLO-labeled dataset (~3500 images)

4 classes:

Helmet

Non-Helmet

Bike

License Plate

80:20 train/validation split

Annotation with LabelImg

⚙️ Tech Stack
Python 3.8+

PyTorch

YOLOv8 / YOLOv10 (Ultralytics)

Vision Transformer (ViT)

EasyOCR

Twilio API

Streamlit

🏆 Results
YOLOv8: 70.1% accuracy, ~90 FPS

YOLOv10: 75.1% accuracy, lightweight, 60 FPS

ViT: 87.1% accuracy in controlled conditions

📜 License
MIT License. Free to use, share, and improve.

🤝 Acknowledgements
EasyOCR

Ultralytics YOLO

Twilio

SASTRA Deemed University
