
# app.py
import streamlit as st
import os
import torch
import cv2
import re
from PIL import Image
from torchvision import transforms, models
import easyocr
from twilio.rest import Client

# Twilio credentials
account_sid = 'AC26ebbe3932c57c0f2f6c60038d1568ee'
auth_token = '8c7dcf96ca934bff6ef3b466513b40fa'
twilio_number = '+19786934978'
client = Client(account_sid, auth_token)

# Mapping of plate to phone number
plate_to_phone = {
    "TN68AJ0661": "+919659339243",
    "49CA9121": "+917339511092",
    "TN47BC1018": "+917339511092"
}
normalized_plate_to_phone = {
    re.sub(r'[^A-Z0-9]', '', k.upper()): v for k, v in plate_to_phone.items()
}

# Labels and model
class_labels = ['Helmet', 'Non-Helmet', 'Bike', 'Licence plate']
import os

model_path = os.path.join("D:", "new pro (av,gr)", "vision transformer", "results 24-04-2025", "resnet18_helmet_detection.pth")

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])
ocr_reader = easyocr.Reader(['en'])

# Streamlit UI
st.set_page_config(page_title="Helmet Violation Detector", layout="wide")
st.title("🚨 Helmet Violation Detection & Auto SMS System")

uploaded_video = st.file_uploader("📤 Upload a Video", type=["mp4"])
run_detection = st.button("▶️ Run Violation Detection")

if run_detection and uploaded_video:
    with open("input_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture("input_video.mp4")
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    plates_sent = set()
    non_helmet_count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred_class = torch.max(outputs, 1)
            predicted_label = class_labels[pred_class.item()]

        # OCR
        plate_number, normalized_plate = None, None
        ocr_results = ocr_reader.readtext(frame)
        for (_, text, prob) in ocr_results:
            if prob > 0.5 and 5 <= len(text) <= 12:
                plate_number = text.upper().replace(" ", "")
                normalized_plate = re.sub(r'[^A-Z0-9]', '', plate_number)
                break

        # Detection logic
        if predicted_label == "Non-Helmet":
            non_helmet_count += 1
            if normalized_plate and normalized_plate in normalized_plate_to_phone and normalized_plate not in plates_sent:
                phone = normalized_plate_to_phone[normalized_plate]
                message = f"🚨 Helmet violation detected for vehicle {normalized_plate}. A fine ₹1000 will be issued."
                try:
                    client.messages.create(body=message, from_=twilio_number, to=phone)
                    plates_sent.add(normalized_plate)
                    cv2.imwrite(f"violation_{normalized_plate}_{frame_count}.jpg", frame)
                except Exception as e:
                    print(f"SMS failed: {e}")

        cv2.putText(frame, f"{predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        if plate_number:
            cv2.putText(frame, f"Plate: {plate_number}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()

    st.video("output_video.mp4")
    st.success(f"✅ Processed {frame_count} frames")
    st.warning(f"🛑 Non-Helmet Detections: {non_helmet_count}")
    st.info(f"📨 Messages sent for plates: {', '.join(plates_sent) if plates_sent else 'None'}")



