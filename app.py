import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
import cv2
import librosa
from sklearn.ensemble import RandomForestClassifier
import joblib

# Define a simple CNN model for cow breed identification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 10)  # Assume 10 cow breeds

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x

# Load pre-trained model (Replace with actual model path if available)
@st.cache_resource
def load_cnn_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load("breed_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

cnn_model = load_cnn_model()

# Load ONNX model for distress detection
@st.cache_resource
def load_onnx_model():
    return ort.InferenceSession("voice_model.onnx")

onnx_session = load_onnx_model()

# Load scikit-learn model for health monitoring
@st.cache_resource
def load_health_model():
    return joblib.load("health_model.pkl")

health_model = load_health_model()

# Streamlit UI
st.title("Silkey Project Prototype")

### 🐄 Cow Breed Identification
st.header("📸 CNN-Based Cow Breed Identification")
uploaded_image = st.file_uploader("Upload a Cow Image", type=["jpg", "png"])
if uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (32, 32))  # Resize for CNN input
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        prediction = cnn_model(image_tensor)
    
    predicted_breed = torch.argmax(prediction).item()
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"🐄 Predicted Breed: **Breed {predicted_breed}**")

### 🏥 AI-Based Health Monitoring
st.header("📊 AI-Based Health Monitoring")
temperature = st.number_input("🐄 Body Temperature (°C)", value=38.0, step=0.1)
movement = st.slider("🐄 Movement Level", 0, 100, 50)
eating_behavior = st.slider("🐄 Eating Behavior", 0, 100, 50)

if st.button("🔍 Predict Health Condition"):
    input_features = np.array([[temperature, movement, eating_behavior]])
    health_prediction = health_model.predict(input_features)
    st.write(f"📢 Health Condition: **{'Normal' if health_prediction[0] == 1 else '⚠️ Warning: Possible Health Issue'}**")

### 🔊 Voice Recognition for Distress Detection
st.header("🎤 Voice Recognition for Distress Detection")
audio_file = st.file_uploader("Upload Cow Sound", type=["wav", "mp3"])
if audio_file:
    audio_data, sr = librosa.load(audio_file, sr=None)
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13).mean(axis=1)
    
    distress_prediction = onnx_session.run(None, {"input": mfcc_features.reshape(1, -1)})[0]
    
    st.write(f"📢 Distress Status: **{'Distressed' if distress_prediction[0] > 0.5 else 'Normal'}**")

# Run the app with: `streamlit run app.py`
