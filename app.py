import sys
sys.path.insert(0, '/path/to/your/project_root') 
from utils.preprocess import preprocess_image
import streamlit as st
import tensorflow as tf
import json
from PIL import Image
import numpy as np
import os
from plant_utils.preprocess import preprocess_image
from .utils.preprocess import preprocess_image  # Note the dot

# Initialize Kaggle dataset
if not os.path.exists('data/train'):
    with st.spinner("📥 Downloading plant dataset..."):
        download_dataset()

# Load model (now from data directory)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('data/plant_model.h5')

# Rest of your existing app code...

# Add Font Awesome for icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

# Load custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header section with enhanced design
st.markdown("""
<div class="header">
    <h2>🌿 Plant Detector Pro</h2>
    <p><strong>Advanced Plant Identification System</strong></p>
    <p>Currently supports <strong>98 medicinal plants</strong> with AI detection</p>
    <div class="social-icons">
        <a href="https://instagram.com/xpushpraj" target="_blank" class="social-icon instagram">
            <i class="fab fa-instagram"></i>
        </a>
        <a href="https://github.com/xpushpraj" target="_blank" class="social-icon github">
            <i class="fab fa-github"></i>
        </a>
        <a href="https://farmfrontier.netlify.app" target="_blank" class="social-icon website">
            <i class="fas fa-globe"></i>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer section
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong> This is an interest project trained on 98 plants. 
While we strive for accuracy, there may be occasional errors in identification. 
Not intended for professional medical or botanical use.
</div>
""", unsafe_allow_html=True)

# Main title
st.title("Plant Identification Portal")

# Load model
model = tf.keras.models.load_model('models/plant_model.h5', compile=False)

# Load class names (must match training order)
class_names = sorted(os.listdir('datasets/train'))

# Load plant info JSON
with open('data/plants_info.json', 'r') as f:
    plant_info = json.load(f)

# Upload section with enhanced styling
st.markdown("""
<div class="upload-section">
    <h3 style="color: #2e7d32; text-align: center;">Upload Plant Image for Identification</h3>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing plant features..."):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)

        class_index = np.argmax(prediction)
        predicted_class = class_names[class_index]
        confidence = np.max(prediction) * 100

        plant = plant_info.get(predicted_class, {
            "common_name": predicted_class,
            "scientific_name": "Unknown",
            "description": "No description available for this plant."
        })

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, use_container_width=True)
        with col2:
            st.markdown(f"""
            <div class="result fade-in">
                <h3 style="color: #2e7d32; margin-top: 0;">Identification Results</h3>
                <p><strong>🌱 Common Name:</strong> {plant["common_name"]}</p>
                <p><strong>🔬 Scientific Name:</strong> <i>{plant["scientific_name"]}</i></p>
                <p><strong>📊 Confidence:</strong> {confidence:.2f}%</p>
                <p style="margin-top: 15px;">{plant["description"]}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Plant Detector App v1.0</p>
    <p>Made with ❤️ by <a href="https://farmfrontier.netlify.app" target="_blank">xpushpz</a></p>
    <p>© 2025 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)