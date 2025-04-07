import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
import random
import os

# Load the model once
model = load_model("mobnet_fine_tuned_model.keras")

# Class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy'
]

# Helper to convert image to base64
def image_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Image paths
image_paths = [
    "apple.jpg", "corn.jpg", "grape.jpg", "potato.jpg", "tomato.jpg",
    "apple2.jpg", "corn2.jpg", "grape1.jpg", "potato1.jpg", "tomato1.jpg",
    "apple.jpg", "corn.jpg", "grape3.jpg", "potato2.jpg", "tomato.jpg"
]

image_s = [
    "apple_hy.jpeg",
    "apple_sb.jpeg",
    "Corn-cr.jpg",
    "grape_br.jpeg",
    "apple_cedar_ar.jpeg"
]

# Sidebar navigation
st.sidebar.title("MENU")
page = st.sidebar.selectbox("", ["Home", "About"])

# Home page
if page == "Home":
    st.header("🌿 PLANT DISEASE DETECTION SYSTEM")

    # Scrollable Image Gallery
first_image_path = image_paths[0]
img_base64 = image_to_base64(first_image_path)
images_html = f'<img src="data:image/png;base64,{img_base64}" style="height: 150px; border-radius: 10px;">'

remaining_images = image_paths[1:]
random.shuffle(remaining_images)

for img_path in remaining_images:
    img_base64 = image_to_base64(img_path)
    images_html += f'<img src="data:image/png;base64,{img_base64}" style="height: 150px; border-radius: 10px;">'

scrolling_html = f"""
<div style="display: flex; overflow-x: auto; padding: 10px; gap: 10px; scroll-behavior: smooth;">
    {images_html}
</div>
"""

st.markdown("### 🌱 Sample Leaves Gallery")
st.components.v1.html(scrolling_html, height=180)

    # Upload or choose test image
    st.markdown("### 🖼️ Upload or Choose a Test Image")
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    
    st.write("Or choose one from test images below:")
    test_image_option = st.selectbox(
        "Select Test Image",
        ["None"] + image_s,
        format_func=lambda x: "None" if x == "None" else os.path.basename(x)
    )

    if test_image_option != "None":
        uploaded_file = test_image_option

    if uploaded_file is not None:
        if isinstance(uploaded_file, str):
            img = Image.open(uploaded_file)
        else:
            img = Image.open(uploaded_file)
        
        st.image(img, caption="Selected Image", width=250)

        if st.button("🔍 Predict"):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))

            st.success(f"🌾 Prediction: **{predicted_class}**")
            st.info(f"🧠 Confidence: **{confidence * 100:.2f}%**")

# About page
elif page == "About":
    st.title("📘 About This App")
    st.header("🔍 Purpose")
    st.write("This app helps farmers and plant lovers detect diseases in leaves using deep learning.")
    st.header("🧠 Model Info")
    st.write("Model: Fine-tuned MobileNetV2 | Dataset: PlantVillage | Crops: Apple, Corn, Grape, Potato, Tomato")
