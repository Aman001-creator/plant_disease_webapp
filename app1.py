# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model once
model = load_model("mobnet_fine_tuned_model.keras")

# Class names
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___healthy'
]

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

# Home page
if page == "Home":
    st.title("🌿 Plant Disease Detection")
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            # Preprocess the image
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Run prediction
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))

            # Display result
            st.success(f"🌱 Prediction: **{predicted_class}**")
            st.info(f"Confidence: **{confidence * 100:.2f}%**")

# About page
elif page == "About":
    st.title("📘 About This App")

    st.header("🔍 Purpose")
    st.write("""
    This app is designed to help farmers, researchers, and plant enthusiasts detect diseases in plant leaves using a deep learning model.
    Just upload an image of a leaf, and the model will identify the disease if present.
    """)

    st.header("🧠 Model Info")
    st.write("""
    The model used in this application is a fine-tuned version of MobileNet, trained on the PlantVillage dataset.
    It can detect diseases in crops like apple, corn, grape, potato, and tomato with high accuracy.
    """)
