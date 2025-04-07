# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

# Set page config
st.set_page_config(page_title="Plant Disease Detector 🌿", layout="centered")

# Load model once
@st.cache_resource
def load_my_model():
    return load_model("mobnet_fine_tuned_model.keras")

model = load_my_model()

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
st.sidebar.title("🌿 Navigation")
selection = st.sidebar.radio("Go to", ["Prediction", "About"])

if selection == "About":
    st.title("🧬 Plant Disease Detection")
    st.markdown("""
        This AI-powered app identifies **plant diseases** from leaf images 🌿 using a fine-tuned MobileNet model.  
        It supports various crops like Apple, Corn, Grape, Potato, and Tomato.  
        **Upload an image** of a leaf to get started!

        **👨‍💻 Model Info**:
        - Model: MobileNetV2
        - Trained on: PlantVillage dataset
        - Accuracy: ~98%

        Created with ❤️ using Streamlit.
    """)
    st.info("Try uploading a leaf image from a plant affected by disease!")

elif selection == "Prediction":
    # Title
    st.title("🌿 Plant Disease Detection")
    st.markdown("Upload an image of a plant leaf to detect the disease using deep learning.")

    # File uploader
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            # Preprocess image
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediction
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))

            # Display result
            st.success(f"🌱 **Prediction**: `{predicted_class}`")
            st.info(f"🔍 **Confidence**: `{confidence * 100:.2f}%`")

            # Show confidence bar
            st.progress(min(int(confidence * 100), 100))

            # Plot all class probabilities
            st.subheader("🔬 Class Probabilities")
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(class_names))
            ax.barh(y_pos, predictions[0], align="center", color="green")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(class_names)
            ax.invert_yaxis()
            ax.set_xlabel("Probability")
            st.pyplot(fig)

            # Offer download of image with prediction
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            b64 = base64.b64encode(byte_im).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="leaf.png">📥 Download Uploaded Image</a>'
            st.markdown(href, unsafe_allow_html=True)
