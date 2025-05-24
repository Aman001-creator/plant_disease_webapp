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

    # Create base64 HTML image string
    first_image_path = image_paths[0]
    img_base64 = image_to_base64(first_image_path)
    images_html = f'<img src="data:image/png;base64,{img_base64}" alt="Image" class="image" style="height: 150px; border-radius: 10px; margin-right: 10px;">'

    remaining_images = image_paths[1:]
    random.shuffle(remaining_images)

    for image_path in remaining_images:
        img_base64 = image_to_base64(image_path)
        images_html += f'<img src="data:image/png;base64,{img_base64}" alt="Image" class="image" style="height: 150px; border-radius: 10px; margin-right: 10px;">'

    # Duplicate for infinite scroll illusion
    duplicated_images_html = images_html + images_html

    scrolling_html = f"""
    <div style="position: relative; height: 180px; overflow: hidden; width: 100%;">
        <style>
            .scrolling-wrapper {{
                display: flex;
                width: fit-content;
                animation: scroll 40s linear infinite;
            }}

            @keyframes scroll {{
                0% {{ transform: translateX(0%); }}
                100% {{ transform: translateX(-50%); }}
            }}
        </style>
        <div class="scrolling-wrapper">
            {duplicated_images_html}
        </div>
    </div>
    """

    st.components.v1.html(scrolling_html, height=200)

    # Upload or choose test image
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
    st.header("About")
    
    # Create a dropdown or radio button to switch between sub-sections in the About page
    about_section = st.radio(
        "Choose a section to explore:",
        ["About the Project", "About the Developers"]
    )
    
    if about_section == "About the Project":
        st.subheader("About the Project")
        st.markdown("""#### Project Overview
The Multiple Plant Disease Detection System leverages cutting-edge deep learning techniques to accurately identify plant diseases through image analysis. Using a fine-tuned MobileNetV2 model, this system is designed to help farmers, agricultural experts, and plant enthusiasts quickly diagnose plant diseases and take timely action. The goal is to reduce crop losses, promote sustainable farming practices, and assist in early disease detection.

#### Key Features
* __Accurate Disease Detection:__  Identifies diseases in plant leaves from a variety of crops including apples, corn, tomatoes, and more.\n
* __User-Friendly Interface:__  Easily upload images of plant leaves to get real-time disease predictions.\n
* __Machine Learning Model:__  Built using a fine-tuned MobileNet V2 model, trained on a large dataset of plant leaf images.\n
* __Multiple Crop Support:__  Capable of detecting diseases in various crops with 19 distinct classes.\n
                      
#### Project Goal
The primary objective of this project is to:

__1. Support Farmers:__ Provide a cost-effective solution for early disease detection.\n
__2. Improve Crop Yield:__ Reduce the time between disease occurrence and identification.\n
__3. Promote Sustainability:__ Encourage efficient resource use by targeting specific diseases.\n
                    
#### Technology Stack
* __Deep Learning Framework:__ TensorFlow and Keras for model training and inference.\n
* __Model Architecture:__ MobileNet V2, a powerful convolutional neural network for image classification.\n
* __Frontend Interface:__ Designed with a user-friendly layout for easy interaction.\n
* __Programming Language:__ Python, used for both model development and deployment.\n
* __Optimization:__ Early stopping and model checkpointing to ensure the best performance during training.\n

#### About the Dataset
The dataset used in this project is a comprehensive collection of plant leaf images categorized into 19 classes, covering both healthy and diseased leaves.\n

Training Images: 35669\n
Validation Images: 8917\n
Test Images: 25 (for performance evaluation) The dataset includes a wide variety of plant diseases and conditions, making it robust for real-world applications.
                            
#### Achieved Accuracy
__I. Training Accuracy: 99.67%__\n
__II. Validation Accuracy: 99.24%__\n
The model achieves excellent results in detecting plant diseases, showcasing the power of deep learning in agriculture.
                    
#### Future Enhancements
__Multi-language Support:__ To cater to users globally, we plan to add multilingual capabilities.\n
__Broader Crop Coverage:__ Adding support for more crops and diseases.\n
__Mobile Application:__ Developing a mobile app for on-the-go disease detection.\n
#### Conclusion
We hope this project inspires innovative solutions in the field of agriculture and plant health. Whether you're a farmer, researcher, or tech enthusiast, this tool is designed to empower you with the power of AI for a better tomorrow.

Together, we can make strides in sustainable farming and secure healthier crops for the future. Thank you for exploring this system, and we look forward to your feedback and ideas to improve and expand its capabilities! 🌱
        """)
        
    elif about_section == "About the Developers":
        st.subheader("About the Developers")
        st.markdown("""
            This project was developed by a team of dedicated students with a shared interest in leveraging technology to tackle real-world challenges in agriculture. Our goal was to build a plant disease detection system using deep learning to help farmers detect plant diseases quickly and accurately.""")


        st.markdown("__Team Members:__")

        st.markdown("__Aman Kumar - Project Lead & Developer__")
        st.markdown("""Aman took the lead in the development of the project, overseeing the technical execution. He focused on training and fine-tuning the Inception V3 model for plant disease detection and integrating the model with the Streamlit interface. Aman ensured the overall flow of the application worked seamlessly from image upload to disease prediction.""")

        st.markdown("__Harsh Pratap Singh - Frontend Developer__")
        st.markdown("""Harsh Pratap Singh was responsible for creating the user interface and ensuring an intuitive and smooth experience for users. They designed the Streamlit frontend, providing a simple yet effective way for users to upload images and receive disease predictions.""")

        st.markdown("__MD. Badrul Hoda - Tester & Quality Assurance__")
        st.markdown("""MD. Badrul Hoda was responsible for testing the system, ensuring that the application performed as expected. They conducted thorough testing to identify bugs, glitches, and user experience issues, providing feedback to the team to enhance the application's quality, stability, and usability.""")

        st.markdown("__Shubham Gupta - Support Specialist__")
        st.markdown("""Shubham Gupta played a crucial supporting role throughout the development of the project. Their responsibility was to assist in various tasks, including dataset organization, documentation, and user interface enhancements. They also helped with troubleshooting and ensuring the smooth progress of the project by assisting all team members as needed.
                """)
        st.subheader("Acknowledgements")
        st.markdown("""We would like to express our gratitude to the researchers and contributors who provided the plant disease dataset. Their work was essential to the success of this project. Additionally, we thank our mentors for their continuous support and guidance throughout the development process.
                """)
        st.markdown("""
        <div style="text-align: right;">
        <P>- Team Members</p>
        </div>
        """, unsafe_allow_html=True)

