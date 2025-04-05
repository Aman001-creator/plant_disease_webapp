from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your model
model = load_model('mobnet_fine_tuned_model.keras')

# Define class names based on your model
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
]  # change as needed

def preprocess(img):
    img = img.resize((224, 224))  # change if your model uses a different size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(io.BytesIO(file.read()))
            img = preprocess(img)
            pred = model.predict(img)
            prediction = class_names[np.argmax(pred)]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
