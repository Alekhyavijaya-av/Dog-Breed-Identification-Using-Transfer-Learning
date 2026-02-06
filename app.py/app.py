import os
import json
import numpy as np
from flask import Flask, render_template, request, url_for, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

app = Flask(__name__)

# 1. Configuration - Set folders for uploads and static files
# Since your C: drive is full, Flask will use the folder on your pendrive
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 2. Load the Model and Labels
model = load_model('dog_breed_model.h5')
with open('labels.json', 'r') as f:
    labels_map = json.load(f)

@app.route('/')
def index():
    # Renders your main upload page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded image to the static/uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image for the AI model
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict the breed
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        
        # Get breed name from our JSON labels
        breed_name = labels_map.get(str(predicted_class_index), "Unknown Breed")

        # RENDER the result.html page with the data
        return render_template('result.html', 
                               prediction=breed_name, 
                               image_path=filepath)

if __name__ == '__main__':
    # Run the server
    app.run(debug=True)