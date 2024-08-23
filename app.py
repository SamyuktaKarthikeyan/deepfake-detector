from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Initialize Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
model.eval()  # Set the model to evaluation mode

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the image
            inputs = process_image(file_path)

            # Predict using the model
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=1).item()

            # Determine if it's a deepfake or not
            result = "Deepfake" if prediction == 1 else "Real"  # Assuming 1 is deepfake and 0 is real

            # Render the result page with the prediction
            return render_template('result.html', result=result, filename=filename)
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
