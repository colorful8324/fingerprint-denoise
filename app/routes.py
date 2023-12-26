import os
import matplotlib.pyplot as plt
from flask import render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from app import app

UPLOAD_FOLDER = 'app/static/uploads'
PREDICTED_FOLDER = 'app/static/predicted'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTED_FOLDER'] = PREDICTED_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
        file.save(filename)

        # Process the image using the pretrained model
        output_filename = process_image(filename)

        return render_template('index.html', input_filename='uploads/input.jpg', output_filename=output_filename)

    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def process_image(input_filename):
    # Load the pretrained H5 model
    model = load_model('models/autoencoder.h5')

    # Load and preprocess the input image
    img = image.load_img(input_filename, color_mode='grayscale', target_size=(72, 72))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # plt.imshow(img_array)
    # plt.show()

    # Create the 'predicted' directory if it doesn't exist
    os.makedirs(app.config['PREDICTED_FOLDER'], exist_ok=True)

    # Model prediction
    prediction = model.predict(img_array)
    plt.imshow(prediction[0])
    plt.show()

    # Save the output image in the 'predicted' folder
    output_filename = os.path.join(app.config['PREDICTED_FOLDER'], 'output.jpg')
    output_img = image.array_to_img(prediction[0] * 255, scale=False)

    output_img.save(output_filename)

    return 'predicted/output.jpg'