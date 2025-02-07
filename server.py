from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model # type: ignore
from PIL import Image
import uuid

# Directories
UPLOAD_FOLDER = "uploads"
MFCC_FOLDER = "mfcc_images"
MODEL_PATH = "models/VGGNet16.h5"
STATIC_FOLDER = "assets"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MFCC_FOLDER, exist_ok=True)

# Load the model
model = load_model(MODEL_PATH, compile=False)

# Initialize Flask app
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path="/assets")
CORS(app)  # Allow frontend requests

# Serve index.html
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# Serve MFCC images
@app.route("/mfcc_images/<filename>")
def get_mfcc_image(filename):
    return send_from_directory(MFCC_FOLDER, filename)

# Function to generate MFCC
def save_mfcc_image(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr)
    plt.axis("off")

    save_filename = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(MFCC_FOLDER, save_filename)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    
    return save_filename

# Function to predict mental state
def predict_mfcc(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((1000, 400))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    level_mapping = {
        0: "Level 0 (No Significant Depressive Symptoms)",
        1: "Level 1 (Mild Depressive Symptoms)",
        2: "Level 2 (Moderate Depressive Symptoms)",
        3: "Level 3 (Moderately Severe Depressive Symptoms)",
        4: "Level 4 (Severe Depressive Symptoms)"
    }

    return level_mapping.get(predicted_class, "Unknown")

# Route to handle file uploads and generate MFCC
@app.route("/generate_mfcc", methods=["POST"])
def generate_mfcc():
    if "audioFile" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    audio_file = request.files["audioFile"]
    filename = f"{uuid.uuid4().hex}.wav"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    audio_file.save(file_path)

    # Generate MFCC and return image URL
    mfcc_filename = save_mfcc_image(file_path)
    return jsonify({"success": True, "mfccImageUrl": f"/mfcc_images/{mfcc_filename}"})

# Route to predict from MFCC
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    mfcc_filename = data.get("mfccImageUrl", "").replace("/mfcc_images/", "")

    image_path = os.path.join(MFCC_FOLDER, mfcc_filename)
    if not os.path.exists(image_path):
        return jsonify({"success": False, "error": "MFCC image not found"}), 400

    prediction = predict_mfcc(image_path)
    return jsonify({"success": True, "prediction": prediction})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
