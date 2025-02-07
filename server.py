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
def save_mfcc_image(file_path, filename, n_mfcc=13):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr)
    plt.axis("off")

    # Use the same name as the audio file, but save as .png
    save_filename = f"{filename}.png"
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

    # Updated level mapping with descriptions
    level_mapping = {
        0: {
            "label": "Level 0 (No Significant Depressive Symptoms)",
            "description": "Scores range from 0 to 4, indicating minimal or no depressive symptoms. "
                        "Individuals in this category generally do not exhibit signs of depression."
        },
        1: {
            "label": "Level 1 (Mild Depressive Symptoms)",
            "description": "Scores range from 5 to 9, representing mild levels of depression. "
                        "Symptoms at this level may include slight changes in mood, sleep, and energy, "
                        "but they typically do not significantly impair daily functioning."
        },
        2: {
            "label": "Level 2 (Moderate Depressive Symptoms)",
            "description": "Scores range from 10 to 14, indicating moderate depression. "
                        "Individuals may experience more noticeable symptoms that can start to impact daily "
                        "activities, like persistent sadness, decreased interest in activities, and changes in appetite or sleep patterns."
        },
        3: {
            "label": "Level 3 (Moderately Severe Depressive Symptoms)",
            "description": "Scores range from 15 to 19, reflecting a higher intensity of depressive symptoms. "
                        "This level often includes more pronounced and disruptive symptoms that markedly affect life, "
                        "such as significant fatigue, feelings of worthlessness, and difficulty concentrating."
        },
        4: {
            "label": "Level 4 (Severe Depressive Symptoms)",
            "description": "Scores range from 20 to 24, indicating severe depression. "
                        "This level is characterized by intense, debilitating symptoms that can include extreme sadness, "
                        "suicidal thoughts, and significant impairment in daily functioning."
        }
    }

    return level_mapping.get(predicted_class, {"label": "Unknown", "description": "No data available."})

# Route to handle file uploads and generate MFCC
@app.route("/generate_mfcc", methods=["POST"])
def generate_mfcc():
    if "audioFile" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    audio_file = request.files["audioFile"]
    original_filename = os.path.splitext(audio_file.filename)[0]  # Extract filename without extension
    filename = f"{original_filename}.wav"  # Save as .wav

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(file_path)

    # Generate MFCC
    mfcc_filename = save_mfcc_image(file_path, original_filename)

    # Delete the uploaded file after processing
    os.remove(file_path)

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

    return jsonify({"success": True, "label": prediction["label"], "description": prediction["description"]})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
