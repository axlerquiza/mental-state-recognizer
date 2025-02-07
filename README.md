
# Mental State Recognizer

This project predicts a person's mental state based on their speech using a deep learning model. It utilizes a hybrid model combining **BiLSTM** and **CNN** to analyze features extracted from audio files.

## Technologies Used

- **Python**: Primary language for development
- **Flask**: For backend and serving the model
- **TensorFlow**: For model building and inference
- **librosa**: For audio processing (MFCC extraction)
- **NumPy, SciPy**: For mathematical operations
- **scikit-learn**: For evaluation and metrics

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/axlerquiza/mental-state-recognizer.git
   cd mental-state-recognizer
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows, use `venv\Scriptsctivate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask server**:
   ```bash
   python server.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000/` to start using the web interface.

## Project Structure

The project follows this directory structure:

```
/mental-state-recognizer
├── /assets
│   ├── /scripts            # JavaScript functionalities
│   │   ├── app.js
│   ├── /styles
│   │   ├── style.css       # CSS styling
├── /models                 # Folder containing the trained models
├── /mfcc_images            # Folder for storing MFCC images (generated from audio)
├── /preprocessed_audio     # Folder with sample preprocessed audio files
│   ├── 308_AUDIO_processed.wav  # Level 4 example
│   ├── 346_AUDIO_processed.wav  # Level 1 example
├── /uploads                # Folder for storing uploaded audio files
├── index.html              # Main HTML file for the web interface
├── server.py               # Flask backend server script
└── requirements.txt        # List of dependencies
```

## Usage

1. **Upload an audio file** (preferably a 10-second speech recording) via the web interface (`index.html`).
2. The backend processes the audio file, extracting MFCC features and passing them through the trained model.
3. The model predicts the mental state, and the result is displayed on the webpage.

## Example

1. Click the "Upload Audio" button on the web interface.
2. Choose a 10-second audio file (preferably a speech recording).
3. Wait for the model to process the file and display the predicted mental state.

### Sample Audio Files

The `/preprocessed_audio` directory contains sample audio files:

- `308_AUDIO_processed.wav`: A **Level 4** example (indicating a specific mental state).
- `346_AUDIO_processed.wav`: A **Level 1** example (indicating a different mental state).

These files can be used for testing the system or understanding different levels of mental states.

