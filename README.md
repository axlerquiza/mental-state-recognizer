
# Mental State Recognizer (GUI Version)

This project predicts a person's mental state based on their speech using a deep learning model. It utilizes a hybrid model combining **BiLSTM** and **CNN** to analyze features extracted from audio files.

## Technologies Used

- **Python**: Primary language for development
- **Tkinter**: For building the graphical user interface (GUI)
- **TensorFlow**: For model building and inference
- **librosa**: For audio processing (MFCC extraction)
- **NumPy**: For numerical operations
- **scikit-learn**: For evaluation and metrics
- **Pillow (PIL)**: For image handling and MFCC image display

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/axlerquiza/mental-state-recognizer.git
   cd mental-state-recognizer
   ```

1. **Switch to the `gui` branch**:
   ```bash
    git switch gui
   ```

3. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows, use `venv\Scripts\activate`
   ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the GUI application**:
   ```bash
   python main.py
   ```

   This will open a Tkinter window where you can upload audio files and predict the mental state.

## Project Structure

The project follows this directory structure:

```
/mental-state-recognizer
├── /mfcc_images            # Folder for storing MFCC images (generated from audio)
├── /models                 # Folder containing the trained models
├── /preprocessed_audio     # Folder with sample preprocessed audio files
│   ├── 308_AUDIO_processed.wav  # Level 4 example
│   ├── 346_AUDIO_processed.wav  # Level 1 example
├── main.py                 # Main Python script to run the GUI
└── requirements.txt        # List of dependencies
```

## Usage

1. **Open the GUI** by running `python main.py`.
2. **Upload an audio file** (preferably a 10-second speech recording) via the file dialog.
3. The model will process the audio file, extract MFCC features, and display the predicted mental state level on the GUI.

## Example

1. Click the "Upload your Audio File" button to select an audio file (e.g., a 10-second speech recording).
2. The system will display the MFCC image and predict the mental state.
3. The result will be shown with a description of the predicted mental state (e.g., Level 0, Level 1, etc.).

### Sample Audio Files

The `/preprocessed_audio` directory contains sample audio files:

- `308_AUDIO_processed.wav`: A **Level 4** example (indicating a specific mental state).
- `346_AUDIO_processed.wav`: A **Level 1** example (indicating a different mental state).

These files can be used for testing the system or understanding different levels of mental states.
