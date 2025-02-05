import tkinter
import tkinter.messagebox
import tkinter as tk
import tkinter.filedialog as filedialog
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import customtkinter
import soundfile as sf
import noisereduce as nr
from keras.models import load_model # type: ignore
import numpy as np 
from PIL import Image, ImageTk
import io
import matplotlib.pyplot as plt

# Set up directories
MFCC_SAVE_DIRECTORY = "mfcc_images"
MODEL_DIRECTORY = "models"

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Callback function
# Pre-defined paths to your saved models
MODEL_PATHS = {
    "VGGNet16": os.path.join(MODEL_DIRECTORY, "VGGNet16.h5"),
    "ResNet50": os.path.join(MODEL_DIRECTORY, "ResNet50.h5"),
    "InceptionV3": os.path.join(MODEL_DIRECTORY, "InceptionV3.h5"),
    "Hybrid VGGNet16": os.path.join(MODEL_DIRECTORY, "Hybrid_VGGNet16.h5"),
    "Hybrid ResNet50": os.path.join(MODEL_DIRECTORY, "Hybrid_ResNet50.h5"),
    "Hybrid InceptionV3": os.path.join(MODEL_DIRECTORY, "Hybrid_InceptionV3.h5")
}

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Mental State Recognition through Speech Analysis")
        self.geometry(f"{1300}x{800}")

        self.label = customtkinter.CTkLabel(self,
                                       text="Mental State Recognition through Speech Analysis Using\nBidirectional Long Short-Term Memory Network and Convolutional Neural Network Hybrid Model",
                                       fg_color="transparent",
                                       font=("Helvetica", 20, "bold"))  # Adjust font family, size, and weight here
        self.label.pack(padx=30, pady=(20, 0))

        self.label = customtkinter.CTkLabel(self,
                                       text="University of Santo Tomas College of Information and Computing Sciences",
                                       fg_color="transparent",
                                       font=("Helvetica", 14))  # Adjust font family, size, and weight here
        self.label.pack(padx=20, pady=(0, 0))

        self.label = customtkinter.CTkLabel(self,
                                      text="Erquiza, Mamaclay, Platon (2023)",
                                      fg_color="transparent",
                                      font=("Helvetica", 14, "italic"))  # Adjust font family, size, and weight here
        self.label.pack(padx=20, pady=(0, 0))

        self.frame_1 = customtkinter.CTkFrame(master=self)
        self.frame_1.pack(pady=(20,70), padx=30, fill="both", expand=True)

        self.label = customtkinter.CTkLabel(self.frame_1,
                                       text="Select Model Architecture:",
                                       fg_color="transparent",
                                       font=("Helvetica", 14))  # Adjust font family, size, and weight here
        self.label.pack(padx=20, pady=(30, 0))

        self.combobox = customtkinter.CTkOptionMenu(self.frame_1, values=["VGGNet16", "ResNet50", "InceptionV3", "Hybrid VGGNet16", "Hybrid ResNet50", "Hybrid InceptionV3"], width=500, fg_color="gray")
        self.combobox.pack(pady=10, padx=0)
        self.combobox.set("Choose your preferred model")

        # Add a progress bar
        self.progress_bar = customtkinter.CTkProgressBar(self.frame_1)
        self.progress_bar.pack(pady=(10, 0), padx=20)
        self.progress_bar.set(0)
        # progress_bar.grid_remove()  # Initially hide the progress bar

        self.label_filename = customtkinter.CTkLabel(self.frame_1, text="No file selected", fg_color="transparent")
        self.label_filename.pack(padx=20, pady=(10,0))

        self.button_1 = customtkinter.CTkButton(master=self.frame_1, command=self.upload_audio_file, text="Upload your Audio File",fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.button_1.pack(pady=(10,0), padx=10)

        self.label = customtkinter.CTkLabel(self.frame_1,
                                       text="Upload a 10-second Audio File(*.mp3 *.wav *.aac)",
                                       fg_color="transparent",
                                       font=("Helvetica", 8, "italic"))  # Adjust font family, size, and weight here
        self.label.pack(padx=20, pady=(0, 0))
        

         # Add a label for the MFCC image
        self.mfcc_image_label = customtkinter.CTkLabel(master=self.frame_1, image=None, text="")
        self.mfcc_image_label.pack(pady=(10, 0), padx=20)

        self.textbox = customtkinter.CTkTextbox(master=self.frame_1, width=800, height=150,text_color="white")
        self.textbox.pack(pady=10, padx=10)

        # Set the medical disclaimer as the initial text
        disclaimer_text = (
            "Medical Disclaimer\n\n"
            "This program is a supplemental tool for mental health assessment and is not intended for standalone diagnostic use. "
            "Mental health diagnostics are complex and should always involve qualified healthcare professionals. Users are advised "
            "to seek professional medical advice for any concerns. The results from this software support, but do not replace, "
            "the judgment of healthcare providers. The use of this program indicates understanding and acceptance of these limitations.\n\n"
            "Consult a healthcare provider for all medical inquiries."
        )

        # Prediction level descriptions
        self.level_descriptions = {
            "Predicted Class: Level 0": (
                "Level 0 (No Significant Depressive Symptoms): \n\n" 
                "Scores range from 0 to 4, indicating minimal or no depressive symptoms. "
                "Individuals in this category generally do not exhibit signs of depression."
            ),
            "Predicted Class: Level 1": (
                "Level 1 (Mild Depressive Symptoms): \n\n" 
                "Scores range from 5 to 9, representing mild levels of depression. "
                "Symptoms at this level may include slight changes in mood, sleep, and energy, but they typically do not significantly impair daily functioning."
            ),
            "Predicted Class: Level 2": (
                "Level 2 (Moderate Depressive Symptoms): \n\n" 
                "Scores range from 10 to 14, indicating moderate depression. "
                "Individuals may experience more noticeable symptoms that can start to impact daily activities, like persistent sadness, decreased interest in activities, and changes in appetite or sleep patterns."
            ),
            "Predicted Class: Level 3": (
                "Level 3 (Moderately Severe Depressive Symptoms): \n\n" 
                "Scores range from 15 to 19, reflecting a higher intensity of depressive symptoms. "
                "This level often includes more pronounced and disruptive symptoms that markedly affect life, such as significant fatigue, feelings of worthlessness, and difficulty concentrating."
            ),
            "Predicted Class: Level 4": (
                "Level 4 (Severe Depressive Symptoms): \n\n" 
                "Scores range from 20 to 24, indicating severe depression. "
                "This level is characterized by intense, debilitating symptoms that can include extreme sadness, suicidal thoughts, and significant impairment in daily functioning."
            )
        }

        self.textbox.insert("0.0", disclaimer_text)
        self.textbox.configure(state="disabled")  # configure textbox to be read-only

        # Add a frame to hold the buttons
        self.button_frame = customtkinter.CTkFrame(master=self.frame_1, border_color=None,fg_color="transparent")
        self.button_frame.pack(pady=(10, 0), padx=20)

        # Reset button
        self.button_reset = customtkinter.CTkButton(master=self.button_frame, text="Reset", command=self.reset_app)
        self.button_reset.pack(side="left", pady=10, padx=10)

        # Button to predict
        self.button_predict = customtkinter.CTkButton(master=self.button_frame, text="Predict Level of Depression", command=self.button_predict_callback)
        self.button_predict.pack(side="left",pady=10, padx=10)

        # Button to display MFCC Image
        self.button_display_mfcc = customtkinter.CTkButton(master=self.button_frame, text="Display MFCC Image", command=self.button_display_mfcc_callback)
        self.button_display_mfcc.pack(side="left", pady=10, padx=10)

        # Add the toggle switch for appearance mode
        self.appearance_mode_switch = customtkinter.CTkSwitch(self, text="Dark Mode", command=self.toggle_appearance_mode)
        self.appearance_mode_switch.place(relx=1.0, rely=1.0, anchor="se", x=-30, y=-30)

        # Set the initial state of the switch
        # Set the switch to the 'on' state by default
        self.appearance_mode_switch.select()
        customtkinter.set_appearance_mode("dark")  # Set the appearance mode to dark initially

    def reset_app(self):
        # Clear the textbox
        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", "end")
        disclaimer_text = (
            "Medical Disclaimer\n\n"
            "This program is a supplemental tool for mental health assessment and is not intended for standalone diagnostic use. "
            "Mental health diagnostics are complex and should always involve qualified healthcare professionals. Users are advised "
            "to seek professional medical advice for any concerns. The results from this software support, but do not replace, "
            "the judgment of healthcare providers. The use of this program indicates understanding and acceptance of these limitations.\n\n"
            "Consult a healthcare provider for all medical inquiries."
        )
        self.textbox.insert("1.0", disclaimer_text)
        self.textbox.configure(state="disabled")

        # Reset the label filename
        self.label_filename.configure(text="No file selected")

        # Reset the combobox to initial state
        self.combobox.set("Choose your preferred model")

        # Remove MFCC image if it exists
        if hasattr(self, 'mfcc_image_label'):
            self.mfcc_image_label.configure(image='')

        # Reset the progress bar
        self.progress_bar.set(0)

        # Remove the uploaded file path
        if hasattr(self, 'uploaded_file_path'):
            del self.uploaded_file_path
    
    
    def display_mfcc_image(self, image_path):
        # Load the image using PIL
        img = Image.open(image_path)
        img = img.resize((250, 75))  # Resize for display, adjust as needed
        img_tk = ImageTk.PhotoImage(img)

        # Update the label
        self.mfcc_image_label.configure(image=img_tk)
        self.mfcc_image_label.image = img_tk  # Keep a reference to avoid garbage collection

    def button_display_mfcc_callback(self):
        if not hasattr(self, 'uploaded_file_path'):
            tk.messagebox.showerror("Error", "No audio file selected")
            return

        # Generate MFCC and get the image path
        mfcc_image_path = self.save_mfcc_image(self.uploaded_file_path)

        # Display the MFCC image
        self.display_mfcc_image(mfcc_image_path)
    
    
    def toggle_appearance_mode(self):
        if self.appearance_mode_switch.get() == 1:  # If switch is on
            customtkinter.set_appearance_mode("dark")
            self.textbox.configure(text_color="white")
        else:  # If switch is off
            customtkinter.set_appearance_mode("light")
            self.textbox.configure(text_color="black")

    def upload_audio_file(self):
        # Display a loading message
        # Show the progress bar
        # progress_bar.grid()
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()

        self.label_filename.configure(text="Uploading...")
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.aac")])
        if file_path:  # If a file was selected
            self.uploaded_file_path = file_path  # Set the uploaded file path
            self.label_filename.configure(text=f"File Uploaded: {file_path.split('/')[-1]}")

            self.progress_bar.start()
            self.progress_bar.configure(mode="determinate", determinate_speed=1)
            self.progress_bar.set(1)
            self.progress_bar.stop()
        else:
            # Reset the label if no file was selected
            self.label_filename.configure(text="No file selected")

            self.progress_bar.start()

    def preprocess_audio(self,file_path, target_duration=10.0):
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)

        # Crop the audio file to the target duration
        if librosa.get_duration(y=y, sr=sr) > target_duration:
            y = y[:int(sr * target_duration)]

        # Apply noise reduction
        y_reduced_noise = nr.reduce_noise(y=y, sr=sr)

        # Apply noise reduction
        y_reduced_noise = nr.reduce_noise(y=y, sr=sr)

        return y_reduced_noise, sr

    # Function to load and return a model
    def load_model_for_prediction(self, model_name):
        model_path = MODEL_PATHS.get(model_name)
        if model_path:
            return load_model(model_path)
        else:
            raise ValueError(f"Model {model_name} not found.")

    def save_mfcc_image(self, file_path, n_mfcc=13):
        """Compute MFCC for a given audio file and save it as an image."""
        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Create the MFCC plot
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sr, x_axis=None, y_axis=None)
        plt.gca().axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        filename = os.path.basename(file_path)
        save_filename = os.path.splitext(filename)[0] + '.png'
        save_path = os.path.join(MFCC_SAVE_DIRECTORY, save_filename)

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return save_path

    def predict_with_model(self, img_path, model_name):
        model_path = MODEL_PATHS.get(model_name)
        if not model_path:
            raise ValueError(f"Model {model_name} not found.")

        model = load_model(model_path)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((1000, 400))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        level_mapping = {0: "Level 0", 1: "Level 1", 2: "Level 2", 3: "Level 3", 4: "Level 4"}
        predicted_level = level_mapping.get(predicted_class, "Unknown")

        return f"Predicted Class: {predicted_level}"

    def button_predict_callback(self):
        selected_model = self.combobox.get()
        if not hasattr(self, 'uploaded_file_path'):
            tk.messagebox.showerror("Error", "No audio file selected")
            return

        # Generate MFCC and get the image path
        mfcc_image_path = self.save_mfcc_image(self.uploaded_file_path)

        # Predict with the selected model
        prediction = self.predict_with_model(mfcc_image_path, selected_model)

        # Get the level description
        level_description = self.level_descriptions.get(prediction, "Unknown level description")

        # Display the prediction and corresponding description
        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", prediction + "\n\n" + level_description)
        self.textbox.configure(state="disabled")

if __name__ == "__main__":
    app = App()
    app.mainloop()
