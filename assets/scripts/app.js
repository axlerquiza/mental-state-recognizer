document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("audioUpload");
    const fileNameDisplay = document.getElementById("fileName");
    const generateMFCCBtn = document.getElementById("generateMFCC");
    const predictBtn = document.getElementById("predict");
    const mfccImage = document.getElementById("mfccImage");
    const assistantFrame = document.getElementById("assistant");

    let selectedFile = null;
    let mfccImageUrl = "";

    // Set the backend URL (Update this if needed)
    const BACKEND_URL = "http://127.0.0.1:5000";

    // Handle file selection
    fileInput.addEventListener("change", function (event) {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            fileNameDisplay.textContent = `File: ${selectedFile.name}`;
        } else {
            fileNameDisplay.textContent = "No file selected";
        }
    });

    // Handle MFCC Generation
    generateMFCCBtn.addEventListener("click", function () {
        if (!selectedFile) {
            alert("Please upload an audio file first.");
            return;
        }

        const formData = new FormData();
        formData.append("audioFile", selectedFile);

        fetch(`${BACKEND_URL}/generate_mfcc`, {
            method: "POST",
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                mfccImageUrl = `${BACKEND_URL}${data.mfccImageUrl}`;
                mfccImage.src = mfccImageUrl;
                mfccImage.style.display = "block";
            } else {
                alert("Error generating MFCC image.");
            }
        })
        .catch(error => {
            console.error("Error:", error);
            alert("Failed to connect to server.");
        });
    });

    // Handle Prediction
    predictBtn.addEventListener("click", function () {
        if (!mfccImageUrl) {
            alert("Generate MFCC image first.");
            return;
        }

        fetch(`${BACKEND_URL}/predict`, {
            method: "POST",
            body: JSON.stringify({ mfccImageUrl: mfccImageUrl.replace(BACKEND_URL, "") }),
            headers: {
                "Content-Type": "application/json"
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                assistantFrame.contentDocument.body.innerHTML = `<p>${data.prediction}</p>`;
            } else {
                alert("Error making prediction.");
            }
        })
        .catch(error => {
            console.error("Error:", error);
            alert("Failed to connect to server.");
        });
    });
});
