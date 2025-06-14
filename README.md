# AI Image Detector Project

This project develops and deploys a deep learning model to classify images as either authentically real or generated by Artificial Intelligence. 
This README provides links to project resources and instructions on how to run the deployed Streamlit application locally.

## Live Demo
<img src="demo.gif">

### Important Project Links

* **Presentation & Demonstration Link:**
    [https://drive.google.com/file/d/16QUFtFIM4G7SNz3Ql5FDXOQTHz8OpVS3/view?usp=sharing](https://drive.google.com/file/d/16QUFtFIM4G7SNz3Ql5FDXOQTHz8OpVS3/view?usp=sharing)

* **Dataset (Original):**
    [https://drive.google.com/file/d/1-O7H-fG_kzTKHGad4L0FbtNJnNr7qGgo/view?usp=sharing](https://drive.google.com/file/d/1-O7H-fG_kzTKHGad4L0FbtNJnNr7qGgo/view?usp=sharing)

* **Dataset (Restructured for this project - `OrganizedDeepGuardDB_V2`):**
    [https://drive.google.com/drive/folders/1g-pcQhOhT_eCKO5gYXh32S9oNtMkmcI-?usp=sharing](https://drive.google.com/drive/folders/1g-pcQhOhT_eCKO5gYXh32S9oNtMkmcI-?usp=sharing)

* **Trained Model File** (The model file is hosted on Google Drive as it's too large for direct upload to the GitHub repository):
    [https://drive.google.com/file/d/1R0xas4a9drGGF1DWTCMXyaYJrQskvRHS/view?usp=sharing](https://drive.google.com/file/d/1R0xas4a9drGGF1DWTCMXyaYJrQskvRHS/view?usp=sharing)

---

## Running the Streamlit Application Locally

This application uses a deep learning model to predict if an image is AI-generated or Real.

### How to Run:

1.  **Get the Files:**

    * Download or clone the project repository.
    * Navigate to the `deployment` folder. This folder should contain:

        * `main2.py` (the Streamlit app script)
        * The downloaded model file (e.g., `ai_detector_resnet50_final.keras`). Ensure the model file from the Google Drive link above is placed here.

2.  **Open Terminal/Command Prompt:**

    * Navigate into the `deployment` folder using the `cd` command. For example:
        ```bash
        cd path/to/your/project/deployment
        ```

3.  **Install Libraries:**
    * In your terminal, run:
        ```bash
        pip install streamlit tensorflow Pillow numpy 
        ```

4.  **Update Model Path in `main2.py`:**

    * Open the `main2.py` file with a text editor.
    * Find the line: `MODEL_PATH = 'your_model_name.keras'` 

    * **Ensure this path correctly points to your model file within the `deployment` folder.** For example, if your model is named `ai_detector_resnet50_final.keras` and it's in the same folder as `main2.py`, the line should be:
        `MODEL_PATH = 'ai_detector_resnet50_final.keras'`

5.  **Run the App:**
    * In your terminal on the deployment folder, run:
        ```bash
        streamlit run main2.py
        ```
    * The application should automatically open in your web browser  at `http://localhost:8501`.

---
