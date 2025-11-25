import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf # TensorFlow is now essential for model loading and preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

# page configuration
st.set_page_config(
    page_title="Ai-Generated vs. Real Image Classification",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# IMPORTANT: Update this path to where your model is saved locally
MODEL_PATH = 'my_ai_detector_resnet50.keras' 
IMAGE_SIZE = (224, 224)  # Should match the input size your model was trained with
CLASS_NAMES = ['AI-Generated (Fake)', 'Real'] # Ensure 0: Fake, 1: Real matches your model's output logic

@st.cache_resource # Caches the loaded model for efficiency
def load_my_model(model_path):

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from: {model_path}") # For console feedback
        # You can show a success message in the UI once, e.g., in the sidebar
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error(f"Please ensure the model file exists at the specified path: {model_path} "
                 f"and is a valid Keras model file.")
        return None

def preprocess_image_for_resnet50(image_pil, target_size):
    """Preprocesses the PIL image for ResNet50 model prediction."""
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    image_pil_resized = image_pil.resize(target_size)
    image_array = np.array(image_pil_resized) # Shape: (H, W, 3), Range: [0, 255]
    
    # Add batch dimension
    image_array_expanded = np.expand_dims(image_array, axis=0) # Shape: (1, H, W, 3)
    
    # Apply ResNet50-specific preprocessing (expects float32)
    processed_image = preprocess_input_resnet50(image_array_expanded.astype('float32')) 
    
    return processed_image

# --- Load the Model ---
model = load_my_model(MODEL_PATH)

# --- Streamlit App UI ---
st.title("Deep Learning Model for Ai-Generated vs. Real Image Classification")
st.markdown(
    "Upload an image (`jpg`, `jpeg`,`png`), and the application will use a "
    "pre-trained ResNet50-based model to predict if it's likely **Real** or **AI-Generated (Fake)**."
)
st.markdown("---")

uploaded_file = st.file_uploader("📁 Choose an image file:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    
    col1, col2 = st.columns([0.6, 0.4]) # 60% for image, 40% for analysis

    with col1:
        st.subheader("🖼️ Uploaded Image")
        st.image(image_pil, caption="Your Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Analysis & Prediction")
        
        st.write("Processing image for the model...")
        processed_image = preprocess_image_for_resnet50(image_pil, IMAGE_SIZE)

        with st.spinner('Classifying, please wait...'):
            try:
                prediction_probs = model.predict(processed_image)
                prob_real = prediction_probs[0][0] # Assuming output is P(Class 1 = Real)

                if prob_real > 0.5:
                    predicted_class_idx = 1 # Real
                    confidence = prob_real
                else:
                    predicted_class_idx = 0 # AI-Generated 
                    confidence = 1 - prob_real # Confidence in the predicted class
                
                predicted_class_name = CLASS_NAMES[predicted_class_idx]

                st.markdown("---") # Visual separator
                if predicted_class_name == CLASS_NAMES[1]: # 'Real'
                    st.markdown(f"<h4 style='text-align: center; color:green;'>✅ Prediction: <strong>{predicted_class_name}</strong></h4>", unsafe_allow_html=True)
                else: # 'AI-Generated (Fake)'
                    st.markdown(f"<h4 style='text-align: center; color:red;'>⚠️ Prediction: <strong>{predicted_class_name}</strong></h4>", unsafe_allow_html=True)
                
                st.metric(label="Model Confidence", value=f"{confidence:.2%}")
                st.info("Note: This is an estimated prediction. Please use critical judgment.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        
    st.markdown("---") # Visual separator before expander
    with st.expander("ℹ️ Understanding the Results & Limitations"):
        st.markdown("""
        * **Confidence Score:** This indicates the model's level of certainty in its prediction for the displayed class.
        * **AI-Generated (Fake):** The model predicts the image exhibits characteristics typical of AI-generated content it has learned.
        * **Real:** The model predicts the image aligns with characteristics of authentic photographs.
        * **Limitations:** This model is a demonstration tool. It has been trained on specific datasets (e.g., `DeepGuardDB`) and may not detect all forms or nuances of AI-generated imagery, nor perfectly classify every real image. Its accuracy is not 100%.
        """)

elif uploaded_file is None:
    st.info("Please upload an image using the file uploader above to start.")


import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
import os

# --- Page Config ---
st.set_page_config(
    page_title="Ai-Generated vs. Real Detector",
    page_icon="/Users/ignacioalarconvarela/Developer/AI-Image-Detector/Deployment/logo.png",
    layout="wide"
)

# --- Path Setup ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'my_ai_detector_resnet50.keras')
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'samples')
FILES_DIR = os.path.join(os.path.dirname(__file__), 'files')
IMAGE_SIZE = (224, 224)

# --- Custom CSS ---
st.markdown("""
    <style>
    /* Global Font Increase */
    html, body, [class*="css"] {
        font-size: 18px !important;
    }
    
    /* Creators Section Styling */
    .creators-text {
        font-size: 20px !important;
        font-weight: bold;
        color: #666;
        margin-bottom: 20px;
    }
    /* Card Common Styles */
    .prediction-card {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Real (Green) Card */
    .real-card {
        background-color: rgba(40, 167, 69, 0.15); 
        border: 2px solid #28a745;
        color: #1e7e34;
    }
    /* Fake (Red) Card */
    .fake-card {
        background-color: rgba(220, 53, 69, 0.15); 
        border: 2px solid #dc3545;
        color: #bd2130;
    }
    /* Typography inside cards */
    .card-title {
        font-size: 24px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .card-result {
        font-size: 50px !important;
        font-weight: 800 !important;
        margin: 10px 0;
    }
    .card-confidence {
        font-size: 28px !important;
        font-weight: bold;
    }

    /* --- UI GAP FIXES --- */

    /* 1. Fix the vertical gap in the Main Content Area */
    .block-container {
        padding-top: 2rem; /* Reduces default gap at the top */
    }

    /* 2. Fix the vertical gap in the Sidebar (Aggressive Fix) */
    [data-testid="stSidebarContent"] {
        padding-top: 0px !important; 
        margin-top: -60px !important; /* Pulls content up to remove fixed margin */
    }

    </style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_my_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Preprocessing ---
def preprocess_image_for_resnet50(image_pil, target_size):
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    image_pil_resized = image_pil.resize(target_size)
    image_array = np.array(image_pil_resized)
    image_array_expanded = np.expand_dims(image_array, axis=0)
    processed_image = preprocess_input_resnet50(image_array_expanded.astype('float32'))
    return processed_image

model = load_my_model(MODEL_PATH)


# --- Sidebar: Project Summary & Files ---
with st.sidebar:
    st.header("Project Background")
    
    st.markdown("""
    **The Generative AI Challenge**
    
    We developed this model thinking of the possible future consequences that we will be facing with Gen-AI. 
    As synthetic media becomes indistinguishable from reality, the need for automated verification tools is critical.
    
    """)
    
    st.markdown("---")

    # 1. SAMPLE SELECTOR
    st.markdown(
        "<h4 style='font-size: 20px; font-weight: bold; margin-bottom: 5px;'>Test the Model:</h4>", 
        unsafe_allow_html=True
    )
    st.markdown("Try a sample image below:")
    
    sample_choice = st.selectbox(
        " ", # Empty label, relying on the markdown above for the title
        ["None", "Sample Real (Organic)", "Sample AI (Synthetic)"],
        label_visibility="collapsed" # Hide the default label area completely
    )
    
    st.markdown("---")
    
    # 2. DOWNLOAD BUTTONS
    st.markdown("### 📥 Resources")
    
    # Download Buttons Logic
    report_path = os.path.join(FILES_DIR, 'report.pdf')
    poster_path = os.path.join(FILES_DIR, 'poster.pdf')

    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button(
                label="📄 Read Full Report",
                data=f,
                file_name="AI_Detection_Report.pdf",
                mime="application/pdf"
            )
    
    if os.path.exists(poster_path):
        with open(poster_path, "rb") as f:
            st.download_button(
                label="🖼️ View Project Poster",
                data=f,
                file_name="AI_Detection_Poster.pdf",
                mime="application/pdf"
            )



# --- Main Layout ---

top_col1, top_col2 = st.columns([0.85, 0.15])

with top_col1:
    st.title("🕵️ AI-Generated vs. Real Image Classification")

with top_col2:
    # Theme Toggle Logic
    current_theme = st.get_option("theme.base")
    toggle_btn = st.button("🌓 Theme")
    if toggle_btn:
        if current_theme == "dark":
            st._config.set_option("theme.base", "light")
        else:
            st._config.set_option("theme.base", "dark")
        st.rerun()

# Creators Section
st.markdown(
    """
    <p class='creators-text'>
    Developed by 
    <a href='https://www.linkedin.com/in/ignacioalarcon/' target='_blank' style='text-decoration: none;'>Ignacio Alarcon</a> & 
    <a href='https://www.linkedin.com/feed/' target='_blank' style='text-decoration: none;'>Bernardo Gandara</a>
    </p>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
    Upload an image to inspect digital artifacts and determine authenticity. 
    **We encourage you to test the model's accuracy against synthetic content, 
    such as images autogenerated by this site: 
    [thispersondoesnotexist.com](https://thispersondoesnotexist.com/)**
    """
)
# --- Image Loading ---
uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png"])
active_image = None

# --- CRASH PROTECTION LOGIC ---
if uploaded_file:
    try:
        active_image = Image.open(uploaded_file)
    except Exception as e:
        st.error("⚠️ Format not supported. Please upload a valid image file (JPG, PNG).")
        # We leave active_image as None, so the code below won't run.
        
elif sample_choice != "None":
    if sample_choice == "Sample Real (Organic)":
        file_path = os.path.join(SAMPLE_DIR, "real.jpg")
    else:
        file_path = os.path.join(SAMPLE_DIR, "fake.jpg")
    
    if os.path.exists(file_path):
        active_image = Image.open(file_path)

# --- Analysis Logic ---
if active_image:
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.image(active_image, caption="Input Image", use_container_width=True)

    with col2:
        st.subheader("Analysis Results")
        
        with st.status("Scanning image artifacts...", expanded=True) as status:
            st.write("Preprocessing image (ResNet50 standard)...")
            processed_image = preprocess_image_for_resnet50(active_image, IMAGE_SIZE)
            
            st.write("Running inference...")
            prediction_probs = model.predict(processed_image)
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        prob_real = float(prediction_probs[0][0])
        
        if prob_real > 0.5:
            pred_class = "REAL"
            sub_text = "Organic Photography"
            confidence_val = f"{prob_real:.2%}"
            
            st.markdown(f"""
            <div class="prediction-card real-card">
                <div class="card-title">Prediction Result</div>
                <div class="card-result">✅ {pred_class}</div>
                <div class="card-confidence">Confidence: {confidence_val}</div>
                <p style="margin-top: 10px;">{sub_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            pred_class = "FAKE"
            sub_text = "AI-Generated / Synthetic"
            confidence_val = f"{(1 - prob_real):.2%}"
            
            st.markdown(f"""
            <div class="prediction-card fake-card">
                <div class="card-title">Prediction Result</div>
                <div class="card-result">⚠️ {pred_class}</div>
                <div class="card-confidence">Confidence: {confidence_val}</div>
                <p style="margin-top: 10px;">{sub_text}</p>
            </div>
            """, unsafe_allow_html=True)

elif not active_image and not uploaded_file:
    # Only show this instruction if nothing is uploaded and no error occurred
    st.info("👈 Upload an image or select a sample from the sidebar to start.")


#to run nacho : streamlit run /Users/ignacioalarconvarela/Developer/AI-Image-Detector/Deployment/main2.py