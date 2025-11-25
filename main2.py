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
    
    This project leverages **ResNet50** to identify subtle digital artifacts that human eyes often miss.
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