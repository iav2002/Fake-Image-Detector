import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# --- Page Config ---
st.set_page_config(
    page_title="Ai-Generated vs. Real Detector",
    page_icon="./logo.png",
    layout="wide"
)

# --- Path Setup ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'my_ai_detector_resnet50.keras')
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'samples')
FILES_DIR = os.path.join(os.path.dirname(__file__), 'files')
IMAGE_SIZE = (224, 224)

# Target conv layer for Grad-CAM (last conv block in ResNet50)
LAST_CONV_LAYER = "conv5_block3_out"

# --- Custom CSS ---
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 18px !important;
    }
    .creators-text {
        font-size: 20px !important;
        font-weight: bold;
        color: #666;
        margin-bottom: 20px;
    }
    .prediction-card {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .real-card {
        background-color: rgba(40, 167, 69, 0.15); 
        border: 2px solid #28a745;
        color: #1e7e34;
    }
    .fake-card {
        background-color: rgba(220, 53, 69, 0.15); 
        border: 2px solid #dc3545;
        color: #bd2130;
    }
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
    .block-container {
        padding-top: 2rem;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 0px !important; 
        margin-top: -60px !important;
    }
    .xai-header {
        font-size: 22px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .xai-description {
        font-size: 14px;
        color: #888;
        margin-bottom: 15px;
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


# =====================
# XAI FUNCTIONS
# =====================

def generate_gradcam(model, img_array, layer_name):
    """Grad-CAM: highlights which broad regions activated the model most."""
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        results = grad_model(img_array)
        conv_output = tf.convert_to_tensor(results[0])
        tape.watch(conv_output)
        predictions = tf.convert_to_tensor(results[1])
        pred_value = tf.reshape(predictions, [-1])

    grads = tape.gradient(pred_value, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def generate_occlusion(model, img_array, patch_size=32, stride=5):
    """
    Occlusion Sensitivity: covers patches of the image with gray
    and measures how the prediction changes per region.
    Faster settings for live app (~6-7 seconds).
    """
    _, h, w, c = img_array.shape
    base_pred = float(tf.squeeze(model(img_array)).numpy())

    out_h = (h - patch_size) // stride + 1
    out_w = (w - patch_size) // stride + 1
    sensitivity = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            occluded = img_array.copy()
            y = i * stride
            x = j * stride
            occluded[0, y:y+patch_size, x:x+patch_size, :] = 128.0

            occ_pred = float(tf.squeeze(model(occluded)).numpy())
            sensitivity[i, j] = base_pred - occ_pred

    s_min, s_max = sensitivity.min(), sensitivity.max()
    if s_max - s_min > 1e-8:
        sensitivity = (sensitivity - s_min) / (s_max - s_min)
    return sensitivity


def create_overlay(original_pil, heatmap, alpha=0.4):
    """Create a colored heatmap overlay on the original image."""
    # Resize heatmap to match original image
    heatmap_resized = tf.image.resize(
        np.expand_dims(np.expand_dims(heatmap, 0), -1),
        (original_pil.size[1], original_pil.size[0])
    ).numpy()[0, :, :, 0]

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    original_arr = np.array(original_pil) / 255.0
    overlay = jet_heatmap * alpha + original_arr * (1 - alpha)
    return np.clip(overlay, 0, 1)


model = load_my_model(MODEL_PATH)

# --- Sidebar ---
with st.sidebar:
    st.header("Project Background")
    st.markdown("""
    **The Generative AI Challenge**
    
    We developed this model thinking of the possible future consequences that we will be facing with Gen-AI. 
    As synthetic media becomes indistinguishable from reality, the need for automated verification tools is critical.
    """)
    st.markdown("---")

    st.markdown(
        "<h4 style='font-size: 20px; font-weight: bold; margin-bottom: 5px;'>Test the Model:</h4>", 
        unsafe_allow_html=True
    )
    st.markdown("Try a sample image below:")
    sample_choice = st.selectbox(
        " ",
        ["None", "Sample Real (Organic)", "Sample AI (Synthetic)"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    st.markdown("### 📥 Resources")
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
    st.title("AI-Generated vs. Real Image Classification")
with top_col2:
    current_theme = st.get_option("theme.base")
    toggle_btn = st.button("🌓 Theme")
    if toggle_btn:
        if current_theme == "dark":
            st._config.set_option("theme.base", "light")
        else:
            st._config.set_option("theme.base", "dark")
        st.rerun()

st.markdown(
    """
    <p class='creators-text'>
    Developed by 
    <a href='https://www.linkedin.com/in/ignacioalarcon/' target='_blank' style='text-decoration: none;'>Ignacio Alarcon</a> & 
    <a href='https://www.linkedin.com/in/bernardo-gandara/' target='_blank' style='text-decoration: none;'>Bernardo Gandara</a>
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

if uploaded_file:
    try:
        active_image = Image.open(uploaded_file)
    except Exception as e:
        st.error("⚠️ Format not supported. Please upload a valid image file (JPG, PNG).")
elif sample_choice != "None":
    if sample_choice == "Sample Real (Organic)":
        file_path = os.path.join(SAMPLE_DIR, "real.jpg")
    else:
        file_path = os.path.join(SAMPLE_DIR, "fake.jpg")
    if os.path.exists(file_path):
        active_image = Image.open(file_path)

# --- Analysis Logic ---
if active_image:
    # Convert to RGB if needed
    if active_image.mode != "RGB":
        active_image = active_image.convert("RGB")

    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.image(active_image, caption="Input Image")

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
            st.caption(
    "Disclaimer: This model is not 100% accurate. Treat the output as a probability and verify before making decisions."
)

    # =====================
    # XAI SECTION
    # =====================
    st.markdown("---")
    st.markdown("<div class='xai-header'>Explainability Analysis (XAI)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='xai-description'>"
        "Understanding what the model focuses on when making its prediction. "
        "Two complementary techniques reveal different levels of detail about the model's decision making process."
        "</div>",
        unsafe_allow_html=True
    )

    xai_col1, xai_col2 = st.columns(2, gap="large")

    # --- Grad-CAM ---
    with xai_col1:
        st.markdown("**Grad-CAM** - Region level activation")
        st.caption("Which broad areas of the image influenced the prediction most?")
        with st.spinner("Generating Grad-CAM..."):
            heatmap = generate_gradcam(model, processed_image, LAST_CONV_LAYER)
            gc_overlay = create_overlay(active_image, heatmap)
        st.image(gc_overlay, use_container_width=True)

    # --- Occlusion Sensitivity ---
    with xai_col2:
        st.markdown("**Occlusion Sensitivity** — Patch level importance")
        st.caption("What happens to the prediction when we block parts of the image?")
        with st.spinner("Running occlusion analysis (~7 seconds)..."):
            occ_map = generate_occlusion(model, processed_image)
            occ_overlay = create_overlay(active_image, occ_map)
        st.image(occ_overlay, use_container_width=True)

    # Brief explanation
    st.caption(
        "🔴 Red/warm regions = high importance for the prediction. "

        "🔵 Blue/cool regions = low importance. "
        "Grad-CAM shows broad activation patterns from the last convolutional layer, "
        "while Occlusion Sensitivity measures the actual prediction change when each region is hidden."
    )

elif not active_image and not uploaded_file:
    st.info("👈 Upload an image or select a sample from the sidebar to start.")
