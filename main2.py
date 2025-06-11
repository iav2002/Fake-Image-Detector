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

