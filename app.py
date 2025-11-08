# Potato Leaf Disease Classification - Streamlit App
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Potato Leaf Disease Classifier",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.2em;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .disease-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .healthy {
        background-color: #C8E6C9;
        border-left: 5px solid #2E7D32;
    }
    .early-blight {
        background-color: #FFE0B2;
        border-left: 5px solid #F57C00;
    }
    .late-blight {
        background-color: #FFCCCC;
        border-left: 5px solid #C62828;
    }
    </style>
    """, unsafe_allow_html=True)

# CLASS NAMES (hardcoded - no dataset needed)
CLASS_NAMES = ['Early_Blight', 'Late_Blight', 'Healthy']

# CACHE & MODEL FUNCTIONS
@st.cache_resource
def load_model():
    """Load pre-trained model from file or Google Drive"""
    model_path = "potato_disease_model.h5"
    
    # Check if model exists locally
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"❌ Model file not found: {model_path}")
        st.info("""
        📌 **To use this app:**
        1. Train your model locally using the dataset
        2. Save the trained model as `potato_disease_model.h5`
        3. Upload the model file to your GitHub repository
        4. Deploy to Streamlit Cloud
        
        **Alternative:** Upload model to Google Drive and use the download option below.
        """)
        
        # Optional: Download from Google Drive
        gdrive_url = st.text_input(
            "Or paste Google Drive share link to download model:",
            help="Make sure the file is publicly accessible"
        )
        
        if gdrive_url and st.button("Download Model"):
            try:
                with st.spinner("Downloading model..."):
                    # Extract file ID from Google Drive URL
                    if 'drive.google.com' in gdrive_url:
                        file_id = gdrive_url.split('/d/')[1].split('/')[0]
                        download_url = f'https://drive.google.com/uc?id={file_id}'
                        gdown.download(download_url, model_path, quiet=False)
                        model = tf.keras.models.load_model(model_path)
                        st.success("✅ Model downloaded and loaded!")
                        st.rerun()
                        return model
            except Exception as e:
                st.error(f"Download failed: {e}")
        
        return None

def predict_image(image, model):
    """Make prediction on uploaded image"""
    # Prepare image
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0])) * 100
    
    return predicted_class, confidence, predictions[0]

def get_disease_info(disease_name):
    """Provide information about each disease"""
    disease_info = {
        'Healthy': {
            'description': '✅ Your potato plant is healthy!',
            'color': 'healthy',
            'emoji': '🌱',
            'details': 'No disease detected. Continue with regular care and monitoring.'
        },
        'Early_Blight': {
            'description': '⚠️ Early Blight Detected',
            'color': 'early-blight',
            'emoji': '🍂',
            'details': 'Early blight is a fungal disease caused by Alternaria solani. Apply fungicides and remove affected leaves. Ensure good air circulation.'
        },
        'Late_Blight': {
            'description': '🚨 Late Blight Detected',
            'color': 'late-blight',
            'emoji': '❌',
            'details': 'Late blight is a serious fungal disease caused by Phytophthora infestans. Apply appropriate fungicides immediately. Increase plant spacing for better air flow.'
        }
    }
    return disease_info.get(disease_name, {
        'description': 'Unknown',
        'color': 'healthy',
        'emoji': '❓',
        'details': 'Classification not recognized.'
    })

# MAIN APP
def main():
    # Header
    st.markdown('<div class="main-header">🥔 Potato Leaf Disease Classifier</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Identify diseases in potato plants using AI</div>', 
                unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Please load a model to continue.")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📸 Upload Image", divider="green")
        
        uploaded_file = st.file_uploader(
            "Choose a potato leaf image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of a potato leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Resize image for model
            image_resized = image.resize((256, 256))
            
            # Make prediction
            with st.spinner("🔍 Analyzing leaf..."):
                predicted_class, confidence, all_predictions = predict_image(
                    image_resized, model
                )
            
            # Display results in second column
            with col2:
                st.subheader("📊 Results", divider="green")
                
                # Get disease information
                disease_info = get_disease_info(predicted_class)
                
                # Display prediction with styling
                st.markdown(f"""
                    <div class="disease-box {disease_info['color']}">
                        <h3>{disease_info['emoji']} {disease_info['description']}</h3>
                        <p style='font-size: 1.1em; font-weight: bold;'>
                            Confidence: {confidence:.2f}%
                        </p>
                        <p>{disease_info['details']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show all predictions
                st.write("### 📈 All Predictions")
                pred_df = {
                    'Class': CLASS_NAMES,
                    'Confidence': [f"{conf*100:.2f}%" for conf in all_predictions]
                }
                st.dataframe(pred_df, use_container_width=True)
        else:
            with col2:
                st.info("👈 Upload an image to see predictions")
    
    # Additional info section
    st.divider()
    st.subheader("📚 About This Classifier")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        **🌱 Healthy Leaves**
        - Green color without spots
        - No visible lesions
        - Normal texture
        """)
    
    with info_col2:
        st.markdown("""
        **🍂 Early Blight**
        - Brown circular lesions
        - Concentric ring pattern
        - Starts from lower leaves
        """)
    
    with info_col3:
        st.markdown("""
        **❌ Late Blight**
        - Large irregular lesions
        - Water-soaked appearance
        - Grayish-white mold underneath
        """)
    
    st.info(
        "💡 **Tip**: For best results, upload clear, well-lit images of individual leaves "
        "taken against a plain background."
    )

if __name__ == "__main__":
    main()
