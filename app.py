# Potato Leaf Disease Classification - Streamlit App
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from pathlib import Path
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

#  CACHE & MODEL FUNCTIONS 
@st.cache_resource
def load_or_train_model(data_dir):
    """Load pre-trained model or train a new one if not available"""
    model_path = "potato_disease_model.h5"
    
    # Check if model exists
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except:
            st.warning("Could not load existing model. Training new model...")
    
    # If model doesn't exist or failed to load, train a new one
    st.info(" Training model from dataset... This may take a few minutes.")
    
    with st.spinner("Training in progress..."):
        # Load datasets from directory
        IMG_SIZE = 256
        BATCH_SIZE = 32
        
        # Create train dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )
        # Store class names immediately — accessing `class_names` after
        # .prefetch() or other dataset transformations can fail because
        # those return a Dataset wrapper that doesn't preserve the
        # .class_names attribute.
        class_names = train_ds.class_names
        
        # Create validation dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )
        
        # Normalize images
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        
        # Prefetch for performance
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Build CNN model
        num_classes = len(class_names)

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # Train model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=15,
            verbose=0
        )

        # Save model
        model.save(model_path)
        st.success(" Model trained and saved successfully!")
    
    return model

@st.cache_resource
def get_class_names(data_dir):
    """Get class names from dataset directory"""
    class_names = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    return class_names

def predict_image(image, model, class_names):
    """Make prediction on uploaded image"""
    # Prepare image
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
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
        'Early Blight': {
            'description': '⚠️ Early Blight Detected',
            'color': 'early-blight',
            'emoji': '🍂',
            'details': 'Early blight is a fungal disease. Apply fungicides and remove affected leaves. Ensure good air circulation.'
        },
        'Late Blight': {
            'description': '🚨 Late Blight Detected',
            'color': 'late-blight',
            'emoji': '❌',
            'details': 'Late blight is a serious fungal disease. Apply appropriate fungicides immediately. Increase plant spacing for better air flow.'
        }
    }
    return disease_info.get(disease_name, {
        'description': 'Unknown',
        'color': 'healthy',
        'emoji': '❓',
        'details': 'Classification not recognized.'
    })

#  MAIN APP 
def main():
    # Header
    st.markdown('<div class="main-header">🥔 Potato Leaf Disease Classifier</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Identify diseases in potato plants using AI</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title(" Configuration")
    
    # Data directory input
    data_dir = st.sidebar.text_input(
        " Dataset Directory Path",
        value="./potato_dataset",
        help="Path containing 'healthy', 'early_blight', and 'late_blight' folders"
    )
    
    # Verify directory exists
    if not os.path.isdir(data_dir):
        st.error(f"❌ Dataset directory not found: {data_dir}")
        st.info("📌 Please ensure your dataset is organized as:\n"
                "```\n"
                "potato_dataset/\n"
                "├── healthy/\n"
                "├── early_blight/\n"
                "└── late_blight/\n"
                "```")
        return
    
    # Load class names
    try:
        class_names = get_class_names(data_dir)
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        return
    
    # Load or train model
    try:
        model = load_or_train_model(data_dir)
    except Exception as e:
        st.error(f"Error loading/training model: {e}")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader(" Upload Image", divider="green")
        
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
            with st.spinner(" Analyzing leaf..."):
                predicted_class, confidence, all_predictions = predict_image(
                    image_resized, model, class_names
                )
            
            # Display results in second column
            with col2:
                st.subheader(" Results", divider="green")
                
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
                st.write("###  All Predictions")
                pred_df = {
                    'Class': class_names,
                    'Confidence': [f"{conf*100:.2f}%" for conf in all_predictions]
                }
                st.dataframe(pred_df, use_container_width=True)
        else:
            with col2:
                st.info(" Upload an image to see predictions")
    
    # Additional info section
    st.divider()
    st.subheader(" About This Classifier")
    
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