# Advanced Streamlit App with Model Training Monitor
# Save this as app_advanced.py for additional features

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Potato Disease Classifier - Advanced",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
    }
    .metric-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_or_train_model_advanced(data_dir):
    """Load or train model with comprehensive logging"""
    model_path = "potato_disease_model.h5"
    history_path = "training_history.npy"
    
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except:
            pass
    
    # Training configuration
    IMG_SIZE = 256
    BATCH_SIZE = 32
    EPOCHS = 15
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load datasets
    status_text.text("📦 Loading dataset...")
    progress_bar.progress(10)
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    # Normalize
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Build model
    status_text.text("🏗️ Building model architecture...")
    progress_bar.progress(20)
    
    num_classes = len(train_ds.class_names)
    
    # Enhanced model with data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    
    model = tf.keras.Sequential([
        data_augmentation,
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Train model
    status_text.text("🤖 Training model...")
    progress_bar.progress(30)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=0
    )
    
    progress_bar.progress(90)
    status_text.text("💾 Saving model...")
    model.save(model_path)
    np.save(history_path, history.history)
    
    progress_bar.progress(100)
    status_text.text("✅ Training complete!")
    
    return model

def main():
    # Header
    st.markdown('<div class="main-header">🥔 Potato Disease Classifier - Advanced</div>', 
                unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Predict", "📊 Model Info", "📈 Training Metrics", "ℹ️ Guide"]
    )
    
    # Initialize session
    with tab1:
        st.subheader("Image Classification")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            data_dir = st.text_input("Dataset path:", "./potato_dataset")
            
            if not os.path.isdir(data_dir):
                st.error(f"Dataset not found: {data_dir}")
                return
            
            model = load_or_train_model_advanced(data_dir)
            class_names = sorted([d for d in os.listdir(data_dir) 
                                 if os.path.isdir(os.path.join(data_dir, d))])
            
            uploaded_file = st.file_uploader("Upload leaf image", 
                                            type=['jpg', 'jpeg', 'png', 'bmp'])
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded", use_column_width=True)
                
                image_resized = image.resize((256, 256))
                img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
                img_array = tf.expand_dims(img_array, 0) / 255.0
                
                predictions = model.predict(img_array, verbose=0)
                
                with col2:
                    st.subheader("Results")
                    pred_class = class_names[np.argmax(predictions[0])]
                    confidence = float(np.max(predictions[0])) * 100
                    
                    st.metric("Prediction", pred_class, f"{confidence:.2f}%")
                    
                    st.write("### All Predictions")
                    for cls, prob in zip(class_names, predictions[0]):
                        st.write(f"{cls}: {prob*100:.2f}%")
                        st.progress(prob)
    
    with tab2:
        st.subheader("Model Architecture")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-box">
                <h3>Model Type</h3>
                <p>Convolutional Neural Network (CNN)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <h3>Input Size</h3>
                <p>256 × 256 × 3</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box">
                <h3>Output Classes</h3>
                <p>3 Classes</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("### Network Layers")
        layers_info = """
        | Layer Type | Filters/Units | Output Shape |
        |---|---|---|
        | Conv2D | 32 | 256×256×32 |
        | MaxPool2D | - | 128×128×32 |
        | Conv2D | 64 | 128×128×64 |
        | MaxPool2D | - | 64×64×64 |
        | Conv2D | 128 | 64×64×128 |
        | MaxPool2D | - | 32×32×128 |
        | Conv2D | 128 | 32×32×128 |
        | MaxPool2D | - | 16×16×128 |
        | Flatten | - | 32768 |
        | Dense | 512 | 512 |
        | Dropout | 0.5 | 512 |
        | Dense (Softmax) | 3 | 3 |
        """
        st.markdown(layers_info)
    
    with tab3:
        st.subheader("Training Information")
        st.write("**Training Parameters:**")
        
        params = {
            "Optimizer": "Adam (lr=0.001)",
            "Loss Function": "Categorical Crossentropy",
            "Batch Size": 32,
            "Epochs": 15,
            "Validation Split": "20%",
            "Data Augmentation": "Yes (Flip, Rotation, Zoom)"
        }
        
        for key, value in params.items():
            st.write(f"- **{key}**: {value}")
    
    with tab4:
        st.subheader("Usage Guide")
        st.write("""
        ### Dataset Structure
        Your dataset must be organized as:
        ```
        potato_dataset/
        ├── healthy/
        ├── early_blight/
        └── late_blight/
        ```
        
        ### Quick Start
        1. Install: `pip install -r requirements.txt`
        2. Run: `streamlit run app.py`
        3. Upload an image to get predictions
        
        ### Disease Indicators
        - **Healthy**: Green leaves without lesions
        - **Early Blight**: Brown lesions with concentric rings
        - **Late Blight**: Large irregular lesions with white mold
        """)

if __name__ == "__main__":
    main()