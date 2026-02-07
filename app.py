import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Dog Breed AI", page_icon="🐾", layout="centered")

# --- 2. THE STABLE LOAD FUNCTION ---
@st.cache_resource
def load_stable_model():
    # Load labels to determine the output size
    try:
        with open('labels.json', 'r') as f:
            labels_map = json.load(f)
        num_classes = len(labels_map)
    except Exception as e:
        st.error(f"Could not load labels.json: {e}")
        return None, {}

    # BUILD THE ARCHITECTURE MANUALLY
    # This bypasses all the DTypePolicy and Flatten errors
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='dense')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax', name='dense_1')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)

    # LOAD WEIGHTS ONLY
    # We use by_name=True to match the 3 saved layers into our new 18-layer model
    if os.path.exists('dog_breed_model.h5'):
        try:
            # We use skip_mismatch=True just in case your layer names are slightly different
            model.load_weights('dog_breed_model.h5', by_name=True, skip_mismatch=True)
            return model, labels_map
        except Exception as e:
            st.error(f"Weight loading error: {e}")
            return None, labels_map
    else:
        st.error("dog_breed_model.h5 not found!")
        return None, labels_map

# --- 3. UI LAYOUT ---
st.title("🐾 Dog Breed Classifier")
st.write("Upload a photo and let the AI identify the breed.")

model, labels_map = load_stable_model()

if model:
    st.sidebar.success("✅ AI Model Ready")
else:
    st.sidebar.error("❌ Model Offline")
    st.stop()

# --- 4. UPLOAD AND PREDICT ---
uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the image
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    
    if st.button("Identify Breed"):
        with st.spinner('Analyzing...'):
            try:
                # Preprocessing
                img = image.load_img(uploaded_file, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Prediction
                preds = model.predict(img_array)
                idx = np.argmax(preds[0])
                confidence = np.max(preds[0]) * 100
                
                breed = labels_map.get(str(idx), "Unknown Breed")

                # Results
                st.divider()
                st.success(f"### Predicted Breed: **{breed}**")
                st.info(f"AI Confidence: {confidence:.2f}%")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Built with Python 3.13, VGG19, and Streamlit.")