import streamlit as st
import numpy as np
import pickle
from PIL import Image
import cv2
import tensorflow as tf 
# # Load the saved model
# with open('history_model_2.pkl', 'rb') as file:
#     model = pickle.load(file)

from tensorflow.keras.models import load_model # type: ignore

#Load the trained model correctly
model = load_model("model_new1.h5") 
print(model.input_shape)

# Title and description for the app
st.title("Brain Tumor Detection App")
st.write("Upload an MRI image to check for the presence of a brain tumor.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Preprocess the image (resize, normalize, convert to array, etc.)
    image = image.resize((150, 150))  # Resize to match model's input shape if needed
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  # Convert to BGR if needed
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Display the result
    labels = {0: "No Tumor", 1: "Meningioma Tumor", 2: "Glioma Tumor", 3: "Pituitary Tumor"}  # Adjust as per your labels
    st.write(f"**Prediction:** {labels[predicted_label]}")
    st.write(f"**Accuracy Level:** {confidence:.2f}%")
