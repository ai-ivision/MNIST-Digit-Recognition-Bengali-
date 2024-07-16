import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
import pickle

# Load the pre-trained model
with open("bengali_mnist_cnn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Function to preprocess the uploaded image
def preprocess_image(image):
    """
    Preprocess the input image to the format required by the model.
    
    Parameters:
    image (PIL.Image): The uploaded image to preprocess.
    
    Returns:
    numpy.ndarray: The preprocessed image.
    """
    # Convert the image to grayscale
    image = ImageOps.grayscale(image)
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the image
    image = image / 255.0
    # Reshape the image for the model
    image = image.reshape(1, 28, 28)
    return image

# Streamlit App
st.title("MNIST Digit Recognition")

# File uploader to allow users to upload an image
upload_image = st.file_uploader("Choose an image to predict digit...", type=["png", "jpg", "jpeg"])

if upload_image is not None:
    # Display the uploaded image
    st.image(upload_image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = Image.open(upload_image)
    processed_image = preprocess_image(image)

    # Make predictions using the pre-trained model
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)

    # Display the predicted digit in a large font
    st.markdown(f"<h1 style='text-align: center; color: cyan;'>Predicted Digit: {predicted_class}</h1>", unsafe_allow_html=True)

    # Display the prediction probabilities in a table
    st.write("Prediction Probabilities:")
    prob_df = pd.DataFrame(predictions[0], columns=["Probability"])
    prob_df["Digit"] = prob_df.index
    st.write(prob_df.sort_values(by="Probability", ascending=False).reset_index(drop=True))

    # Display a bar chart of prediction probabilities
    st.bar_chart(prob_df.set_index("Digit"))

else:
    st.write("Please upload an image to get the prediction.")

