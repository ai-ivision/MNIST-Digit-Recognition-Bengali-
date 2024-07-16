# Bengali MNIST Digit Recognition

This project implements a Bengali MNIST digit recognition web application using Streamlit. The app allows users to upload images of Bengali digits in PNG, JPG, or JPEG format and predicts the digit using a pre-trained Convolutional Neural Network (CNN) model.

## Table of Contents
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [File Structure](#file-structure)
- [Acknowledgements](#acknowledgements)

## Demo
![Demo](demo.mp4)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ai-ivision/MNIST-Digit-Recognition-Bengali-.git
    cd MNIST-Digit-Recognition-Bengali
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Upload an image**:
    - Choose a PNG, JPG, or JPEG file containing a Bengali digit.
    - The app will display the uploaded image and the predicted digit along with the prediction probabilities.

## Model

The model used in this project is a Convolutional Neural Network (CNN) trained on the Bengali MNIST dataset. The model is saved as a pickle file (`bengali_mnist_cnn_model.pkl`) and loaded in the Streamlit app for making predictions.

## File Structure

bengali-mnist-digit-recognition/
│
├── app.py # Streamlit app script
├── bengali_mnist_cnn_model.pkl # Pre-trained CNN model
├── requirements.txt # List of required packages
├── README.md # Project documentation
└── demo.gif # Demo mp4 for README


## Acknowledgements

- [Streamlit](https://streamlit.io/) for providing the web application framework.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the deep learning libraries.
- [Bengali MNIST dataset](https://www.kaggle.com/datasets/truthr/banglamnist) for the digit images used in training the model.
- The open-source community for continuous support and resources.

---

Feel free to customize this README as per your specific project details and requirements.
