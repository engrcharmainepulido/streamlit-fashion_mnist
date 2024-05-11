import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from PIL import Image
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('fashion_mnist_classifier.h5')
    return model

model = load_model()

st.write("# Fashion Item Classifier")

file = st.file_uploader("Upload an image of a clothing item", type=["jpg", "png"])

def preprocess_image(image):
    # Resize image to 28x28 and convert to grayscale
    image = image.resize((28, 28)).convert('L')
    img_array = np.asarray(image)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if file is None:
    st.text("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the uploaded image
    processed_image = preprocess_image(image)

    # Perform prediction
    prediction = model.predict(processed_image)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Get predicted class label
    predicted_class = np.argmax(prediction)
    predicted_item = class_names[predicted_class]

    st.success(f"Predicted Item: {predicted_item}")
