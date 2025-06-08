import tensorflow as tf
import streamlit as st
import os
from keras.models import load_model
import numpy as np

# Set page title
st.header('🌸 Flower Recognition Model')

# Define the classes
flower_name = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

# Define model path relative to current script
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'archive', 'flower_recognition_model.h5')

# Check if model file exists and load
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at {model_path}. Please upload the model to this location.")
    st.stop()  # Stop execution if model is missing
else:
    model = load_model(model_path)

# Define the prediction function
def call(input_image_path):
    input_image = tf.keras.utils.load_img(input_image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image) / 255.0  # Normalize pixel values
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'This image belongs to **{}** with a score of **{:.2f}%**'.format(
        flower_name[np.argmax(result)],
        np.max(result) * 100
    )
    return outcome

# Upload file
upload_file = st.file_uploader("📤 Upload your image here", type=["jpg", "jpeg", "png"])

# Fancy sidebar
st.sidebar.markdown("## 🌸 Flower Classifier")
st.sidebar.markdown("This app predicts which flower is in the image.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='background-color:#000000;padding:10px;border-radius:10px'>
        <b>💡 Tip:</b> Use clear flower images <br>
        Recommended size: <i>180x180 px</i>
    </div>
    """, unsafe_allow_html=True
)
st.sidebar.markdown("")
st.sidebar.markdown("### 🌼 Flower Classes:")
st.sidebar.markdown("- 🌺 **Lilly**\n- 🌸 **Lotus**\n- 🏵️ **Orchid**\n- 🌻 **Sunflower**\n- 🌷 **Tulip**")
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Bipradeep")

# Process uploaded file
if upload_file is not None:
    # Ensure the 'upload' folder exists
    os.makedirs('upload', exist_ok=True)

    # Save the uploaded file
    save_path = os.path.join('upload', upload_file.name)
    with open(save_path, 'wb') as f:
        f.write(upload_file.getbuffer())

    # Display the uploaded image
    st.image(upload_file, width=200, caption="Uploaded Image")

    # Run prediction and display result
    prediction = call(save_path)
    st.success(prediction)

    st.markdown(prediction)
   


