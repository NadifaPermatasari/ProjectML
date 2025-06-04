import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Fungsi load model TFLite
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="xception_deepfake_image.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Fungsi prediksi
def predict_image(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Pastikan input sesuai ukuran model
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Fungsi preprocessing gambar
def preprocess_image(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalisasi
    return img_array

# Streamlit UI
st.title("Deteksi Gambar Deepfake")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Diupload", use_column_width=True)

    img_array = preprocess_image(image)

    interpreter = load_model()
    prediction = predict_image(interpreter, img_array)

    confidence = prediction[0][0]  # anggap output scalar, sesuaikan jika multi-class
    if confidence > 0.5:
        st.error(f"🧪 Deepfake terdeteksi! Skor: {confidence:.2f}")
    else:
        st.success(f"✅ Gambar asli. Skor: {confidence:.2f}")
