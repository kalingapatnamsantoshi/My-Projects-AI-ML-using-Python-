import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import streamlit as st
from PIL import Image, ImageOps  # Added missing imports

#st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('glaucoma-model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (100, 100), Image.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image).astype(np.float32) / 255.0
    img_reshape = image[np.newaxis, ...]  # Adding batch dimension
    
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
    
    if model:
        prediction = model.predict(img_reshape)
        return prediction
    else:
        return None

st.title("🩺 *Glaucoma Detector*")
st.write("### Classification of glaucoma through fundus image of the eye")

file = st.file_uploader("📸 Please upload an eye fundus image (JPG)", type=["jpg"])

if file:
    imageI = Image.open(file)
    prediction = import_and_predict(imageI, model)

    if prediction is not None:
        pred = prediction[0][0]

        if pred > 0.5:
            st.success("✅ *Prediction:* Your eye is Healthy. Great!!")
            st.balloons()
        else:
            st.error("⚠ *Prediction:* You are affected by Glaucoma. Please consult a doctor.")
    else:
        st.warning("⚠ Model could not make a prediction. Please check the model file.")
else:
    st.info("📂 You haven't uploaded a JPG image file.")