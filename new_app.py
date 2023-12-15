import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
sm.set_framework('tf.keras')
tf.keras.backend.set_image_data_format('channels_last')

st.title('🌕 Mars Exploration Using Machine Learning.')



def preprocess_input(input_img):
    test_img = input_img.resize((480,480))
    test_img = np.asarray(test_img, dtype = np.float32) / 255.0
    return test_img

@st.cache_data
def model_prediction(test_img,_quantized_model):
    # Preprocess input image
    input_tensor_index = quantized_model.get_input_details()[0]['index']
    quantized_model.set_tensor(input_tensor_index, np.expand_dims(test_img, axis=0))
    # Run inference
    quantized_model.invoke()
    # Get the output
    output = quantized_model.get_tensor(quantized_model.get_output_details()[0]['index'])
    # Converting the predicted probabilities into a mask of integer labels
    predicted_mask = np.argmax(output, axis=-1)
    # Extracting the predicted mask for the input image
    predicted_mask = predicted_mask[0]
    return predicted_mask

import base64
from io import BytesIO
@st.cache_data
def data_url(pil_img1,pil_img2=None):

    fig = plt.figure()
    plt.imshow(pil_img1.resize((480,480)))
    if pil_img2 is not None:
        plt.imshow(pil_img2,alpha=0.3)
    plt.axis('OFF')
    data = BytesIO()
    fig.savefig(data, format="png",bbox_inches='tight',pad_inches = 0)
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/png;base64,'+data64.decode('utf-8')

col1, col2,col3 = st.columns(3,gap="medium")

with col1:
    st.header("Input")
    st.image("Inputs/IMG1.jpg")

with col2:
    st.header("Prediction")
    st.image("predicted_mask.png")

with col3:
    st.header("Output")
    st.image('demotrans.png')

st.divider()
st.write("""
#### 🗺️ Exploration legend : 
#### 🟡 Yellow area - Danger 💀
#### 🟣 Purple area - Safe 😀 
""")

st.divider()
st.write("""
         
### 📷 Choose Images from Below: 
""")

# uploaded_file = st.file_uploader("Select Images from [Inputs](https://github.com/ShrirangKanade/ML-for-mars-exploration/tree/master/Inputs):")
# if uploaded_file is not None:

col1, col2,col3 = st.columns(3,gap="medium")

with col1:
    st.image("Inputs/IMG1.jpg",caption='Image 1')

with col2:
    st.image("Inputs/IMG2.jpg",caption='Image 2 ')

with col3:
    st.image('Inputs/IMG3.jpg',caption='Image 3')

selected_image = st.selectbox("Select an image", ["IMG1.jpg", "IMG2.jpg", "IMG3.jpg"])

if selected_image in ["IMG1.jpg", "IMG2.jpg", "IMG3.jpg"]:
    input_imgs = Image.open(uploaded_file).convert('RGB')
    test_imgs = preprocess_input(input_imgs)

    quantized_model = tf.lite.Interpreter(model_content=open('quantized_model.tflite', 'rb').read())

    input_details = quantized_model.get_input_details()
   
    quantized_model.allocate_tensors()

    predicted_mask = model_prediction(test_imgs,quantized_model)

    new_img = Image.fromarray((predicted_mask * 255).astype(np.uint8))


    col1, col2,col3 = st.columns(3,gap="medium")

    with col1:
        st.header("Input")
        st.image(input_imgs)

    with col2:
        st.header("Prediction")
        plt.axis('off')
        st.image(data_url(new_img))
    with col3:
        st.header("Output")
   
        st.image(data_url(input_imgs,new_img))

    st.divider()
    st.header("🗗 Segmented Output")
    st.image(data_url(input_imgs,new_img))
    st.divider()
    st.write("## Proudly made by Shrirang Kanade 😎")
