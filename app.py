import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st
import numpy as np
st.markdown("<h1 style='text-align: center'>LEAF CLASSIFICATION MODEL</h1>", unsafe_allow_html=True)
model= load_model('C:\\python_project1\\Image_classification\\Image_classify.keras')
data_cat=['Amla',
 'Amruta_Balli',
 'Arali',
 'Ashoka',
 'Ashwagandha',
 'Avacado',
 'Bamboo',
 'Basale',
 'Betel',
 'Betel_Nut',
 'Brahmi',
 'Castor',
 'Curry_Leaf',
 'Doddapatre',
 'Ekka',
 'Ganike',
 'Geranium',
 'Henna',
 'Hibiscus',
 'Honge',
 'Insulin',
 'Jasmine',
 'Lemon',
 'Lemon_grass',
 'Mango',
 'Mint',
 'Nagadali',
 'Neem',
 'Nithyapushpa',
 'Nooni',
 'Pappaya',
 'Pepper',
 'Pomegranate',
 'Raktachandini',
 'Rose',
 'Sapota',
 'Tulasi',
 'Wood_sorel']

img_height=256
img_width=256

image_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image_load = tf.keras.utils.load_img(image_file, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = np.expand_dims(img_arr, 0)

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    st.image(image_file)
st.markdown("<h4 style='text-align: center'>Leaf image classified is " + data_cat[np.argmax(score)] + "</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center'>With Accuracy of " + str(np.max(score)*100) + "%</h4>", unsafe_allow_html=True)
