#!/usr/bin/env python
# coding: utf-8

# In[49]:
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
from tensorflow.keras import backend, layers
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.pexels.com/photos/1054221/pexels-photo-1054221.jpeg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
html_temp = """ 
  <div style="background-color:orange ;padding:7px">
  <h2 style="color:black;text-align:center;"><b>AgriSage- An AI tool for smart agriculture<b></h2>
  </div>
  """ 
class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)
st.markdown(html_temp,unsafe_allow_html=True)
activities=['SECTION 1-Crop recommendation','SECTION 2-Crop disease identification']
option=st.sidebar.selectbox('choose the options displayed below',activities) 
st.subheader(option) 
if option=='SECTION 1-Crop recommendation':
    model1= pickle.load(open(r"trained_model.pkl", 'rb'))
    x1=st.number_input('Enter the value of soil nitrogen',0,140)
    x2=st.number_input('Enter the value of soil phosphorus',5,145)
    x3=st.number_input('Enter the value of soil potassium',5,205)
    x4=st.number_input('Enter the value of soil temperature',0.1,43.7,0.1)
    x5=st.number_input('Enter the value of soil humidity',0.1,100.1,0.1)
    x6=st.number_input('Enter the value of soil ph',0.01,9.94,0.01)
    x7=st.number_input('Enter the value of soil rainfall',0.1,299.1,0.1)
    inp=pd.DataFrame([[x1,x2,x3,x4,x5,x6,x7]],columns=['N','P','K','temperature','humidity','ph','rainfall'])
    if st.button('RECOMMEND CROP'):
        out=model1.predict(inp)
        if out==0:
            st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is rice</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==1:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is maize</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==2:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is chickpea</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==3:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is kidneybeans</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==4:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is pigeonpeas</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==5:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is mothbean</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==6:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is mungbean</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==7:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is blackgram</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==8:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is lentil</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==9:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is pomogranate</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==10:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is banana</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==11:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is mango</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==12:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is grape</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==13:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is watermelon</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==14:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is muskmelon</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==15:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is apple</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==16:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is orange</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==17:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is papaya</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==18:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is coconut</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==19:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is cotton</h2>
      </div>
      """ ,unsafe_allow_html=True)
        elif out==20:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is jute</h2>
      </div>
      """ ,unsafe_allow_html=True)
        else:
                    st.markdown(""" 
      <div style="background-color: green;padding:3px;border: 3px solid;">
      <h2 style="color:white;text-align:center;">The recommended crop is coffee</h2>
      </div>
      """ ,unsafe_allow_html=True)
elif option=='SECTION 2-Crop disease identification':
    @st.cache(allow_output_mutation=True)
    def load_model():
        model=tf.keras.models.load_model(r"CNN for plant disease.h5",custom_objects={'FixedDropout':FixedDropout(rate=0.2)},compile=False)
        return model
    with st.spinner('Model is being loaded..'):
        model=load_model()
    file = st.file_uploader("Please upload any image from the local machine in case of computer or upload camera image in case of mobile.", type=["jpg","jpeg"])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is None:
         st.text("Please upload an image file within the allotted file size")
    else:
        img = Image.open(file)
        st.image(img, use_column_width=True)
        size = (224,224)    
        image = ImageOps.fit(img, size, Image.LANCZOS)
        imag = np.asarray(image)
        imaga = np.expand_dims(imag,axis=0) 
        predictions = model.predict(imaga)
        a=np.argmax(predictions,axis=1)
        if st.button('PREDICT DISEASE'):
            if a==0:
                st.markdown(""" 
          <div style="background-color: red;padding:3px;border: 3px solid;">
          <h2 style="color:white;text-align:center;">The predicted condition is scab disease in apple</h2>
          </div>
          """ ,unsafe_allow_html=True)
            elif a==1:
                            st.markdown(""" 
          <div style="background-color: red;padding:3px;border: 3px solid;">
          <h2 style="color:white;text-align:center;">The predicted condition is black rot disease in apple</h2>
          </div>
          """ ,unsafe_allow_html=True)
            elif a==2:
                            st.markdown(""" 
          <div style="background-color: red;padding:3px;border: 3px solid;">
          <h2 style="color:white;text-align:center;">The predicted condition is rust disease in apple</h2>
          </div>
          """ ,unsafe_allow_html=True)
            elif a==3:
                            st.markdown(""" 
          <div style="background-color: green;padding:3px;border: 3px solid;">
          <h2 style="color:white;text-align:center;">The predicted condition is healthy apple</h2>
          </div>
          """ ,unsafe_allow_html=True)
            elif a==4:
                            st.markdown(""" 
          <div style="background-color: green;padding:3px;border: 3px solid;">
          <h2 style="color:white;text-align:center;">The predicted condition is healthy blueberry</h2>
          </div>
          """ ,unsafe_allow_html=True)
            elif a==5:
                            st.markdown(""" 
          <div style="background-color: red;padding:3px;border: 3px solid;">
          <h2 style="color:white;text-align:center;">The predicted condition is powdery mildew disease in cherry</h2>
          </div>
          """ ,unsafe_allow_html=True)
            elif a==6:
                            st.markdown(""" 
          <div style="background-color: green;padding:3px;border: 3px solid;">
          <h2 style="color:white;text-align:center;">The predicted condition is healthy cherry</h2>
          </div>
          """ ,unsafe_allow_html=True)
            elif a==7:
                            st.markdown(""" 
          <div style="background-color: red;padding:3px;border: 3px solid;">
          <h2 style="color:white;text-align:center;">The predicted condition is leaf spot disease in corn</h2>
          </div>
          """ ,unsafe_allow_html=True)
            elif a==8:
                            st.markdown(""" 
          <div style="background-color: red;padding:3px;border: 3px solid;">
          <h2 style="color:white;text-align:center;">The predicted condition is rust disease in corn</h2>
          </div>
          """ ,unsafe_allow_html=True)
            elif a==9:
                            st.markdown(""" 
          <div style="background-color: red;padding:3px;border: 3px solid;">
          <h2 style="color:white;text-align:center;">The predicted condition is leaf blight disease in corn</h2>
          </div>
          """ ,unsafe_allow_html=True)
            elif a==10:
                            st.markdown(""" 
          <div style="background-color: green;padding:3px;border: 3px solid;">
          <h2 style="color:white;text-align:center;">The predicted condition is healthy corn</h2>
          </div>
          """ ,unsafe_allow_html=True)
            else:
                            st.markdown(""" 
          <div style="background-color: yellow;padding:3px;border: 3px solid;">
          <h2 style="color:white;text-align:center;">The predicted condition is not yet included!!</h2>
          </div>
          """ ,unsafe_allow_html=True)
