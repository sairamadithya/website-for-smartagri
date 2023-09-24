#!/usr/bin/env python
# coding: utf-8

# In[49]:


get_ipython().run_cell_magic('smartagri-ML.py', '', 'import streamlit as st\nimport numpy as np\nimport pandas as pd\nimport pickle\nimport tensorflow as tf\nimport numpy as np\nfrom tensorflow.keras.preprocessing import image\nfrom PIL import Image, ImageOps\nfrom tensorflow.keras import backend, layers\nst.markdown(\n         f"""\n         <style>\n         .stApp {{\n             background-image: url("https://images.pexels.com/photos/1054221/pexels-photo-1054221.jpeg");\n             background-attachment: fixed;\n             background-size: cover\n         }}\n         </style>\n         """,\n         unsafe_allow_html=True\n     )\nhtml_temp = """ \n  <div style="background-color:orange ;padding:7px">\n  <h2 style="color:black;text-align:center;"><b>AgriSage- An AI tool for smart agriculture<b></h2>\n  </div>\n  """ \nclass FixedDropout(layers.Dropout):\n        def _get_noise_shape(self, inputs):\n            if self.noise_shape is None:\n                return self.noise_shape\n\n            symbolic_shape = backend.shape(inputs)\n            noise_shape = [symbolic_shape[axis] if shape is None else shape\n                           for axis, shape in enumerate(self.noise_shape)]\n            return tuple(noise_shape)\nst.markdown(html_temp,unsafe_allow_html=True)\nactivities=[\'SECTION 1-Crop recommendation\',\'SECTION 2-Crop disease identification\']\noption=st.sidebar.selectbox(\'choose the options displayed below\',activities) \nst.subheader(option) \nif option==\'SECTION 1-Crop recommendation\':\n    model1= pickle.load(open(r"trained_model.pkl", \'rb\'))\n    x1=st.number_input(\'Enter the value of soil nitrogen\',0,140)\n    x2=st.number_input(\'Enter the value of soil phosphorus\',5,145)\n    x3=st.number_input(\'Enter the value of soil potassium\',5,205)\n    x4=st.number_input(\'Enter the value of soil temperature\',0.1,43.7,0.1)\n    x5=st.number_input(\'Enter the value of soil humidity\',0.1,100.1,0.1)\n    x6=st.number_input(\'Enter the value of soil ph\',0.01,9.94,0.01)\n    x7=st.number_input(\'Enter the value of soil rainfall\',0.1,299.1,0.1)\n    inp=pd.DataFrame([[x1,x2,x3,x4,x5,x6,x7]],columns=[\'N\',\'P\',\'K\',\'temperature\',\'humidity\',\'ph\',\'rainfall\'])\n    if st.button(\'RECOMMEND CROP\'):\n        out=model1.predict(inp)\n        if out==0:\n            st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is rice</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==1:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is maize</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==2:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is chickpea</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==3:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is kidneybeans</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==4:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is pigeonpeas</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==5:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is mothbean</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==6:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is mungbean</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==7:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is blackgram</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==8:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is lentil</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==9:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is pomogranate</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==10:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is banana</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==11:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is mango</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==12:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is grape</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==13:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is watermelon</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==14:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is muskmelon</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==15:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is apple</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==16:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is orange</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==17:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is papaya</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==18:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is coconut</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==19:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is cotton</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        elif out==20:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is jute</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\n        else:\n                    st.markdown(""" \n      <div style="background-color: green;padding:3px;border: 3px solid;">\n      <h2 style="color:white;text-align:center;">The recommended crop is coffee</h2>\n      </div>\n      """ ,unsafe_allow_html=True)\nelif option==\'SECTION 2-Crop disease identification\':\n    @st.cache(allow_output_mutation=True)\n    def load_model():\n        model=tf.keras.models.load_model(r"CNN for plant disease.h5",custom_objects={\'FixedDropout\':FixedDropout(rate=0.2)},compile=False)\n        return model\n    with st.spinner(\'Model is being loaded..\'):\n        model=load_model()\n    file = st.file_uploader("Please upload any image from the local machine in case of computer or upload camera image in case of mobile.", type=["jpg","jpeg"])\n    st.set_option(\'deprecation.showfileUploaderEncoding\', False)\n    if file is None:\n         st.text("Please upload an image file within the allotted file size")\n    else:\n        img = Image.open(file)\n        st.image(img, use_column_width=True)\n        size = (224,224)    \n        image = ImageOps.fit(img, size, Image.LANCZOS)\n        imag = np.asarray(image)\n        imaga = np.expand_dims(imag,axis=0) \n        predictions = model.predict(imaga)\n        a=np.argmax(predictions,axis=1)\n        if st.button(\'PREDICT DISEASE\'):\n            if a==0:\n                st.markdown(""" \n          <div style="background-color: red;padding:3px;border: 3px solid;">\n          <h2 style="color:white;text-align:center;">The predicted condition is scab disease in apple</h2>\n          </div>\n          """ ,unsafe_allow_html=True)\n            elif a==1:\n                            st.markdown(""" \n          <div style="background-color: red;padding:3px;border: 3px solid;">\n          <h2 style="color:white;text-align:center;">The predicted condition is black rot disease in apple</h2>\n          </div>\n          """ ,unsafe_allow_html=True)\n            elif a==2:\n                            st.markdown(""" \n          <div style="background-color: red;padding:3px;border: 3px solid;">\n          <h2 style="color:white;text-align:center;">The predicted condition is rust disease in apple</h2>\n          </div>\n          """ ,unsafe_allow_html=True)\n            elif a==3:\n                            st.markdown(""" \n          <div style="background-color: green;padding:3px;border: 3px solid;">\n          <h2 style="color:white;text-align:center;">The predicted condition is healthy apple</h2>\n          </div>\n          """ ,unsafe_allow_html=True)\n            elif a==4:\n                            st.markdown(""" \n          <div style="background-color: green;padding:3px;border: 3px solid;">\n          <h2 style="color:white;text-align:center;">The predicted condition is healthy blueberry</h2>\n          </div>\n          """ ,unsafe_allow_html=True)\n            elif a==5:\n                            st.markdown(""" \n          <div style="background-color: red;padding:3px;border: 3px solid;">\n          <h2 style="color:white;text-align:center;">The predicted condition is powdery mildew disease in cherry</h2>\n          </div>\n          """ ,unsafe_allow_html=True)\n            elif a==6:\n                            st.markdown(""" \n          <div style="background-color: green;padding:3px;border: 3px solid;">\n          <h2 style="color:white;text-align:center;">The predicted condition is healthy cherry</h2>\n          </div>\n          """ ,unsafe_allow_html=True)\n            elif a==7:\n                            st.markdown(""" \n          <div style="background-color: red;padding:3px;border: 3px solid;">\n          <h2 style="color:white;text-align:center;">The predicted condition is leaf spot disease in corn</h2>\n          </div>\n          """ ,unsafe_allow_html=True)\n            elif a==8:\n                            st.markdown(""" \n          <div style="background-color: red;padding:3px;border: 3px solid;">\n          <h2 style="color:white;text-align:center;">The predicted condition is rust disease in corn</h2>\n          </div>\n          """ ,unsafe_allow_html=True)\n            elif a==9:\n                            st.markdown(""" \n          <div style="background-color: red;padding:3px;border: 3px solid;">\n          <h2 style="color:white;text-align:center;">The predicted condition is leaf blight disease in corn</h2>\n          </div>\n          """ ,unsafe_allow_html=True)\n            elif a==10:\n                            st.markdown(""" \n          <div style="background-color: green;padding:3px;border: 3px solid;">\n          <h2 style="color:white;text-align:center;">The predicted condition is healthy corn</h2>\n          </div>\n          """ ,unsafe_allow_html=True)\n            else:\n                            st.markdown(""" \n          <div style="background-color: yellow;padding:3px;border: 3px solid;">\n          <h2 style="color:white;text-align:center;">The predicted condition is not yet included!!</h2>\n          </div>\n          """ ,unsafe_allow_html=True)')

