import streamlit as st
from streamlit_drawable_canvas import st_canvas
# import cv2
import pickle
import requests
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import time

# prg = st.progress(0)
  
# for i in range(100):
#     time.sleep(0.1)
#     prg.progress(i+1)

st.title("Number Classifier with my unsupervised package 🔢")
st.write("                                     ")
st.write("This data app could classify the number 0 and 8 from mnist dataset")

col0, col1 = st.columns(2)
# choose an unique option
options = {'PCA svd':'svd', 'PCA eigen':'eigen'}

# make a box for the choize 
unique_selection = col0.radio('Unique option', options.keys())
method = options[unique_selection]

# print the selected option:
st.write('You choose :', method)

#write n_components:
n_components=col1.slider("n_components", 0, 28, 10)

#Input data

st.subheader("How do you want to do your input?")

col0, col1 = st.columns(2)
# options2 = ['Upload a file', 'URL', 'Draw your number']
options2 = ['Upload a file', 'URL']
unique_selection2 = col0.radio('Unique option', options2)

def do_predict(img):
#     with open('modelo.pkl', 'rb') as archivo:
#         model = pickle.load(archivo)

#     pred = model.predict(img.reshape(1, 784))

    with st.spinner('Wait for it...'):

        # X=pd.read_csv("XMatrix")
        # y=pd.read_csv("YMatrix")
        X=pd.read_csv("https://raw.githubusercontent.com/DavidZap/Dimensionality-Reduction-ML-2/main/XMatrix.csv"")
        X=X.drop(X.columns[0],axis=1) 

        from model import myPCA
        myPCA = myPCA(n_components=n_components,method=method)

        myPCA.fit(X)

        # transform the data using the PCA object
        X_transformed = myPCA.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
        model = LogisticRegression(penalty='none', solver='saga')
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        st.write(f'Test accuracy PCA - {method} : %.2f%%' % (accuracy * 100))

        img2=img.reshape(1,-1)
        img_reduced = myPCA.fit_transform(img2)
        pred = model.predict(img_reduced)
        return st.write('the prediction is:', pred[0])

    st.success('Done!')

if unique_selection2 == "Upload a file":
    
    #Upload a file:
    image = col1.file_uploader("Upload a image", type=["jpg", "jpeg", "png"])
    if image:
        st.image(image, caption='loaded', use_column_width=True)
        img = Image.open(image).convert('L')
        img = img.resize((28, 28))
        img = np.array(img).astype('float32') / 255.0
        do_predict(img)
        # st.write(img)
        
elif unique_selection2 == "URL":
    url = col1.text_input("Type image URL")
    if url:
        st.image(url, caption='Imagen cargada desde URL', use_column_width=True)
        response = requests.get(url, verify=False)
        img = Image.open(BytesIO(response.content)).convert('L')
        img = img.resize((28, 28))
        img = np.array(img).astype('float32') / 255.0
        do_predict(img)

# else:
#     SIZE = 150
#     mode = st.checkbox("Draw (or Delete)?", True)
#     canvas_result = st_canvas(
#         fill_color='#000000',
#         stroke_width=10,
#         stroke_color='#FFFFFF',
#         background_color='#000000',
#         width=SIZE,
#         height=SIZE,
#         drawing_mode="freedraw" if mode else "transform",
#         key='canvas')
    
#     if canvas_result.image_data is not None:
#         img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
#         rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
#         st.write('Model Input')
#         st.image(rescaled)
#         input_numpy_array = np.array(canvas_result.image_data)
#         input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
#         input_image_gs = input_image.convert('L')
#         img = input_image_gs.resize((28, 28))
#         img = np.array(img).astype('float32') / 255.0
#         do_predict(img)
#         # st.write(img)
        
