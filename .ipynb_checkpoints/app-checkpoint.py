import streamlit as st
from streamlit_drawable_canvas import st_canvas
# import cv2
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from model import myPCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

st.title("Number Classifier with my unsupervised package 🔢")

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

if unique_selection2 == "Upload a file":
    
    #Upload a file:
    image = col1.file_uploader("Upload a image", type=["jpg", "jpeg", "png"])
    if image:
        st.image(image, caption='loaded', use_column_width=True)
        img = Image.open(image).convert('L')
        img = img.resize((28, 28))
        img = np.array(img).astype('float32') / 255.0
        # st.write(img)
        
elif unique_selection2 == "URL":
    url = col1.text_input("Introduce la URL de la imagen")
    if url:
        st.image(url, caption='Imagen cargada desde URL', use_column_width=True)
        response = requests.get(url, verify=False)
        img = Image.open(BytesIO(response.content)).convert('L')
        img = img.resize((28, 28))
        img = np.array(img).astype('float32') / 255.0

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
#         # st.write(img)
        
#Mpdel 


# Descargar el conjunto de datos MNIST
mnist = fetch_openml('mnist_784')
X = mnist.data / 255.0  # Normalizar los datos
y = mnist.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(penalty='none', solver='saga')
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
st.write('Test accuracy: %.2f%%' % (accuracy * 100))


pred = modelo.predict(img.reshape(1, 784))

st.write('the prediction is:', pred[0])





#

# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_openml

# mnist = fetch_openml('mnist_784')
# X, y = mnist["data"], mnist["target"]

# # seleccionamos algunos números al azar del conjunto de datos
# random_indices = np.random.choice(X.shape[0], size=25, replace=False)
# some_digits = X.iloc[random_indices]
# some_digits_labels = y[random_indices]

# # mostramos las imágenes de los números seleccionados
# fig, ax = plt.figure(figsize=(10, 10))
# fig, ax = plt.subplots()
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     fig.imshow(some_digits.iloc[i].values.reshape(28, 28), cmap="binary")
#     plt.axis("off")
#     plt.title(some_digits_labels.iloc[i])
# # plt.show()
# st.pyplot(fig)

    
#Make PCA



# # Descargar el conjunto de datos MNIST
# mnist = fetch_openml('mnist_784')
# X = mnist.data / 255.0  # Normalizar los datos
# y = mnist.target

# # Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # Realizar PCA para reducir el número de características
# pca = PCA(n_components=50) # Reducir a 50 características
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# # Entrenar un clasificador logístico en los datos de PCA
# clf = LogisticRegression(random_state=0, max_iter=1000)
# clf.fit(X_train_pca, y_train)

# # Predecir en los datos de prueba y calcular la precisión
# y_pred = clf.predict(X_test_pca)
# accuracy = accuracy_score(y_test, y_pred)

# print("Precisión del clasificador logístico con PCA: {:.2f}%".format(accuracy * 100))