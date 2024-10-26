import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('/home/santiago/Documentos/Estudio/MinTIC/Interligencia Artificial/mlp_model.h5')

# Función para preprocesar la imagen
def preprocess_image(image):
    # Convertir la imagen a escala de grises y redimensionar a 28x28
    image = image.convert('L')          # Convertir a escala de grises
    image = image.resize((28, 28))      # Redimensionar a 28x28 píxeles
    image = np.array(image).reshape(1, 28 * 28).astype('float32') / 255
    return image

# Titulo de la aplicación
st.title('Clasificación de digitos manuscritos')

# Cargar la imagen
uploaded_file = st.file_uploader("Cargar una imagen de un digito (0-9)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
        # Mostrar la imagen cargada
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        # Preprocesar la imagen
        processed_image = preprocess_image(image)
        # Hacer la predicción
        prediction = model.predict(processed_image)
        predicted_digit =  np.argmax(prediction)
        # Mostrar la predicción
        st.write(f"Predicción: **{predicted_digit}**")
        # Mostrar la probabilidad de cada clase
        for i in range(10):
            st.write(f"Digito {i}: {prediction[0][i]:.4f}")
        # Mostrar imagen procesada para ver el preprocesamiento
        plt.imshow(np.squeeze(processed_image.reshape(28, 28)),cmap='gray')
        plt.axis('off')
        st.pyplot(plt)
        