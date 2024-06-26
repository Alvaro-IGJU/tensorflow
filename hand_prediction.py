import tensorflow as tf
from image_utils import load_and_preprocess_image

# Función para predecir si una imagen de mano está rota o no
def predict_hand_fracture(image_path, model_path):
    # Cargar el modelo previamente entrenado desde el archivo
    model = tf.keras.models.load_model(model_path)

    # Cargar y preprocesar la imagen
    img = load_and_preprocess_image(image_path)
    img = tf.expand_dims(img, axis=0)  # Añadir una dimensión adicional para batch

    # Realizar la predicción
    prediction = model.predict(img)[0][0]

    # Calcular el porcentaje de probabilidad de que la mano esté rota
    fracture_percentage = (1 - prediction) * 100
    not_fracture_percentage = prediction * 100

    # Imprimir el resultado
    print(f'Probabilidad de mano rota: {fracture_percentage:.2f}%')
    print(f'Probabilidad de mano no rota: {not_fracture_percentage:.2f}%')

# Ejemplo de uso
image_path = 'dataset/photos/images.jpg'
model_path = 'hand_fracture_detection_model.h5'
predict_hand_fracture(image_path, model_path)
