import os
import numpy as np
import tensorflow as tf
from image_utils import load_and_preprocess_image

# Directorio que contiene las imágenes de entrenamiento
train_dir = 'dataset/train'
classes = ['broken', 'not_broken']

# Lista para almacenar las imágenes y etiquetas
images = []
labels = []

# Iterar sobre las clases
for i, class_name in enumerate(classes):
    class_dir = os.path.join(train_dir, class_name)
    file_names = os.listdir(class_dir)
    # Iterar sobre los archivos de imagen en la clase
    for file_name in file_names:
        image_path = os.path.join(class_dir, file_name)
        img = load_and_preprocess_image(image_path)
        images.append(img)
        labels.append(i)  # Etiqueta de la clase (0 o 1)

# Convertir las listas a matrices numpy
images = np.array(images)
labels = np.array(labels)

# Crear un modelo secuencial
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(images, labels, epochs=20, batch_size=20)

# Guardar el modelo entrenado en formato h5
model.save('hand_fracture_detection_model.h5')

# Función para predecir si una imagen de mano está rota o no
def predict_hand_fracture(image_path):
    # Cargar y preprocesar la imagen
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Añadir una dimensión adicional para batch

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
predict_hand_fracture(image_path)
