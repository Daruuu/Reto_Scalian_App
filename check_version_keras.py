import tensorflow as tf

# Verificar la versión de TensorFlow
print("TensorFlow version:", tf.__version__)

# Verificar si Keras está disponible
print("Keras version:", tf.keras.__version__)

# Intentar cargar un modelo de Keras para asegurarse de que todo está bien
try:
    model = tf.keras.models.load_model('my_model_manos16.keras')
    print("Modelo cargado exitosamente")
except Exception as e:
    print("Error al cargar el modelo:", e)

