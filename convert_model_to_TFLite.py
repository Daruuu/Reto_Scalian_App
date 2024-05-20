import  tensorflow as tf
import os

#cargarmos el modelo
model = tf.keras.models.load_model('my_model_manos16.keras')

# convertir a TFlite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
#save model TFLite
open('model_android/model.tflite', 'wb').write(tflite_model)
