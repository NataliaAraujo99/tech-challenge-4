import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model('models/modelo_lstm.h5')
    return model