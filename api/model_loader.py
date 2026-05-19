import tensorflow as tf
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model():
    model_path = os.path.join(BASE_DIR, '../models/modelo_lstm.keras')
    return tf.keras.models.load_model(model_path)

def load_scaler():
    scaler_path = os.path.join(BASE_DIR, '../models/scaler.pkl')
    return joblib.load(scaler_path)