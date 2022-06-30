import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import accuracy_metrics


tf.keras.metrics.TopKCategoricalAccuracy(
    k=5, name="top_k_categorical_accuracy", dtype=None
)
"""
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[tf.keras.metrics.TopKCategoricalAccuracy()])
"""