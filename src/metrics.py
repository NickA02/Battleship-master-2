from tensorflow import keras
from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential

inTop5 = lambda x, y : top_k_categorical_accuracy(x, y, k=5)