from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(200, input_dim=81, activation = 'relu'))
model.add(Dense(81, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=20, epochs=50, validation_split=0.1)
model.summary()

model.evaluate(x_test, y_test)

best_guesses = np.zeros(81)

heat_map = np.reshape(best_guesses, (9, 9))

fig, ax = plt.subplots(figsize=[10,10])
im = ax.imshow(heat_map[:,:], cmap='gray', vmin=0, vmax=1)

for i in range(9):
    for j in range(9):
        text = ax.text(j, i, np.format_float_positional(heat_map[i,j], precision=3),
                       ha="center", va="center", color="blue")


def model_predict(board):
    best_guesses = model.predict(board)
    new_best = np.reshape(best_guesses, (9,9))
    print(zip(*np.where(new_best == 1)))