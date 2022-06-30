import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout

pd.options.display.max_rows = 9999
n = 5
skip_func = lambda x: x%n != 0
df1 = pd.read_csv("checkerboard.csv", skiprows = skip_func, header = None)
df2 = pd.read_csv("dia.csv", skiprows = skip_func, header = None)
df3 = pd.read_csv("hunt.csv", skiprows = skip_func, header = None)

flattened_x = []
flattened_y = []
for i in range(len(df1)):
    flattened_x.append(np.array([int(x) for x in list(df1.loc[i][0])]))
    flattened_y.append(np.array([int(x) for x in list(df1.loc[i][1])]))
    
p1x = []
p1x.append(np.zeros(81))
p2x = []
p2x.append(np.zeros(81))
for i in range(len(flattened_x)):
    if i % 2 == 0:
        p1x.append(flattened_x[i])
    else:
        p2x.append(flattened_x[i])

for i in range(len(p1x)):
    if np.count_nonzero(1) + np.count_nonzero(2) == 1:
        p1x[i-1] = np.zeros(81)

for i in range(len(p2x)):
    if np.count_nonzero(1) + np.count_nonzero(2) == 1:
        p2x[i-1] = np.zeros(81)
        
p1y = []
p2y = []
for i in range(len(flattened_y)):
    if i % 2 == 0:
        p1y.append(flattened_y[i])
    else:
        p2y.append(flattened_y[i])

p1x.pop()
p2x.pop()


model = Sequential()
#model.add(Dense(81, activation='relu'))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(81, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

p1x = np.array(p1x)
p1y = np.array(p1y)
print(p1x.shape)
print(p1y.shape)
print(p1y.sum(axis=1))

model.fit(p1x, p1y, batch_size=20, epochs=50, validation_split=0.1)
model.summary()

model.save('cb_ai.h5')

model.evaluate(p1x, p1y)

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