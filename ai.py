from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from numpy import unravel_index

model = keras.models.load_model('hunt_ai.h5')

"""board =[0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,2,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0]

best_guesses = model.predict(np.reshape(board, (1,81)))

heat_map = np.reshape(best_guesses, (9, 9))

fig, ax = plt.subplots(figsize=[10,10])
im = ax.imshow(np.reshape(board, (9,9)))
plt.show()
#im = ax.imshow(heat_map[:,:])
plt.imshow(heat_map[:,:])"""


#for i in range(9):
#    for j in range(9):
#        text = ax.text(j, i, np.format_float_positional(heat_map[i,j], precision=3),
#                       ha="center", va="center", color="blue")
#plt.colorbar()
#plt.show()
def model_predict(board):
    for x in range(len(board)):
        board[x] = int(board[x])
        #print(board(x))
    board = np.array(board)
    board = np.reshape(board, (1, 81))
    model = keras.models.load_model('cb_ai.h5')
    best_guesses = model.predict(board)
    #best_guesses[unravel_index(best_guesses.argmax(), best_guesses.shape)] = 0
    new_best = np.reshape(best_guesses, (9,9))
    ret = unravel_index(new_best.argmax(), new_best.shape)
    print(ret)