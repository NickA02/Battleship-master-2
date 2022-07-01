from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from numpy import unravel_index
import plot

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
    model = keras.models.load_model('hunt_ai.h5')
    best_guesses = model.predict(board)
    #best_guesses[unravel_index(best_guesses.argmax(), best_guesses.shape)] = 0
    new_best = np.reshape(best_guesses, (9,9))
    plot.gen_heat_map(new_best)
    ret = unravel_index(new_best.argmax(), new_best.shape)
    while plot.check_if_invalid_position(ret[0],ret[1],board):
        new_best[ret[1]][ret[0]] = 0
        ret = unravel_index(new_best.argmax(), new_best.shape)
    x3= str(ret[0] + 1) 
    y = ret[1]
    x2 = ""
    if y == 0:
        x2 = "A"
    if y == 1:
        x2 = "B"
    if y == 2:
        x2 = "C"
    if y == 3:
        x2 = "D"
    if y == 4:
        x2 = "E"
    if y == 5:
        x2 = "F"
    if y == 6:
        x2 = "G"
    if y == 7:
        x2 = "H"
    if y == 8:
        x2 = "I"
    
    print("(" + x2 + "," + x3 + ")")