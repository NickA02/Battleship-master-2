import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

def gen_heat_map(preds):
    """Generates Heat Map"""

    fig, ax = plt.subplots(figsize=[10,10])
    for i in range(9):
        for j in range(9):
            text = ax.text(j, i, np.format_float_positional(preds[i][j], precision=3),
                        ha="center", va="center", color="blue")
    im = ax.imshow(preds[:,:]) 
    plt.show()

def check_if_invalid_position(x,y,board):
    board = np.reshape(board, (9,9))
    if board[x][y] != 0:
        print("Hit! Original guess: (" + str(x) + ', ' + str(y) + ')')
        return True
    return False

    """    while plot.check_if_invalid_position(ret[0],ret[1],board):
        new_best[ret[0]][ret[1]] = 0
        ret = unravel_index(new_best.argmax(), new_best.shape)
        
        This is the insert for redundancy inspection on guesses
    """