import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

def gen_heat_map(preds):
    print(preds.shape)
    print(preds)
    fig, ax = plt.subplots(figsize=[10,10])
    for i in range(9):
        for j in range(9):
            text = ax.text(j, i, np.format_float_positional(preds[i][j], precision=3),
                        ha="center", va="center", color="blue")
    im = ax.imshow(preds[:,:]) 
    plt.show()
