import pandas as pd
import numpy as np

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
