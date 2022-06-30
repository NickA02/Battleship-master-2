import pandas as pd
import numpy as np

pd.options.display.max_rows = 9999
n = 5
skip_func = lambda x: x%n != 0
df1 = pd.read_csv("checkerboard.csv", skiprows = skip_func, header = None)
df2 = pd.read_csv("dia.csv", skiprows = skip_func, header = None)
df3 = pd.read_csv("hunt.csv", skiprows = skip_func, header = None)

flattened_x = np.array()
flattened_y = np.array()
for i in range(len(df1)):
    flattened_x.append(np.array([int(x) for x in list(df1.loc[i][0])]))
    flattened_y.append(np.array([int(x) for x in list(df1.loc[i][1])]))
    
p1x = np.array()
p1x.append(np.zeros(81))
p2x = np.array()
p2x.append(np.zeros(81))
for i in range(len(flattened_x)):
    if i % 2 == 0:
        p1x.append(flattened_x[i])
    else:
        p2x.append(flattened_x[i])

for i in range(len(p1x)):
    if np.count_nonzero() == 1:
        p1x[i-1] = np.zeros(81)

for i in range(len(p2x)):
    if np.count_nonzero() == 1:
        p2x[i-1] = np.zeros(81)
        
p1y = np.array()
p2y = np.array()
for i in range(len(flattened_y)):
    if i % 2 == 0:
        p1y.append(flattened_y[i])
    else:
        p2y.append(flattened_y[i])

p1x.pop()
p2x.pop()

p1_df = pd.DataFrame(p1x, p1y, columns=['x', 'y'])
p2_df = pd.DataFrame(p2x, p2y, columns=['x', 'y'])
p1_df.to_csv("p1.csv")
p2_df.to_csv("p2.csv")