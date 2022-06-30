import pandas as pd
import numpy as np

pd.options.display.max_rows = 9999
n = 5
skip_func = lambda x: x%n != 0
df1 = pd.read_csv("checkerboard.csv", skiprows = skip_func, header = None)
df2 = pd.read_csv("dia.csv", skiprows = skip_func, header = None)
df3 = pd.read_csv("hunt.csv", skiprows = skip_func, header = None)


np.array([int(x) for x in list(df1.loc[0][0])]) 



