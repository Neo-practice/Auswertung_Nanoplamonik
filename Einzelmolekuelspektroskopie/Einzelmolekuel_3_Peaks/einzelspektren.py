import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob



x = 0
# for filename in glob.glob('*.csv'):
filename = "01.csv"

values = pd.read_csv(filename, skiprows=1, delimiter=',', names=['x', 'y'])


# plt.plot(values.x, values.y)
print(x)
plt.show()
