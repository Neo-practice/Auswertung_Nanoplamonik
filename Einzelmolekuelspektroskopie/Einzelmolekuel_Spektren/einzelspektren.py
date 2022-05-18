import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

values = pd.read_csv("01.csv", skiprows=1, delimiter=',', names=['x', 'y'])
values = pd.read_csv("22.csv", skiprows=1, delimiter=',', names=['x', 'y'])

plt.plot(values.x, values.y)

plt.show()
