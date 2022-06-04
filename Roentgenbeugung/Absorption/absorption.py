import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data1 = pd.read_csv("data/HG4_1.STN",
                   delimiter=r"\s+",
                   #delimiter="/t",
                   skiprows=22,
                   names=["angle", "value", "untg"],
                   engine="python",
                   skipinitialspace=True
                   )
data2 = pd.read_csv("data/HG4_2.STN",
                   delimiter=r"\s+",
                   #delimiter="/t",
                   skiprows=22,
                   names=["angle", "value", "untg"],
                   engine="python",
                   skipinitialspace=True
                   )
data3 = pd.read_csv("data/HG4_3.STN",
                   delimiter=r"\s+",
                   #delimiter="/t",
                   skiprows=22,
                   names=["angle", "value", "untg"],
                   engine="python",
                   skipinitialspace=True
                   )
data4 = pd.read_csv("data/HG4_4.STN",
                   delimiter=r"\s+",
                   #delimiter="/t",
                   skiprows=22,
                   names=["angle", "value", "untg"],
                   engine="python",
                   skipinitialspace=True
                   )

# data1         ohne Folie 1s
# data2         Folie 7 4sec
# data3         Folie 7 1sec
# data4         Folie 1 4sec


# plt.plot(data1.angle, data1.value)
# plt.plot(data1.angle, data1.untg, label="untergrund")
plt.plot(data1.angle, data1.value-data1.untg)

plt.show()

plt.plot(data1.angle, data1.value)
# plt.plot(data1.angle, data1.untg, label="untergrund")
plt.show()