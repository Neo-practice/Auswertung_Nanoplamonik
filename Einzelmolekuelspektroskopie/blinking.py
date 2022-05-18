import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

arr = np.loadtxt("/Users/mariuskaiser/Desktop/PPD/Auswertung_PPD/Einzelmolekuelspektroskopie/bearbeitetes/blinking_13.csv",
                        delimiter=",", dtype=str, skiprows=1)

frame, area, mean, min, max, rest1, rest2, slice = np.hsplit(np.array(arr, dtype=float), 8)
frame = np.transpose(frame).reshape((len(frame)))
mean = np.transpose(mean).reshape((len(mean)))
max = np.transpose(max).reshape((len(max)))

plt.plot(mean)
plt.show()
