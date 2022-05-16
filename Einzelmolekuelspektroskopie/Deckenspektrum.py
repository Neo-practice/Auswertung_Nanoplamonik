import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

def lin_func(x, a, b):
    return a+b*x


arr = np.loadtxt("/Users/mariuskaiser/Desktop/PPD/Auswertung_PPD/Einzelmolekuelspektroskopie/bearbeitetes/deckenleuchte01.csv",
                 delimiter=",", dtype=str, skiprows=1)

### Kosmetik
pixel, spec = np.hsplit(np.array(arr, dtype=float), 2)
spec = np.transpose(spec).reshape((len(spec)))
pixel = np.transpose(pixel).reshape((len(pixel)))

hoehe = np.array([276., 555., 61., 61., 538., 56.], dtype=float)

peaks = signal.find_peaks(spec, height=56)
ids= peaks[0][[0,2,3,4,7,8]]

stuetzen = np.array([542.5, 546.2, 577.0, 579.1, 611., 631.])

parameters, covariance_matrix = curve_fit(lin_func, pixel[ids], stuetzen, p0=[ 540, 0.6])
wavelength = lin_func(pixel, *parameters)

np.savetxt("wellenlaengenskala.txt", wavelength)