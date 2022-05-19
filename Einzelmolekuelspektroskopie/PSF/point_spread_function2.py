import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import glob, os

def lorentz( x, w0, gamma, d):
    return 1/( (x**2 - w0**2 )**2 + gamma**2*w0**2)+d

def lorentz1( x, w0, gamma, d, fac):
    return fac * 1/( (x**2 - w0**2 )**2 + gamma**2*w0**2)+d

def gaussian(x, mu, sig, fac):
    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian1(x, mu, sig, fac, d):
    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) +d

def faltung( x, w0, gamma, mu, sig, fac , offset):
    return lorentz(x, w0, gamma, offset)*gaussian1(x, mu, sig, fac, offset)

def FWHM(x, y):
    fullmax = np.max(y) - np.min(y)
    fullmax_id = np.argmax(y)
    diff = np.abs(y-(np.min(y)+fullmax/2))
    left_id = np.argmin(diff[0:fullmax_id])
    right_id = np.argmin(diff[fullmax_id:])+ fullmax_id - 1
    fwhm = x[right_id]-x[left_id]
    return fwhm, left_id, right_id


# os.chdir("PSF"), uses current folder
# for filename in glob.glob("*.csv"):
    # print(filename)

filename = "5.csv"
data = pd.read_csv(filename, skiprows=1, names=["x", "y"])

zoom = 47
pixelsize = 6.45 * 10**(-6)

# w0, gamma, mu, sig, fac, offset
param = [22.0, 10.0, 22.0, 100000.0, 100000000.0, 820.0]
parameters, covariance_matrix = curve_fit(faltung, data.x, data.y, p0=param)
w0, gamma, mu, sig, fac, offset = parameters


def error_fwhm(par, cov, func, x):
    greater_par = []
    lower_par = []
    for i in range(len(par)):
        greater_par += [par[i] + np.sqrt(cov[i][i])]
        lower_par += [par[i] - np.sqrt(cov[i][i])]
    greater_fwhm = FWHM(x, func(x, *greater_par))
    lower_fwhm = FWHM(x, func(x, *lower_par))
    return np.abs((greater_fwhm[0]-lower_fwhm[0])/2)

# Plot auf Abstand geeicht
xaxis = (data.x-w0) * pixelsize / zoom

def xaxis_inverse(xaxis):
    data = xaxis * zoom /pixelsize + w0
    return data

# recalibrating fit-x-Axis
fitaxis = np.arange(np.min(xaxis), np.max(xaxis), 0.0000000005)
fitplot = faltung(xaxis_inverse(fitaxis), w0, gamma, mu, sig, fac, offset)

# calculating FWHM
fwhm, index_links, index_rechts = FWHM(fitaxis, faltung(xaxis_inverse(fitaxis), *parameters))
error = error_fwhm(parameters, covariance_matrix, faltung, fitaxis)





# Plotting
plt.plot(fitaxis[index_links], fitplot[index_links], marker="+", color="red")
plt.plot(fitaxis[index_rechts], fitplot[index_rechts], marker="+", color="red")
plt.plot(fitaxis, faltung(xaxis_inverse(fitaxis), w0, gamma, mu, sig, fac, offset), label="Fit Voigt-Funktion")
plt.plot(xaxis, data.y, label="Messwerte")
plt.legend()
plt.ylabel("Intensität")
plt.xlabel("Abstand")
plt.show()