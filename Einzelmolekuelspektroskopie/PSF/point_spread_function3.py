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

def error_fwhm(par, cov, func, x):
    greater_par = []
    lower_par = []
    for i in range(len(par)):
        greater_par += [par[i] + np.sqrt(np.abs(cov[i][i]))]
        lower_par += [par[i] - np.sqrt(np.abs(cov[i][i]))]
    greater_fwhm = FWHM(x, func(x, *greater_par))
    lower_fwhm = FWHM(x, func(x, *lower_par))
    return np.abs((greater_fwhm[0]-lower_fwhm[0])/2)


zoom = 47
pixelsize = 6.45 * 10**(-6)  # m

def xaxis_inverse(xaxis):
    data = xaxis * zoom /pixelsize + w0
    return data

all_fwhm = []
all_errors = []

# os.chdir("PSF"), uses current folder
# for filename in ["2.csv"]:
# for filename in glob.glob("*.csv"):

for i in range(1, 16, 1):
    if i in [2, 3, 7, 10]: # aus irgendeinem Grund schmeißen die hier fehler...
        continue
    filename = str(i) + ".csv"
    # print(filename)
    data = pd.read_csv(filename, skiprows=1, names=["x", "y"])

    # w0, gamma, mu, sig, fac, offset

    param = [22.0, 10.0, 22.0, 100000.0, 100000000.0, 820.0]

    if i in [1, 5, 7, 9, 14]:
        param = [22.0, 10.0, 22.0, 100000.0, 100000000.0, 820.0]
    elif i in [4, 11, 13, 15]:
        param = [22.0, 10.0, 22.0, 100000.0, 1000000000.0, 820.0]
    elif i in [8, 10, 12]:
        param = [22.0, 10.0, 22.0, 500000.0, 100000000.0, 820.0]
    elif i in [3]:
        param = [22.0, 10.0, 22.0, 80000.0, 100000000.0, 820.0]
    elif i == 2:
        param = [22.0, 10.0, 22.0, 100000.0, 50000000.0, 820.0]
    elif i == 6:
        param = [22.0, 10.0, 22.0, 80000.0, 50000000.0, 820.0]

    # [2, 3, 6, 8, 10, 12]
    # param = [22.0, 10.0, 22.0, 100000.0, 100000000.0, 820.0]
    # works for 1, 5, 7, 9, 14,
    # doesnt calculate cov.ma for 3, 10
    # fails for 2, 4, 6, 8, 11, 12, 13, 15

    parameters, covariance_matrix = curve_fit(faltung, data.x, data.y, p0=param)
    w0, gamma, mu, sig, fac, offset = parameters

    # recalibrating x-axis
    xaxis = (data.x-w0) * pixelsize / zoom

    # refining fit-x-Axis
    fitaxis = np.arange(np.min(xaxis), np.max(xaxis), 0.0000000005)
    fitplot = faltung(xaxis_inverse(fitaxis), *parameters)

    # calculating FWHM and error
    fwhm, index_links, index_rechts = FWHM(fitaxis, faltung(xaxis_inverse(fitaxis), *parameters))
    # error = 0

    # print(i)
    # print(covariance_matrix)
    error = error_fwhm(parameters, covariance_matrix, faltung, fitaxis)

    all_fwhm += [fwhm]
    all_errors += [error]

    a = np.max(faltung(xaxis_inverse(fitaxis), w0, gamma, mu, sig, fac, offset))
    # Plotting
    plt.plot(fitaxis, faltung(xaxis_inverse(fitaxis), w0, gamma, mu, sig, fac, offset)/a, label="Fit Voigt-Funktion")
    plt.plot(xaxis, data.y/a, label="Messwerte")
    plt.plot(fitaxis[index_links], fitplot[index_links]/a, marker="+", color="black")
    plt.plot(fitaxis[index_rechts], fitplot[index_rechts]/a, marker="+", color="black")
    plt.legend()
    plt.ylabel("Normierte Intensität")
    plt.xlabel("Aufweitung in m")
    # plt.show()
    file = "test/" + str(i) + ".png"
    plt.savefig(file)
    plt.close('all')

print("Mittelwert:    ", np.mean(all_fwhm))
print("Fehler:        ", np.mean(all_errors))
print("Anzahl Kurven: ", len(all_fwhm))

# print(all_fwhm)

