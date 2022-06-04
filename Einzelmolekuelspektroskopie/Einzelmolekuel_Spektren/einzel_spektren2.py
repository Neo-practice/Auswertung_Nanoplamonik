import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import pandas as pd


pd.set_option('use_inf_as_na', True)

def gauss(x, a, mu, sig):
    return a/(np.sqrt(2*np.pi*sig**2))*np.exp(-(x-mu)**2/(2*sig**2))

def double_gauss(x, a1, mu1, sig1, a2, mu2, sig2):
    return gauss(x, a1, mu1, sig1) + gauss(x, a2, mu2, sig2)

def triple_gauss(x, a1, mu1, sig1, a2, mu2, sig2, a3, mu3, sig3):
    return gauss(x, a1,  mu1, sig1) + gauss(x, a2, mu2, sig2) + gauss(x, a3, mu3, sig3)


wavelength = np.loadtxt("wellenlaengenskala.txt")
wavelength = np.array(wavelength, dtype=float)
# wavelength = wavelength -10


peak1 = []
peak2 = []
peak3 = []

# 3 Peaks
for filename in glob.glob("3peaks/*.csv"):
    data = pd.read_csv(filename, delimiter=",", skiprows=1, names=["pixel", "spec"])
    wavelength = 0.51289659*data.pixel + 423.58859879
    data = data.fillna(method="ffill")
    pixel = data.pixel
    spec = data.spec

    spec = spec-np.mean(spec[0:150])

    parameters, covariance_matrix = curve_fit(triple_gauss, wavelength, spec, p0=[ 1400,551,7,   1036,591,13,  500, 650, 5])
    fit = triple_gauss(wavelength, *parameters)
    [a1, mu1, sig1, a2, mu2, sig2, a3, mu3, sig3] = parameters
    fit1 = gauss(wavelength, a1, mu1, sig1)
    fit2 = gauss(wavelength, a2, mu2, sig2)
    fit3 = gauss(wavelength, a3, mu3, sig3)

    peak1 += [mu1]
    peak2 += [mu2]
    peak3 += [mu3]

    plt.plot(wavelength, spec, label="Messdaten")
    plt.plot(wavelength, fit, label="Summe aller Fits")
    plt.plot(wavelength, fit1, label="Gauss-Fit 1")
    plt.plot(wavelength, fit2, label="Gauss-Fit 2")
    plt.plot(wavelength, fit3, label="Gauss-Fit 3")

    plt.ylabel('Intensity')
    plt.xlabel('$\lambda$ in nm')
    plt.legend()

    filename = filename.replace("3peaks", "")
    file = "plots/" + filename.replace(".csv", "") + ".png"
    plt.savefig(file)
    plt.close("all")


peak1_2 = []
peak2_2 = []

# 2 Peaks
for filename in glob.glob("2peaks/*.csv"):
    data = pd.read_csv(filename, delimiter=",", skiprows=1, names=["pixel", "spec"])
    data = data.fillna(method="ffill")
    pixel = data.pixel
    spec = data.spec
    spec = spec-np.mean(spec[0:150])

    parameters, covariance_matrix = curve_fit(double_gauss, wavelength, spec, p0=[1400, 551, 7, 1036, 591, 13])
    fit = double_gauss(wavelength, *parameters)
    [a1, mu1, sig1, a2, mu2, sig2] = parameters
    fit1 = gauss(wavelength, a1, mu1, sig1)
    fit2 = gauss(wavelength, a2, mu2, sig2)

    peak1_2 += [mu1]
    peak2_2 += [mu2]

    plt.plot(wavelength, spec, label="Messdaten")
    plt.plot(wavelength, fit, label="Summe der Fits")
    plt.plot(wavelength, fit1, label="Gauss-Fit 1")
    plt.plot(wavelength, fit2, label="Gauss-Fit 2")

    plt.ylabel('Intensity')
    plt.xlabel('$\lambda$ in nm')
    plt.legend()

    filename = filename.replace("2peaks", "")
    file = "plots/" + filename.replace(".csv", "") + ".png"
    plt.savefig(file)
    plt.close("all")




bin_range = (530, 650)
bin_num = 20

# plt.hist(peak1+peak2+peak3+peak1_2+peak2_2, range=bin_range, bins=bin_num)

plt.hist(peak1+peak1_2, color="#969696", range=bin_range, bins=bin_num, label="Zusätzliche Messungen")
plt.hist(peak1, color="red", range=bin_range, bins=bin_num, label=r"Übergang 0* $\to$ 0")

plt.hist(peak2+peak2_2, color="#969696", range=bin_range, bins=bin_num)
plt.hist(peak2, color="green", range=bin_range, bins=bin_num, label=r"Übergang 0* $\to$ 1")

plt.hist(peak3, color="blue", range=bin_range, bins=bin_num, label=r"Übergang 0* $\to$ 2")

plt.title("")
plt.xlabel("Wellenlänge in nm")
plt.ylabel("Häufigkeit")
plt.legend()
plt.savefig("plots/histogramm.png")

ensemble = pd.read_csv("Emmisionspektrum_PBI.txt", delimiter="\s", names=["x", "y"], on_bad_lines="skip", engine="python")
# ensemble = np.loadtxt("Emmisionspektrum_PBI.txt", delimiter=" ", dtype="float")


plt.plot(ensemble.x, ensemble.y*16)
plt.xlim([500, 675])
plt.show()