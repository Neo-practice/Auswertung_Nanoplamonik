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

delta = 13
# 3 Peaks
farbe="#7dcff5"
i = 1
for filename in glob.glob("3peaks/*.csv"):
    data = pd.read_csv(filename, delimiter=",", skiprows=1, names=["pixel", "spec"])
    wavelength = 0.51289659*data.pixel + 423.58859879 - delta
    data = data.fillna(method="ffill")
    pixel = data.pixel
    spec = data.spec

    spec = spec-np.mean(spec[0:150])

    parameters, covariance_matrix = curve_fit(triple_gauss, wavelength, spec, p0=[ 1400,551-delta,7,   1036,591-delta,13,  500, 650-delta, 5])
    fit = triple_gauss(wavelength, *parameters)
    [a1, mu1, sig1, a2, mu2, sig2, a3, mu3, sig3] = parameters
    fit1 = gauss(wavelength, a1, mu1, sig1)
    fit2 = gauss(wavelength, a2, mu2, sig2)
    fit3 = gauss(wavelength, a3, mu3, sig3)

    # plt.plot(wavelength, spec, label="Messdaten")
    plt.plot(wavelength, fit/np.max(fit), color=(0.1+0.02*i, 0.2+0.02*i, 0.5+0.02*i))
    # plt.plot(wavelength, fit1, label="Gauss-Fit 1")
    # plt.plot(wavelength, fit2, label="Gauss-Fit 2")
    # plt.plot(wavelength, fit3, label="Gauss-Fit 3")

    i += 1

plt.plot(wavelength, fit/np.max(fit), color=(0.1+0.02*i, 0.2+0.02*i, 0.5+0.02*i), label="Einzelmessungen")


plt.ylabel('Intensität')
plt.xlabel('$\lambda$ in nm')



ensemble = pd.read_csv("Emmisionspektrum_PBI.txt", delimiter="\s", names=["x", "y"], on_bad_lines="skip",
                           engine="python")
plt.plot(ensemble.x, ensemble.y/np.max(ensemble.y), label="Ensemblemessung", color="red")
plt.xlim([500, 675])

plt.legend()

plt.show()
plt.savefig("plots/ensemblemessung.png")