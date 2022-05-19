import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import pandas as pd


pd.set_option('use_inf_as_na', True)

def gauss(x, a, mu, sig):
    return a/(np.sqrt(2*np.pi*sig**2))*np.exp(-(x-mu)**2/(2*sig**2))

def triple_gauss(x, a1, mu1, sig1, a2, mu2, sig2, a3, mu3, sig3):
    return gauss(x, a1,  mu1, sig1) + gauss(x, a2, mu2, sig2) + gauss(x, a3, mu3, sig3)



wavelength = np.loadtxt("wellenlaengenskala.txt")
wavelength = np.array(wavelength, dtype=float)


plt.figure(1, dpi=130, figsize=(10,6.666))

# numbers = range(1,33)
# for i in numbers:
#     if i < 10:
#         arr = np.loadtxt("Einzelmolekuel_Spektren/0"+str(i)+".csv",
#                         delimiter=",", dtype=str, skiprows=1)
#     else:
#         arr = np.loadtxt(
#             "Einzelmolekuel_Spektren/" + str(i) + ".csv",
#             delimiter=",", dtype=str, skiprows=1)

i = 1
for filename in glob.glob("*.txt"):
    data = pd.read_csv(filename, delimiter=",", skiprows=1, names=["pixel", "spec"])
    data = data.fillna(method="ffill")
    pixel = data.pixel
    spec = data.spec
    # spec = spec.fillna(method='ffill')

    #pixel, spec = np.hsplit(np.array(arr, dtype=float), 2)
    #spec = np.transpose(spec).reshape((len(spec)))
    #pixel = np.transpose(pixel).reshape((len(pixel)))

    spec = spec-np.mean(spec[0:150])

    print(np.any(np.isnan(wavelength)))
    print(np.any(np.isnan(spec)))

    parameters, covariance_matrix = curve_fit(triple_gauss, wavelength, spec, p0=[ 1400,551,7,   1036,591,13,  500, 650, 5])
    fit = triple_gauss(wavelength, *parameters)
    [a1, mu1, sig1, a2, mu2, sig2, a3, mu3, sig3] = parameters
    fit1 = gauss(wavelength, a1, mu1, sig1)
    fit2 = gauss(wavelength, a2, mu2, sig2)
    fit3 = gauss(wavelength, a3, mu3, sig3)

    plt.subplot(6, 6, i)
    plt.plot(wavelength, spec)
    plt.plot(wavelength, fit)
    plt.plot(wavelength, fit1)
    plt.plot(wavelength, fit2)
    plt.plot(wavelength, fit3)

    plt.ylabel('Intensity')
    plt.xlabel('$\lambda$ in nm')

    i+=1


plt.show()
