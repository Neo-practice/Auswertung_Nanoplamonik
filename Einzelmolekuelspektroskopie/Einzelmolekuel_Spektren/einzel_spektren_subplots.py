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

peak1_er = []
peak2_er = []
peak3_er = []



fig, axs = plt.subplots(2, 2)
j = 0

# 3 Peaks
delta = 13
for i, filename in enumerate([r"3peaks/22.csv", r"3peaks/01.csv", r"3peaks/21_02.csv"]):
    data = pd.read_csv(filename, delimiter=",", skiprows=1, names=["pixel", "spec"])
    wavelength = 0.51289659*data.pixel + 423.58859879 - delta
    data = data.fillna(method="ffill")
    pixel = data.pixel
    spec = data.spec
    spec = spec-np.mean(spec[0:150])

    # a1, mu1, sig1, a2, mu2, sig2, a3, mu3, sig3
    parameters, covariance_matrix = curve_fit(triple_gauss, wavelength, spec, p0=[ 1400,551-delta,7,   1036,591-delta,13,  500, 650-delta, 5])
    fit = triple_gauss(wavelength, *parameters)
    [a1, mu1, sig1, a2, mu2, sig2, a3, mu3, sig3] = parameters
    fit1 = gauss(wavelength, a1, mu1, sig1)
    fit2 = gauss(wavelength, a2, mu2, sig2)
    fit3 = gauss(wavelength, a3, mu3, sig3)

    peak1 += [mu1]
    peak2 += [mu2]
    peak3 += [mu3]

    peak1_er += [np.sqrt(np.abs(covariance_matrix[1][1]))]
    peak2_er += [np.sqrt(np.abs(covariance_matrix[4][4]))]
    peak3_er += [np.sqrt(np.abs(covariance_matrix[7][7]))]

    a = np.max(spec)

    if i == 0:
        ax = axs[0, 0]
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel('Intensität')
    elif i == 1:
        ax = axs[0, 1]
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    elif i == 2:
        ax = axs[1, 0]
        ax.set_xlabel('$\lambda$ in nm')
        ax.set_ylabel('Intensität')

    ax.set_xlim(441, 765)
    ax.set_ylim(-0.1, 1.1)

    ax.plot(wavelength, spec/a, label="_Messdaten", color="#969696")
    ax.plot(wavelength, fit1/a, label=r"_Gauss-Fit an Peak 0* $\to$ 0", color="red", linestyle="--")
    ax.plot(wavelength, fit2/a, label=r"_Gauss-Fit an Peak 0* $\to$ 1", color="green", linestyle="--")
    ax.plot(wavelength, fit3/a, label=r"_Gauss-Fit an Peak 0* $\to$ 2", color="blue", linestyle="--")
    # ax.plot(wavelength, fit / a, label="Summe aller Fits", color="black")

    j = i

    #filename = filename.replace("3peaks", "")
    #file = "plots/" + filename.replace(".csv", "") + ".png"
    #plt.savefig(file)
    #plt.close("all")


peak1_2 = []
peak2_2 = []

peak1_2_er = []
peak2_2_er = []

# 2 Peaks
for i, filename in enumerate([r"2peaks/14_02.csv"]):

    data = pd.read_csv(filename, delimiter=",", skiprows=1, names=["pixel", "spec"])
    data = data.fillna(method="ffill")
    pixel = data.pixel
    spec = data.spec
    spec = spec-np.mean(spec[0:150])

    parameters, covariance_matrix = curve_fit(double_gauss, wavelength, spec, p0=[1400, 551-delta, 7, 1036, 591-delta, 13])
    fit = double_gauss(wavelength, *parameters)
    [a1, mu1, sig1, a2, mu2, sig2] = parameters
    fit1 = gauss(wavelength, a1, mu1, sig1)
    fit2 = gauss(wavelength, a2, mu2, sig2)

    peak1_2 += [mu1]
    peak2_2 += [mu2]

    peak1_2_er += [np.sqrt(np.abs(covariance_matrix[1][1]))]
    peak2_2_er += [np.sqrt(np.abs(covariance_matrix[4][4]))]

    a = np.max(spec)

    ax = axs[1,1]

    ax.set_xlim(441, 765)
    ax.set_ylim(-0.1, 1.1)
    ax.yaxis.set_ticklabels([])

    ax.plot(wavelength, spec / a, label="Messdaten", color="#969696")
    ax.plot(wavelength, fit1 / a, label=r"Gauss-Fit an Peak 0* $\to$ 0", color="red", linestyle="--")
    ax.plot(wavelength, fit2 / a, label=r"Gauss-Fit an Peak 0* $\to$ 1", color="green", linestyle="--")
    ax.plot([], [], label=r"Gauss-Fit an Peak 0* $\to$ 2", color="blue", linestyle="--") # damit Fit 3 angezeigt wird
    # ax.plot(wavelength, fit / a, label="Summe aller Fits", color="black")

    ax.set_xlabel('$\lambda$ in nm')

# fig.legend()
plt.tight_layout(rect=[0, 0.1, 1, 1])
fig.legend(bbox_to_anchor=(0.105, 0, 0.884, 0.1), loc="lower left", mode="expand", bbox_transform=fig.transFigure, ncol=2)

plt.show()
fig.savefig(r"plots/4plots.png")
#plt.close("all")







