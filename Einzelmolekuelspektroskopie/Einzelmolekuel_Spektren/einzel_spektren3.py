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


# 3 Peaks
delta = 13
for filename in glob.glob("3peaks/*.csv"):
    data = pd.read_csv(filename, delimiter=",", skiprows=1, names=["pixel", "spec"])
    wavelength = 0.51289659*data.pixel + 423.58859879 - delta
    data = data.fillna(method="ffill")
    pixel = data.pixel
    spec = data.spec
#
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

    plt.plot(wavelength, spec/a, label="Messdaten", color="#969696")
    plt.plot(wavelength, fit/a, label="Summe aller Fits", color="black")
    plt.plot(wavelength, fit1/a, label=r"Gauss-Fit an Peak 0* $\to$ 0", color="red", linestyle="--")
    plt.plot(wavelength, fit2/a, label=r"Gauss-Fit an Peak 0* $\to$ 1", color="green", linestyle="--")
    plt.plot(wavelength, fit3/a, label=r"Gauss-Fit an Peak 0* $\to$ 2", color="blue", linestyle="--")
    # plt.plot(wavelength, fit / a, label="Summe aller Fits", color="black")

    plt.ylabel('Intensität')
    plt.xlabel('$\lambda$ in nm')
    plt.legend()

    filename = filename.replace("3peaks", "")
    file = "plots/" + filename.replace(".csv", "") + ".png"
    plt.savefig(file)
    plt.close("all")


peak1_2 = []
peak2_2 = []

peak1_2_er = []
peak2_2_er = []

# 2 Peaks
for filename in glob.glob("2peaks/*.csv"):
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

    plt.plot(wavelength, spec / a, label="Messdaten", color="#969696")
    plt.plot(wavelength, fit / a, label="Summe aller Fits", color="black")
    plt.plot(wavelength, fit1 / a, label=r"Gauss-Fit an Peak 0* $\to$ 0", color="red", linestyle="--")
    plt.plot(wavelength, fit2 / a, label=r"Gauss-Fit an Peak 0* $\to$ 1", color="green", linestyle="--")


    plt.ylabel('Intensity')
    plt.xlabel('$\lambda$ in nm')
    plt.legend()

    filename = filename.replace("2peaks", "")
    file = "plots/" + filename.replace(".csv", "") + ".png"
    plt.savefig(file)
    plt.close("all")


peak1cm = (np.mean(peak1+peak1_2))*10**(-6) # in cm
peak2cm = (np.mean(peak2+peak2_2))*10**(-6) # in cm
peak3cm = (np.mean(peak3))*10**(-6) # in cm
uebergang1=peak1cm
uebergang2=peak2cm
uebergang3=peak3cm


peak1nm = (np.mean(peak1+peak1_2)) # in nm
peak2nm = (np.mean(peak2+peak2_2)) # in nm
peak3nm = (np.mean(peak3)) # in nm



# Energiedifferenz der Übergänge
delta1 = (1/peak1nm-1/peak2nm) * 1/1.000272 * 10**7 # von nm in cm-1
delta2 = (1/peak2nm-1/peak3nm) * 1/1.000272 * 10**7
fehler1 = np.sqrt(
        (1/(peak2nm*peak2nm) * 1/1.000272 * 10**7 * 4)**2 +
        (1/(peak1nm*peak1nm) * 1/1.000272 * 10**7 * 3)**2
        )
fehler2 = np.sqrt(
        (1/(peak2nm*peak2nm) * 1/1.000272 * 10**7 * 4)**2 +
        (1/(peak3nm*peak3nm) * 1/1.000272 * 10**7 * 6)**2
        )

print("------------Berechnete Fehler (nm)--------------")
print("Übergang nach 0:", 1/uebergang1, " cm-1, ", np.mean(peak1+peak1_2), " nm")
print("Fehler: ", 1/(uebergang1 - np.mean(peak1_er+peak1_2_er)*10**(-6))-1/uebergang1, " cm-1", np.mean(peak1_er+peak1_2_er), " nm")
print("Übergang nach 1:", 1/uebergang2, " cm-1, ", np.mean(peak2+peak2_2), " nm")
print("Fehler: ", 1/(uebergang2-np.mean(peak2_er+peak2_2_er)*10**(-6))-1/uebergang2, " cm-1", np.mean(peak2_er+peak2_2_er), " nm")
print("Übergang nach 2:", 1/uebergang3, " cm-1, ", np.mean(peak3), " nm")
print("Fehler: ", 1/(uebergang3-np.mean(peak3_er)*10**(-6))-1/uebergang3, " cm-1", np.mean(peak3_er), " nm")
print("----------Selbstgewählte Fehler (nm)------------")
print("Übergang nach 0:", 1/uebergang1, " cm-1, ", np.mean(peak1+peak1_2), " nm")
print("Fehler: ", 1/(uebergang1 -3*10**(-6))-1/uebergang1, " cm-1", np.mean(peak1_er+peak1_2_er), " nm")
print("Übergang nach 1:", 1/uebergang2, " cm-1, ", np.mean(peak2+peak2_2), " nm")
print("Fehler: ", 1/(uebergang2-4*10**(-6))-1/uebergang2, " cm-1", np.mean(peak2_er+peak2_2_er), " nm")
print("Übergang nach 2:", 1/uebergang3, " cm-1, ", np.mean(peak3), " nm")
print("Fehler: ", 1/(uebergang3-6*10**(-6))-1/uebergang3, " cm-1", np.mean(peak3_er), " nm")
print("--------Energie mit selbstgew. Fehlern-----------")
print("E_12 = ", delta1, " +/- ", fehler1)
print("E_23 = ", delta2, " +/- ", fehler2)

bin_range = (530, 650)
bin_num = 20

# plt.hist(peak1+peak2+peak3+peak1_2+peak2_2, range=bin_range, bins=bin_num)

plt.hist(peak1+peak1_2, color="#969696", range=bin_range, bins=bin_num, label="Zusätzliche Messungen")
plt.hist(peak1, color="red", range=bin_range, bins=bin_num, label=r"Übergang 0* $\to$ 0")

plt.hist(peak2+peak2_2, color="#969696", range=bin_range, bins=bin_num)
plt.hist(peak2, color="green", range=bin_range, bins=bin_num, label=r"Übergang 0* $\to$ 1")

plt.hist(peak3, color="blue", range=bin_range, bins=bin_num, label=r"Übergang 0* $\to$ 2")

plt.title("")
plt.xlabel(r"$\lambda$ in nm")
plt.ylabel("Häufigkeit")
plt.legend()
plt.xlim([500, 675])
plt.savefig("plots/histogramm.png")

ensemble = pd.read_csv("Emmisionspektrum_PBI.txt", delimiter="\s", names=["x", "y"], on_bad_lines="skip", engine="python")
# ensemble = np.loadtxt("Emmisionspektrum_PBI.txt", delimiter=" ", dtype="float")


plt.plot(ensemble.x, ensemble.y*16, label="Ensemblemessung", color="black")
plt.xlim([500, 675])
plt.legend()
# plt.show()
plt.savefig("plots/histogramm_ensemble.png")
# plt.show()

print("-----------------Peaks-Infos---------------------")
print("Anzahl gemeinsame Peaks: ", len(peak1+peak1_2))
print("Anzahl mit Peak 3:       ", len(peak1))
