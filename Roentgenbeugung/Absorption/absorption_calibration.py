import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#######################################################################################################
# constants
#######################################################################################################
e = 1.602176634e-19 # C
c = 299792458 # m/s
h = 6.62607015e-34

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


trans_data = pd.read_csv("data/wolfram_transitions.txt",
                   delimiter=r"\s+",
                   #delimiter="/t",
                   skiprows=23,
                   names=["Ele", "trans",
                          "value", "Unc1",
                          "Direct", "Unc2",
                          "Combined", "Unc3",
                          "Vapor", "Unc4",
                          "Blend", "Ref"],
                   engine="python",
                   skipinitialspace=True
                   )
# übergänge "trans" in eV zu Angström
trans_energy = trans_data.value.to_numpy() # in eV
trans_lambda = h * c / (trans_energy * e) # in m
d = {'transition': list(trans_data.trans), 'lambda': list(trans_lambda)}
df = pd.DataFrame(data=d)
df.to_csv('data/wolfram_transitions_in_angstroem.txt', header=None, index=None, sep=' ')


#plt.plot(data1.angle, data1.value)
# plt.plot(data1.angle, data1.untg, label="untergrund")
#plt.show()

# Umrechnung der Winkel in Wellenlängen:
a = 5.463e-10 # m
d = np.sqrt(a*a / (2*2+2*2+0*0)) # d_hkl in m
thet = data1.angle.to_numpy() * (2*np.pi/360)
lamb = a / np.sqrt(2) * np.sin(thet) # in Angström


#plt.plot(lamb, data1.value)
#plt.show()

peaks_data = np.array([0.9931, 1.0280, 1.0646, 1.1870, 1.2115, 1.2466, 1.4435,
              2.0235, 2.0922, 2.1683, 2.4646, 2.5035, 2.5424])*10**(-10)
peaks_theo = np.array([1.06186, 1.06796, 1.09862, 1.2626, 1.24445, 1.28187,
              1.476424, 2.123, 2.136, 2.197, 2.488, 2.526, 2.561])*10**(-10)
peaks_trans= [r"$L_\gamma_3$", r"$L_\gamma_2$", r"$L_\gamma_1$", r"$L_\beta_3$",
              r"$L_\beta_2$", r"$L_\beta_1$", r"$L_\alpha_1$", r"$L_\gamma_3*",
              r"$L_\gamma_2*", r"$L_\gamma_1*", r"$L_\beta_3*", r"$L_\beta_2*",
              r"$L_\beta_1*"]

def f(x, m, b):
    return m*x+b

(m, b), covariance_matrix = curve_fit(f, xdata=peaks_data, ydata=peaks_theo, p0=[1.0, 1.0])
# m = 0.9876039005116924
# b = 6.371626387990272e-12

plt.plot(peaks_data, peaks_theo, marker="+", linestyle='None', label="Peaks von Wolfram")
plt.plot(peaks_data, f(peaks_data, m, b), label="Fitgerade")
plt.xlabel("$\lambda_{mess}$ in $\AA$")
plt.ylabel("$\lambda_{theo}$ in $\AA$")
plt.legend()
plt.savefig("plots/calibration_fit.png")
#plt.show()
plt.close("all")

def to_wavelength(x):
    # takes dataframe-slice with angles
    # returns calibrated wavelength in Angström
    thet = x.to_numpy() * (2 * np.pi / 360)
    lamb = a / np.sqrt(2) * np.sin(thet)
    return lamb*m+b

plt.plot(to_wavelength(data1.angle), data1.value, label="data")
plt.savefig("plots/calibration_data.png")
plt.show()
plt.close(("all"))
