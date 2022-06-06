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


# Umrechnung der Winkel in Wellenlängen:
a = 5.463e-10 # m
d = np.sqrt(a*a / (2*2+2*2+0*0)) # d_hkl in m
# Konstanten zur Kalibierierung
m = 0.9876039005116924
b = 6.371626387990272e-12

def to_wavelength(x):
    # takes dataframe-slice with angles
    # returns calibrated wavelength in Angström
    thet = x.to_numpy() * (2 * np.pi / 360)
    lamb = a / np.sqrt(2) * np.sin(thet)
    return lamb*m+b




plt.xlabel("$\lambda_{mess}$ in $\AA$")
plt.ylabel("$\lambda_{theo}$ in $\AA$")
plt.plot(to_wavelength(data4.angle), data4.value-data1.value, label="mit Folie 1")
#plt.plot(to_wavelength(data2.angle), data2.value, label="mit Folie 7")
plt.plot(to_wavelength(data1.angle), data1.value, label="ohne Folie")
plt.legend()
plt.savefig("plots/folie7_4s.png")
plt.show()
plt.close(("all"))
