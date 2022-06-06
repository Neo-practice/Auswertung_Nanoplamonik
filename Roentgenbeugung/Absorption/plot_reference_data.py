import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

e = 1.602176634e-19 # C
c = 299792458 # m/s
h = 6.62607015e-34


# Umrechnung der Winkel in Wellenlängen:
a = 5.463e-10 # m
d = np.sqrt(a*a / (2*2+2*2+0*0)) # d_hkl in m

def eV_to_wavelength(x):
    return h*c/(x*e)

ag = pd.read_csv("reference_data/ag.txt",
                   delimiter=r"\s+",
                   names=["energy", "y1", "y2"],
                   engine="python"
                   )

plt.plot(eV_to_wavelength(ag.energy), ag.y1, label="y1")
plt.plot(eV_to_wavelength(ag.energy), ag.y2, label="y2")
plt.show()

