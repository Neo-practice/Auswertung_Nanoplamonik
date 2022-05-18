import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def lorentz( x, w0, gamma, d):
    return 1/( (x**2 - w0**2 )**2 + gamma**2*w0**2)+d

def gaussian(x, mu, sig, fac):
    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian1(x, mu, sig, fac, d):
    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) +d

def faltung( x, w0, gamma, mu, sig, fac , offset):
    return lorentz(x, w0, gamma, offset)*gaussian1(x, mu, sig, fac, offset)

def FWHM( x, y , orig): # Gibt Float-Array zurück [ fwhm, fwhm_fehler, index links, index rechts ]
    ID = np.zeros(2, dtype = int)
    half_values = np.zeros(2, dtype=float)
    ID_max = list(y).index(y.max())
    half_values[0] = find_nearest(y[0:ID_max-1],y.max()/2)
    half_values[1] = find_nearest(y[ID_max + 1:-1], y.max() / 2)
    ID[0] = list(y).index(half_values[0])
    ID[1] = list(y).index(half_values[1])
    #### berechne Unsicherheit
    ## berechne jede Abweichung in w-Richtung und Intensitätsrichtung (zweiteres müsste noch)
    bereich=1     #Bereich um halbes Maximum
    abweichung_links = np.zeros(2*bereich)     ## Abweichung links/rechts durch Rauschen
    abweichung_rechts = np.zeros(2*bereich)
    abweichung_oben = np.zeros(2 * bereich) ## Abweichung Modell zur Peak-Hoehe
    j = 0
    while j < bereich: # besetze Array mit echten Abweichungen
        abweichung_links[bereich+j] = np.abs(y[ID[0]+j]-find_nearest(orig[0:ID_max], y[ID[0]+j]))
        abweichung_links[bereich - j-1] = np.abs(y[ID[0]-j-1]-find_nearest(orig[ID_max:-1], y[ID[0]-j-1]))
        abweichung_rechts[bereich+j] = np.abs(y[ID[1]+j]-find_nearest(orig[0:ID_max], y[ID[1]+j]))
        abweichung_rechts[bereich - j-1] = np.abs(y[ID[1]-j-1]-find_nearest(orig[ID_max:-1], y[ID[1]-j-1]))
        abweichung_oben[bereich+j] = np.abs(y[ID_max+j]-find_nearest(orig, y[ID_max+j]))
        abweichung_oben[bereich - j -1] = np.abs(y[ID_max - j-1]-find_nearest(orig, y[ID_max - j-1]))
        j += 1
    delta_w = np.sqrt(np.max(abweichung_links)**2 + np.max(abweichung_rechts)**2 + np.max(abweichung_oben)**2)
    #print(delta_w)
    fwhm = np.array( [np.abs(x[ID[0]]-x[ID[1]]), delta_w, ID[0], ID[1]  ], dtype = float )
    #print(ID)
    return fwhm



filename = "15.csv"

data = pd.read_csv(filename, skiprows=1, names=["x", "y"])

# w0, gamma, mu, sig, fac, offset
param = [23.0, 1.0, 23.0, 1.0, 2000.0, 700.0]
parameters, covariance_matrix = curve_fit(faltung, data.x, data.y, p0=param)
w0, gamma, mu, sig, fac, offset = parameters
plt.plot(faltung(data.x, w0, gamma, mu, sig, fac, offset))

plt.plot(data.x, data.y)
plt.show()



