import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# werte = np.loadtxt("dips4.txt")
# x, y, b, nichts, c, d = np.hsplit(werte, 6)


# Define Gauß Function (not normalized)
def func1(x, mu, sig, fac, d):
    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) +d

def gaussian(x, mu, sig, fac):
    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

x = np.arange(-1000, 1000, 0.5)
# plt.plot(x, gaussian(x, 650, 50, 0.6))



# Define left and right borders to fit
# l = 580
# r = 1070

# load data
referenz_0 = np.loadtxt('spectrometer_0_referenz_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
wavelength, ref_0 = np.hsplit(referenz_0, 2)

hybrid_0_20nm = np.loadtxt('spectrometer_0_20nm_hybrid_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
wavelength, intensity = np.hsplit(hybrid_0_20nm, 2)

einzel_0_100nm = np.loadtxt('spectrometer_0_100nm_einzel_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
wavelength, intensity = np.hsplit(einzel_0_100nm, 2)

einzel_90_100nm = np.loadtxt('spectrometer_90_100nm_einzel_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
wavelength, intensity = np.hsplit(einzel_90_100nm, 2)



# Calculate absorption abs
abs = 1-intensity/ref_0

# Plot data
plt.plot(wavelength, abs, label='Absorption Daten')

# Muss noch richtig formatiert werden (warum auch immer, sollte eig. schon richtig sein)
wavelength = np.squeeze(wavelength)
abs = np.squeeze(abs)

# Fit gaussian
parameters, covariance_matrix = curve_fit(gaussian, wavelength, abs, p0=[650, 50, 0.6])
mu, sig, fac = parameters
plt.plot(wavelength, gaussian(wavelength, mu, sig, fac), label="Fit Peak 1")

# fit mit d (zus. Verschiebung in y Richtung)
parameters, covariance_matrix = curve_fit(func1, wavelength, abs, p0=[650, 50, 0.6, 0.1])
mu, sig, fac, d = parameters
plt.plot(wavelength, func1(wavelength, mu, sig, fac, d), label="Fit Peak 2")

# calculate fwhm
fwhm = 2 * np.sqrt(2*np.log(2)) * sig


plt.legend()
plt.show()




