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

einzel_0_100nm = np.loadtxt('spectrometer_0_70nm_einzel_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
wavelength, intensity1 = np.hsplit(einzel_0_100nm, 2)

einzel_90_100nm = np.loadtxt('spectrometer_90_70nm_einzel_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
wavelength, intensity2 = np.hsplit(einzel_90_100nm, 2)


# nehme einfach mal für intensität beide Polarisationen
intensity = intensity1+intensity2

# constants
h = 6.626*np.power(10,-34.)     # J s
c = 299792458.                   # m/s
e = 1.6 * np.power(10,-19.)     # C

# berechne Energieskala
energyscale = np.zeros(wavelength.size)
for i in np.arange(wavelength.size):
    energyscale[i] = h*c/(e*np.array(wavelength[i])*np.power(10,-9.))

# Calculate absorption abs
abs = 1-intensity/ref_0
abs1 = 1-intensity1/ref_0
abs2 = 1-intensity2/ref_0

# Plot data
#plt.plot(wavelength, abs, label='Absorption Daten')
#plt.plot(energyscale, abs, label='Absorption Daten')
plt.plot(energyscale, abs1, label='0 Grad Polarisation')
plt.plot(energyscale, abs2, label='90 Grad Polariation')
plt.xlabel('Energy (eV)')

# Muss noch richtig formatiert werden (warum auch immer, sollte eig. schon richtig sein)
wavelength = np.squeeze(wavelength)
abs = np.squeeze(abs)

# Fit gaussian
#parameters, covariance_matrix = curve_fit(gaussian, wavelength, abs, p0=[650, 50, 0.6])
#mu, sig, fac = parameters
#plt.plot(wavelength, gaussian(wavelength, mu, sig, fac), label="Fit Peak 1")

# fit mit d (zus. Verschiebung in y Richtung)
parameters, covariance_matrix = curve_fit(func1, wavelength, abs, p0=[650, 50, 0.6, 0.1])
mu, sig, fac, d = parameters
#plt.plot(wavelength, func1(wavelength, mu, sig, fac, d), label="Fit Peak 2")



# calculate fwhm
fwhm = 2 * np.sqrt(2*np.log(2)) * sig

print(fwhm)


plt.legend()
plt.show()




