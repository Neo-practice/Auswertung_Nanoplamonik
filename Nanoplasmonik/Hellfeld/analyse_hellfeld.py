import matplotlib.pyplot
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# werte = np.loadtxt("dips4.txt")
# x, y, b, nichts, c, d = np.hsplit(werte, 6)


# Define Gauß Function (not normalized)
def func1(x, mu, sig, fac, d):
    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) +d

def lorentz( x, w0, gamma):
    return 1/( (x**2 - w0**2 )**2 + gamma**2*w0**2)

def gaussian(x, mu, sig, fac):
    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def faltung( x, w0, gamma, mu, sig, fac ):
    return lorentz(x, w0, gamma)*gaussian(x, mu, sig, fac)

#def gaussian(x, mu, sig, fac):
#    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# constants
h = 6.626*10**-34.     # J s
c = 299792458.         # m/s
e = 1.6 * 10**-19.     # C

x = np.arange(-1000, 1000, 0.5)
# plt.plot(x, gaussian(x, 650, 50, 0.6))



# Define left and right borders to fit
# l = 580
# r = 1070



# load referenzes
referenz_0 = np.loadtxt('spectrometer_0_referenz_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
wavelength, ref_0 = np.hsplit(referenz_0, 2)

referenz_90 = np.loadtxt('spectrometer_90_referenz_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
wavelength, ref_90 = np.hsplit(referenz_0, 2)

# berechne Energie bzw. Winkelfrequenzskala
energyscale = np.zeros(wavelength.size)
w = np.zeros(wavelength.size)
for i in np.arange(wavelength.size):
    energyscale[i] = h * c / (e * np.array(wavelength[i]) * np.power(10, -9.))
    w[i] = 2 * np.pi * c / (wavelength[i] * 10 ** (-9)) * 10 ** (-15.)

########## Auswertung Einzel-Partikel
##################################################################

groesse = ['100', '70']
print(np.arange(len(groesse)))

for i in np.arange(len(groesse)):

    einzel_0 = np.loadtxt('spectrometer_0_'+groesse[i]+'nm_einzel_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
    wavelength, intensity1 = np.hsplit(einzel_0, 2)
    einzel_90 = np.loadtxt('spectrometer_90_'+groesse[i]+'nm_einzel_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
    wavelength, intensity2 = np.hsplit(einzel_90, 2)

    abs1 = 1-intensity1/ref_0
    abs2 = 1-intensity2/ref_90
    plt.figure(i)
    plt.plot(w, abs1, label='0 Grad Polarisation')
    plt.plot(w, abs2, label='90 Grad Polariation')
    plt.xlabel('omega *10^(15)')
    plt.title('Extinktionsspektrum für '+groesse[i]+' nm Nanopartikel')

    w = np.squeeze(w)
    abs1 = np.squeeze(abs1)
    abs2 = np.squeeze(abs2)

    start = round(w.size*2/5)
    end = w.size
    range_help = [*range(start,end,1) ]

    lorentz_parameter = np.array([2.8, 0.5, 0.1], dtype=float)
    gauss_parameter = np.array([3.0, 0.1, 0.2, 0.1], dtype=float)

    fwhm = np.zeros(2)

    #plt.plot(w,lorentz(w,2.9, 0.5))
    #plt.plot(w,gaussian(w,3.4, 0.3, 0.3))
    #plt.plot(w,lorentz(w,3.0, 0.4)*gaussian(w,3.3, 0.5, 0.3))
    parameters, covariance_matrix = curve_fit(faltung, w, abs1, p0=[ 3.0, 0.4, 3.3, 0.5, 0.3 ])
    w0, gamma, mu, sig, fac = parameters
    fwhm[0] = np.sqrt( w0**2+gamma*w0 ) - np.sqrt( w0**2-gamma*w0 )
    plt.plot(w, faltung(w, w0, gamma, mu, sig, fac),
             label="Fit 0 Grad, FWHM = "+str(round(fwhm[0]*100)/100)+" *10^(15)")

    parameters, covariance_matrix = curve_fit(faltung, w, abs2, p0=[3.0, 0.4, 3.3, 0.5, 0.3])
    w0, gamma, mu, sig, fac = parameters
    fwhm[1] = np.sqrt(w0 ** 2 + gamma * w0) - np.sqrt(w0 ** 2 - gamma * w0)
    plt.plot(w, faltung(w, w0, gamma, mu, sig, fac),
             label="Fit 0 Grad, FWHM = " + str(round(fwhm[1] * 100) / 100) + " *10^(15)")

    print(fwhm, " in 10^(15)")
    plt.legend()


## Load Hybridisierte



option_größe = ['20' , '30' , '40' , '50' , '60']


hybrid_0_20nm = np.loadtxt('spectrometer_0_20nm_hybrid_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
wavelength, intensity = np.hsplit(hybrid_0_20nm, 2)




# nehme einfach mal für intensität beide Polarisationen
intensity = intensity1+intensity2





# Calculate absorption abs
abs = 1-intensity/ref_0


# Plot data
#plt.plot(wavelength, abs, label='0 Grad Polarisation')

#plt.plot(wavelength, abs1, label='0 Grad Polarisation')
#plt.plot(wavelength, abs2, label='90 Grad Polariation')
#plt.xlabel('Wellenlänge (nm)')

#plt.plot(energyscale, abs1, label='0 Grad Polarisation')
#plt.plot(energyscale, abs2, label='90 Grad Polariation')
#plt.xlabel('Energy (eV)')



# Muss noch richtig formatiert werden (warum auch immer, sollte eig. schon richtig sein)
wavelength = np.squeeze(wavelength)

abs = np.squeeze(abs)


# Fit gaussian
#parameters, covariance_matrix = curve_fit(gaussian, wavelength, abs, p0=[650, 50, 0.6])
#mu, sig, fac = parameters
#plt.plot(wavelength, gaussian(wavelength, mu, sig, fac), label="Fit Peak 1")

# fit mit d (zus. Verschiebung in y Richtung)
parameters, covariance_matrix = curve_fit(func1, wavelength, abs1, p0=[650, 50, 0.6, 0.1])
mu, sig, fac, d = parameters
#plt.plot(wavelength, func1(wavelength, mu, sig, fac, d), label="Fit Peak 2")





#range_help = [*range(1300,wavelength.size,1) ]
#print(range_help)
#parameters, covariance_matrix = curve_fit(func1, wavelength[range_help], abs2[range_help], p0=[650, 50, 0.6, 0.1])
#mu, sig, fac, d = parameters
#plt.plot(wavelength, func1(wavelength, mu, sig, fac, d), label="Fit Peak 2")



# calculate fwhm
#fwhm = 2 * np.sqrt(2*np.log(2)) * sig
#print(fwhm)

plt.legend()
plt.show()




