import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# werte = np.loadtxt("dips4.txt")
# x, y, b, nichts, c, d = np.hsplit(werte, 6)


# Define Gauß Function (not normalized)
def gaussian1(x, mu, sig, fac, d):
    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) +d

def lorentz( x, w0, gamma, d):
    return 1/( (x**2 - w0**2 )**2 + gamma**2*w0**2)+d

def gaussian(x, mu, sig, fac):
    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def faltung( x, w0, gamma, mu, sig, fac , offset):
    return lorentz(x, w0, gamma, offset)*func1(x, mu, sig, fac, offset)

def gauss_sum2( x, mu1, sig1, fac1 , mu2, sig2, fac2, offset):
    return gaussian(x, mu1, sig1, fac1)+gaussian(x, mu2, sig2, fac2)+0.1



def f_L(w0, gamma):
    return np.sqrt( w0**2+gamma*w0 ) - np.sqrt( w0**2-gamma*w0 )
def f_G(sigma):
    return 2 * np.sqrt(2*np.log(2)) * sigma

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

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
wavelength, ref_90 = np.hsplit(referenz_90, 2)

# berechne Energie bzw. Winkelfrequenzskala
energyscale = np.zeros(wavelength.size)
w = np.zeros(wavelength.size)
for i in np.arange(wavelength.size):
    energyscale[i] = h * c / (e * np.array(wavelength[i]) * np.power(10, -9.))
    w[i] = 2 * np.pi * c / (wavelength[i] * 10 ** (-9)) * 10 ** (-15.)


### Auswertung Hybrid-Plasmonen
## Load Hybridisierte

option_abstand = ['20' , '30' , '40' , '50' , '60']

for i in np.arange(1):#len(option_groesse)):
    plt.figure(i, dpi=100)  # figsize= [15,10])

    hybrid_0 = np.loadtxt('spectrometer_0_'+option_abstand[i]+'nm_hybrid_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
    wavelength, intensity_0 = np.hsplit(hybrid_0, 2)
    hybrid_90 = np.loadtxt('spectrometer_90_'+option_abstand[i]+'nm_hybrid_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
    wavelength, intensity_90 = np.hsplit(hybrid_90, 2)

    abs_0 = 1 - intensity_0 / ref_0
    abs_90 = 1 - intensity_90 / ref_90

    plt.plot(w, abs_0, label='0 Grad Polarisation')
    #plt.plot(w, abs_90, label='90 Grad Polariation')
    plt.xlabel('omega *10^(15)')
    plt.title('Extinktionsspektrum bei ' + option_abstand[i] + ' nm Abstand der Hybrid-Nanopartikel')

    w = np.squeeze(w)
    abs_0 = np.squeeze(abs_0)
    abs_90 = np.squeeze(abs_90)

    ## Gausssummen Fit
    #plt.plot(w, gaussian1(w, 2.75, 0.1, 0.5, 0.05)+gaussian1(w, 3.0, 0.2, 0.3, 0.05))
    #plt.plot(w, gaussian1(w, 3.0, 0.2, 0.4, 0.05))
    parameters, covariance_matrix = curve_fit(gauss_sum2, w, abs_0, p0=[2.75, 0.1, 0.5, 3.0, 0.2, 0.3, 0.05])
    #std_parameters = np.sqrt(np.diag(covariance_matrix))
    # Gibt laut Python-Documentation die Standardabweichung der Parameter an
    # Guter Fehler für w ist vlt sqrt( sdt_w0^2 +std_mu^2 )
    # delta_w = np.sqrt( std_parameters[0]**2. + std_parameters[2]**2. )
    # print('Aus Parameter: ' , delta_w)
    mus1, sig1, fac1, mus2, sig2, fac2, offset = parameters
    func = gauss_sum2(w, mus1, sig1, fac1, mus2, sig2, fac2, offset)
    #fwhm = FWHM(w, func, abs_0)
    plt.plot(w, func,
             label="Lorentz*Gauss-Fit 0 Grad")
    #plt.plot(w[[int(fwhm[2]), int(fwhm[3])]], func[[int(fwhm[2]), int(fwhm[3])]], lw=1, ls='--',
    #         label="FWHM = " + str(round(fwhm[0] * 1000) / 1000) + "± " + str(
    #             round(fwhm[1] * 1000) / 1000) + " *10^(15)")
    plt.plot(w, gaussian1(w, mus1, sig1, fac1 , offset ))
    plt.plot(w, gaussian1(w, mus2, sig2, fac2, offset ))
    plt.legend(loc='upper right')

    #parameters, covariance_matrix = curve_fit(gauss_sum2, w, abs_90, p0=[2.6, 0.4, 0.5, 3.0, 0.4, 0.5, 0.1])
    #mus1, sig1, fac1, mus2, sig2, fac2, offset = parameters
    #func = gauss_sum2(w, mus1, sig1, fac1, mus2, sig2, fac2, offset)
    #fwhm = FWHM(w, func, abs_90)
    #plt.plot(w, func,
    #         label="Lorentz*Gauss-Fit 90 Grad")
    #plt.plot(w[[int(fwhm[2]), int(fwhm[3])]], func[[int(fwhm[2]), int(fwhm[3])]], lw=1, ls='--',
    #         label="FWHM = " + str(round(fwhm[0] * 1000) / 1000) + "± " + str(
    #             round(fwhm[1] * 1000) / 1000) + " *10^(15)")
    #plt.plot(gaussian1(w, mus1, sig1, fac1, offset ))
    #plt.plot(gaussian1(w, mus2, sig2, fac2, offset))
    plt.legend(loc='upper right')





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
#parameters, covariance_matrix = curve_fit(func1, wavelength, abs_0, p0=[650, 50, 0.6, 0.1])
#mu, sig, fac, d = parameters
#plt.plot(wavelength, func1(wavelength, mu, sig, fac, d), label="Fit Peak 2")





#range_help = [*range(1300,wavelength.size,1) ]
#print(range_help)
#parameters, covariance_matrix = curve_fit(func1, wavelength[range_help], abs2[range_help], p0=[650, 50, 0.6, 0.1])
#mu, sig, fac, d = parameters
#plt.plot(wavelength, func1(wavelength, mu, sig, fac, d), label="Fit Peak 2")



# calculate fwhm
#fwhm = 2 * np.sqrt(2*np.log(2)) * sig
#print(fwhm)

#plt.legend()
plt.show()




