import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# werte = np.loadtxt("dips4.txt")
# x, y, b, nichts, c, d = np.hsplit(werte, 6)


# Define Gauß Function (not normalized)
def func1(x, mu, sig, fac, d):
    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) +d

def lorentz( x, w0, gamma, d):
    return 1/( (x**2 - w0**2 )**2 + gamma**2*w0**2)+d

def gaussian(x, mu, sig, fac):
    return fac * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def faltung( x, w0, gamma, mu, sig, fac , offset):
    return lorentz(x, w0, gamma, offset)*func1(x, mu, sig, fac, offset)



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
    bereich=400     #Bereich um halbes Maximum
    abweichung_links = np.zeros(2*bereich)
    abweichung_rechts = np.zeros(2*bereich)
    abweichung_oben = np.zeros(2 * bereich) ## Abweichung Modell zur Peak-Hoehe
    j = 0
    while j < bereich:
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



########## Auswertung Einzel-Partikel
##################################################################

groesse = ['100', '70']

fig = plt.figure(1, dpi=130, figsize=(10,6.666))
for i in np.arange(len(groesse)):

    einzel_0 = np.loadtxt('spectrometer_0_'+groesse[i]+'nm_einzel_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
    wavelength, intensity_0 = np.hsplit(einzel_0, 2)
    einzel_90 = np.loadtxt('spectrometer_90_'+groesse[i]+'nm_einzel_at_650.00nm_cut_at_1184.50Y_01.dat', skiprows = 6)
    wavelength, intensity_90 = np.hsplit(einzel_90, 2)

    abs_0 = 1-intensity_0/ref_0
    abs_90 = 1-intensity_90/ref_90

    plt.subplot(1, 2, i+1)
    plt.plot(w, abs_0, color='blue', label='0° Polarisation')
    plt.axis([w[-1], w[0], 0, 1])
    plt.xlabel('$\omega \cdot 10^{15}$ s$^{-1}$')
    plt.title('Extinktionsspektrum für '+groesse[i]+' nm Nanopartikel')

    w = np.squeeze(w)
    abs_0 = np.squeeze(abs_0)
    abs_90 = np.squeeze(abs_90)

    # lorentz_parameter = np.array([2.8, 0.5, 0.1], dtype=float)
    # gauss_parameter = np.array([3.0, 0.1, 0.2, 0.1], dtype=float)

    ## Lorentz*Gauss Fit
    parameters, covariance_matrix = curve_fit(faltung, w, abs_0, p0=[ 3.0, 0.4, 3.3, 0.5, 0.3, 0.1 ])
    std_parameters = np.sqrt(np.diag(covariance_matrix))
    # Gibt laut Python-Documentation die Standardabweichung der Parameter an
    # Guter Fehler für w ist vlt sqrt( sdt_w0^2 +std_mu^2 )
    #delta_w = np.sqrt( std_parameters[0]**2. + std_parameters[2]**2. )
    #print('Aus Parameter: ' , delta_w)
    w0, gamma, mu, sig, fac, offset = parameters
    func = faltung(w, w0, gamma, mu, sig, fac, offset)
    fwhm = FWHM(w,func, abs_0)
    plt.plot(w, func, color='green',
             label="Lorentz-Gauss-Fit 0°")
    plt.plot( w[[int(fwhm[2]),int(fwhm[3])]] , func[[int(fwhm[2]),int(fwhm[3])]] ,  lw=1 ,ls='--',
              color='green',
              label="$\Delta\omega_{0°}$ = "+str(round(fwhm[0]*1000)/1000)+"$\pm$ "+str(round(fwhm[1]*1000)/1000)+" $\cdot 10^{15}$ s$^{-1}$")
    plt.legend(loc='upper right')

    plt.plot(w, abs_90, color='red', label='90° Polariation')
    parameters, covariance_matrix = curve_fit(faltung, w, abs_90, p0=[2.65, 0.3, 4.2, 0.8, 2., 0.1])
    w0, gamma, mu, sig, fac , offset= parameters
    func = faltung(w, w0, gamma, mu, sig, fac, offset)
    fwhm = FWHM(w,func, abs_90)
    plt.plot(w, func, color='orange',
             label="Lorentz-Gauss-Fit 90°")
    plt.plot(w[[int(fwhm[2]), int(fwhm[3])]], func[[int(fwhm[2]), int(fwhm[3])]], lw=1, ls='--',
             color='orange',
             label="$\Delta\omega_{90°}$ = " + str(round(fwhm[0] * 1000) / 1000)+"$\pm$ "+str(round(fwhm[1]*1000)/1000)+" $\cdot 10^{15}$ s$^{-1}$")
    plt.legend(loc='upper right')
    if i == 0:
        plt.ylabel('Extinktionsspektrum')
    else:
        plt.yticks([])

    #### Rest der Überlegungen
    ## Lorentz Fit
    # parameters, covariance_matrix = curve_fit(lorentz, w[range_help], abs_0[range_help], p0=[3.0, 0.4])
    # w0, gamma = parameters
    # fwhm[0] = f_L(w0, gamma)
    # plt.plot(w, lorentz(w, w0, gamma),
    #             label="Lorentz-Fit 0 Grad, FWHM = " + str(round(fwhm[0] * 100) / 100) + " *10^(15)")

    ## Gauss Fit
    # parameters, covariance_matrix = curve_fit(gaussian, w[range_help], abs_0[range_help], p0=[3.3, 0.5, 0.3])
    # mu, sig, fac = parameters
    # fwhm[1] = f_G(sig)
    # plt.plot(w, gaussian(w, mu, sig, fac),
    #             label="Gauss-Fit 0 Grad, FWHM = " + str(round(fwhm[1] * 100) / 100) + " *10^(15)")

plt.savefig('Einzelpartikelplamonen')
plt.show()





