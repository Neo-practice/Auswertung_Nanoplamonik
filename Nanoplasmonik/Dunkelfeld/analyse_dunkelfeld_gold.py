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
    bereich=700     #Bereich um halbes Maximum
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
referenz = np.loadtxt('spectrometer_weißlichtspektrum_at_650.00nm_cut_at_1436.50Y_01.dat', skiprows = 6)
wavelength, ref = np.hsplit(referenz, 2)

# berechne Energie bzw. Winkelfrequenzskala
energyscale = np.zeros(wavelength.size)
w = np.zeros(wavelength.size)
for i in np.arange(wavelength.size):
    energyscale[i] = h * c / (e * np.array(wavelength[i]) * np.power(10, -9.))
    w[i] = 2 * np.pi * c / (wavelength[i] * 10 ** (-9)) * 10 ** (-15.)


########## Auswertung Einzel-Partikel
##################################################################


einzel_70 = np.loadtxt('spectrometer_dunkelfeld_70nm_einzel_at_650.00nm_cut_at_1436.50Y_01.dat', skiprows = 6)
wavelength, intensity_70 = np.hsplit(einzel_70, 2)

einzel_100 = np.loadtxt('spectrometer_dunkelfeld_100nm_einzel_at_650.00nm_cut_at_1436.50Y_01.dat', skiprows = 6)
wavelength, intensity_100 = np.hsplit(einzel_100, 2)
'''
correction_100 = (16.503/49.086)
correction_70 = (16.503/24.962)

plt.figure(9)
plt.plot(w,ref*correction_70)
plt.plot(w,ref*correction_100)
plt.plot(w,intensity_100)
plt.plot(w,intensity_70)

abs_100 = intensity_100/ref
abs_70 = intensity_70/ref

plt.figure(10)
plt.plot(w, abs_100)
plt.plot(w, abs_70)
'''


maximal_val = np.max(intensity_100)

intensity_100=intensity_100/maximal_val
intensity_70=intensity_70/maximal_val

ref =  ref/maximal_val





abs_100 = intensity_100
abs_70 = intensity_70


fwhm_val = np.zeros((2,2), dtype=float)

w = np.squeeze(w)
abs_70 = np.squeeze(abs_70)
abs_100 = np.squeeze(abs_100)

## Lorentz*Gauss Fit
i=2
plt.figure(1, dpi=130, figsize=(10,6.666))
plt.subplot(1,2,2)
#abs_70 = -(intensity_70-ref)
plt.plot(w, ref, label='Weißlichtquelle', color='grey')
plt.plot(w, abs_70, color='blue', label='Signal')
plt.axis([w[-1], w[0], -0.1, 1.1])
plt.yticks([])
plt.xlabel('$\omega \cdot 10^{15}$ s$^{-1}$')
plt.title('Dunkelfeldspektrum für 70 nm Nanopartikel')

parameters, covariance_matrix = curve_fit(faltung, w, abs_70, p0=[ 3.0, 0.4, 3.3, 0.5, 0.3, 0.1 ])
std_parameters = np.sqrt(np.diag(covariance_matrix))
w0, gamma, mu, sig, fac, offset = parameters
func = faltung(w, w0, gamma, mu, sig, fac, offset)
fwhm = FWHM(w,func, abs_70)
tau = 1/fwhm[0]
s_tau = fwhm[1]/(fwhm[0])**2
print("w_70 = "+ str(fwhm[0])+"\pm"+str(fwhm[1])+"\ntau_70 = "+str(tau)+"\pm"+str(s_tau))

plt.plot(w, func, color='green',
             label="Lorentz-Gauss-Fit")
plt.plot( w[[int(fwhm[2]),int(fwhm[3])]] , func[[int(fwhm[2]),int(fwhm[3])]] ,  lw=1 ,ls='--',
              color='green',
              label="$\Delta\omega$ = "+str(np.round(fwhm[0]*1000)/1000)+"$\pm$ "
                    +str(round(fwhm[1]*1000)/1000)+" $\cdot 10^{15}$ s$^{-1}$\n"
                                                   "τ = "+str(np.round(tau*100)/100)+"$\pm$"+str(np.round(s_tau*100)/100)+" fs")
plt.legend(loc='upper right')

plt.subplot(1,2,1)
plt.plot(w, ref, label='Weißlichtquelle', color='grey')
plt.plot(w, abs_100, color='blue', label='Signal')
plt.axis([w[-1], w[0], -0.1, 1.1])
plt.xlabel('$\omega \cdot 10^{15}$ s$^{-1}$')
plt.title('Dunkelfeldspektrum für 100 nm Nanopartikel')


parameters, covariance_matrix = curve_fit(faltung, w, abs_100, p0=[ 3.0, 0.4, 3.3, 0.5, 0.3, 0.1 ])
std_parameters = np.sqrt(np.diag(covariance_matrix))
w0, gamma, mu, sig, fac, offset = parameters
func = faltung(w, w0, gamma, mu, sig, fac, offset)
fwhm = FWHM(w,func, abs_100)
#fwhm_val[0,i] = fwhm[0]
#fwhm_val[1, i] = fwhm[1]
tau = 1/fwhm[0]
s_tau = fwhm[1]/(fwhm[0])**2
print("w_100 = "+ str(fwhm[0])+"\pm"+str(fwhm[1])+"\ntau_100 = "+str(tau)+"\pm"+str(s_tau))

plt.plot(w, func, color='green',
             label="Lorentz-Gauss-Fit")
plt.plot( w[[int(fwhm[2]),int(fwhm[3])]] , func[[int(fwhm[2]),int(fwhm[3])]] ,  lw=1 ,ls='--',
              color='green',
              label="$\Delta\omega$ = "+str(np.round(fwhm[0]*1000)/1000)+"$\pm$ "
                    +str(round(fwhm[1]*1000)/1000)+" $\cdot 10^{15}$ s$^{-1}$\n"
                                                   "τ = "+str(np.round(tau*100)/100)+"$\pm$"+str(np.round(s_tau*100)/100)+" fs")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.))
plt.ylabel('Intensität')

'''
tau = np.zeros((2,2))

tau[0,0] = 1/(fwhm_val[0,0]*10**(15))
tau[1,0] = 1/(fwhm_val[0,1]*10**(15))
tau[0,1] = fwhm_val[1,0]*10**(15)/(fwhm_val[0,0]*10**(15))**2
tau[1,1] = fwhm_val[1,1]*10**(15)/(fwhm_val[0,1]*10**(15))**2


print(tau)

#np.savetxt("einzel_lebenszeiten.txt", tau)



'''
plt.savefig('dunkelfeldspektren_gold')


fluoreszenz_0 = np.loadtxt("/Users/mariuskaiser/Desktop/PPD/Auswertung_PPD/Nanoplasmonik/Hellfeld/einzel_fluoreszenzspektrum_70.dat")
w_0, intensity_0 = np.hsplit(np.transpose(fluoreszenz_0), 2)
print(intensity_0)

plt.figure(100)
#plt.plot(w_0,intensity_0)
abs = np.array(intensity_70/intensity_0, dtype = float)
plt.plot(w, abs)
'''
parameters, covariance_matrix = curve_fit(faltung, w, abs, p0=[ 3.0, 0.4, 3.3, 0.5, 0.3, 0.1 ])
std_parameters = np.sqrt(np.diag(covariance_matrix))
w0, gamma, mu, sig, fac, offset = parameters
func = faltung(w, w0, gamma, mu, sig, fac, offset)
fwhm = FWHM(w,func, abs_100)
#fwhm_val[0,i] = fwhm[0]
#fwhm_val[1, i] = fwhm[1]
tau = 1/fwhm[0]
s_tau = fwhm[1]/(fwhm[0])**2
plt.plot(w, func, color='green',
             label="Lorentz-Gauss-Fit")
plt.plot( w[[int(fwhm[2]),int(fwhm[3])]] , func[[int(fwhm[2]),int(fwhm[3])]] ,  lw=1 ,ls='--',
              color='green',
              label="$\Delta\omega$ = "+str(np.round(fwhm[0]*1000)/1000)+"$\pm$ "
                    +str(round(fwhm[1]*1000)/1000)+" $\cdot 10^{15}$ s$^{-1}$\n"
                                                   "τ = "+str(np.round(tau*100)/100)+"$\pm$"+str(np.round(s_tau*100)/100)+" fs")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.))
'''
plt.show()
