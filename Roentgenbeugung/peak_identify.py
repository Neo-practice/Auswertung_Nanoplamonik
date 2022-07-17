import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit


def polynom6(x, a, b, c, d, e, f, g):
    return a+b*x+c*x**2+d*x**3+e*x**4 + f*x**5 +g*x**6

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def linear(x, a, b):
    return a+b*x

def baseline_corr( x, y , a,b):
    param = [a,b]
    ID_max = list(y).index(np.max(y))
    bereich = list(range(0,ID_max-20))+list(range(ID_max+21,len(x)))
    x_help = x[bereich]
    y_help = y[bereich]
    parameters, covariance_matrix = curve_fit(linear, x_help ,y_help, p0=param)
    #print(parameters)
    y_fit = linear(x, *parameters)
    return y_fit
'''
def _1Voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1):
    return (np.abs(ampG1)*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG1)**2)/(2*sigmaG1**2)))) +\
              ((np.abs(ampL1)*widL1**2/((x-cenL1)**2+widL1**2)) )
'''
def _G1_V(x, ampG1, cen1, sigmaG1 ):
    return (np.abs(ampG1) * (1 / (sigmaG1 * (np.sqrt(2 * np.pi)))) * (np.exp(-((x - cen1) ** 2) / (2 * sigmaG1 ** 2))))
def _L1_V(x, ampL1, cenL1, widL1):
    return  ((np.abs(ampL1)*widL1**2/((x-cenL1)**2+widL1**2)) )
def _1Voigt(x, ampG1, cen1, sigmaG1, ampL1, cenL1, widL1):#, ampG2, cen2, sigmaG2, ampL2, widL2):
    return (np.abs(ampG1)*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cen1)**2)/(2*sigmaG1**2)))) +\
              ((np.abs(ampL1)*widL1**2/((x-cenL1)**2+widL1**2)) ) #+ \
           #(np.abs(ampG2) * (1 / (sigmaG2 * (np.sqrt(2 * np.pi)))) * (
           #    np.exp(-((x - cen2) ** 2) / (2 * sigmaG2 ** 2)))) + \
           #((np.abs(ampL2) * widL2 ** 2 / ((x - cen2) ** 2 + widL2 ** 2)))

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

    bereich=10     #Bereich um halbes Maximum
    abweichung_links = np.zeros(2*bereich)
    abweichung_rechts = np.zeros(2*bereich)
    #abweichung_oben = np.zeros(2 * bereich) ## Abweichung Modell zur Peak-Hoehe
    j = 0
    while j < bereich:
        abweichung_links[bereich+j] = np.abs(y[ID[0]+j]-find_nearest(orig[0:ID_max], y[ID[0]+j]))
        abweichung_links[bereich - j-1] = np.abs(y[ID[0]-j-1]-find_nearest(orig[ID_max:-1], y[ID[0]-j-1]))
        abweichung_rechts[bereich+j] = np.abs(y[ID[1]+j]-find_nearest(orig[0:ID_max], y[ID[1]+j]))
        abweichung_rechts[bereich - j-1] = np.abs(y[ID[1]-j-1]-find_nearest(orig[ID_max:-1], y[ID[1]-j-1]))
        #abweichung_oben[bereich+j] = np.abs(y[ID_max+j]-find_nearest(orig, y[ID_max+j]))
        #abweichung_oben[bereich - j -1] = np.abs(y[ID_max - j-1]-find_nearest(orig, y[ID_max - j-1]))
        j += 1
    delta_w = np.sqrt(np.max(abweichung_links)**2 + np.max(abweichung_rechts)**2)# + np.max(abweichung_oben)**2)

    print(abweichung_links)

    plt.figure(100)
    plt.plot(x,y)
    #plt.plot(x,orig, marker = '.')
    plt.plot(x[ID_max], y[ID_max], ls='', marker = 'x', color='black', markersize = 4)
    plt.plot(x[ID], half_values, ls='', marker='x', color='black', markersize=4)

    #print('s_fwhm = ',delta_w)

    fwhm = np.array( [np.abs(x[ID[0]]-x[ID[1]]), delta_w, ID[0], ID[1]  ], dtype = float ) #
    #print(ID)
    return fwhm

def give_max( y, orig ):
    ID_max = list(y).index(y.max())
    upper_value = max( orig[ID_max-10:ID_max+10] )
    #print(upper_value, y.max())
    maximum = y.max()
    s_maximum = np.abs(maximum-upper_value)
    return [maximum, s_maximum]

def calc_Table( theta, s_theta, signal, fit1 , opt):#, fit2
    two_theta_list = []
    fwhm = []
    s_fwhm = []
    intensity = []
    for i in range(len(theta[:,1])):
        theta_mean1 = theta[i, list(fit1[i,:]).index(np.max(fit1[i,:]))]
        #theta_mean2 = theta[i, list(fit2[i, :]).index(np.max(fit2[i,:]))]
        fwhm1 = FWHM( theta[i,:], fit1[i,:], signal[i,:] )
        #fwhm2 = FWHM( theta[i,:], fit2[i,:], signal[i,:] )
        two_theta_list.append(theta_mean1)
        #two_theta_list.append(theta_mean2)
        fwhm.append(fwhm1[0])
        s_fwhm.append(fwhm1[1])
        #fwhm.append(fwhm2[0])
        intensity.append(fit1[i,:].max())
        #intensity.append(fit2[i, :].max())
    print('fwhm = ', fwhm)
    print('s_fwhm = ', s_fwhm)
    if opt ==1:
        display_Table1( two_theta_list, s_theta, fwhm, intensity )

def display_Table1( _2theta, s_2theta, fwhm, intensity ):
    print('Peak-Nr. & $2\\theta$ in ° & FWHM & Intensität & Fit-Fkt.\\\\ \\hline')
    for i in range(len(_2theta)):
        print( str(i+1), ' & ', str(np.round(_2theta[i]*1000)/1000), ' $\\pm$ '
               , str( np.round(s_2theta[i]*1000)/1000 ) , ' & '
               , str(np.round(fwhm[i]*1000)/1000), ' & ', str(np.round(intensity[i])), ' & Gauss \\\\' )

def calc_Table2( _2theta, hkl ):
    _lambda = 1.540593  # Angstroem
    sin_theta2 = np.zeros( len(_2theta))
    constante = np.zeros( len(_2theta) )
    a = np.zeros( len(_2theta) )
    for i in range(len(_2theta)):
        sin_theta2[i] = np.sin(0.5*_2theta[i] * np.pi/180)**2
        #sin_theta2[i,1] =
        constante[i] = sin_theta2[i]/hkl[i]
        a[i] = _lambda/2*( constante[i] )**(-1./2)
    s_a = np.sqrt( np.sum( (a-np.mean(a))**2 )/len(a) )
    #print('a = ', np.mean(a), ' \pm ', s_a)
    berechne_Z(a,s_a)
    return [ sin_theta2, constante, a, s_a]

def berechne_Z(a ,s_a):
    a = np.mean(a) * 10 ** (-8)  # cm
    s_a = s_a * 10 ** (-8)
    N_A = 6.02214076 * 10 ** (23)  # 1/mol
    M_M = 58.44  # g/mol
    rho = 2.17  # g/cm^3
    Z = N_A * rho * (a) ** 3 / (M_M)
    s_Z = N_A * rho / M_M * 3 * a ** 2 * s_a
    print('Z = ', Z, ' \\pm ', s_Z)
hkl = [ 4,8, 11,12 ,16, 20, 22, 26, 27, 32, 36, 38]
kombi = [ '2,0,0', '2,2,0', '3,1,1', '2,2,2', '4,0,0'
    , '4,2,0', '2,3,3', '5,1,0','5,1,1', '4,4,0', '6,0,0', '6,1,1' ]
def display_Table2( _2theta, d, s_2theta, s_d, hkl, kombis ):
    [ sin2, const, a, s_a ] = calc_Table2( _2theta, hkl)
    print('Peak-Nr. & $2\\theta$ in ° & $d$ in \\si{\\AA} & $\\sin^2(\\theta)$ & $\\frac{\\lambda^2}{4a^2}$ & $h^2+k^2+l^2$ & $a$ in \\si{\\AA} & $hkl$ \\\\ \\hline')
    for i in range(len(_2theta)):
        print( str(i+1), ' & ', str(np.round(_2theta[i]*1000)/1000), ' $\\pm$ '
               , str( np.round(s_2theta[i]*1000)/1000 ) ,' & '
               , str(np.round(d[i]*10000)/10000) ,' & ', str(np.round(sin2[i]*1000000)/1000000)
               ,' & ' , str(np.round(const[i]*1000000)/1000000) , ' & ', str(hkl[i])
               ,' & ', str(np.round(a[i]*10000)/10000) , ' &  ( ', kombis[i] ,' )\\\\' )
    #print(_2theta)
    #print(s_2theta)
def n_func( theta, d ):
    lam = 1.540593
    return 2*np.sin( 0.5*theta  *np.pi/180)/lam*d

def d_func(theta):
    lam = 1.540593
    return lam/(2 * np.sin(0.5 * theta * np.pi / 180))
def s_d_func(theta, s_theta):
    lam = 1.540593
    return lam/(2 * np.sin(0.5 * theta * np.pi / 180))**2*np.cos(0.5* theta * np.pi / 180)*s_theta

data = np.loadtxt('Data_Copie/HG_NACL_.DAT')
wave, spec = np.hsplit(data, 2)

shift = np.array([wave[0]]*len(wave))

spec = np.array(np.transpose(spec)[0], dtype = float)
wave = np.array(np.transpose(wave-shift)[0], dtype = float)

print('Wave_ende = ', wave[-1])

ID = list(spec).index(np.max(spec))

# liste für die einzelnen gefundenen peak indices
liste = [ [974, 1100], [1450,1600] , [1739,1830], [1830, 1980],
          [2100, 2300], [2400, 2500], [2500, 2650], [2750, 2900],
          [ 3000, 3150], [3300, 3500], [3600, 3710], [3710, 3900]]

liste2 = [ 1033, 1515, 1802, 1895, 2233, 2467, 2552, 2859, 3080, 3452, 3673, 3767]
#wave = wave[ID:-1-60]
#spec = spec[ID:-1-60]
#i = 0
#wave1 = wave[liste[i][0]:liste[i][1]]
#spec1 = spec[liste[i][0]:liste[i][1]]
#corr = baseline_corr(wave1, spec1, 2000, -50)
#spec1 -= corr
#x_help = np.array(range(len(spec1)), dtype=float)
#file = open("peak1.dat","w")
#data = [ np.transpose(wave[liste2[0]-60:liste2[0]+60]), np.transpose(spec[liste2[0]-60:liste2[0]+60]) ]
#np.disp( np.transpose(wave[liste2[0]-60:liste2[0]+60]))
#file.write()
#file.close()

#plt.plot(spec)

'''
plt.figure(1, dpi=130, figsize=(8.0,8.0))
start = 300
end = len(spec)-50
theta_simple = wave[start: end]
spec_simple_raw = spec[start: end]
parameter, covariance_matrix1 = curve_fit( polynom6, theta_simple, spec_simple_raw, p0=[6000, 1, 1, 1, 1, 1, 1])
spec_simple_fit1 = polynom6(theta_simple, *parameter)
spec_simple = spec_simple_raw-spec_simple_fit1
peaks = list(signal.find_peaks(spec_simple, height=255, width=6)[0])

plt.plot(theta_simple, spec_simple )
plt.plot(theta_simple[peaks], spec_simple[peaks], ls = '', marker = 'o')

theta_peak = theta_simple[peaks]
print(theta_peak)
n = np.array(range(1,len(theta_simple[peaks])))
#n = np.array([3,4,6,8,9,11,14,15])
print(n)

d = []
for i in range(len(theta_peak)):
    parameters, covar_matrix = curve_fit(n_func, np.array([theta_peak[i]]),1, [1.5])
    d.append(list(parameters)[0])
print(d)
'''

plt.figure(1, dpi=130, figsize=(12.0,8.0))
plt.plot(wave, spec)
plt.text(28, 3622, '('+kombi[0]+')',bbox=dict(facecolor='none',edgecolor='none',boxstyle='square'))
plt.text(42, 2670, '('+kombi[1]+')',bbox=dict(facecolor='none',edgecolor='none',boxstyle='square'))
plt.text(50, 826,  '('+kombi[2]+')',bbox=dict(facecolor='none',edgecolor='none',boxstyle='square'))
plt.text(54, 1393, '('+kombi[3]+')',bbox=dict(facecolor='none',edgecolor='none',boxstyle='square'))
plt.text(64, 880, '('+kombi[4]+')',bbox=dict(facecolor='none',edgecolor='none',boxstyle='square'))
plt.text(70, 616, '('+kombi[5]+')',bbox=dict(facecolor='none',edgecolor='none',boxstyle='square'))
plt.text(73, 2014, '('+kombi[6]+')',bbox=dict(facecolor='none',edgecolor='none',boxstyle='square'))
plt.text(82, 1700, '('+kombi[7]+')',bbox=dict(facecolor='none',edgecolor='none',boxstyle='square'))
plt.text(90, 643, '('+kombi[8]+')',bbox=dict(facecolor='none',edgecolor='none',boxstyle='square'))
plt.text(100, 1035, '('+kombi[9]+')',bbox=dict(facecolor='none',edgecolor='none',boxstyle='square'))
plt.text(106, 721, '('+kombi[10]+')',bbox=dict(facecolor='none',edgecolor='none',boxstyle='square'))
plt.text(109, 1545, '('+kombi[11]+')',bbox=dict(facecolor='none',edgecolor='none',boxstyle='square'))

plt.xlabel('2${\\theta}$ in °')
plt.ylabel('Intensität')

plt.savefig('Spektrum_roh')

plt.figure(2, dpi=130, figsize=(8.0,8.0))
wave1 = np.zeros( (12, 120) )
part = np.zeros( (12, 120) )
parameters = np.zeros( (12,6) )
parameterse = np.zeros( (12,3) )
param_error = np.zeros( (12,3) )
fit_G1 = np.zeros( (12,120) )
fit_L1 = np.zeros( (12,120) )
fit_gauss = np.zeros( (12,120) )
maximum = np.zeros( (12, 2) )

for i in range(len(liste)):
    #wave1 = 0; part = 0; x_help = 0; corr = 0; fit = 0;
    #wave1 = wave[liste[i][0]:liste[i][1]]
    wave1[i,:] = wave[liste2[i]-60:liste2[i]+60]
    #spec1 = spec[liste[i][0]:liste[i][1]]
    part[i,:] = spec[liste2[i]-60:liste2[i]+60]
    corr = baseline_corr(wave1[i,:], part[i,:], 2000, -50)
    part[i,:] -= corr
    x_help = np.array(range(len(part[i,:])), dtype=float)
    x_help2 = np.array( np.arange(len(part[i,:]), 0.1) )

    plt.subplot( 4,3,i+1 )

    # nur Gauss oder Lorenzfit
    param = [1000, 60, 1]
    parameterse[i,:], covariance_matrix1 = curve_fit(_G1_V, x_help, part[i, :], p0=param)
    param_error[i,:] = np.sqrt(np.diag(covariance_matrix1))
    #[ ampL1, cenL1, gamma1] = parameterse[]
    fit_gauss[i,:] = _G1_V(x_help, *parameterse[i,:])



    #print(fit2)

    #print(parameterse)

    plt.plot( wave1[i,:], part[i,:], lw=1, label='Signalpeak') #wave1[i,:],
    plt.plot(wave1[i,:], fit_gauss[i,:], lw = 2, c = 'tab:orange', label='Gaußfit')



    ### Alternativ noch mehr auswertung möglich

    peaks = [int(find_nearest(x_help, parameters[i, 1])), int(find_nearest(x_help, parameters[i, 4]))]
    # print(peaks)
    # ampG1, cen1, sigmaG1, ampL1, widL1, ampG2, cen2, sigmaG2, ampL2, widL2
    if i in [1, 3, 4]:
        ampG1 = 1300;
        cen1 = 70;
        sig1 = 3;
        ampL1 = 1000;
        gamma1 = 0.0007;
        cenL1 = 45
    else:
        ampG1 = 1300;
        cen1 = 60;
        sig1 = 3;
        ampL1 = 1000;
        gamma1 = 0.0007;
        cenL1 = 54  # ampG2 = 1000; cen2 = 50; sig2 = 3; ampL2 = 1000; gamma2 = 0.0001;
    param = [ampG1, cen1, sig1, ampL1, cenL1,
             gamma1]  # , ampG2, cen2, sig2, ampL2, gamma2]  # ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1
    parameters[i, :], covariance_matrix1 = curve_fit(_1Voigt, x_help, part[i, :], p0=param)
    [ampG1, cen1, sig1, ampL1, cenL1, gamma1] = parameters[i, :]
    fit = _1Voigt(x_help, *parameters[i, :])
    fit_G1[i, :] = _G1_V(x_help, ampG1, cen1, sig1)
    fit_L1[i, :] = _L1_V(x_help, ampL1, cenL1, gamma1)

    maximum[i,:] = give_max(fit_G1[i, :], part[i, :])

    plt.plot(wave1[i, :], fit, lw=1, c='black', label='Voigt-Fit')  # wave1[i,:],
    plt.plot(wave1[i, :], fit_G1[i, :], lw=1, ls='--', c='black', label='Einzelpeaks aus Voigt')  # wave1[i,:],
    plt.plot(wave1[i, :], fit_L1[i, :], lw=1, ls='--', c='black')  # wave1[i,:],
    plt.plot([wave1[i, peaks[0]], wave1[i, peaks[0]]], [-50, 50], c='black', lw=1)
    plt.plot([wave1[i, peaks[1]], wave1[i, peaks[1]]], [-50, 50], c='black', lw=1)
    # plt.axis([wave1[i,0], wave1[i,-1], -50, 1850])
    #############################################################






    #if i+1 in [2,3,5,6,8,9, 11, 12]:
        #plt.yticks([])
    if i+1 in [ 10, 11, 12 ]:
        plt.xlabel('2${\\theta}$ in °')
    if i+1 in [ 1, 4, 7, 10 ]:
        plt.ylabel( 'Intensität' )
    if i + 1 in [2]:
        plt.legend(loc='best', bbox_to_anchor=(0.5, 1.5))

plt.savefig('Einzel_peaks')

thetapeaks = parameterse[:,1]
thetapeaks_error = param_error[:,1]

#print(maximum[:,1])

#print(peak_intens,'\n', peak_intens_error)


theta = []
s_theta = []
d = []
s_d = []
for j in range(len(thetapeaks)):
    theta.append( wave1[j,round(thetapeaks[j])] )
    s_theta.append( thetapeaks_error[j] )
    d.append(d_func(theta[j]))
    s_d.append(s_d_func(theta[j], s_theta[j]))

#display_Table2(theta, d, s_theta, s_d, hkl, kombi)
#print( ' Z_{Jana} = ',  2.17*6.02214076 * 10 ** (23)*(5.6339*10**(-8))**3/58.44)

calc_Table(wave1, s_theta, part, fit_gauss, 0)
plt.show()


