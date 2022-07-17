import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit

def lorentzfaktor( theta ):
    return 1./(np.sin(theta)*np.cos(theta))
def polfaktor(_2theta):
    return (1+np.cos(_2theta))/2
def atomformfaktor( theta , atom):
    _lambda = 1.540593  # m
    a = np.zeros(4, dtype = float)
    b = np.zeros(4, dtype = float)
    c = np.zeros(1, dtype=float)
    if atom == 'Na+':
        a[:] = [ 3.25650, 3.93620, 1.39980, 1.00320 ]
        b[:] = [ 2.66710, 6.11530, 0.200100, 14.0390 ]
        c[:] = 0.404
    elif atom == 'Cl-':
        a[:] = [ 18.2915, 7.20840, 6.53370, 2.33860 ]
        b[:] = [ 0.0066, 1.1717, 19.5424, 60.4486 ]
        c[:] =-16.378
    elif atom == 'Na':
        a[:] =[ 4.7626, 3.1736, 1.26740, 1.1128  ]
        b[:]= [ 3.285, 8.8422, 0.3136, 129.424 ]
        c[:] = 0.676
    elif atom == 'Cl':
        a[:] = [ 11.4604, 7.1964, 6.2556, 1.64550 ]
        b[:] =[ 0.0104, 1.1662, 18.5194, 47.7784 ]
        c[:] = -9.5574
    else:
        print('Error: wrong string input, only ( Na+, Na, Cl-, Cl ) are accepted.')
        return 0
    f = 0
    for i in range(4):
        f = f + a[i]*np.exp( -b[i]*(np.sin(theta* np.pi/180)/_lambda)**2 )
        #print(f)
    f += c
    return f[0]
def strukturamplitude( f_Na, f_Cl, option, hkl ):
    if option == 'A':
        xyz = np.array( [[0,0,0],[0.5,0,0.5],[0,0.5,0.5],[0.5,0.5,0] ,
                         [0.5,0.5,0.5],[0.5,0,0], [0,0.5,0], [0,0,0.5]])
        change = 4
        n_Na = 1
        n_Cl = 1
    if option == 'B':
        xyz = np.array([[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0],
                        [0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.75, 0.25, 0.25],
                        [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75], [0.75, 0.75, 0.75]])
        change = 4
        n_Na = 1
        n_Cl = 0.5
    F_re = 0
    F_im = 0
    F = np.zeros(2)
    for j in range(len(xyz)):
        f = f_Na
        n = n_Na
        if j >= change:
            f = f_Cl
            n = n_Cl
        F_re += n * f * np.cos(2 * np.pi * ( hkl[0]*xyz[j,0] + hkl[1]*xyz[j,1] + hkl[2]*xyz[j,2] ))
        F_im += n * f * np.sin(2 * np.pi * (hkl[0] * xyz[j, 0] + hkl[1] * xyz[j, 1] + hkl[2] * xyz[j, 2]))
        #F = np.sqrt(F_re**2 + F_im**2)
    F[0] = F_re
    F[1] = F_im
    return F
def strukturfaktor( F ):
    f = complex(F[0] ,F[1])
    f_conj = complex(F[0] ,-F[1])
    F_2 = np.real(f*f_conj)
    return F_2
def calc_inensity(L,P, F2):
    return L*P*F2
def print_table_strukture( _2theta, hkl, Hf , opt):
    theta = _2theta/2.
    L = []
    P = []
    f_Na = []
    f_Cl = []
    F_A = []
    F_B = []
    F2_A =[]
    F2_B = []
    if opt == 1:
        print( 'Peak-Nr. & $2\\theta$ & $hkl$ & $L$ & $P$ & $H_{hkl}$ & $f_\\text{Na}$ & $f_\\text{Cl}$ & $F_{A}$ & $F_B$ & $|F_A|^2$ & $|F_B|^2$ \\\\ \\hline' )
    for i in range(len(_2theta)):
        L.append( lorentzfaktor(theta[i] * np.pi / 180.) )
        P.append( polfaktor(_2theta[i] * np.pi / 180.) )
        f_Na.append( atomformfaktor(theta[i], 'Na+') )
        f_Cl.append( atomformfaktor(theta[i], 'Cl-') )
        F_A.append( strukturamplitude(f_Na[i], f_Cl[i], 'A', hkl[i]) )
        F_B.append( strukturamplitude(f_Na[i], f_Cl[i], 'B', hkl[i]) )
        F2_A.append(strukturfaktor(F_A[i]))
        F2_B.append(strukturfaktor(F_B[i]))
        if opt == 1:
            print(str(i+1), ' & $',
              str(np.round(_2theta[i]*100)/100)               , '$ & $'
              '( ', str(hkl[i][0]),',',str(hkl[i][1]),',',str(hkl[i][2]), ' )'  , '$ & $',
              str(np.round(L[i]*100)/100)                                       , '$ & $',
              str(np.round(P[i]*100)/100)                                       , '$ & $',
              str(Hf[i])                                                        , '$ & $',
              str(np.round(f_Na[i] * 100) / 100)                                , '$ & $',
              str(np.round(f_Cl[i] * 100) / 100)                                , '$ & $',
              str(np.round(F_A[i][0] * 100) / 100), '$ & $',
              str(np.round(F_B[i][0] * 100) / 100), '$ & $',
              str(np.round(F2_A[i] * 100) / 100), '$ & $',
              str(np.round(F2_B[i] * 100) / 100), '$ \\\\'
              )
    return [np.transpose(F2_A), np.transpose(F2_B)]
def print_table_intensity(_2theta, intens, hkl, Hf, F2, opt):
    theta = _2theta/2.
    L=[]
    P=[]
    I_A = []
    I_B = []
    for i in range(len(intens)):
        L.append(lorentzfaktor(theta[i] * np.pi / 180.))
        P.append(polfaktor(_2theta[i] * np.pi / 180.))
        I_A.append(calc_inensity(L[i], P[i], F2[0][i]) )
        I_B.append(calc_inensity(L[i], P[i], F2[1][i]))
    if opt == 1:
        print('Peak-Nr. & $hkl$ & $I_\\text{Messung}$ & $I_A$ & $I_B$ & $I^\\text{norm}_\\text{Mess}$ & $I^\\text{norm}_A$ & $I^\\text{norm}_B$ \\\\ \\hline')
        for i in range(len(intens)):
            print(str(i + 1), ' & $',
                  '( ', str(hkl[i][0]), ',', str(hkl[i][1]), ',', str(hkl[i][2]), ' )', '$ & $',
                  str( int(np.round(intens[i])) ), '$ & $',
                  str(int(np.round(I_A[i]))), '$ & $',
                  str(int(np.round(I_B[i]))), '$ & $',
                  str(np.round(intens[i]/intens[0]*100)/100), '$ & $',
                  str(np.round(I_A[i] / I_A[0] * 100) / 100), '$ & $',
                  str(np.round(I_B[i] / I_B[0] * 100) / 100), '$ \\\\'
              )
    return [np.transpose(intens/intens[0]), np.transpose(I_A / I_A[0]), np.transpose(I_B / I_B[0]) ]
Hf = [ 6,12, 24,8,6,24,24,24,24,12,6,24 ]

_2theta = np.array([31.212000000000003, 45.759, 54.56100000000001, 57.283, 67.505, 74.673, 77.032, 86.226, 93.001, 104.282, 111.208, 113.658])
s_2theta = np.array([0.15726956481153873, 0.10485497947961288, 0.37722269533529207, 0.13450951610463863, 0.1844340634908383, 0.7954957747927066, 0.11616596331232076, 0.15515337550949332, 0.7867916603753885, 0.2788530567859875, 0.764323745615404, 0.307396186018411])

theta = _2theta/2.

peak_intens = np.array([1504.254743905688, 1584.2623933762159, 120.98909581807797, 708.8415733434098 ,143.0657181453868 ,82.49781778086393, 1260.8339528513254, 1030.413074024467, 117.93000730903586 ,389.8789289136629, 88.15085077267919, 767.685904742168])
s_peak_intens = np.array([158.04975722, 171.54006845,  45.17963621,  60.96177146, 276.87452341,30.70518039, 174.78667193, 158.0696006,   52.65978216,  77.55012603, 38.73995262, 170.32289177])

hkl = [ [2,0,0], [2,2,0], [3,1,1], [2,2,2], [4,0,0]
    , [4,2,0], [2,3,3], [5,1,0],[5,1,1], [4,4,0], [6,0,0], [6,1,1] ]


F2 = print_table_strukture(_2theta, hkl, Hf,0)  # letzte Option: 1 Ausgabe der Tabelle in Latexform
#print(F2)
I = print_table_intensity(_2theta,peak_intens, hkl, Hf, F2, 0)

plt.figure(1, dpi=150, figsize=(9.0,5.0))
peak_nr1 = np.arange(1,(len(peak_intens))*5+1,5)
peak_nr2 = np.arange(2,(len(peak_intens))*5+1,5)
peak_nr3 = np.arange(3,(len(peak_intens))*5+1,5)


plt.bar(peak_nr1,I[0], label='${I_{Mess}}$')
plt.bar(peak_nr2,I[1], label='${I_A}$')
plt.bar(peak_nr3,I[2], label='${I_B}$')
plt.yscale('log')
plt.axis([0,59, 10**(-4),10**2])
plt.xticks(np.arange(2,(len(peak_intens))*5+1,5), range(1,len(peak_intens)+1))
plt.xlabel('Peak-Nr.')
plt.ylabel('Normierte Intensitäten')
plt.legend()
plt.savefig('Modell-Histogramm')


plt.show()

#print( atomformfaktor(theta[2], 'Na+'))
'''
for i in range(len(_2theta)):
    f_Na = atomformfaktor(theta[i], 'Na+')
    f_Cl = atomformfaktor(theta[i], 'Cl-')
    F = strukturamplitude(f_Na, f_Cl, 'A', hkl[i])
    F_2 = strukturfaktor(F)
    print( F_2 )
#print()
#print(atomformfaktor(theta[0], 'Na'))
'''