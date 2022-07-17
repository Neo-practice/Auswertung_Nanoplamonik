import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit

def lin_func(x, a,b):
    return a+ b*x


_2theta_unser = np.array([31.212000000000003, 45.759, 54.56100000000001, 57.283, 67.505, 74.673, 77.032, 86.226, 93.001, 104.282, 111.208, 113.658])
_2theta_jana =  np.array([ 31.7391, 45.5005, 53.9322, 56.5399, 66.3088, 73.1632, 75.3888 ])

_2theta_unser = _2theta_unser[0:7]
kombi = [ '(2,0,0)', '(2,2,0)', '(3,1,1)', '(2,2,2)', '(4,0,0)', '(4,2,0)', '(2,3,3)' ]

difference = np.zeros((len(_2theta_jana)))
for i in range(len(difference)):
    difference[i] = np.abs( _2theta_jana[i]-_2theta_unser[i] )


parameter, covariance_matrix1 = curve_fit(lin_func, np.arange(7), difference, [0.2,1])
diff_fit = lin_func(np.arange(7), *parameter)

plt.figure(1, dpi=150, figsize=(9.0,5.0))
plt.plot( difference, ls='', marker='x', mew=2 , label='Abweichung von Sollpositionen')
plt.plot(np.arange(7), diff_fit, c='k', label= 'linearer Fit f(x) = a+b${\cdot}$x\na = '
                                               +str(np.round(parameter[0]*100)/100)+
                                               '° , b = '+str(np.round(parameter[1]*100)/100)+'°' )
plt.xlabel('Peak')
plt.ylabel('$|2{\\theta - 2\\theta_{Jana}}|$ in °')
plt.xticks(np.arange(7), kombi)
plt.legend()

plt.savefig('Abweichungsplot')

plt.show()