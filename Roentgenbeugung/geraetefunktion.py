import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit


def geraetefunc( theta, U, V, W):
    return np.sqrt( U*np.tan(theta)**2 +V*np.tan(theta) +W )
def theta_min( U, V ):
    return np.arctan( -V / (2*U) )*180/np.pi


fwhm = [0.3930000000000007, 0.3330000000000055, 0.18099999999999739, 0.36299999999999955, 0.33200000000000784, 0.4240000000000066, 0.4539999999999935, 0.5440000000000111, 0.6659999999999968, 0.695999999999998, 0.695999999999998, 0.7860000000000014]
fwhm = np.array(fwhm,dtype=float)/2.
_2theta = np.array([31.212000000000003, 45.759, 54.56100000000001, 57.283, 67.505, 74.673, 77.032, 86.226, 93.001, 104.282, 111.208, 113.658])
s_2theta = np.array([0.15726956481153873, 0.10485497947961288, 0.37722269533529207, 0.13450951610463863, 0.1844340634908383, 0.7954957747927066, 0.11616596331232076, 0.15515337550949332, 0.7867916603753885, 0.2788530567859875, 0.764323745615404, 0.307396186018411])
theta = _2theta/2.

jana = np.array([[27.3971, 0.1652] , [31.7391, 0.1695], [45.5005, 0.1842], [53.9322, 0.1944], [56.5399, 0.1978], [66.3088, 0.2116], [73.1632, 0.2226], [75.3888, 0.2264]], dtype = float)
jana = jana*0.5

parameter, covariance_matrix1 = curve_fit(geraetefunc, theta*np.pi/180, fwhm, p0=[0.01,-0.01,0.01])
theta_help = np.arange(10,60,0.01)*np.pi/180
#param = [0.0397, -0.011, 0.0131]
fit = geraetefunc(theta_help, *parameter)

print('U V W = ', parameter)

parameter_jana, covariance_matrix2 = curve_fit(geraetefunc, jana[:,0]*np.pi/180, jana[:,1], p0=[0.005,-0.005,0.005])
fit_jana = geraetefunc(theta_help, *parameter_jana)
print('U V W jana = ', parameter_jana)

plt.figure(1, dpi=150, figsize=(9.0,5.0))
plt.plot(theta, fwhm, ls='', marker='o', color='tab:blue', label='Messdaten')
plt.plot(theta_help*180/np.pi, fit, color='tab:blue', label='Halbwertsbreiten-Fit für Messdaten')
plt.plot(jana[:,0], jana[:,1], ls='', marker='o', color='tab:green', label='Vergleichsdaten aus Jana')
plt.plot(theta_help*180/np.pi, fit_jana, color='tab:green',label='Halbwertsbreiten-Fit für Vgl.-Daten')
plt.xlabel('${\\theta}$ in °')
plt.ylabel('FWHM in °')
plt.legend()

plt.savefig('Geraetefunktion-Plot')
print('Theta_min = ', theta_min(parameter[0],parameter[1] ))
print('FWHM( Min ) = ', geraetefunc(theta_min(parameter[0],parameter[1] )*np.pi/180, *parameter))
print('')
print('Theta_min = ', theta_min(parameter_jana[0],parameter_jana[1] ))
print('FWHM( Min )_Jana = ', geraetefunc(theta_min(parameter_jana[0],parameter_jana[1] )*np.pi/180, *parameter_jana))
print('FWHM( 0 )_Jana = ', geraetefunc(0, *parameter_jana))


plt.show()





