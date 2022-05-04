import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


x = np.arange(0,100,0.1, dtype=float)


x0 = 30
gamma = 5

y1 = 1/((x**2.-x0**2)**2.+gamma**2.*x0**2.)

a = 4.4*10**(-5)
mu = 45
sig = 7
y2 = a*np.exp(-(x-mu)**2./(2*sig**2))

plt.figure(1)
plt.plot(x,y1)
plt.plot(x,y2)

fy1 = fft(y1)
fy2 = fft(y2)

falt = ifft( fy1*fy2 )

plt.figure(2)
plt.plot(x,y1*y2)


plt.show()