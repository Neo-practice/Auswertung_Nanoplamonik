import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit


a = 5.633174166071078 *10**(-8)      # cm
s_a = 0.05724653330346782 * 10**(-8)
N_A = 6.02214076*10**(23)   # 1/mol
M_M = 58.44                 # g/mol
rho = 2.17                  # g/cm^3

Z = N_A*rho*(a)**3/(M_M)
s_Z = N_A*rho/M_M * 3*a**2*s_a

print( 'Z = ', Z, ' \\pm ', s_Z )