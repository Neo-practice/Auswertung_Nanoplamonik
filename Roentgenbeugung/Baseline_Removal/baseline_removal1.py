import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BaselineRemoval import BaselineRemoval
from scipy.signal import lfilter

################################################################################################
# reading data (NA CL)
################################################################################################

data = pd.read_csv("data/HG_NACL_.DAT",
                   delimiter=r"\s+",
                   #delimiter="/t",
                   names=["x", "y"],
                   engine="python",
                   )

left=270
right=4045

input_array=data.y[left:right]
x = data.x[left:right]

n = 12  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

baseObj=BaselineRemoval(input_array)
Zhangfit_output=baseObj.ZhangFit()
plt.plot(x, Zhangfit_output, label="Zhangfit baseline removal")

smooth = lfilter(b,a,Zhangfit_output)

plt.plot(x, smooth, label="Zhangfit baseline removal (smoothed)")

plt.plot(x, input_array, label="data")

plt.title("NaCl")
plt.legend()
#plt.show()
plt.savefig("plots/nacl.png")
plt.close("all")

x_array = np.asmatrix(x)
y_array = np.asmatrix(smooth)
new_data = np.hstack((x_array.T, y_array.T))
np.savetxt('data_corrected_filter/HG4_NACL.dat', new_data)


################################################################################################
# reading data (DEXT)
################################################################################################

data = pd.read_csv("data/HG1_DEXT.DAT",
                   delimiter=r"\s+",
                   #delimiter="/t",
                   names=["x", "y"],
                   engine="python",
                   )

left=270
right=4045

input_array=data.y[left:right]
x = data.x[left:right]

n = 12  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

baseObj=BaselineRemoval(input_array)
Zhangfit_output=baseObj.ZhangFit()
plt.plot(x, Zhangfit_output, label="Zhangfit baseline removal")

smooth = lfilter(b,a,Zhangfit_output)
plt.plot(x, smooth, label="Zhangfit baseline removal (smoothed)")

plt.plot(x, input_array, label="data")

plt.title("Dextrose")
plt.legend()
#plt.show()
plt.savefig("plots/dextrose.png")
plt.close("all")

x_array = np.asmatrix(x)
y_array = np.asmatrix(smooth)
new_data = np.hstack((x_array.T, y_array.T))
np.savetxt('data_corrected_filter/HG4_DEXT.dat', new_data)