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

plt.plot(np.arange(len(data.y)), data.y)
# plt.show() # for showing data points by index
plt.savefig("plots/nacl_data.png")
plt.close("all")

left=270
right=4045

input_array=data.y[left:right]
x = data.x[left:right]

################################################################################################
# baseline correction without filtering (NA CL)
################################################################################################

polynomial_degree=5 #only needed for Modpoly and IModPoly algorithm

baseObj=BaselineRemoval(input_array)
Modpoly_output=baseObj.ModPoly(polynomial_degree)
Imodpoly_output=baseObj.IModPoly(polynomial_degree)
Zhangfit_output=baseObj.ZhangFit()

plt.plot(x, input_array, label="data")
plt.plot(x, Modpoly_output, label="Modpoly")
plt.plot(x, Imodpoly_output, label="IModpoly")
plt.plot(x, Zhangfit_output, label="Zhangfit")

plt.legend()
#plt.show()
plt.savefig("plots/nacl_corr.png")
plt.close("all")

x_array = np.asmatrix(x)
y_array = np.asmatrix(Zhangfit_output)
new_data = np.hstack((x_array.T, y_array.T))
np.savetxt('data_corrected/HG_NACL_.dat', new_data)

################################################################################################
# baseline correction with filtering (NA CL)
################################################################################################

n = 12  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
y_filter = lfilter(b, a, Zhangfit_output)
plt.plot(x, y_filter, label="data")
plt.legend()
#plt.show()
plt.savefig("plots/nacl_corr_filter.png")
plt.close("all")

x_array = np.asmatrix(x)
y_array = np.asmatrix(y_filter)
new_data = np.hstack((x_array.T, y_array.T))
np.savetxt('data_corrected_filter/HG_NACL_.dat', new_data)


################################################################################################
# reading data (DEXT)
################################################################################################

data = pd.read_csv("data/HG1_DEXT.DAT",
                   delimiter=r"\s+",
                   #delimiter="/t",
                   names=["x", "y"],
                   engine="python",
                   )
x = np.arange(0, len(data.y))
plt.plot(x, data.y)
#plt.show() # for showing datapoints by index
plt.savefig("plots/dext_data.png")
plt.close("all")

left=270
right=4045

input_array=data.y[left:right]
x = data.x[left:right]

################################################################################################
# baseline correction without filtering (DEXT)
################################################################################################

polynomial_degree=5 #only needed for Modpoly and IModPoly algorithm

baseObj=BaselineRemoval(input_array)
Modpoly_output=baseObj.ModPoly(polynomial_degree)
Imodpoly_output=baseObj.IModPoly(polynomial_degree)
Zhangfit_output=baseObj.ZhangFit()

plt.plot(x, input_array, label="data")
plt.plot(x, Modpoly_output, label="Modpoly")
plt.plot(x, Imodpoly_output, label="IModpoly")
plt.plot(x, Zhangfit_output, label="Zhangfit")

plt.legend()
plt.savefig("plots/dext_corr.png")
plt.close("all")

x_array = np.asmatrix(x)
y_array = np.asmatrix(Zhangfit_output)
new_data = np.hstack((x_array.T, y_array.T))
np.savetxt('data_corrected/HG1_DEXT_.dat', new_data)

################################################################################################
# baseline correction with filtering (DEXT)
################################################################################################

left=270
right=4045

input_array=data.y[left:right]
x = data.x[left:right]

n = 12  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
y_filter = lfilter(b,a,input_array)

baseObj=BaselineRemoval(y_filter)
Modpoly_output=baseObj.ModPoly(polynomial_degree)
Imodpoly_output=baseObj.IModPoly(polynomial_degree)
Zhangfit_output=baseObj.ZhangFit()
plt.plot(x, y_filter, label="data")
plt.plot(x, Modpoly_output, label="Modpoly")
plt.plot(x, Imodpoly_output, label="IModpoly")
plt.plot(x, Zhangfit_output, label="Zhangfit")
plt.legend()
#plt.show()
plt.savefig("plots/dext_corr_filter.png")
plt.close("all")

x_array = np.asmatrix(x)
y_array = np.asmatrix(Zhangfit_output)
new_data = np.hstack((x_array.T, y_array.T))
np.savetxt('data_corrected_filter/HG1_DEXT.dat', new_data)