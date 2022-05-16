import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2

startfile = "spectrometer_vergroeßerung_ende.txt"
# endfile = "spectrometer_vergroeßerung_start_txt.txt"

image = np.loadtxt(startfile, skiprows=5)
plt.imshow(image, aspect="auto")
plt.show()
