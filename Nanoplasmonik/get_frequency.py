import numpy as np
import matplotlib.pyplot as plt



xwave = np.array([400, 425, 450, 475, 500, 510, 525, 540, 550, 575, 600, 610, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900])
blue = np.array([45, 65, 85, 88, 80, 58, 38, 28, 21, 17, 13, 12, 11, 12, 13, 17, 17, 18, 25, 45, 52, 42, 33, 23])
green = np.array([8, 9, 12, 19, 30, 59, 83, 91, 87, 65, 44, 33, 26, 20, 22, 28, 30, 33, 37, 48, 52, 42, 34, 27])
red = np.array([12, 7, 4, 5, 6, 9, 13, 15, 16, 50, 90, 100, 94, 85, 73, 68, 62, 60, 58, 56, 56, 47, 37, 28])

# stuetz = xwave

# rrel = np.interp(stuetz,[400, 450, 550, 620, 650, 651, 900], [12, 5, 25, 100, 85, 0, 0])
# grel = np.interp(stuetz,[400, 450, 540, 635, 650, 651, 900], [8, 12, 93, 20, 20, 0, 0])
# brel = np.interp(stuetz,[400, 450, 550, 620, 650, 651, 900], [45, 5, 25, 100, 85, 0, 0])

# rrel = np.interp(lamp[0:2068,0],[400, 450, 550, 620, 650, 651, 900], [12, 5, 25, 100, 85, 0, 0])
# grel = np.interp(lamp[0:2068,0],[400, 450, 540, 635, 650, 651, 900], [8, 12, 93, 20, 20, 0, 0])
# brel = np.interp(lamp[0:2068,0],[400, 450, 550, 620, 650, 651, 900], [45, 5, 25, 100, 85, 0, 0])


plt.plot(xwave, blue)
plt.plot(xwave, green)
plt.plot(xwave, red)
plt.show()
