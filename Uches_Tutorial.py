from pba import *
from matplotlib import pyplot as plt


# Construct a C box
cb = cBox(2,10)
plotcBox(cb)

# Computer 10 focal elements

computeFocalElements(cb,npoints=10,show_plot=True)

computeConfidenceInterval(cBox(1,2), show_plot = True)

cb2 = cBox(1, 3)
plotcBox(cb2)

plt.show()