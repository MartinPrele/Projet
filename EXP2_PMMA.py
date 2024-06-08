import matplotlib.pyplot as plt
import numpy as np

FC=[13.6,6.3,4.3,3.2,1.3]
R=[0,2,3,4,5]

fitFC=np.polyfit(R[1:4],FC[1:4],1)
FITFC=[fitFC[1]+i*fitFC[0] for i in R]

plt.figure(dpi=400)
plt.plot(R,FC,'r*')
#plt.plot(R,FITFC,color='coral')
plt.grid()