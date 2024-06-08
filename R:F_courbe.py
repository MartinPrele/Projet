import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
import numpy as np

alpha=np.linspace(0,25,1000)
facteur=1/(2*np.tan(alpha*np.pi/180))

fig, ax = plt.subplots(dpi=300)

ax.plot(alpha,facteur,'b',lw=2)
C='black'
plt.plot(13.75,2.04,'o',color=C)
plt.plot([13.75,13.75],[0,2.04],linestyle='--',color=C)
plt.plot([0,13.75],[2.04,2.04],linestyle='--',color=C)
ax.add_patch(Rectangle((15.1,-1),10,100,edgecolor = 'red',  fill = False, hatch='///'))
ax.add_patch(Rectangle((0,-1),8.9,100,edgecolor = 'red', fill = False, hatch='///'))

ax.text(-0.83, 1.5, '2 ',fontsize=10.5,color=C)

ax.text(0.10, 0.8, 'Rupture came', transform=ax.transAxes, fontsize=12, verticalalignment='top',bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
ax.text(0.777, 0.8, 'Glissement', transform=ax.transAxes, fontsize=12, verticalalignment='top',bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))


plt.title('Rapport de multiplicité entre F et N',fontsize=14)
plt.xlabel('$\\alpha$ (en °)',fontsize=12,labelpad=11)
plt.ylabel('N/F',fontsize=13,labelpad=13,rotation=0)
plt.ylim(0,30)
plt.xlim(0,20)
plt.grid()



plt.show()




