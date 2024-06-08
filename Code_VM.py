''' Pression coinceur à cames trou '''
import numpy as np
import matplotlib.pyplot as plt

stress=np.array([1.238,1.262,1.259,1.267,1.731,2.515,6.055])*100
rayon=np.array([0,1,2,3,4,4.5,5])

def UN(x):
    masse_base = 30
    rho = 2.7
    xun = (5*np.pi*(x*0.1)**2*0.5)/(masse_base/rho)*100
    return xun

Vol = UN(rayon)

x=np.linspace(0,18,100)
def f(x):
    return 120+np.exp(0.37*(x-0.995))

plt.figure(dpi=400)
plt.plot(x,f(x),color='dodgerblue')
plt.plot(Vol,stress,'ob')

##ligne
plt.axhline(y=570,color='red',linestyle='--')
plt.axhline(y=450,color='forestgreen',linestyle='--')

plt.plot(16.65 ,450,color='forestgreen',marker='D')
plt.plot([16.65,16.65],[450,0],color='forestgreen',linestyle='--')

plt.text(1,580,'$R_r$ (limite à la rupture)',color='red',fontsize =13)
plt.text(1,460,'$R_e$ (limite élastique)',color='forestgreen',fontsize =13)

##Paramètre figure
plt.ylim(0,630)
plt.xlabel('Gain massique (%)')
plt.title('Contrainte maximale pour une force de 5kN')
plt.ylabel('Contrainte maximale Von-Mises (MPa)')
plt.grid()
plt.legend()
plt.show()