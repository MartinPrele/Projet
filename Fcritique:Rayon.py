import numpy as np
import matplotlib.pyplot as plt 

Fmax=[15.3,15.1,15,14.5,13.7,12,9.7,7.7,5.5,3.1,1.4]
R=np.array([0,1,2,3,4,4.25,4.5,4.75,5,5.25,5.5])
Vol=[(30/2.7)/(30/2.7-5*np.pi*(i*0.1)**2*0.5) for i in R]

Fmax_exp=[25,20,17.2,15.5]
R_exp=np.array([4,4.5,4.75,5])

def UN(x):
    masse_base = 30
    rho = 2.7
    xun = (5*np.pi*(x*0.1)**2*0.5)/(masse_base/rho)*100
    return xun

Vol = UN(R)
Vol_exp = UN(R_exp)

'''Simulation'''

##première regression
fit1=np.polyfit(Vol[:5],Fmax[:5],1)

x1=np.linspace(0,11.5,100)
FIT1=fit1[0]*x1+fit1[1]

fit1=np.polyfit(Vol[:5],Fmax[:5],1)

##deuxième regression
fit2=np.polyfit(Vol[5:-1],Fmax[5:-1],1)

x2=np.linspace(11.3,22,100)
FIT2=fit2[0]*x2+fit2[1]


''' Expérience '''
##regression
exp=np.linspace(0,25,1000)
fit_exp=np.polyfit(Vol_exp,Fmax_exp,1)
FIT_exp=fit_exp[0]*exp+fit_exp[1]

plt.figure(dpi=400)

plt.title('Force critique en fonction du rayon des perçages')

##figure expérimentale

plt.plot(exp,FIT_exp,color='cornflowerblue')
plt.plot(Vol_exp,Fmax_exp,'bv')
##figure simultaion
plt.plot(x2,FIT2,color='coral',linestyle='-')
plt.plot(x1,FIT1,color='coral',linestyle='-')
plt.plot(Vol,Fmax,'or')

##ligne 
# plt.axhline(y=6.75,xmin=0,xmax=0.8,color='forestgreen',linestyle='--')
# plt.plot([16.7,16.7],[6.75,0],color='forestgreen',linestyle='--')
# plt.plot(16.7 ,6.7,color='forestgreen',marker='D')
##Paramètre de la figure
plt.grid()
plt.ylabel('Force critique (kN)')
plt.xlabel('Gain massique (%)')
plt.xlim(-0.4,21)
plt.ylim(0,27)
plt.show()