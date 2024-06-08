
import numpy as np
import matplotlib.pyplot as plt

# Paramètres de la spirale
r0=200
f=np.tan(13.75*np.pi/180)

# Nombre de points équidistants
num_points = 100

# Paramètre t avec des valeurs équidistantes
t_values = np.linspace(0,2.05*np.pi, num_points)

# Calcul des coordonnées x et y
x_values = r0 * np.exp(f * t_values) * np.cos(t_values)
y_values = -r0 * np.exp(f * t_values) * np.sin(t_values)

# Calcul des distances entre les points pour avoir une distribution équidistante
distances = np.cumsum(np.sqrt(np.diff(x_values)**2 + np.diff(y_values)**2))
t_values=t_values[:-1]
print(len(distances))
print(len(t_values))
total_distance = distances[-1]

# Nombre de points équidistants désiré
equidistant_points = 10

# Paramètre t pour les points équidistants
equidistant_t_values = np.interp(np.linspace(0, total_distance, equidistant_points), distances, t_values)

# Interpolation pour obtenir les coordonnées des points équidistants
equidistant_x = r0 * np.exp(f * equidistant_t_values) * np.cos(equidistant_t_values)
equidistant_y = -r0 * np.exp(f * equidistant_t_values) * np.sin(equidistant_t_values)



# point moyen M

def M(A,B):
    Mx=(A[0]+B[0])/2
    My=(A[1]+B[1])/2
    return [Mx,My]

def vect_normale(A,B):
    a=B[0]-A[0]
    b=B[1]-A[1]
    return [-b,a]
    

def centre_inter(norm1,norm2,M1,M2):
    A1=norm1[1]/norm1[0]
    B1=-A1*M1[0]+M1[1]
    A2=norm2[1]/norm2[0]
    B2=-A2*M2[0]+M2[1]
    xc=(B2-B1)/(A1-A2)
    return [xc,A1*xc+B1]

def tout(equidistant_x,equidistant_y):
    centre_x=[]
    centre_y=[]
    for i in range(1,len(equidistant_x)-1):
        A=[equidistant_x[i-1],equidistant_y[i-1]]
        B=[equidistant_x[i],equidistant_y[i]]
        C=[equidistant_x[i+1],equidistant_y[i+1]]
    
        ci = centre_inter(vect_normale(A,B),vect_normale(B,C),M(A,B),M(B,C))
        centre_x.append(ci[0])
        centre_y.append(ci[1])
    return centre_x,centre_y


X_centre,Y_centre=tout(equidistant_x,equidistant_y)

moy_x=[M([equidistant_x[i],equidistant_y[i]],[equidistant_x[i+1],equidistant_y[i+1]])[0] for i in range(equidistant_points-1)]
moy_y=[M([equidistant_x[i],equidistant_y[i]],[equidistant_x[i+1],equidistant_y[i+1]])[1] for i in range(equidistant_points-1)]

normale=[vect_normale([equidistant_x[i],equidistant_y[i]],[equidistant_x[i+1],equidistant_y[i+1]]) for i in range(equidistant_points-1)]

trait_x=[]
trait_y=[]
for i in range(2*len(moy_x)-1):
    if i%2==0:
        trait_x.append(moy_x[i//2])
        trait_y.append(moy_y[i//2])
    else :
        trait_x.append(X_centre[i//2])
        trait_y.append(Y_centre[i//2])
        


plt.figure(dpi=300)
plt.plot(trait_x,trait_y,'b-')
plt.plot(X_centre,Y_centre,'g*',label='centre de courbure')
plt.plot(moy_x,moy_y,'r*',label='point médiant')



 
# Tracer la spirale
plt.plot(x_values, y_values,'blue')

# Tracer les points équidistants
plt.plot(equidistant_x, equidistant_y,'rD', label='Points équidistants')



# Ajouter des labels et une légende
plt.title('Centre de courbure de la spirale logarithmique')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-500,1200)
plt.ylim(-400,1000)
plt.grid()
plt.legend()

# Afficher le tracé
plt.show()