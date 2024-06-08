import matplotlib.pyplot as plt
import numpy as np

image=plt.imread('/Users/prelemartin/Desktop/TIPE/CAM_python_aire.jpeg')


####
''' échelle 1cm en pixel '''

for i in range(len(image)):
    for j in range(len(image[i])):
        if image[i][j][0]>200 and image[i][j][1]<130 and image[i][j][2]<130:
            image[i][j]=[0,0,255]
        

def centre(xmin,xmax,ymin,ymax):
    x=0
    y=0
    s=0
    for i in range(ymin,ymax):
        for j in range(xmin,xmax):
            if image[i][j][0]==0:
                x+=j
                y+=i
                s+=1
    return (x//s , y//s)

def norme(a,b):
    return int(np.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2))
M1=centre(200,350,600,750)
M2=centre(200,350,760,900)

cm=norme(M1,M2)  # distance pixel en cm
print('1cm <=>',cm,'pixels')

image=np.flip(image)


####

# Paramètres de la spirale
r0=351
f=np.tan(6*np.pi/180)

# Nombre de points équidistants
num_points = 100

# Paramètre t avec des valeurs équidistantes
A=1.375*np.pi
B=2.077*np.pi
t_values = np.linspace(A,B, num_points)


x_offset=773-88
y_offset=81
# Calcul des coordonnées x et y
x_values = -r0 * np.exp(f * t_values) * np.cos(t_values)+x_offset
y_values = -r0 * np.exp(f * t_values) * np.sin(t_values)+y_offset

# Calcul des distances entre les points pour avoir une distribution équidistante
distances = np.cumsum(np.sqrt(np.diff(x_values)**2 + np.diff(y_values)**2))
t_values=t_values[:-1]
total_distance = distances[-1]

# Nombre de points équidistants désiré -1
equidistant_points =10

# Paramètre t pour les points équidistants
equidistant_t_values = np.interp(np.linspace(0, total_distance, equidistant_points), distances, t_values)

# Interpolation pour obtenir les coordonnées des points équidistants
equidistant_x = -r0 * np.exp(f * equidistant_t_values) * np.cos(equidistant_t_values)+x_offset
equidistant_y = -r0 * np.exp(f * equidistant_t_values) * np.sin(equidistant_t_values)+y_offset


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




''' passage en mm '''

cx=[i*10/cm for i in X_centre]
cy=[i*10/cm for i in Y_centre]
mx=np.array([i*10/cm for i in moy_x])
my=np.array([i*10/cm for i in moy_y])

rx=[X_centre[i]-moy_x[i] for i in range(len(X_centre))]
ry=[Y_centre[i]-moy_y[i] for i in range(len(Y_centre))]
r=[np.sqrt(rx[i]**2+ry[i]**2) for i in range(len(rx))]
Rx=[cx[i]-mx[i] for i in range(len(cx))]
Ry=[cy[i]-my[i] for i in range(len(cy))]
R=[np.sqrt(Rx[i]**2+Ry[i]**2) for i in range(len(Rx))]


print('point courbe x(mm):',mx)
print('point courbe y(mm):',my)


print('centre valeur x(mm) :',cx)
print('centre valeur y(mm) :',cy)


print('rayon en mm :', R)
trait_x=[]
trait_y=[]
for i in range(2*len(mx)-1):
    if i%2==0:
        trait_x.append(mx[i//2])
        trait_y.append(my[i//2])
    else :
        trait_x.append(cx[i//2])
        trait_y.append(cy[i//2])
        
# def d_f(i):
#     ang1=np.arctan2((moy_y[i]-Y_centre[i]),(moy_x[i]-X_centre[i]))
#     ang2=np.arctan2((moy_y[i+1]-Y_centre[i]),(moy_x[i+1]-X_centre[i]))
#     return ang1 , ang2

            
# def cerc(Xc,Yc,rc,num,a,b):
#     theta=np.linspace(a,b,num)
#     l_x=Xc+rc*np.cos(theta)
#     l_y=Yc+rc*np.sin(theta)
#     return list(l_x) , list(l_y)
Z=np.sqrt((mx-np.array([x_offset*10/cm for i in range(len(mx))]))**2+(my-np.array([y_offset*10/cm for i in range(len(mx))]))**2)
print(Z)
           
#%%
plt.figure(dpi=300)
k=1.29
plt.imshow(image,extent=[0,80*k,60*k,0])

plt.plot(trait_x,trait_y,'lightcoral')
plt.plot(cx,cy,'b*')
plt.plot(x_offset*10/cm,y_offset*10/cm,'ro')
plt.plot(mx,my,'r--o')

print('bobi', x_offset*10/cm,y_offset*10/cm)
# for i in range(len(X_centre)):
#     lx,ly = cerc(X_centre[i],Y_centre[i],r[i],100,d_f(i)[0],d_f(i)[1])
#     plt.plot(lx,ly,'b--')

# Tracer la spirale
# plt.plot(x_values, y_values,'gold')

# Tracer les points équidistants
#plt.plot(equidistant_x, equidistant_y,'rD')



# Ajouter des labels et une légende
plt.title('Centre de courbure de la spirale logarithmique')

plt.xlabel('x(mm)')
plt.ylabel('y(mm)')
plt.xlim(0,110)
plt.ylim(-10,80)



plt.legend()

# Afficher le tracé
plt.show()



