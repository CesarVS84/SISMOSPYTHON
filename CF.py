
import numpy as np
#Dominio espacial
Lx = Ly = 25000.
#Dominio temporal
tfin = 15.
#Esfuerzos inciales
six = 60.
siy = 80.
sixy = 50.


# # Discretización

# In[3]:


#Espacial
nx = ny = 500
dx = Lx/nx
dy = Ly/ny
# Nodos de velocidad en x = i*dx (igual espaciamiento que usa el solver)
x = np.arange(nx)*dx
y = np.arange(ny)*dy

X,Y = np.meshgrid(x,y,indexing = 'ij')

xx = np.linspace(0,Lx,nx+1)
yy = np.linspace(0,Ly,ny+1)
#Temporal
nt = 15000
T = np.linspace(0,tfin,nt+1)
dt = tfin/nt
print('dx:',dx,'dy:',dy,'dt:',dt)


# # Criterio de Estabilidad

# In[4]:

# Y_gran = 59300000000.
# nu_gran = 0.23
# lambda_gran = Y_gran*nu_gran/((1+nu_gran)*(1-2*nu_gran))
# mu_gran =Y_gran/(2*(1+nu_gran))

# print(lambda_gran)
# print(mu_gran)

#Buscar el número de Courant
Rho = 2800.
Lambda = 3e10
Mu = 3e10
vp = np.sqrt((Lambda+2*Mu)/Rho)
vs = np.sqrt(Mu/Rho)
# Límite de estabilidad del esquema (derivadas promediadas en celdas):
# vp*dt/min(dx,dy) <= 1 (verificado numéricamente: estable en 1.0, inestable en 1.1)
dtestable = min(dx,dy)/vp
if dt < dtestable:
    print('El dt cumple con el criterio')
else:
    print('Debes escoger otro dt')
print('el dt que cumple con el criterio corresponde a: ', dtestable,'\nmientras que el dt que utilizaremos es: ', dt)
print('Velocidad onda P: ', vp)
print('Velocidad onda S: ', vs)
# Frecuencia máxima que la malla resuelve sin dispersión numérica importante
# (10 puntos por longitud de onda de la onda S, la más lenta)
fmax_malla = vs/(10*max(dx,dy))
print('Frecuencia máxima recomendada para la malla (10 pts/longitud de onda S): ', fmax_malla, 'Hz')
