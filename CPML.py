#!/usr/bin/env python
# coding: utf-8

# ## SISMO ANALIZADO EN UN TIEMPO $T = 1$ SEGUNDO

# Vamos a modelar la onda resolviendo primero que todo el sistema de ecuaciones diferenciales lineales acoplado:
# 
# \begin{equation}
# \begin{split}
# \frac{\partial v_x}{\partial t} &= \frac{1}{\rho} \left[ \frac{\partial \sigma_{XX}}{\partial x} + \frac{\partial \sigma_{XY}}{\partial y} \right] \\
# \frac{\partial v_y}{\partial t} &= \frac{1}{\rho} \left[ \frac{\partial \sigma_{XY}}{\partial x} + \frac{\partial \sigma_{YY}}{\partial y} \right] \\
# \frac{\partial \sigma_{XX}}{\partial t} &= (\lambda + 2\mu) \frac{\partial v_x}{\partial x} + \lambda \frac{\partial v_y}{\partial y}\\
# \frac{\partial \sigma_{YY}}{\partial t} &= (\lambda + 2\mu) \frac{\partial v_y}{\partial y} + \lambda \frac{\partial v_x}{\partial x}\\
# \frac{\partial \sigma_{XY}}{\partial t} &= \mu \left[ \frac{\partial v_x}{\partial y} + \frac{\partial v_y}{\partial x}\right]
# \end{split}
# \end{equation}

# El cual se discretizará utilizando diferencias finitas adelantadas, es decir: 

# $\dfrac{\partial v}{\partial t} \approx \dfrac{v[i+1]-v[i]}{\Delta t}$ 

# Así la discretización de cada una de las ecuaciones: 

# VELOCIDAD EN DIRECCION X

# \begin{equation}
# \begin{split}
# v^{k+1}_{x_[i,j]} = v^k_{x_{[i,j]}} + \frac{\Delta t}{\rho} \left[ \frac{\sigma^k_{{{XX}}_{[i+1,j]}}- \sigma^k_{{XX}_{[i,j]}}}{ \Delta x} \right] + \frac{\Delta t}{\rho} \left[ \frac{\sigma^k_{{XY}_{[i,j+1]}}- \sigma^k_{{XY}_{[i,j]}}}{ \Delta y} \right]
# \end{split}
# \end{equation}

# VELOCIDAD EN DIRECCION Y

# \begin{equation}
# \begin{split}
# vy^{k+1}_{[i,j]} = vy^k_{[i,j]} + \frac{\Delta t}{\rho} \left[ \frac{\sigma^k_{{XY}_{[i+1,j]}}- \sigma^k_{{XY}_{[i,j]}}}{ \Delta x} \right] + \frac{\Delta t}{\rho} \left[ \frac{\sigma^k_{{YY}_{[i,j+1]}}- \sigma^k_{{YY}_{[i,j]}}}{ \Delta y} \right]
# \end{split}
# \end{equation}

# SIGMA XX

# \begin{equation}
# \begin{split}
# \sigma^{k+1}_{{XX}_{[i,j]}} = \sigma^{k}_{{XX}_{[i,j]}} + \Delta t (\lambda + 2\mu) \left[ \frac{vx^k_{[i+1,j]}- vx^k_{[i,j]}}{\Delta x} \right] + \Delta t \lambda \left[ \frac{vy^k_{[i,j+1]} - vy^k_{[i,j]}}{\Delta y} \right] 
# \end{split}
# \end{equation}

# SIGMA YY

# \begin{equation}
# \begin{split}
# \sigma^{k+1}_{{YY}_{[i,j]}} = \sigma^{k}_{{YY}_{[i,j]}} + \Delta t (\lambda + 2\mu) \left[ \frac{vy^k_{[i,j+1]}- vy^k_{[i,j]}}{ \Delta y} \right] + \Delta t \lambda \left[ \frac{vx^k_{[i+1,j]} - vx^k_{[i,j]}}{\Delta x} \right] 
# \end{split}
# \end{equation}

# SIGMA XY

# \begin{equation}
# \begin{split}
# \sigma^{k+1}_{{XY}_{[i,j]}} = \sigma^{k}_{{XY}_{[i,j]}} + \Delta t \mu \left[ \frac{vx^k_{[i,j+1]} - vx^k_{[i,j]}}{\Delta y} + \frac{vy^k_{[i+1,j]} - vy^k_{[i,j]}}{\Delta x} \right] 
# \end{split}
# \end{equation}


# # Cargando Librerias

# In[1]:


import numpy as np
import time
from numba import config, njit, prange, set_num_threads
from TOP import TOP, too,F,f,lambdaa, mu, rho, sour1
from prueba import identificador_de_pendientes,elevation, boundary_flag_velocity
from math import exp,sqrt
#from fuente import Top,top, vx1, vx2

# MacBook Air M4: 10 hilos. Si la máquina tiene menos, se usan los disponibles.
NUM_HILOS = 10
set_num_threads(min(NUM_HILOS, config.NUMBA_NUM_THREADS))

# # Cargando datos iniciales

# In[2]:


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
nt = 7500
T = np.linspace(0,tfin,nt+1)
dt = tfin/nt
print('dx:',dx,'dy:',dy,'dt:',dt)
# El solver usa dx tanto en x como en y (aa, b, c, d están divididos por dx)
assert dx == dy, 'El esquema de este programa supone dx == dy'


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
# Límite de estabilidad del esquema: vp*dt/min(dx,dy) <= 1
dtestable = min(dx,dy)/vp
if dt < dtestable:
    print('El dt cumple con el criterio')
else:
    print('Debes escoger otro dt')
print('el dt que cumple con el criterio corresponde a: ', dtestable,'\nmientras que el dt que utilizaremos es: ', dt)
print('Velocidad onda P: ', vp)
print('Velocidad onda S: ', vs)


# # Constantes Auxiliares

# In[5]:
LAMBDA = lambdaa
MU = mu
RHO = rho

# Modelo consistente en la superficie libre. Una celda es de roca solo si sus 4
# nodos lo son (en TOP.py la fila de celdas sobre la superficie también queda
# como roca), y la masa de cada nodo es proporcional a la cantidad de celdas de
# roca que lo rodean: aa = 4/(rho*n_celdas). Con aa = 1/rho en todos los nodos la
# onda de Rayleigh viaja un 5 % más lenta (0.872 vs. 0.919 de la velocidad S,
# medido en una superficie plana). Ponga False para el modelo original.
MODELO_CONSISTENTE = True
if MODELO_CONSISTENTE:
    _roca = (RHO.reshape(ny, nx) > 0)                       # [fila j, columna i]
    _celda = _roca[:-1, :-1] & _roca[:-1, 1:] & _roca[1:, :-1] & _roca[1:, 1:]
    LAMBDA = np.where(_celda, LAMBDA.reshape(ny-1, nx-1), 0.).ravel()
    MU = np.where(_celda, MU.reshape(ny-1, nx-1), 0.).ravel()
    _cuenta = np.zeros((ny, nx))
    for _dj in (0, 1):
        for _di in (0, 1):
            _cuenta[_dj:ny-1+_dj, _di:nx-1+_di] += _celda
    aa = np.zeros(nx*ny)
    _m = (_cuenta > 0).ravel()
    aa[_m] = (4./(RHO[_m]*_cuenta.ravel()[_m]))/dx
else:
    aa = np.zeros(nx*ny)
    mask = RHO != 0
    aa[mask] = (1/RHO[mask])/dx

# Coeficientes para el interior (ya divididos por dx: las derivadas se usan sin dividir)
b = (LAMBDA+2*MU)/dx
c = LAMBDA/dx
d = MU/dx
# Coeficientes para la CPML. cpml_stress y cpml_vel dividen ellas mismas por dx,
# por lo que reciben los valores SIN dividir.
aa_pml = aa*dx
b_pml = b*dx
c_pml = c*dx
d_pml = d*dx
# # Matrices Auxiliares


#Angulo  de incidencia respecto al eje x
#debido a la topografia 
thetag = 30.
def rad(g):
    return g*np.pi/180

theta = rad(thetag)

fa = sour1

# In[6]:


#Esfuerzos interiores
Sxx = np.zeros((nx-1)*(ny-1))
Syy = np.zeros((nx-1)*(ny-1))
Sxy = np.zeros((nx-1)*(ny-1))
Sxxn = np.zeros((nx-1)*(ny-1))
Syyn = np.zeros((nx-1)*(ny-1))
Sxyn = np.zeros((nx-1)*(ny-1))
SXXN = np.zeros((nx-1)*(ny-1))
SXYN = np.zeros((nx-1)*(ny-1))
SYYN = np.zeros((nx-1)*(ny-1))


#Esfuerzos con borde
Sbxx = np.zeros((nx+1)*(ny+1))
Sbyy = np.zeros((nx+1)*(ny+1))
Sbxy = np.zeros((nx+1)*(ny+1))


# Velocidades con borde

vx_out = np.zeros(nx*ny)
vy_out = np.zeros(nx*ny)
#Velocidades
vx = np.zeros(nx*ny)
vy = np.zeros(nx*ny)
vxout = np.zeros(nx*ny)
vyout = np.zeros(nx*ny)
Vx = np.zeros([nx*ny])
Vy = np.zeros([nx*ny])

VXX2 = np.zeros([])
VXX3 = np.zeros([])

#Estaciones
e1 = [170,356]
e2 = [190,360]
e3 = [210,368]
e4 = [261,392]
e5 = [281,381]
e6 = [301,369]
e7 = [321,361]
e8 = [200,250]    
e9 = [300,250] 
fa = sour1
estaciones = [e1,e2,e3,e4,e5,e6,e7,e8,e9,fa]
# Registro de las estaciones (se llena en el bucle temporal): [estación, paso, (vx, vy)]
registro = np.zeros((len(estaciones), nt+1, 2))

#Desplazamientos:
Uoldx = np.zeros(nx*ny)
Uoldy = np.zeros(nx*ny)
Unewx = np.zeros(nx*ny)
Unewy = np.zeros(nx*ny)

Velx = np.zeros((1,nx*ny))
Vely = np.zeros((1,nx*ny))
Uxxx = np.zeros((1,nx*ny))
Uyyy = np.zeros((1,nx*ny))


# Definiendo los valores en del stress interior en el punto $(305,175)$

# In[7]:


def R(i,j):
    return i+(nx-1)*j
w= int(R(200,175))
#print(w)

def e(i,j):
    return int(i+nx*j)



# In[8]:


Sxx[w] = Sxxn[w] = six
Syy[w] = Syyn[w] = siy
Sxy[w] = Sxyn[w] = sixy


# In[9]:


'''Las funciones que antes leían arreglos GLOBALES (Sbxxdx, vxdx, ...) están 
ahora dentro de los kernels de abajo, que reciben los arreglos como argumentos.
Numba "congela" los arreglos globales en el momento de compilar: con esas 
funciones los esfuerzos y velocidades que se leían eran siempre los del primer 
paso de tiempo.'''

#Esfuerzos con borde: Sb[i,j] = S[i-1,j-1]; el contorno vale cero
@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def construir_Sb(Sbxx, Sbyy, Sbxy, Sxxn, Syyn, Sxyn, nx, ny):
    for j in prange(1, ny):
        for i in range(1, nx):
            Sbxx[i+(nx+1)*j] = Sxxn[i-1+(nx-1)*(j-1)]
            Sbyy[i+(nx+1)*j] = Syyn[i-1+(nx-1)*(j-1)]
            Sbxy[i+(nx+1)*j] = Sxyn[i-1+(nx-1)*(j-1)]


#Velocidades en todo el dominio sin borde topográfico (no incluye el efecto de la CPML)
@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def wave(vx, vy, vx_out, vy_out, Sbxx, Sbxy, Sbyy, aa, dt, damp, nx, ny):
    for j in prange(ny):
        for i in range(nx):
            p = i+nx*j
            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                vx_out[p] = 0.
                vy_out[p] = 0.
            else:
                Sbxxdx = (Sbxx[i+1+(nx+1)*j] - Sbxx[i+(nx+1)*j] + Sbxx[i+1+(nx+1)*(j+1)] - Sbxx[i+(nx+1)*(j+1)])
                Sbxydy = (Sbxy[i+(nx+1)*(j+1)] - Sbxy[i+(nx+1)*j] + Sbxy[i+1+(nx+1)*(j+1)] - Sbxy[i+1+(nx+1)*j])
                Sbxydx = (Sbxy[i+1+(nx+1)*j] - Sbxy[i+(nx+1)*j] + Sbxy[i+1+(nx+1)*(j+1)] - Sbxy[i+(nx+1)*(j+1)])
                Sbyydy = (Sbyy[i+(nx+1)*(j+1)] - Sbyy[i+(nx+1)*j] + Sbyy[i+1+(nx+1)*(j+1)] - Sbyy[i+1+(nx+1)*j])
                vx_out[p] = vx[p]*(1-damp) + 0.5*aa[p]*dt*(Sbxxdx + Sbxydy)
                vy_out[p] = vy[p]*(1-damp) + 0.5*aa[p]*dt*(Sbxydx + Sbyydy)


#Esfuerzos en el centro de cada celda (a partir de las velocidades nuevas)
@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def esfuerzos(Sxxn, Syyn, Sxyn, Sxx, Syy, Sxy, vx_out, vy_out, b, c, d, dt, nx, ny):
    for j in prange(ny-1):
        for i in range(nx-1):
            q = i+(nx-1)*j
            k = i+nx*j
            vxdx = (vx_out[k+1] - vx_out[k] + vx_out[k+1+nx] - vx_out[k+nx])
            vydy = (vy_out[k+nx] - vy_out[k] + vy_out[k+1+nx] - vy_out[k+1])
            vxdy = (vx_out[k+nx] - vx_out[k] + vx_out[k+1+nx] - vx_out[k+1])
            vydx = (vy_out[k+1] - vy_out[k] + vy_out[k+1+nx] - vy_out[k+nx])
            Sxxn[q] = Sxx[q] + 0.5*dt*(b[q]*vxdx + c[q]*vydy)
            Syyn[q] = Syy[q] + 0.5*dt*(b[q]*vydy + c[q]*vxdx)
            Sxyn[q] = Sxy[q] + 0.5*dt*d[q]*(vxdy + vydx)


# In[10]:


#Definiendo la función de fuente en Vx:
w = 1256.5
z = 0.0001
h1 = 5.
h2 = 15.
h3 = 5/3

def r(t):
    return w*(1*np.exp((-1+np.sqrt(z))/2*(h1*10*t))-1*np.exp((-1-np.sqrt(z))/2*(h1*10*t)))

def s(t):
    return -(w*(1*np.exp((-1+np.sqrt(z))/2*(h2*10*t))-1*np.exp((-1-np.sqrt(z))/2*(h2*10*t))))

def o(t):
    return w*(1*np.exp((-1+np.sqrt(z))/2*(h3*10*t))-1*np.exp((-1-np.sqrt(z))/2*(h3*10*t)))

# # Agregando las condiciones PML

# In[11]:

from CBO import cboveljih_inplace
from CPMLD import damp_profiles, cpml_stress, cpml_vel, init_memory_velocity, init_memory_stress


# In[12]:

damp = 1e-3
npow = 2.0
pi = 3.141592654
'''revisar topografia si coincide con los parametros del cpml'''
pml_points_x1 = 100
pml_points_x2 = 100
pml_points_y1 = 100
pml_points_y2 = 100
assert 2*pml_points_x1 < nx and 2*pml_points_y1 < ny, 'La CPML no cabe en la malla'

# Velocidad P que usa la capa. Antes: cp = 3300. (distinta de la velocidad P
# del modelo, que es vp). Si el modelo de TOP.py tiene una velocidad P mayor,
# use esa.
cp = vp


vel_damping = 5e-3
Rcoef = 0.001
f0=7.0
k_max_pml = 3.0
alpha_max_pml = 2.0*pi*(f0/2.0);
# d0 = -(N+1) cp ln(R) / (2L): logaritmo natural (antes log, que en numpy es el natural; ver CPMLD)
d0x = -(npow + 1)*cp*np.log(Rcoef) / (2.*pml_points_x1*dx)
d0y = -(npow + 1)*cp*np.log(Rcoef) / (2.*pml_points_y1*dy)
a_x = np.zeros(nx*ny)
alpha_x = np.zeros(nx*ny)
k_x = np.ones(nx*ny)
b_x = np.zeros(nx*ny)
d_x = np.zeros(nx*ny)
a_y = np.zeros(nx*ny)
alpha_y = np.zeros(nx*ny)
k_y = np.ones(nx*ny)
b_y = np.zeros(nx*ny)
d_y = np.zeros(nx*ny)

a_x_half = np.zeros((nx-1)*(ny-1))
alpha_x_half = np.zeros((nx-1)*(ny-1))
k_x_half = np.ones((nx-1)*(ny-1))
b_x_half = np.zeros((nx-1)*(ny-1))
d_x_half = np.zeros((nx-1)*(ny-1))
a_y_half = np.zeros((nx-1)*(ny-1))
alpha_y_half = np.zeros((nx-1)*(ny-1))
k_y_half = np.ones((nx-1)*(ny-1))
b_y_half = np.zeros((nx-1)*(ny-1))
d_y_half = np.zeros((nx-1)*(ny-1))
vxout = np.zeros(nx*ny)
vyout = np.zeros(nx*ny)

sigmaxx_out = np.zeros((nx-1)*(ny-1))
sigmayy_out = np.zeros((nx-1)*(ny-1))
sigmaxy_out = np.zeros((nx-1)*(ny-1))
mem_dvx_dx =  np.zeros((nx-1)*(ny-1))
mem_dvx_dy =  np.zeros((nx-1)*(ny-1))
mem_dvy_dx =  np.zeros((nx-1)*(ny-1))
mem_dvy_dy =  np.zeros((nx-1)*(ny-1))
mem_dvx_dx_out =  np.zeros((nx-1)*(ny-1))
mem_dvx_dy_out =  np.zeros((nx-1)*(ny-1))
mem_dvy_dx_out =  np.zeros((nx-1)*(ny-1))
mem_dvy_dy_out =  np.zeros((nx-1)*(ny-1))


mem_dsigmaxx_dx = np.zeros((nx)*(ny))
mem_dsigmaxy_dx = np.zeros((nx)*(ny))
mem_dsigmaxy_dy = np.zeros((nx)*(ny))
mem_dsigmayy_dy = np.zeros((nx)*(ny))
mem_dsigmaxx_dx_out = np.zeros((nx)*(ny))
mem_dsigmaxy_dx_out = np.zeros((nx)*(ny))
mem_dsigmaxy_dy_out = np.zeros((nx)*(ny))
mem_dsigmayy_dy_out = np.zeros((nx)*(ny))


#inicializar en:
alpha_x[:] = 1.1*alpha_max_pml
alpha_y[:] = 1.1*alpha_max_pml
b_x[:] = exp(-1.1*alpha_max_pml*dt)
b_y[:] = exp(-1.1*alpha_max_pml*dt)

alpha_x_half[:] = 1.1*alpha_max_pml
alpha_y_half[:] = 1.1*alpha_max_pml
b_x_half[:] = exp(-1.1*alpha_max_pml*dt)
b_y_half[:] = exp(-1.1*alpha_max_pml*dt)
        
# In[13]:


#Profile
damp_profiles(a_x, a_x_half, a_y, a_y_half, d_x, k_x, alpha_x, b_x, 
              d_x_half, k_x_half, alpha_x_half, b_x_half, d_y, k_y,
              alpha_y, b_y, d_y_half, k_y_half, alpha_y_half, b_y_half,
              npow, k_max_pml, alpha_max_pml, pml_points_x1, pml_points_x2,
              pml_points_y1, pml_points_y2, d0x, d0y, dx, dy, nx, ny, dt) 

init_memory_stress(Sbxx, Sbxy, Sbyy, mem_dsigmaxx_dx, mem_dsigmaxy_dx,
               mem_dsigmaxy_dy, mem_dsigmayy_dy, k_x, k_y, pml_points_x1,
               pml_points_x2, pml_points_y1, pml_points_y2, dx, dy, nx, ny)


init_memory_velocity(vx,vy,mem_dvx_dx, mem_dvy_dx, mem_dvx_dy, mem_dvy_dy,
                 k_x_half, k_y_half, pml_points_x1, pml_points_x2, pml_points_y1,
                 pml_points_y2, dx, dy, nx, ny)


# In[14]:


#Definiendo las velocidades 

# Nodos de la fuente, buscados una sola vez (antes se recorría toda la malla en cada paso)
fuente50 = np.argwhere(np.asarray(F) == 50)
fuente90 = np.argwhere(np.asarray(F) == 90)

@njit(cache=True)
def aplicar_fuente(vx, vy, fuente50, fuente90, rc, rs, nx):
    for n in range(fuente50.shape[0]):
        p = fuente50[n, 0] + nx*fuente50[n, 1]
        vx[p] = rc
        vy[p] = -rs
    for n in range(fuente90.shape[0]):
        p = fuente90[n, 0] + nx*fuente90[n, 1]
        vx[p] = -rc
        vy[p] = rs

# Fuente suave. Con FUENTE_SIGMA = None la velocidad de la fuente se impone solo en los nodos
# F == 50 y F == 90 (como en el original): esos nodos quedan fijos en cero después del pulso,
# lo que actúa como un obstáculo rígido, y la fuente excita ondas de 2 a 4 nodos de largo que
# la CPML no absorbe (medido: el eco con la capa sube de -66 dB a -23 dB). Con un número
# (ancho en nodos), la velocidad se reparte en una gaussiana alrededor de cada nodo.
FUENTE_SIGMA = 1.5

def _pesos_fuente(sigma):
    pesos = {}
    R = int(np.ceil(4*sigma))
    for valor, signo in ((50, 1.), (90, -1.)):
        for i0, j0 in np.argwhere(np.asarray(F) == valor):
            for j in range(max(j0-R, 1), min(j0+R+1, ny-1)):
                for i in range(max(i0-R, 1), min(i0+R+1, nx-1)):
                    pesos[i+nx*j] = pesos.get(i+nx*j, 0.) + signo*np.exp(-((i-i0)**2+(j-j0)**2)/(2*sigma**2))
    idx = np.array(list(pesos.keys()), dtype=np.int64)
    return idx, np.array([pesos[k] for k in idx])

@njit(cache=True)
def asignar_fuente(vx, vy, idx, peso, rc, rs):
    for n in range(idx.shape[0]):
        vx[idx[n]] = peso[n]*rc
        vy[idx[n]] = -peso[n]*rs

if FUENTE_SIGMA:
    idx_fuente, peso_fuente = _pesos_fuente(FUENTE_SIGMA)

def source(t):
    rt = r(t)
    if FUENTE_SIGMA:
        asignar_fuente(vx, vy, idx_fuente, peso_fuente, rt*np.cos(theta), rt*np.sin(theta))
    else:
        aplicar_fuente(vx, vy, fuente50, fuente90, rt*np.cos(theta), rt*np.sin(theta), nx)


# Instantáneas: (tiempo, sufijo del nombre del archivo). Se guardan en el paso de
# tiempo más cercano (antes se comparaba t == 0.1 con números decimales, que
# casi nunca coinciden exactamente).
INSTANTANEAS = (
    (0.1, '1e1'), (0.2, '2e1'), (0.3, '3e1'), (0.4, '4e1'), (0.5, '5e1'), (0.6, '6e1'),
    (0.7, '7e1'), (0.8, '8e1'), (0.9, '9e1'), (1, '1'), (1.1, '11'), (1.2, '12'),
    (1.3, '13'), (1.4, '14'), (1.5, '15'), (1.6, '16'), (1.7, '17'), (1.8, '18'),
    (1.9, '19'), (2, '2'), (2.102, '21'), (2.2, '22'), (2.3, '23'), (2.4, '24'),
    (2.5, '25'), (2.6, '26'), (2.7, '27'), (2.8, '28'), (2.9, '29'), (3, '3'),
    (3.3, '33'), (3.2, '32'), (3.4, '34'), (3.5, '35'), (3.6, '36'), (3.7, '37'),
    (3.8, '38'), (3.9, '39'), (4, '4'), (4.1, '41'), (4.2, '42'), (4.3, '43'),
    (4.4, '44'), (4.5, '45'), (4.6, '46'), (4.7, '47'), (4.8, '48'), (4.9, '49'),
    (5, '5'), (5.1, '51'), (5.2, '52'), (5.3, '53'), (5.4, '54'), (5.5, '55'),
    (5.6, '56'), (5.7, '57'), (5.8, '58'), (5.9, '59'), (6, '6'), (6.1, '61'),
    (6.2, '62'), (6.3, '63'), (6.4, '64'), (6.5, '65'), (6.6, '66'), (6.7, '67'),
    (6.8, '68'), (6.9, '69'), (7.005, '7'), (7.1, '71'), (7.2, '72'), (7.3, '73'),
    (7.4, '74'), (7.5, '75'), (7.6, '76'), (7.705, '77'), (7.8, '78'), (7.9, '79'),
    (8, '8'), (8.1, '81'), (8.2, '82'), (8.3, '83'), (8.4, '84'), (8.5, '85'),
    (8.6, '86'), (8.7, '87'), (8.8, '88'), (8.9, '89'), (9, '9'), (9.105, '91'),
    (9.2, '92'), (9.3, '93'), (9.4, '94'), (9.5, '95'), (9.6, '96'), (9.7, '97'),
    (9.8, '98'), (9.9, '99'), (10, '10'), (10.1, '101'), (10.2, '102'), (10.3, '103'),
    (10.4, '104'), (10.505, '105'), (10.6, '106'), (10.7, '107'), (10.8, '108'), (10.9, '109'),
    (11, '110'), (11.1, '1101'), (11.2, '1102'), (11.3, '1103'), (11.4, '1104'), (11.5, '1105'),
    (11.6, '1106'), (11.7, '1107'), (11.8, '1108'), (11.9, '1109'), (12, '120'), (12.1, '1201'),
    (12.2, '1202'), (12.3, '1203'), (12.4, '1204'), (12.5, '1205'), (12.605, '1206'), (12.7, '1207'),
    (12.8, '1208'), (12.9, '1209'), (13, '130'), (13.1, '1301'), (13.2, '1302'), (13.3, '1303'),
    (13.4, '1304'), (13.5, '1305'), (13.6, '1306'), (13.7, '1307'), (13.8, '1308'), (13.9, '1309'),
    (14.005, '140'), (14.1, '1401'), (14.2, '1402'), (14.3, '1403'), (14.4, '1404'), (14.5, '1405'),
    (14.6, '1406'), (14.705, '1407'), (14.8, '1408'), (14.9, '1409'),
)
paso_guardado = {int(round(tg/dt)): sufijo for tg, sufijo in INSTANTANEAS}

def guardar_instantanea(sufijo):
    np.save('sxxt'+sufijo,Sbxx)
    np.save('syyt'+sufijo,Sbyy)
    np.save('sxyt'+sufijo,Sbxy)
    np.save('vyt'+sufijo,vy_out)
    np.save('vxt'+sufijo,vx_out)

# In[ ]:


#Loop temporal
t_inicio = time.time()
for n in range(nt+1):
    t = T[n]

    #Sbxx
    #(se construye a partir de Sxx, Syy, Sxy: los esfuerzos vigentes tras el intercambio de arreglos)
    construir_Sb(Sbxx, Sbyy, Sbxy, Sxx, Syy, Sxy, nx, ny)

    #Fuente
    source(t)
    
    #Loop para la velocidad x e y sin borde topografico
    wave(vx, vy, vx_out, vy_out, Sbxx, Sbxy, Sbyy, aa, dt, damp, nx, ny)
    
    #Borde topográfico (Jih). Lee una copia de las velocidades: leer y escribir
    #el mismo arreglo desde varios hilos da resultados que dependen del orden.
    cboveljih_inplace(elevation, boundary_flag_velocity, too, vx_out, vy_out,
                      vp, vs, dx, dy, nx, ny)
    
    #vx, vy son las velocidades del paso anterior y vx_out las nuevas (arreglos distintos)
    cpml_vel(vx_out, vy_out, Sbxx, Sbxy, Sbyy,vx, vy, mem_dsigmaxx_dx, 
             mem_dsigmaxy_dx, mem_dsigmaxy_dy, mem_dsigmayy_dy, mem_dsigmaxx_dx_out,
             mem_dsigmaxy_dx_out, mem_dsigmaxy_dy_out, mem_dsigmayy_dy_out, 
             a_x, a_y, k_x, alpha_x, b_x, k_y, alpha_y, b_y, pml_points_x1, 
             pml_points_x2, pml_points_y1, pml_points_y2, aa_pml, dx, dy, dt, 
             vel_damping, nx, ny)
    
    #Estaciones (e1..e9 y la fuente)
    for k, est in enumerate(estaciones):
        registro[k, n, 0] = vx_out[e(est[0], est[1])]
        registro[k, n, 1] = vy_out[e(est[0], est[1])]
    
    #Sxxn, Syyn, Sxyn
    esfuerzos(Sxxn, Syyn, Sxyn, Sxx, Syy, Sxy, vx_out, vy_out, b, c, d, dt, nx, ny)
    
    cpml_stress(Sxxn,Sxyn,Syyn,Sxx,Sxy,Syy,vx_out,vy_out,mem_dvx_dx, mem_dvy_dx, 
                mem_dvx_dy,mem_dvy_dy, mem_dvx_dx_out,mem_dvy_dx_out, 
                mem_dvx_dy_out, mem_dvy_dy_out, a_x_half, a_y_half, k_x_half, 
                alpha_x_half, b_x_half, k_y_half, alpha_y_half, b_y_half, 
                pml_points_x1, pml_points_x2, pml_points_y1,
                pml_points_y2, b_pml, c_pml, d_pml, dx,dy,dt,nx,ny)
    
    #Desplazamientos
    Unewx += vx_out*dt
    Unewy += vy_out*dt
    
    if n in paso_guardado:
        guardar_instantanea(paso_guardado[n])
    
    if n == nt:
        np.save('sxxtfin',Sbxx)
        np.save('syytfin',Sbyy)
        np.save('sxytfin',Sbxy)
        np.save('vytfin',vy_out)
        np.save('vxtfin',vx_out)
        np.save('Dfin',Unewx)
        np.save('Dfiny',Unewy)
    
    if n % 500 == 0:
        print('paso', n, 'de', nt, ' t =', round(t,3), ' tiempo transcurrido:', round(time.time()-t_inicio,1), 's')
    
    #Se intercambian los arreglos (en vez de hacer vx = vx_out, que los vuelve el
    #mismo arreglo y hace que la CPML reciba velocidades ya actualizadas).
    vx, vx_out = vx_out, vx
    vy, vy_out = vy_out, vy
    Sxx, Sxxn = Sxxn, Sxx
    Syy, Syyn = Syyn, Syy
    Sxy, Sxyn = Sxyn, Sxy
    mem_dvx_dx, mem_dvx_dx_out = mem_dvx_dx_out, mem_dvx_dx
    mem_dvy_dx, mem_dvy_dx_out = mem_dvy_dx_out, mem_dvy_dx
    mem_dvx_dy, mem_dvx_dy_out = mem_dvx_dy_out, mem_dvx_dy
    mem_dvy_dy, mem_dvy_dy_out = mem_dvy_dy_out, mem_dvy_dy
    mem_dsigmaxx_dx, mem_dsigmaxx_dx_out = mem_dsigmaxx_dx_out, mem_dsigmaxx_dx
    mem_dsigmaxy_dx, mem_dsigmaxy_dx_out = mem_dsigmaxy_dx_out, mem_dsigmaxy_dx
    mem_dsigmaxy_dy, mem_dsigmaxy_dy_out = mem_dsigmaxy_dy_out, mem_dsigmaxy_dy
    mem_dsigmayy_dy, mem_dsigmayy_dy_out = mem_dsigmayy_dy_out, mem_dsigmayy_dy

print('Simulación terminada en', round(time.time()-t_inicio,1), 's')

e1x, e1y = registro[0,:,0], registro[0,:,1]
e2x, e2y = registro[1,:,0], registro[1,:,1]
e3x, e3y = registro[2,:,0], registro[2,:,1]
e4x, e4y = registro[3,:,0], registro[3,:,1]
e5x, e5y = registro[4,:,0], registro[4,:,1]
e6x, e6y = registro[5,:,0], registro[5,:,1]
e7x, e7y = registro[6,:,0], registro[6,:,1]
e8x, e8y = registro[7,:,0], registro[7,:,1]
e9x, e9y = registro[8,:,0], registro[8,:,1]
fax, fay = registro[9,:,0], registro[9,:,1]

np.save('estacion1vx',e1x)
np.save('estacion1vy',e1y)

np.save('estacion2vx',e2x)
np.save('estacion2vy',e2y)


np.save('estacion3vx',e3x)
np.save('estacion3vy',e3y)


np.save('estacion4vx',e4x)
np.save('estacion4vy',e4y)

np.save('estacion5vx',e5x)
np.save('estacion5vy',e5y)

np.save('estacion6vx',e6x)
np.save('estacion6vy',e6y)


np.save('estacion7vx',e7x)
np.save('estacion7vy',e7y)


np.save('estacion8vx',e8x)
np.save('estacion8vy',e8y)

np.save('estacion9vx',e9x)
np.save('estacion9vy',e9y)

np.save('favx',fax)
np.save('favy',fay)
