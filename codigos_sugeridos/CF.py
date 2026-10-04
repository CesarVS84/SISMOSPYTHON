# -*- coding: utf-8 -*-
"""
CF.py: parámetros de la simulación (único lugar donde se definen).

Todos los demás módulos (CBA, CPMLD, CBO, TOP, prueba, fuente, CPML) leen sus
parámetros de aquí. En los archivos originales la malla, dt, la velocidad P de
la capa absorbente, etc. estaban repetidos y con valores distintos en CF.py,
CPML.py y CPMLD.py.
"""

import numpy as np

# ------------------------------------------------------------------ Dominio
#Dominio espacial
Lx = Ly = 25000.
#Dominio temporal
tfin = 15.
#Esfuerzos inciales (se aplican en una celda si USAR_ESFUERZO_INICIAL)
six = 60.
siy = 80.
sixy = 50.
USAR_ESFUERZO_INICIAL = False

# ------------------------------------------------------------ Discretización
#Espacial
nx = ny = 500
dx = Lx/nx
dy = Ly/ny
# Nodos de velocidad en x = i*dx, y = j*dy. j crece hacia ARRIBA: j = 0 es la
# base del modelo y el aire queda en los valores grandes de j (como en TOP.py).
x = np.arange(nx)*dx
y = np.arange(ny)*dy
X, Y = np.meshgrid(x, y, indexing='ij')
xx = np.linspace(0, Lx, nx+1)
yy = np.linspace(0, Ly, ny+1)

#Temporal
# El esquema es estable para vp*dt/min(dx,dy) <= 1 (verificado numéricamente).
# Con una fuente de hasta 8 Hz basta un paso mucho mayor que 0.001: dt = 0.005
# (número de Courant 0.57) usa 2.5 veces menos pasos que el programa original (dt = 0.002)
# y permite guardar el campo cada 0.01 s (= 2 pasos).
nt = 3000
T = np.linspace(0, tfin, nt+1)
dt = tfin/nt

# ----------------------------------------------------------------- Material
Rho = 2800.
Lambda = 3e10
Mu = 3e10
vp = np.sqrt((Lambda+2*Mu)/Rho)
vs = np.sqrt(Mu/Rho)

# ------------------------------------------------- Criterio de estabilidad
dtestable = min(dx, dy)/vp
courant = vp*dt/min(dx, dy)

# ------------------------------------------------------------------- Fuente
#Frecuencia máxima de la fuente. La forma temporal es la misma del programa
#original (diferencia de dos exponenciales); ver fuente.py.
FMAX_FUENTE = 8.0
#Nivel (dB, respecto del máximo del espectro) al que cae el espectro a FMAX_FUENTE.
#-20 dB = a 8 Hz queda el 10 % de la amplitud (1 % de la energía). Con -40 dB el pulso
#es tan lento (~2 s) que las ondas P y S se funden en una sola onda suave y el
#sismograma se ve plano, sin coda; con -20 dB se separan y aparece la coda.
#Opciones medidas: -40 (muy suave), -30, -20 (por defecto), -12 (más contenido sobre 8 Hz).
NIVEL_FUENTE_DB = -20.0
#Ancho (desviación estándar, en nodos) de la gaussiana en que se reparte la velocidad
#impuesta en cada nodo de la fuente (F = 50 y F = 90). Con None se impone solo en los
#nodos F, como en el programa original; eso deja los nodos fijos en cero después del
#pulso (un obstáculo rígido) y excita ondas de 2 a 4 nodos de largo que la CPML no
#absorbe: el eco medido con la capa sube de -66 dB a -23 dB.
FUENTE_SIGMA = 1.5
#Ángulo de la dirección de apertura de la fuente respecto del eje x (grados)
thetag = 30.

# --------------------------------------------------------------- Absorbente
npow = 2.0
# Espesor (en nodos) de la capa en cada lado. En el lado del aire (j grande) es 0:
# allí el campo es nulo y la superficie libre no debe tener capa absorbente.
pml_points_x1 = 100       # izquierda
pml_points_x2 = 100       # derecha
pml_points_y1 = 100       # base del modelo (j pequeño)
pml_points_y2 = 0         # aire (j grande)
Rcoef = 0.001             # reflexión teórica de la capa
k_max_pml = 3.0
#Frecuencia que fija alpha_max = pi*f0 de la CPML. La fuente tiene casi toda su
#energía por debajo de ~2 Hz, y la CPML absorbe mal por debajo de f0.
f0 = 1.0
#Amortiguamiento viscoso por paso (v <- v*(1-damp)). Se usa el MISMO valor en
#toda la malla: un valor distinto dentro de la capa genera reflexiones.
damp = 1e-3
vel_damping = damp

# ---------------------------------------------------------------- Superficie
#Si es True se aplican además las condiciones de Jih (ver CBO.py). Con False
#la superficie libre se modela con aire de densidad y rigidez nulas (con masa
#efectiva por nodo: la onda de Rayleigh sale a menos del 1 % del valor teórico).
#EXPERIMENTAL: las fórmulas de Jih de CBO.py no son robustas; en las pruebas
#hechas con el volcán sintético, activadas todas juntas, la amplitud crece 10^6
#veces en 1000 pasos. Déjelo en False salvo que las esté depurando.
USAR_JIH = False

# ------------------------------------------------------------- Instantáneas
# Campos completos: se guardan vx y vy cada SNAPSHOT_CADA segundos en dos archivos,
# salida/campo_vx.npy y salida/campo_vy.npy, de forma (n_instantáneas, ny, nx) en simple
# precisión, junto con salida/tiempos.npy. Con 0.01 s y 15 s son 1501 instantáneas, 1.5 GB
# por campo (3 GB en total). Con 0.05 s serían 0.6 GB. None desactiva el guardado.
SNAPSHOT_CADA = 0.01
# Formato antiguo (un archivo por instantánea, nombres sxxt..., vxt..., cada 0.1 s)
GUARDAR_INSTANTANEAS_ANTIGUAS = False
GUARDAR_ESFUERZOS = True  # (solo formato antiguo) además de vx, vy se guardan Sbxx, Sbyy, Sbxy
SNAPSHOT_FLOAT32 = True   # (solo formato antiguo) guarda en simple precisión

# ------------------------------------------------------------------ Hilos
#MacBook Air M4: 10 hilos
NUM_HILOS = 10


def informe():
    """Imprime el resumen de la discretización y de los criterios."""
    print('dx:', dx, 'dy:', dy, 'dt:', dt, 'nt:', nt)
    if dt < dtestable:
        print('El dt cumple con el criterio')
    else:
        print('Debes escoger otro dt')
    print('el dt que cumple con el criterio corresponde a: ', dtestable,
          '\nmientras que el dt que utilizaremos es: ', dt,
          '(Courant = %.2f)' % courant)
    print('Velocidad onda P: ', vp)
    print('Velocidad onda S: ', vs)
    ppl = vs/(FMAX_FUENTE*max(dx, dy))
    print('Puntos por longitud de onda S a %.1f Hz: %.1f (recomendado >= 8)' % (FMAX_FUENTE, ppl))


if __name__ == '__main__':
    informe()
