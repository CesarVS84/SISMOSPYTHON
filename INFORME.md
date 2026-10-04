# Informe de revisión

Contenido del repositorio:

* **Carpeta raíz**: tus archivos revisados y corregidos, con los mismos nombres y la
  misma estructura (`CF`, `CBA`, `CBACPML2`, `CBO`, `CBVO`, `CPML`, `CPMLD`,
  `CPMLNN`, `TOP`, `prueba`).
* **`codigos_sugeridos/`**: la misma estructura, pero autocontenida y rápida
  (incluye un `TOP.py`/`prueba.py` sintéticos porque `snif.npy` no estaba), con la
  fuente limitada a 8 Hz.
* `tests/` y `codigos_sugeridos/tests/`: pruebas. Se ejecutan por separado, cada una desde
  su carpeta (`python -m pytest tests`): ambas carpetas tienen módulos con el mismo nombre.

Todo lo que se afirma como "medido" se ejecutó en este entorno (4 hilos, no el M4).

## 1. Errores encontrados en el código original

### Programa principal `CPML.py` (los más graves)

1. **Arreglos globales congelados por Numba.** `Sbxxdx`, `vxdx`, etc. son funciones
   `@njit` que leen arreglos *globales*; Numba los copia al compilar. Comprobado con
   un ejemplo mínimo: tras cambiar el arreglo en Python, la función sigue viendo el
   valor del primer paso. Es decir, la velocidad y el esfuerzo interiores se
   calculaban siempre con los campos del primer paso de tiempo.
2. **`vx = vx_out` (y lo mismo para esfuerzos y memorias)** hace que ambos nombres sean
   el mismo arreglo; `cpml_vel`/`cpml_stress` recibían entonces valores ya
   actualizados y la capa sumaba dos incrementos.
3. **Unidades**: `aa`, `b`, `c`, `d` ya estaban divididos por `dx`, pero `cpml_vel` y
   `cpml_stress` vuelven a dividir al derivar: en la capa el material efectivo era
   `dx` (50) veces más blando.
4. `cboveljih(..., vx_out, vy_out, vx_out, vy_out, ...)` lee y escribe el mismo
   arreglo desde varios hilos (resultado según el orden de los hilos).
5. Las instantáneas se guardaban con `if t == 0.1` (comparación exacta de decimales),
   `np.append` dentro del bucle de tiempo (cuadrático) y bucles de Python sobre
   250 000 nodos varias veces por paso.
6. `cp = 3300` en la capa, distinto de `vp` (5669 m/s); `dt` distinto entre `CF.py`
   (0.001) y `CPML.py` (0.002).

### Módulos CPML (`CPMLD`, `CPMLNN`, `CBACPML2`, `CBA`)

* `d0` usaba `log10`; la fórmula es con logaritmo **natural** (d0 sale 2.3 veces menor).
* Perfiles **asimétricos**: capa derecha/inferior desfasada un nodo, perfiles de medias
  celdas con `espesor - 1` (metros menos uno), `nx` en lugar de `ny`, `thick_pml_x1`
  en la capa 2, `b_y` en lugar de `b_y_half`. Ahora son simétricos (comprobado).
* `CBACPML2.cpml_vel`: la cadena `if/elif` está rota, así que las esquinas se
  actualizaban dos veces (con la derivada ya modificada). `aa` se indexaba con
  `(nx+1)`.
* `CPMLNN.cpml_stress` no tenía `@njit`; la capa derecha de `cpml_vel` usaba
  `pml_points_x1`; `init_memory_*` tenía índices cruzados.
* `CBA.damp_profiles` estaba vacía.
* `init_memory_stress` usaba `dx` en derivadas en `y`.
* `set_num_threads(4)` falla si hay menos de 4 hilos; `THREADING_LAYER='threadsafe'`
  falla si no hay TBB/OpenMP.

Los kernels nuevos recorren la malla una sola vez en paralelo. Se compararon con los
originales sobre datos aleatorios: coinciden a 1e-15 en todo salvo en lo corregido
a propósito.

### `CBO.py` / `CBVO.py` (Jih)

* Caso 1: `vy_ij_rot` sumaba (`+`) donde las demás fórmulas restan.
* Caso 4: la rotación inversa de `vy` tenía el signo de `cos` cambiado.
* Caso 5: `m == 1` con decimales (nunca se cumple, `tan 45°` no es exactamente 1) y en
  esa rama faltaba el factor `1-2(vs/vp)²`.
* Los bucles iban hasta `nx+1`/`ny+1` y escribían fuera del arreglo
  (corrupción de memoria). Divisiones por cero cuando el ángulo es 0.
* `CBVO` (guvectorize) no podía funcionar y tenía `vs=200, vp=4500` escritos a mano.
  Ahora delega en `CBO`.
* Verificado: un campo de velocidad uniforme se conserva en los 6 casos.
* **No verificado**: que las fórmulas coincidan con Jih (1988); no tengo el artículo.
  Una prueba con un campo lineal que cumple la superficie libre exacta **no** se
  reprodujo, así que no puedo afirmar que sean correctas.

### `TOP.py` y `prueba.py` (los que enviaste después)

* `prueba.py`: la clasificación se ejecutaba en cada fila `k` aunque no se hubieran
  encontrado los tres nodos, usando `tan_left`/`tan_right` de la iteración anterior
  (sin inicializar en la primera) y escribiendo en la columna 0. Con un perfil
  sintético el original marcó 25 nodos de más como caso 4 donde no aplica ningún caso.
* `prueba.py` termina con `boundary_flag_velocity[::-1]` y `elevation[::-1]`: esto
  refleja el campo en `i` y en `j` (giro de 180°) mientras `too` no se invierte.
  Las banderas se calculan sobre los nodos de `too`, así que tras invertir **no queda
  ninguna sobre la superficie (0 de 500 nodos)** y las condiciones de Jih no se
  aplicaban. Sin invertir y activadas, la simulación explota (ver abajo).
* `TOP.py`: bucles triples de Python (vectorizados, mismos resultados), `np.reshape`
  que exigía exactamente `nx` nodos de superficie, `G` posiblemente sin definir, código
  muerto. Los arreglos exportados son idénticos al original (comprobado).
* Observación sin cambiar: las celdas (`Lam`, `Mu`) se marcan por el nodo de su esquina
  inferior, así que la fila de celdas sobre la superficie cuenta como roca. Efecto
  medido despreciable.
* `CBO` se leía con `domain_flag[i + nx*j]`; el tamaño real de `too` es
  `(nx+1)*(ny+1)`, así que se mantiene `i + (nx+1)*j`.

## 2. Hallazgos de física (medidos)

* **Estabilidad (CFL)**: este esquema es estable hasta `vp·dt/dx = 1` (1.0 estable,
  1.1 inestable). El criterio de `CF.py` era correcto.
* **Absorción de la CPML**: con una fuente suave la CPML corregida llega a **−44 dB**
  (−54 dB con `Rcoef = 1e-5`) frente a −6.5 dB de una pared rígida.
* **Rebotes que parecen de la CPML y vienen de la fuente**: la capa corregida absorbe
  bien (−66 dB con una fuente suave), pero con tu fuente de dos nodos (50 en `(i,j)`, 90
  en `(i+1,j+1)`, con la velocidad impuesta solo en esos nodos) el eco medido es de
  −23 dB y se ven rebotes por todo el dominio. Causas: (1) tras el pulso los nodos
  quedan fijos en velocidad cero, un obstáculo rígido que dispersa; (2) una fuente
  puntual excita ondas de 2 a 4 nodos de largo, incluido el modo de tablero
  `(-1)^(i+j)` del esquema (derivada nula: no se propaga ni se absorbe), que ninguna
  capa absorbe. No depende de `k_max`, `f0`, `Rcoef`, la potencia del perfil ni del
  espesor. Solución: repartir la misma velocidad impuesta en una gaussiana (`sigma`
  1 a 2 nodos); con `sigma = 1.5` el eco baja a −65 dB y la energía residual en el
  núcleo a los 10 s pasa de 1.2·10⁻² a 7·10⁻⁶ del pico. Está activada por defecto
  (`CF.FUENTE_SIGMA` en `codigos_sugeridos`, `FUENTE_SIGMA` en `CPML.py` de la raíz;
  `None` restituye la fuente original). Una variante con fuerzas de cuerpo se descartó:
  irradia una forma de onda distinta de la del original.
* **Amortiguamiento**: `vel_damping = 5e-3` en la capa y `damp = 1e-3` fuera
  introducen una discontinuidad y empeoran la CPML ~9 dB. En `damp = 1e-3` por paso el
  campo se atenúa como `e^(-0.5 t)` con `dt = 0.002`, o sea 1/1800 a los 15 s.
* **Superficie libre**: con `aa = 1/rho` en todos los nodos la onda de Rayleigh viaja a
  0.872·vs (teórico 0.919). Con masa efectiva por nodo, `aa = 4/(rho·n_celdas)`, a
  0.911·vs. Está en `CPML.py` raíz (`MODELO_CONSISTENTE = True`) y en `TOP.py` sugerido.
* **Jih activo en el volcán sintético**: cada caso por separado cambia los resultados
  (el caso 3 multiplica por 5 la amplitud en una estación) y todos juntos hacen crecer
  la amplitud ~10⁶ veces en 1000 pasos. Con el reflejo en `j` (el kernel supone el
  sólido hacia `j` creciente, tu modelo tiene `j` hacia arriba) se evita el uso de
  celdas de aire, pero sigue siendo inestable. Por eso queda **desactivado** en ambas
  carpetas.

## 3. Velocidad

`codigos_sugeridos/CPML.py`: 3.3 ms por paso con 4 hilos (3750 pasos ≈ 12 s), medido
sin contar la compilación. Con `dt = 0.004` (Courant 0.45, mitad de pasos que `0.002`).
Del original solo pude reproducir 2 de sus bucles de Python: ya cuestan 0.43 s por
paso (7500 pasos ≈ 0.9 h como mínimo). **No pude medir el M4**: usa 10 hilos
(`CF.NUM_HILOS`); con 4 núcleos de rendimiento y 6 de eficiencia el reparto en partes
iguales no escala linealmente.

## 4. Fuente de 8 Hz

`codigos_sugeridos/fuente.py`: misma forma
`w(exp(-a1 t) - exp(-a2 t))` y mismo máximo (9.245); solo cambia la escala de tiempo.
"Frecuencia máxima" = la frecuencia a la que el espectro cae `CF.NIVEL_FUENTE_DB`
respecto de su máximo. Con el valor por defecto **−20 dB a 8 Hz** (`h = 3.351`), comprobado
con FFT. Con −40 dB (versión anterior, `h = 1.01`) el pulso quedaba tan largo y suave que
las ondas P y S se fundían y los sismogramas se veían planos; con −20 dB P y S se separan
y aparece coda. Ajustable en `CF.FMAX_FUENTE` y `CF.NIVEL_FUENTE_DB` (−40, −30, −20 y
−12 medidos): menos negativo = pulso más corto y sismogramas más "sísmicos", pero más
energía sobre 8 Hz (más dispersión numérica). Con 8 Hz la malla da 8.2 puntos por
longitud de onda S.

**Fuente tipo sismo (por defecto).** Un solo pulso no es un sismo: el sismograma muestra
apenas el pulso pasando y rebotando en la topografía (en un medio homogéneo no hay coda).
Con `CF.TIPO_FUENTE = 'sismo'` (`fuente.sismo`) la fuente es una ruptura de duración finita:
un evento principal en t = 0 seguido de `CF.SISMO_N` = 80 subeventos en `CF.SISMO_DURACION`
= 5 s, con amplitud que decae como `exp(-t/SISMO_TAU)`, signo aleatorio (semilla fija) y
cada uno con la forma del pulso original. Está normalizada a la misma amplitud máxima
(9.245) y su contenido a 8 Hz es ≤ −20 dB (medido −27 dB), así que el límite de 8 Hz se
mantiene. Resultado: llegada impulsiva, tren de ondas de unos 5 s y decaimiento
(`'pulso'` recupera la fuente anterior). Con esto no se simula coda por dispersión en
heterogeneidades; si la quieres, habría que añadir variaciones aleatorias de velocidad
al modelo.

Animaciones: las estaciones cambian de color a la llegada teórica de la onda P
(`r/vp`, naranja) y de la S (`r/vs`, magenta), y esas llegadas se marcan en los
sismogramas. `animacion.py --ventana-sismograma S` limita la ventana de tiempo mostrada.

## 5. Decisiones que necesito de ti

1. ¿Qué hace realmente `prueba.py[::-1]`? Si tu `snif.npy` ya está en otro marco, dime
   cómo está definido.
2. ¿Quieres seguir con Jih? Habría que validar las fórmulas contra el artículo y
   decidir los índices de los casos 4 y 5 (`prueba.py` guarda en la columna `c` el
   ángulo del tramo `c → c+1`; los casos 3 y 6 de `CBO` ya lo usan así, el 4 y el 5 no).
3. Umbral entre los casos 1 y 2: el código usa `tan >= 2`, los comentarios dicen 1.
4. `cp`: ¿la velocidad P máxima de tu modelo?

## 6. Visualización (`codigos_sugeridos/`)

Todo se ejecuta después de `python CPML.py` y lee la carpeta `salida/`. Requiere
`matplotlib` (y `ffmpeg` para los MP4); no usa `scipy`.

**Datos guardados.** Con `CF.SNAPSHOT_CADA = 0.01` el programa guarda vx y vy cada 0.01 s
mientras corre, en `salida/campo_vx.npy` y `salida/campo_vy.npy` (forma
`(1501, 500, 500)`, simple precisión, 1.5 GB cada uno) y `salida/tiempos.npy`. Para eso
`dt` pasó a 0.005 (Courant 0.57; 0.01 s son 2 pasos). Las estaciones se registran en cada
paso. La simulación completa de 15 s, escribiendo los 3 GB, tomó 43 s con 4 hilos.
Para menos disco, suba `SNAPSHOT_CADA` (con 0.05 s son 0.6 GB) o ponga `None`.

| Script | Qué genera |
|---|---|
| `graficas.py --t 3.5` | mapa de vx, vy y \|v\| en ese tiempo, con topografía, aire, capa CPML y estaciones |
| `animacion.py` | tres videos separados (`onda_vx`, `onda_vy`, `onda_modulo`): mapa a la izquierda y, a la derecha, el sismograma de la estación de superficie más cercana a la fuente y el de la fuente (pulso), con cursor |
| `sismograma_espectrograma.py` | una figura por estación (sismograma vx, vy y espectrograma) y `resumen_estaciones.png` con todas |
| `animacion_estacion.py` | un video por estación y uno para la fuente, con el sismograma y el espectrograma construyéndose en el tiempo y la posición de la estación |
| `espectrograma.py` | rejilla de espectrogramas de todas las estaciones y sus espectros de amplitud |
| `visual_comun.py` | funciones compartidas (lectura, STFT, dibujo del modelo) |

`animacion.py` usa por defecto una instantánea de cada 5 (0.05 s) a 20 cuadros por
segundo, es decir, el video va en tiempo real (15 s). En el volcán sintético E3 y E4 están
a igual distancia de la fuente; se elige la primera. Con la topografía real se calcula con
las estaciones de `TOP.estaciones` que están sobre la superficie.
