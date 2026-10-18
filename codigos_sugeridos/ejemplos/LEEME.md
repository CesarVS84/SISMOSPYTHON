# Ejemplos de salida

Generados con `codigos_sugeridos` sobre **tu `snif.npy`** (con tus estaciones y tu fuente): simulación
completa de 15 s, campos guardados cada 0.01 s.

* `onda_vx.mp4`, `onda_vy.mp4`, `onda_modulo.mp4`: animación del campo con el sismograma de la estación
  de superficie más cercana a la fuente (E3) y el del nodo junto a la fuente. Fuente: doble cupla (tensor de momento) y roca con heterogeneidad del 20 % (coda), ver INFORME.md §4.
* `animaciones_estaciones/estacion_*.mp4`: sismograma y espectrograma por estación (E1–E9) y fuente (F).
* `validacion_analitica.png`: simulación contra la solución analítica de una doble cupla (8 receptores).
* `figuras/estacion_*.png`, `figuras/resumen_estaciones.png`: sismograma + espectrograma.

Para generarlos con sus propios datos: `python CPML.py`, luego `python animacion.py`,
`python animacion_estacion.py` y `python sismograma_espectrograma.py`.
