# Ejemplos de salida

Generados con `codigos_sugeridos` sobre el **volcán sintético** (no con tu `snif.npy`): simulación
completa de 15 s, campos guardados cada 0.01 s.

* `onda_vx.mp4`, `onda_vy.mp4`, `onda_modulo.mp4`: animación del campo con el sismograma de la estación
  de superficie más cercana a la fuente (E3) y el de la fuente.
* `animaciones_estaciones/estacion_*.mp4`: sismograma y espectrograma por estación (E1–E9) y fuente (F).
* `figuras/estacion_*.png`, `figuras/resumen_estaciones.png`: sismograma + espectrograma.

Para generarlos con sus propios datos: `python CPML.py`, luego `python animacion.py`,
`python animacion_estacion.py` y `python sismograma_espectrograma.py`.
