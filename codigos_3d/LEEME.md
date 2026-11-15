# codigos_3d: topografía del volcán Osorno para la simulación 3-D

* `dem_osorno.py`: descarga las teselas SRTM de 1 arco-segundo (~30 m) S41W073 y S42W073
  (AWS Terrain Tiles, https://registry.opendata.aws/terrain-tiles/; SRTM, NASA/NGA) y las remuestrea
  a una malla regular de 50 m, de 25 x 25 km, centrada en la cima. `python dem_osorno.py --descargar`.
* `osorno_dem_50m.npz`: resultado (z[j, i] en metros, j hacia el norte, i hacia el este; dx, lat0, lon0).
* `osorno_dem.png`: mapa sombreado, vista 3-D y perfiles por la cima.

Cima: 41.104° S, 72.496° O, 2664 m en SRTM (la cota oficial es 2652 m). Cota del dominio: 53 a 2664 m.
Las zonas planas son los lagos Llanquihue (~51 m, oeste y noroeste) y Todos los Santos (~190 m, este);
SRTM da la altura de la superficie del agua. Pendiente máxima 69°, media 25° sobre los 1000 m.
