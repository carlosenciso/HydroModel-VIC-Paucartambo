# Sobre este proyecto — VIC Cuenca Paucartambo

## Qué hace este proyecto

Este proyecto implementa una cadena completa de modelado hidrológico para la **cuenca del Río Paucartambo** (Pasco, Perú), desde la descarga de datos satelitales hasta la generación de caudales en la **Central Hidroeléctrica Yuncan** y la visualización de la dinámica de la cuenca.

La cadena incluye:

1. **Descarga automatizada de datos** desde Google Earth Engine (GEE)
2. **Preparación de parámetros** del suelo y la vegetación a partir de datos de teledetección
3. **Simulación hidrológica** con el modelo VIC 5 Image Driver
4. **Enrutamiento de caudal** desde cada celda de la grilla hasta el outlet en Yuncan
5. **Visualización**: series temporales, balance hídrico, curva de duración de caudales y animaciones

---

## El modelo VIC

**VIC (Variable Infiltration Capacity)** es un modelo hidrológico macroescala de balance de agua y energía desarrollado originalmente por Xu Liang en la Universidad de Washington (1994). Es uno de los modelos hidrológicos más usados a nivel global para estudios de clima, recursos hídricos y cambio climático.

### ¿Por qué VIC Image Driver?

VIC tiene tres modos de operación:
- **Classic Driver**: procesa celdas de forma secuencial, I/O en ASCII
- **Image Driver** (`vic_image.exe`): procesa toda la grilla en paralelo con MPI, I/O en NetCDF4 — más eficiente para cuencas grandes
- **CESM Driver**: acoplado con el modelo climático CESM

Se usó el **Image Driver** porque permite trabajar con grillas de alta resolución (~1 km) de manera eficiente y genera salidas en NetCDF directamente compatibles con las herramientas de análisis (xarray, Python).

### Física del modelo

VIC resuelve el balance hídrico en cada celda de la grilla con estas componentes:

```
Precipitación = Evapotranspiración + Escorrentía superficial + Flujo base + ΔAlmacenamiento
```

- **Infiltración**: esquema de Capacidad Variable de Infiltración (curva de Arno)
- **Evapotranspiración**: Penman-Monteith con resistencia de canopy por tipo de vegetación
- **Flujo base**: esquema no-lineal de Francini-Pacciani (ARNO/VIC)
- **Nieve**: balance de energía de la manto de nieve (importante en zonas > 4000 msnm)
- **Suelo**: 3 capas con propiedades hidráulicas derivadas de textura

---

## La cuenca de Paucartambo

| Característica | Valor |
|---|---|
| Localización | Pasco, Peru |
| Área aproximada | ~3,400 km² |
| Elevación | 200 m (outlet) – 4,800 m (cabeceras) |
| Outlet | Central Hidroeléctrica Yuncan (-10.745°S, -75.586°W) |
| Tipo de cuenca | Andino-amazónica |
| Régimen hidrológico | Tropical estacional (lluvias Nov–Abr) |

La cuenca presenta un gradiente altitudinal pronunciado con tres zonas ecológicas:
- **Selva alta / Yungas** (< 1500 m): bosque tropical húmedo, precipitación alta
- **Ceja de selva / Nublada** (1500–3500 m): bosque de neblina, alta humedad
- **Puna** (> 3500 m): pastizales de ichu, precipitación estacional

---

## Cómo se construyó el proyecto

### 1. Obtención de datos — Google Earth Engine

Todos los datos se obtienen automáticamente desde GEE sin necesidad de descarga manual. Se usa autenticación por service account (token en variable de entorno `EE_SERVICE_ACCOUNT_JSON_B64`) que elimina el login interactivo.

La librería **wxee** permite descargar colecciones de imágenes GEE directamente como `xarray.Dataset`, lo que facilita la integración con el resto del pipeline Python.

**Datos estáticos (descargados una sola vez):**

| Dato | Fuente | Uso |
|---|---|---|
| DEM 30m | SRTM (USGS) | Delineación de cuenca, elevación por celda, routing |
| Textura del suelo | ISRIC SoilGrids 250m v2.0 | Parámetros hidráulicos VIC |
| Cobertura vegetal | ESA WorldCover 2021 (10m) | Tipo y fracción de vegetación |
| LAI mensual | MODIS MOD15A2H (500m) | Índice de área foliar por mes |

**Datos temporales (por año, 2008–2020):**

| Dato | Fuente | Variables |
|---|---|---|
| Precipitación | CHIRPS Daily (~5 km) | mm/día |
| Temperatura | ERA5-Land Daily (~11 km) | T2m máx, T2m mín |
| Viento | ERA5-Land Daily | u10, v10 → velocidad |
| Radiación | ERA5-Land Daily | Solar (J/m²→W/m²), Longwave |
| Presión | ERA5-Land Daily | Presión superficial (Pa→kPa) |
| Presión de vapor | ERA5-Land Daily | Calculada desde temperatura de rocío |

### 2. Preparación del dominio

El dominio VIC es una grilla regular de `0.009°` (~1 km) que cubre el bounding box de la cuenca. La máscara de cuenca se genera con **pysheds** delineando la cuenca desde el punto de salida (Yuncan) usando el algoritmo D8 sobre el DEM SRTM.

El archivo `domain.nc` contiene:
- `mask`: 1 = celda activa, 0 = fuera de la cuenca
- `area`: área de cada celda en m² (varía con la latitud)
- `frac`: fracción activa de la celda (1.0 para grilla regular)

### 3. Derivación de parámetros del suelo

Los parámetros hidráulicos de VIC se derivan de la textura del suelo (arena, arcilla, limo) usando las **pedotransfer functions de Cosby et al. (1984)**:

| Parámetro VIC | Fórmula |
|---|---|
| Conductividad hidráulica saturada (Ksat) | `7.06×10⁻⁶ × exp(3.52×sand − 2.54×clay)` m/s |
| Exponente b de Campbell | `3.10 + 15.7×clay − 0.3×sand` |
| Presión de burbuja (ψₛ) | `0.01 × exp(1.54 − 0.0095×sand + 0.0063×silt)` m |
| Porosidad (θₛ) | `0.489 − 0.00126×sand` |
| Humedad en punto de marchitez | `θₛ × (ψₛ/150)^(1/b)` |
| Humedad en capacidad de campo | `θₛ × (ψₛ/3.36)^(1/b)` |

Los parámetros de infiltración (B, Ds, Ws) se asignan con valores típicos para cuencas andino-amazónicas húmedas y pueden calibrarse en pasos posteriores.

### 4. Derivación de parámetros de vegetación

La cobertura vegetal de **ESA WorldCover** (11 clases) se mapea a 9 clases de vegetación del VIC adaptadas a la zona andino-amazónica:

| Clase VIC | Descripción | Elevación típica |
|---|---|---|
| 1 | Bosque tropical húmedo | < 1500 m |
| 2 | Bosque de neblina / Yungas | 1500–3000 m |
| 3 | Pastizal Puna / Ichu | 3000–4200 m |
| 4 | Agricultura / Cultivos | variable |
| 5 | Arbustos andinos | 2000–3500 m |
| 6 | Páramo / Jalca | > 4200 m |
| 7 | Suelo desnudo / Roca | variable |
| 8 | Nieve / Glaciar | > 4800 m |
| 9 | Cuerpos de agua | variable |

El LAI mensual se extrae de la climatología MODIS (2010–2020).

### 5. Preparación de forzantes

Los forzantes se procesan con correcciones físicas antes de ingresarlos a VIC:

- **Corrección topográfica de temperatura**: aplica gradiente adiabático `-6.5°C/km` para corregir las temperaturas ERA5 (resolución ~11 km) al relieve de 1 km del DEM
- **Presión atmosférica por elevación**: `P = 101.3 × exp(−z/8500)` para celdas sin datos ERA5
- **Presión de vapor**: calculada desde temperatura de punto de rocío ERA5 usando la ecuación de Magnus
- **Radiación de onda larga**: estimada desde temperatura y nubosidad cuando no hay datos directos
- **Suavizado espacial gaussiano**: ERA5 (~11 km) suavizado antes de interpolar a 1 km para evitar artefactos

### 6. Simulación VIC

VIC se ejecuta en modo **balance de agua** (`FULL_ENERGY=FALSE`) a paso diario con 24 sub-pasos internos para el balance de nieve y escorrentía. El período incluye 2 años de spin-up (2008–2009) para que los estados internos del suelo converjan.

El ejecutable `vic_image.exe` usa **MPI** para paralelizar el procesamiento por filas de la grilla, lo que permite simular ~4000 celdas activas de 1 km en tiempo razonable.

### 7. Enrutamiento

VIC genera escorrentía y flujo base por celda (mm/día) pero no caudal en el cauce. El **enrutamiento lineal por tiempo de viaje** convierte estas salidas en caudal en Yuncan:

1. Calcular la distancia de cada celda al outlet (Yuncan) usando transformada de distancia sobre la máscara
2. Convertir distancia a tiempo de viaje: `t = d × tortuosidad / (v × 86400)` donde `v = 1.5 m/s` y `tortuosidad = 1.5`
3. Convolucionar la escorrentía de cada celda con un retardo discreto

También está disponible **RVIC** (Routing VIC) como método de enrutamiento alternativo, más riguroso, que usa hidrógrafas unitarias de convolución.

### 8. Tiempo de viaje a Yuncan (opcional)

Se puede calcular el tiempo aproximado de desplazamiento del agua desde cualquier punto (embalse, tributario) hasta la Central Yuncan, útil para análisis de pronóstico de caudal y gestión hídrica en tiempo real.

---

## Stack tecnológico

| Componente | Tecnología | Justificación |
|---|---|---|
| Modelo hidrológico | VIC 5.0.1 Image Driver | Estándar científico, MPI, NetCDF4 |
| Compilación | GCC 11 + MPI | Disponible en Ubuntu 22.04 |
| Datos satelitales | Google Earth Engine | Acceso a petabytes de datos geoespaciales |
| Descarga GEE | wxee + earthengine-api | Descarga directa como xarray |
| Procesamiento numérico | xarray + numpy + scipy | Estándar en ciencias atmosféricas |
| Geoespacial | pysheds, rasterio, geopandas | Delineación de cuenca y operaciones ráster |
| Visualización | matplotlib + cartopy + cmocean | Mapas con proyección geográfica |
| Animaciones | matplotlib.animation + imageio | GIF y MP4 |
| Contenerización | Docker + docker-compose | Reproducibilidad total |
| Análisis interactivo | JupyterLab | Exploración post-simulación |

---

## Limitaciones y posibles mejoras

**Parámetros de suelo sin calibración**
Los parámetros B (infilt), Ds, Ws se asignan con valores típicos. Una calibración con datos observados de caudal (SENAMHI) mejoraría significativamente los resultados.

**Resolución de ERA5 (~11 km)**
Los forzantes de temperatura y radiación vienen de ERA5 a 0.1°. Para la heterogeneidad topográfica de los Andes, sería ideal usar datos de estaciones para downscaling estadístico.

**Routing simple**
El enrutamiento lineal es una aproximación. Implementar RVIC con parámetros calibrados mejoraría la representación de la onda de crecida.

**Glaciares**
VIC puede representar glaciares con el módulo `FROZEN_SOIL=TRUE` y bandas de elevación (`SNOW_BAND`). Activar estos módulos mejoraría la simulación de caudal en la estación seca.

**Validación**
Comparar con registros históricos de caudal del SENAMHI en estaciones de la cuenca permitiría evaluar el desempeño del modelo (NSE, KGE, PBIAS).

---

## Referencias

- Liang, X., et al. (1994). *A simple hydrologically based model of land surface water and energy fluxes for general circulation models*. Journal of Geophysical Research.
- Hamman, J.J., et al. (2018). *The Variable Infiltration Capacity model version 5 (VIC-5): infrastructure improvements for new applications and reproducibility*. Geoscientific Model Development.
- Cosby, B.J., et al. (1984). *A statistical exploration of the relationships of soil moisture characteristics to the physical properties of soils*. Water Resources Research.
- Funk, C., et al. (2015). *The climate hazards infrared precipitation with stations — a new environmental record for monitoring extremes*. Scientific Data (CHIRPS).
- Muñoz-Sabater, J., et al. (2021). *ERA5-Land: a state-of-the-art global reanalysis dataset for land applications*. Earth System Science Data.
