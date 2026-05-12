# VIC Hydrological Model — Cuenca Paucartambo, Pasco, Perú

Simulación hidrológica de la cuenca del Río Paucartambo usando el modelo **VIC 5 (Variable Infiltration Capacity)** con datos de **Google Earth Engine** (ERA5-Land + CHIRPS), ejecutado en **Docker**, con enrutamiento de caudal (RVIC) hasta la **Central Hidroeléctrica Yuncan**.

---

## Estado actual del proyecto

| Paso | Script | Estado | Período simulado |
|------|--------|--------|-----------------|
| 1. Descarga GEE | `01_download_gee.py` | ✅ Completado | 2014–2015 |
| 2. Dominio | `02_prepare_domain.py` | ✅ Completado | — |
| 3. Parámetros | `03_prepare_parameters.py` | ✅ Completado | — |
| 4. Forzantes | `04_prepare_forcings.py` | ✅ Completado | 2014–2015 |
| 5. VIC | `05_run_vic.py` | ✅ Completado | 2014–2015 |
| 6. Routing RVIC | `06_run_routing.py` | ✅ Completado | 2014–2015 |
| 7. Gráficos | `07_plot_timeseries.py` | ✅ Completado | — |
| 8. Animaciones | `08_create_animation.py` | ✅ Completado | — |
| 9. Red drenaje | `09_plot_routing_network.py` | ✅ Completado | — |

> **Próximo paso**: Ampliar período 2000–2025 con spin-up 2000–2002 (ver sección [Simulación período largo](#simulación-período-largo-recomendado)).

### Resultados actuales (2014–2015 — prueba, sin spin-up)

| Estadístico | Valor |
|-------------|-------|
| Caudal medio (RVIC) | 6.0 m³/s |
| Caudal máximo | 23.8 m³/s (ene 2014) |
| Caudal mínimo | ~1 m³/s |
| Q95 | ~18 m³/s |

> **Nota**: Los caudales de la prueba 2014–2015 están subestimados por falta de spin-up (warm-up). El modelo parte con suelos vacíos y necesita 2–3 años para equilibrarse. Enero 2014 (estación húmeda) tiene valores más representativos. Ver [Simulación período largo](#simulación-período-largo-recomendado) para la configuración correcta.

---

## Guía rápida

```bash
# 1. Configurar token GEE (necesario solo para descarga)
export EE_SERVICE_ACCOUNT_JSON_B64=$(cat \
  /home/cenciso/Documents/WORK/ENGIE/DashBoard/windShortTermForecast/key_b64.txt)

# 2. Construir imagen Docker (solo la primera vez)
cd /home/cenciso/Documents/WORK/ENGIE/2026/VIC_project
docker build -t vic-paucartambo -f docker/Dockerfile .

# 3. Correr pipeline completo
docker compose -f docker/docker-compose.yml run --rm vic bash run_all.sh
```

> **Nota**: usar `docker compose` (v2) sin guión, no `docker-compose`.

---

## Tabla de contenidos

1. [Requisitos](#requisitos)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Paso a paso detallado](#paso-a-paso-detallado)
4. [Simulación período largo (recomendado)](#simulación-período-largo-recomendado)
5. [Configuración de la cuenca](#configuración-de-la-cuenca)
6. [Salidas del modelo](#salidas-del-modelo)
7. [Notas técnicas de implementación](#notas-técnicas-de-implementación)
8. [Solución de problemas conocidos](#solución-de-problemas-conocidos)

---

## Requisitos

| Herramienta | Versión mínima | Notas |
|---|---|---|
| Docker | 24.0+ | Motor de contenedores |
| Docker Compose | V2 (plugin) | `docker compose`, sin guión |
| Espacio en disco | ~20 GB (prueba) / ~80 GB (2000–2025) | Imagen Docker + datos GEE |
| RAM | 8 GB mínimo | 16 GB recomendado |
| Internet | — | Para descarga GEE |

**Token GEE** (disponible en el sistema):
```bash
export EE_SERVICE_ACCOUNT_JSON_B64=$(cat \
  /home/cenciso/Documents/WORK/ENGIE/DashBoard/windShortTermForecast/key_b64.txt)
```

---

## Estructura del proyecto

```
VIC_project/
├── docker/
│   ├── Dockerfile              # VIC5 + Python stack (xarray, RVIC, cartopy, etc.)
│   └── docker-compose.yml      # Servicios: vic, pipeline, jupyter, visualize
├── config/
│   ├── basin_config.yml        # EDITAR AQUÍ: bbox, resolución, fechas, outlet
│   ├── global_params_template.txt  # Template parámetros VIC Image Driver
│   └── vic_veglib.txt          # Biblioteca de vegetación (9 clases)
├── scripts/
│   ├── utils/
│   │   ├── gee_auth.py         # Auth GEE sin login interactivo
│   │   └── vic_utils.py        # Pedotransfer functions, utilidades NetCDF
│   ├── 01_download_gee.py      # Descarga DEM, suelo, cobertura, ERA5, CHIRPS
│   ├── 02_prepare_domain.py    # Genera domain.nc (máscara de cuenca, D8 FDR)
│   ├── 03_prepare_parameters.py # Genera params.nc (suelo + vegetación)
│   ├── 04_prepare_forcings.py  # Convierte GEE → forzantes VIC NetCDF
│   ├── 05_run_vic.py           # Genera global_params.txt y ejecuta vic_image.exe
│   ├── 06_run_routing.py       # RVIC routing: escurrimiento → caudal en Yuncan
│   ├── 07_plot_timeseries.py   # Gráficos: caudal, balance hídrico, curva duración
│   ├── 08_create_animation.py  # Animaciones GIF/MP4 (variables VIC + caudal)
│   └── 09_plot_routing_network.py # Mapa de red de drenaje D8 + acumulación flujo
├── data/                       # GENERADO durante la ejecución (no subir a git)
│   ├── gee_raw/                # Datos crudos de GEE
│   ├── domain/                 # domain.nc, flow_direction.nc, DEM
│   ├── parameters/             # params.nc
│   ├── forcings/               # forcing_YYYY.nc (uno por año)
│   ├── outputs/                # Salidas VIC: fluxes_YYYY.nc
│   └── routing/                # RVIC outputs, streamflow_yuncan.nc/.csv
├── plots/                      # Gráficos PNG generados
├── animations/                 # Animaciones GIF/MP4
├── logs/                       # Logs de ejecución
├── notebooks/
│   └── analysis.ipynb          # Análisis interactivo post-simulación
├── requirements.txt
└── run_all.sh                  # Pipeline completo (todos los scripts en orden)
```

---

## Paso a paso detallado

### Paso 0 — Construir la imagen Docker

Solo es necesario la **primera vez** o cuando se modifique el Dockerfile.

```bash
cd /home/cenciso/Documents/WORK/ENGIE/2026/VIC_project
docker build -t vic-paucartambo -f docker/Dockerfile .
# Tiempo: ~15 minutos (compila VIC5 + instala dependencias Python)
```

Verificar que VIC quedó compilado:
```bash
docker run --rm vic-paucartambo vic_image.exe -v
# Debe mostrar: VIC Driver: Image, VIC Version: 5.0.1
```

---

### Paso 1 — Descargar datos de Google Earth Engine

```bash
export EE_SERVICE_ACCOUNT_JSON_B64=$(cat \
  /home/cenciso/Documents/WORK/ENGIE/DashBoard/windShortTermForecast/key_b64.txt)

docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/01_download_gee.py \
    --start 2000-01-01 \
    --end 2025-12-31
```

**Datasets descargados:**

| Dataset | Fuente GEE | Variable | Resolución nativa |
|---|---|---|---|
| DEM | `USGS/SRTMGL1_003` | Elevación | 30 m |
| Suelo | `ISRIC/SoilGrids250m/v2_0` | Arena, arcilla, limo, BD | 250 m |
| Cobertura | `ESA/WorldCover/v200` | Clase + fracciones | 10 m |
| LAI | `MODIS/006/MOD15A2H` | LAI climatología mensual | 500 m |
| Precipitación | `UCSB-CHG/CHIRPS/DAILY` | mm/día | ~5 km |
| T, Viento, Rad | `ECMWF/ERA5_LAND/DAILY_AGGR` | T2m, u10, v10, radiación | ~11 km |

> Los archivos ya descargados se saltan automáticamente (reanudable).
> Tiempo estimado: 30–120 minutos para 25 años.

---

### Paso 2 — Preparar el dominio VIC

```bash
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/02_prepare_domain.py
```

**Salidas en `data/domain/`:**
- `domain.nc` — Máscara de cuenca, área de celdas, fracciones, elevación
- `flow_direction.nc` — Dirección de flujo D8, acumulación, cuenca delineada, elevación

El script usa DEM SRTM para delinear la cuenca desde el outlet de Yuncan (-10.745°S, -75.586°W) con algoritmo Priority-Flood + D8.

---

### Paso 3 — Preparar parámetros del suelo y vegetación

```bash
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/03_prepare_parameters.py
```

**Salidas:** `data/parameters/params.nc`

- **Suelo**: infilt (B), Ds, Ds_max, Ws, Ksat, expt, bubble, quartz desde SoilGrids usando pedotransfer functions de Cosby et al. (1984)
- **Vegetación**: Cv (fracción), LAI mensual, albedo, root_depth/frac desde ESA WorldCover + MODIS LAI

---

### Paso 4 — Preparar forzantes meteorológicos

```bash
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/04_prepare_forcings.py \
    --start 2000-01-01 \
    --end 2025-12-31
```

**Salidas:** `data/forcings/forcing_YYYY.nc` (un archivo por año)

Variables: `PRCP`, `AIR_TEMP`, `TMAX`, `TMIN`, `WIND`, `SHORTWAVE`, `LONGWAVE`, `PRESSURE`, `VP`

---

### Paso 5 — Ejecutar el modelo VIC

```bash
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/05_run_vic.py \
    --start 2000-01-01 \
    --end 2025-12-31
```

**Salidas en `data/outputs/`:**
- `fluxes_YYYY.nc` — Variables hidrológicas diarias por año

---

### Paso 6 — Enrutamiento hidrológico (RVIC)

```bash
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/06_run_routing.py
```

El script ejecuta el pipeline RVIC completo:
1. Genera `flow_direction.nc` con D8 desde DEM
2. Crea la UH box y pour points
3. Corre `rvic parameters` (calcula hidrogramas unitarios para cada celda)
4. Corre `rvic convolution` (convoluciona escurrimiento VIC con UH)
5. Si RVIC falla, usa routing lineal de fallback

**Salidas en `data/routing/`:**
- `streamflow_yuncan.nc` — Caudal diario en Yuncan (m³/s)
- `streamflow_yuncan.csv` — Misma serie en CSV
- `flow_direction.nc` — Red D8 + acumulación de flujo

---

### Paso 7 — Gráficos y series temporales

```bash
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/07_plot_timeseries.py
```

**Gráficos en `plots/`:**
- `streamflow_timeseries.png` — Serie de caudal en Yuncan
- `water_balance.png` — Balance hídrico: P, ET, escorrentía, SWE
- `flow_duration_curve.png` — Curva de duración de caudales

---

### Paso 8 — Animaciones de dinámica hidrológica

```bash
# Animación de escorrentía total + caudal enrutado (RECOMENDADO)
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/08_create_animation.py \
    --streamflow \
    --freq 7D \
    --format gif \
    --fps 6

# Animación de variable VIC (humedad del suelo, precipitación, etc.)
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/08_create_animation.py \
    --variable OUT_SOIL_MOIST \
    --year 2015 \
    --freq 7D \
    --format gif

# Grid de snapshots mensuales (alternativa estática)
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/08_create_animation.py \
    --variable OUT_RUNOFF \
    --year 2015 \
    --snapshots

# Todas las variables disponibles
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/08_create_animation.py --all-vars --year 2015
```

**Variables disponibles:** `OUT_PREC`, `OUT_SOIL_MOIST`, `OUT_RUNOFF`, `OUT_BASEFLOW`, `OUT_EVAP`, `OUT_SWE`

**Modos de animación:**
- `--streamflow`: Mapa de escorrentía + timeseries de caudal enrutado con cursor + estadísticas en tiempo real
- `--variable VAR`: Mapa espacial de variable VIC con cursor temporal

---

### Paso 9 — Mapa de red de drenaje

```bash
# Mapa de red de ríos + acumulación de flujo (ambos en un solo comando)
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/09_plot_routing_network.py

# Con flechas de dirección de flujo (más lento pero más informativo)
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/09_plot_routing_network.py --arrows --threshold 150

# Solo mapa de acumulación de flujo
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/09_plot_routing_network.py --accumulation-only
```

**Salidas en `plots/`:**
- `routing_network_acc100.png` — Red de ríos D8 (4 órdenes: quebradas → río principal) sobre hillshade + elevación
- `flow_accumulation_map.png` — Acumulación de flujo (log-scale) mostrando conectividad del drenaje

---

### Análisis interactivo con Jupyter Lab

```bash
docker compose -f docker/docker-compose.yml up jupyter
# Abrir en el navegador: http://localhost:8888
```

---

## Simulación período largo (recomendado)

Para producción, usar el período 2000–2025 con spin-up descartado.

### ¿Por qué es necesario el spin-up?

VIC inicia con condiciones de suelo por defecto (generalmente suelos secos). Los primeros 2–3 años el modelo consume escorrimiento para llenar los perfiles de suelo en lugar de generar caudal real. Por eso la prueba 2014–2015 da caudales bajos excepto en enero 2014 (plena estación húmeda).

### Configuración recomendada

Editar `config/basin_config.yml`:

```yaml
simulation:
  start_date: "2000-01-01"    # Inicio del spin-up
  end_date:   "2025-12-31"
  spinup_years: 3             # 2000-2002 se descartan del análisis
  analysis_start: "2003-01-01"  # Período de análisis real
```

### Pipeline para 2000–2025

```bash
# 1. Descargar datos GEE 2000-2025 (si no están descargados)
export EE_SERVICE_ACCOUNT_JSON_B64=$(cat \
  /home/cenciso/Documents/WORK/ENGIE/DashBoard/windShortTermForecast/key_b64.txt)

docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/01_download_gee.py \
    --start 2000-01-01 \
    --end 2025-12-31

# 2. Preparar forzantes para todo el período
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/04_prepare_forcings.py \
    --start 2000-01-01 \
    --end 2025-12-31

# 3. Correr VIC 2000-2025
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/05_run_vic.py \
    --start 2000-01-01 \
    --end 2025-12-31

# 4. Routing (descarta automáticamente spin-up si se configura analysis_start)
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/06_run_routing.py

# 5. Análisis del período 2003-2025 (sin spin-up)
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/07_plot_timeseries.py

# 6. Animación del caudal para el período completo
docker compose -f docker/docker-compose.yml run --rm vic \
  python3 scripts/08_create_animation.py --streamflow --freq 7D --format gif
```

### Criterios para descartar el spin-up

El spin-up está completo cuando la variación interanual de humedad del suelo (OUT_SOIL_MOIST) se estabiliza. Regla práctica: descartar los primeros 3 años (2000–2002), analizar desde 2003.

---

## Configuración de la cuenca

Editar `config/basin_config.yml` para cambiar parámetros:

```yaml
basin:
  name: "Paucartambo"
  department: "Pasco"
  country: "Peru"
  bbox:
    lat_min: -11.80
    lat_max: -10.20
    lon_min: -76.80
    lon_max: -74.80
  resolution: 0.009    # ~1 km (mantener este valor)
  outlet:
    name: "Central Hidroeléctrica Yuncan"
    lat: -10.745
    lon: -75.586

simulation:
  start_date: "2000-01-01"
  end_date:   "2025-12-31"
```

---

## Salidas del modelo

### Variables VIC (en `data/outputs/fluxes_YYYY.nc`)

| Variable | Descripción | Unidades |
|---|---|---|
| `OUT_PREC` | Precipitación total | mm/día |
| `OUT_EVAP` | Evapotranspiración | mm/día |
| `OUT_RUNOFF` | Escorrentía superficial | mm/día |
| `OUT_BASEFLOW` | Flujo base | mm/día |
| `OUT_SOIL_MOIST` | Humedad del suelo (3 capas) | mm |
| `OUT_SWE` | Equivalente agua en nieve | mm |
| `OUT_SNOW_DEPTH` | Profundidad de nieve | m |
| `OUT_EVAP_CANOP` | Evaporación del canopy | mm/día |
| `OUT_TRANSP_VEG` | Transpiración de vegetación | mm/día |
| `OUT_SURF_TEMP` | Temperatura superficial | °C |

### Caudal enrutado (en `data/routing/`)

| Archivo | Descripción |
|---|---|
| `streamflow_yuncan.nc` | Caudal diario en Yuncan (m³/s) |
| `streamflow_yuncan.csv` | Misma serie en CSV (columnas: time, streamflow_m3s) |
| `flow_direction.nc` | Red D8: dirección, acumulación, cuenca, elevación |
| `rvic_params/params/*.nc` | Parámetros RVIC (hidrogramas unitarios por celda) |
| `rvic_conv/hist/*.nc` | Historia de convolucion RVIC (caudal diario) |

### Gráficos y animaciones

| Archivo | Descripción |
|---|---|
| `plots/streamflow_timeseries.png` | Serie de caudal diario/mensual/estacional |
| `plots/water_balance.png` | Balance hídrico |
| `plots/flow_duration_curve.png` | Curva de duración de caudales |
| `plots/routing_network_acc100.png` | Red de ríos D8 sobre topografía |
| `plots/flow_accumulation_map.png` | Acumulación de flujo (log-scale) |
| `animations/animation_STREAMFLOW_all.gif` | Animación escorrentía+caudal |
| `animations/animation_OUT_SOIL_MOIST_all.gif` | Animación humedad del suelo |

---

## Notas técnicas de implementación

### Routing RVIC — fixes aplicados al código fuente

El script `06_run_routing.py` parchea automáticamente 6 archivos de RVIC al inicio para compatibilidad con pandas ≥1.0, NumPy ≥1.24 y netCDF4 ≥1.6:

| Archivo | Fix |
|---|---|
| `rvic/parameters.py` | `pour_points.ix[` → `.loc[`; `np.finfo(np.float)` → `np.finfo(float)` |
| `rvic/core/param_file.py` | `dtype='S...'` → `dtype='U...'` (Unicode); `np.finfo(np.float)` → `np.finfo(float)` |
| `rvic/core/share.py` | `valid_range='0, 86400'` (string) → `[0.0, 86400.0]` (lista numérica) |
| `rvic/core/variables.py` | `stringtochar(np.array(...))` → `np.frombuffer(..., dtype='S1')` |
| `rvic/core/history.py` | `char_names = stringtochar(self._outlet_name)` → `np.ma.filled(...)` |

### UH box — unidades en SEGUNDOS

El tiempo en la UH box debe estar en **segundos** (no horas). Si se usan horas, RVIC crea arrays de ~180,000 timesteps causando un consumo de ~47 GB de RAM.

```csv
time,uh
0.0,0.5
86400.0,0.5   # ← 24 horas en segundos
```

### Pour point — celda con máxima acumulación de flujo

El pour point se hace snap a la celda con **máxima acumulación de flujo** dentro de `basin_id == 1`. Esto garantiza que RVIC puede trazar el camino desde todas las celdas upstream hasta el outlet.

### Forzantes RVIC — época temporal común

Todos los archivos anuales de forzantes RVIC usan la misma época temporal:
```python
time_units = f"days since {year_start}-01-01 00:00:00"
```
Si cada año usa su propia época (`days since YYYY-01-01`), RVIC falla con `ValueError: Units do not match`.

---

## Solución de problemas conocidos

### `docker compose` no encontrado
```bash
# Verificar versión
docker compose version
# Si no funciona, verificar que Docker Desktop o el plugin está instalado
```

### Error de autenticación GEE
```bash
export EE_SERVICE_ACCOUNT_JSON_B64=$(cat \
  /home/cenciso/Documents/WORK/ENGIE/DashBoard/windShortTermForecast/key_b64.txt)
# Verificar que el token no está vacío
echo $EE_SERVICE_ACCOUNT_JSON_B64 | head -c 20
```

### RVIC parameters falla con `ValueError: min() arg is an empty sequence`
El pour point cayó fuera de `basin_id == 1`. El script hace snap automático a la celda outlet (max acumulación). Si persiste, verificar `data/domain/flow_direction.nc`.

### RVIC falla con `KeyError: 'GRIDID'`
Error en la config de RVIC parameters. El script genera la config automáticamente con todas las claves requeridas incluyendo `GRIDID`.

### Caudales muy bajos (< 2 m³/s)
Sin spin-up. Correr el período completo recomendado (2000–2025) y descartar 2000–2002.

### Memoria insuficiente durante VIC
```yaml
# En config/basin_config.yml, reducir resolución:
resolution: 0.025   # ~2.5 km en lugar de 1 km
```

### Log de RVIC con `Cannot convert masked element to a Python int`
Warning benigno — no afecta los resultados. Es un bug en el logger de RVIC con valores enmascarados.
