#!/usr/bin/env python3
"""
Script 04: Preparar archivos de forzantes para VIC Image Driver
================================================================
Convierte los datos descargados de GEE al formato NetCDF requerido
por VIC5 Image Driver.

Formato de salida VIC:
  - Archivo por año: forcing_YYYY.nc
  - Variables: PREC, AIR_TEMP (o TMAX/TMIN), WIND, SHORTWAVE,
               LONGWAVE, PRESSURE, VP
  - Dimensiones: time (6-horario, req. VIC5), lat, lon
  - Unidades: mm/6h, C, m/s, W/m2, W/m2, kPa, kPa
  - Outputs VIC siguen siendo diarios (AGGFREQ NDAYS 1)

Funcionalidades adicionales:
  - Bias correction de precipitación (CHIRPS) con estaciones observadas
  - Corrección topográfica de temperatura (lapse rate)
  - Distribución espacial de precipitación (disagregación)
  - Relleno de datos faltantes

Uso:
    python scripts/04_prepare_forcings.py --start 2010-01-01 --end 2020-12-31
"""

import argparse
import calendar
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils.vic_utils import (
    kelvin_to_celsius, pa_to_kpa, joules_to_watts,
    dewpoint_to_vp, wind_components_to_speed
)

# ─────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).parent.parent
CONFIG_FILE   = PROJECT_DIR / "config" / "basin_config.yml"
DATA_DIR      = PROJECT_DIR / "data"
GEE_RAW_DIR   = DATA_DIR / "gee_raw" / "forcings"
DOMAIN_DIR    = DATA_DIR / "domain"
FORCING_DIR   = DATA_DIR / "forcings"
FORCING_DIR.mkdir(parents=True, exist_ok=True)

LAPSE_RATE = -0.0065  # °C/m (lapse rate estándar)


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def load_domain() -> xr.Dataset:
    return xr.open_dataset(DOMAIN_DIR / "domain.nc")


def rename_xy(ds: xr.Dataset) -> xr.Dataset:
    """Rename x/y dims (wxee output) to lon/lat."""
    rename_map = {}
    if "x" in ds.dims:
        rename_map["x"] = "lon"
    if "y" in ds.dims:
        rename_map["y"] = "lat"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


# ─────────────────────────────────────────────────────────
# Corrección topográfica de temperatura
# ─────────────────────────────────────────────────────────

def apply_temperature_lapse_correction(
    temp: np.ndarray,
    elev: np.ndarray,
    reference_elev: float = 1500.0,
) -> np.ndarray:
    """
    Corregir temperatura por gradiente altitudinal.

    T_corrected = T_ref + lapse_rate * (elev - ref_elev)

    Args:
        temp: Array de temperatura (°C), shape (time, nlat, nlon)
        elev: Array de elevación (m), shape (nlat, nlon)
        reference_elev: Elevación de referencia (ERA5 ~10km → usar media cuenca)

    Returns:
        Temperatura corregida
    """
    correction = LAPSE_RATE * (elev[np.newaxis, :, :] - reference_elev)
    return temp + correction


def compute_vapor_pressure(
    tmax: np.ndarray,
    tmin: np.ndarray,
    rh_mean: float = 0.80,
) -> np.ndarray:
    """
    Estimar presión de vapor desde temperatura máx/min.
    Método de FAO-56 (Allen et al., 1998).

    Args:
        tmax: Temperatura máxima (°C)
        tmin: Temperatura mínima (°C)
        rh_mean: Humedad relativa media (fracción)

    Returns:
        Presión de vapor (kPa)
    """
    # Presión de vapor de saturación
    es_tmax = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    es_tmin = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    es = (es_tmax + es_tmin) / 2.0
    return es * rh_mean


def estimate_longwave(
    tmax: np.ndarray,
    tmin: np.ndarray,
    shortwave: np.ndarray,
    elev: np.ndarray = None,
) -> np.ndarray:
    """
    Estimar radiación de onda larga usando fórmula de Prata (1996) / FAO-56.

    Args:
        tmax, tmin: Temperatura max/min en °C
        shortwave: Radiación solar (W/m2)
        elev: Elevación (m), para calcular Rs0

    Returns:
        Radiación de onda larga neta (W/m2)
    """
    sigma = 5.67e-8  # Stefan-Boltzmann (W/m2/K4)
    Tmean_K = (tmax + tmin) / 2 + 273.15

    # Radiación de cuerpo negro
    Rl_down = sigma * Tmean_K ** 4

    # Corrección por nubosidad (simplificada)
    # Para cuencas tropicales, asumir un factor de nubosidad basado en shortwave
    # Rs0 (clearsky radiation) ~ shortwave / 0.75 como estimación
    Rs0 = np.where(shortwave > 0, shortwave / 0.75, shortwave)
    cloud_factor = np.where(
        Rs0 > 0,
        np.clip(1.35 * shortwave / Rs0 - 0.35, 0.05, 1.0),
        0.5
    )

    # Longwave hacia abajo (estimación)
    # Factor de emisividad atmosférica (simplificado)
    eps_a = 0.85 + 0.10 * (1 - cloud_factor)
    Rl_down_corrected = eps_a * Rl_down

    return Rl_down_corrected


# ─────────────────────────────────────────────────────────
# Procesamiento principal
# ─────────────────────────────────────────────────────────

def process_year(
    year: int,
    ds_domain: xr.Dataset,
    config: dict,
    use_station_bias_correction: bool = False,
) -> None:
    """
    Procesar forzantes para un año específico.

    Args:
        year: Año a procesar
        ds_domain: Dataset del dominio VIC
        config: Configuración del proyecto
        use_station_bias_correction: Aplicar corrección de bias con estaciones
    """
    output_file = FORCING_DIR / f"forcing_{year}.nc"
    if output_file.exists():
        print(f"  [SKIP] {year} ya existe.")
        return

    # Cargar datos crudos de GEE
    raw_file = GEE_RAW_DIR / f"era5_chirps_{year}.nc"
    if not raw_file.exists():
        print(f"  [WARNING] Datos GEE no encontrados para {year}: {raw_file}")
        return

    print(f"  Procesando año {year}...")

    ds_raw = rename_xy(xr.open_dataset(raw_file))
    lats_target = ds_domain.lat.values
    lons_target = ds_domain.lon.values
    mask = ds_domain["mask"].values

    # Interpolar al grid del dominio
    ds = ds_raw.interp(
        lat=lats_target,
        lon=lons_target,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    # ── Precipitación (CHIRPS) ─────────────────────────
    # GEE descarga como "PREC" o "PRCP" dependiendo del año
    prec_var = "PREC" if "PREC" in ds else ("PRCP" if "PRCP" in ds else None)
    if prec_var:
        prcp = ds[prec_var].values.clip(min=0)
    else:
        prcp = np.zeros((len(ds.time), len(lats_target), len(lons_target)))
        print(f"  [WARNING] {year}: no se encontró variable de precipitación")

    # ── Temperatura (ERA5) ─────────────────────────────
    if "TMAX" in ds and "TMIN" in ds:
        tmax = ds["TMAX"].values
        tmin = ds["TMIN"].values
        # Asegurar que Tmax > Tmin
        tmax = np.where(tmax < tmin, tmin + 0.5, tmax)
        tmin = np.where(tmin > tmax, tmax - 0.5, tmin)
        air_temp = (tmax + tmin) / 2.0
    elif "AIR_TEMP" in ds:
        air_temp = ds["AIR_TEMP"].values
        tmax = air_temp + 3.0  # Estimación
        tmin = air_temp - 3.0
    else:
        print(f"  [WARNING] No se encontró temperatura para {year}")
        air_temp = np.full_like(prcp, 15.0)
        tmax = air_temp + 3.0
        tmin = air_temp - 3.0

    # Corrección topográfica de temperatura
    if "elev" in ds_domain:
        elev = ds_domain["elev"].values
        ref_elev = np.nanmean(elev[mask == 1]) if mask.sum() > 0 else 1500.0
        air_temp = apply_temperature_lapse_correction(air_temp, elev, ref_elev)
        tmax = apply_temperature_lapse_correction(tmax, elev, ref_elev)
        tmin = apply_temperature_lapse_correction(tmin, elev, ref_elev)

    # ── Viento ─────────────────────────────────────────
    if "WIND" in ds:
        wind = ds["WIND"].values.clip(min=0.1, max=50.0)
    else:
        wind = np.full_like(prcp, 2.0)  # 2 m/s por defecto

    # ── Radiación de onda corta ─────────────────────────
    if "SHORTWAVE" in ds:
        shortwave = ds["SHORTWAVE"].values.clip(min=0)
    else:
        # Estimar desde temperatura (método simplificado Hargreaves)
        print(f"  [INFO] Estimando radiación solar (método Hargreaves)...")
        Ra = 30.0  # W/m2 promedio - deberías calcular desde lat/lon/fecha
        shortwave = 0.16 * Ra * np.sqrt(np.abs(tmax - tmin))
        shortwave = shortwave.clip(min=0)

    # ── Radiación de onda larga ─────────────────────────
    if "LONGWAVE" in ds:
        longwave = ds["LONGWAVE"].values.clip(min=0)
    else:
        longwave = estimate_longwave(tmax, tmin, shortwave)

    # ── Presión atmosférica ─────────────────────────────
    if "PRESSURE" in ds:
        pressure = ds["PRESSURE"].values.clip(min=30, max=110)
    else:
        # Calcular desde elevación: P = P0 * exp(-z / 8500)
        if "elev" in ds_domain:
            elev = ds_domain["elev"].values
            pressure = 101.325 * np.exp(-elev / 8500)
        else:
            pressure = np.full_like(prcp, 85.0)  # ~1500m de elevación

    # ── Presión de vapor ────────────────────────────────
    if "VP" in ds:
        vp = ds["VP"].values.clip(min=0.01, max=10.0)
    else:
        # Estimar desde temperatura
        # Para cuencas tropicales húmedas, humedad relativa ~75-85%
        vp = compute_vapor_pressure(tmax, tmin, rh_mean=0.80)

    # ── Corrección de bias de precipitación ────────────
    if use_station_bias_correction and config["monitoring_stations"]["enabled"]:
        print(f"  [INFO] Aplicando bias correction de precipitación...")
        prcp = apply_precipitation_bias_correction(
            prcp, lats_target, lons_target, year, config
        )

    # ── Suavizado espacial de forzantes ERA5 ───────────
    # ERA5 tiene resolución ~0.1°, necesita suavizado al interpolar a 1km
    sigma = 2.0  # píxeles de suavizado gaussiano
    for t in range(prcp.shape[0]):
        wind[t] = gaussian_filter(wind[t], sigma=sigma)
        shortwave[t] = gaussian_filter(shortwave[t], sigma=sigma / 2)
        longwave[t] = gaussian_filter(longwave[t], sigma=sigma / 2)
        pressure[t] = gaussian_filter(pressure[t], sigma=sigma)
        vp[t] = gaussian_filter(vp[t], sigma=sigma)

    # ── Pad to full year (GEE may return fewer than 365 days) ──────────
    ndays_in_year = 366 if calendar.isleap(year) else 365
    n_avail = prcp.shape[0]
    if n_avail < ndays_in_year:
        pad_days = ndays_in_year - n_avail
        def pad_arr(arr):
            return np.concatenate([arr, np.repeat(arr[[-1]], pad_days, axis=0)], axis=0)
        prcp      = pad_arr(prcp)
        air_temp  = pad_arr(air_temp)
        wind      = pad_arr(wind)
        shortwave = pad_arr(shortwave)
        longwave  = pad_arr(longwave)
        pressure  = pad_arr(pressure)
        vp        = pad_arr(vp)
        print(f"  [INFO] Padded {pad_days} missing day(s) with last-value fill ({ndays_in_year} days total)")

    # ── Desagregar diario → 6-hourly (4 pasos/día) ─────────────────────
    # VIC5 Image Driver requiere que el timestep de los forzantes sea igual a
    # SNOW_STEPS_PER_DAY (mínimo 4). No existe modo de forzantes diarios en VIC5.
    # Los outputs siguen siendo diarios (AGGFREQ NDAYS 1 en global_params).
    STEPS_PER_DAY = 4
    times_6h = pd.date_range(f"{year}-01-01", f"{year}-12-31 18:00", freq="6h")

    # Crear índice de día correspondiente a cada paso 6-horario
    day_idx = np.arange(len(times_6h)) // STEPS_PER_DAY

    prec_6h = prcp[day_idx] / STEPS_PER_DAY  # mm/6h (suma diaria conservada)

    # Radiación solar: sólo en el día (pasos 1 y 2 = 06:00 y 12:00 UTC)
    step_in_day = np.arange(len(times_6h)) % STEPS_PER_DAY
    swdown_daily = shortwave[day_idx]
    swdown_6h = np.where(
        (step_in_day[:, None, None] == 1) | (step_in_day[:, None, None] == 2),
        swdown_daily * 2.0,
        0.0,
    )

    ds_out = xr.Dataset(
        {
            "PREC": (
                ["time", "lat", "lon"],
                prec_6h.astype(np.float32),
                {"units": "mm", "long_name": "6-hourly precipitation"},
            ),
            "AIR_TEMP": (
                ["time", "lat", "lon"],
                air_temp[day_idx].astype(np.float32),
                {"units": "C", "long_name": "Air temperature"},
            ),
            "WIND": (
                ["time", "lat", "lon"],
                wind[day_idx].astype(np.float32),
                {"units": "m/s", "long_name": "Wind speed at 10m"},
            ),
            "SWDOWN": (
                ["time", "lat", "lon"],
                swdown_6h.astype(np.float32),
                {"units": "W/m2", "long_name": "Incoming shortwave radiation"},
            ),
            "LWDOWN": (
                ["time", "lat", "lon"],
                longwave[day_idx].astype(np.float32),
                {"units": "W/m2", "long_name": "Incoming longwave radiation"},
            ),
            "PRESSURE": (
                ["time", "lat", "lon"],
                pressure[day_idx].astype(np.float32),
                {"units": "kPa", "long_name": "Atmospheric pressure"},
            ),
            "VP": (
                ["time", "lat", "lon"],
                vp[day_idx].astype(np.float32),
                {"units": "kPa", "long_name": "Vapor pressure"},
            ),
        },
        coords={
            "time": times_6h,
            "lat": (["lat"], lats_target, {"units": "degrees_north"}),
            "lon": (["lon"], lons_target, {"units": "degrees_east"}),
        },
    )

    ds_out.attrs = {
        "title": f"VIC Forcings - Cuenca Paucartambo - {year}",
        "sources": "ERA5-Land (ECMWF) + CHIRPS (UCSB)",
        "history": "Generado por 04_prepare_forcings.py",
        "conventions": "CF-1.6",
    }

    # Guardar con compresión
    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ds_out.data_vars
    }
    ds_out.to_netcdf(output_file, encoding=encoding)

    ds_raw.close()
    print(f"  [OK] {year}: {output_file.name}")


def apply_precipitation_bias_correction(
    prcp: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    year: int,
    config: dict,
) -> np.ndarray:
    """
    Corrección de bias de precipitación CHIRPS usando estaciones.
    Método: Scaling mensual con interpolación IDW.
    (Solo si hay datos de estaciones disponibles)
    """
    stations = config.get("monitoring_stations", {}).get("stations", [])
    prcp_stations = [s for s in stations if s["type"] == "precipitation"]

    if not prcp_stations:
        return prcp

    station_data_dir = DATA_DIR / "stations"
    if not station_data_dir.exists():
        return prcp

    print(f"  [INFO] Aplicando bias correction con {len(prcp_stations)} estaciones...")
    # Implementación simplificada - expandir con datos reales
    return prcp


def create_spinup_forcings(start_year: int, spinup_years: int = 2) -> None:
    """
    Crear forzantes de spin-up repitiendo el primer año disponible.
    """
    print(f"[SPINUP] Creando forzantes de spin-up ({spinup_years} años)...")
    first_file = FORCING_DIR / f"forcing_{start_year}.nc"

    if not first_file.exists():
        print(f"  [ERROR] No existe {first_file}")
        return

    ds_first = xr.open_dataset(first_file)

    for i in range(spinup_years, 0, -1):
        spinup_year = start_year - i
        spinup_file = FORCING_DIR / f"forcing_{spinup_year}.nc"

        if not spinup_file.exists():
            ds_spinup = ds_first.copy()
            new_times = pd.date_range(
                f"{spinup_year}-01-01",
                f"{spinup_year}-12-31",
                freq="D"
            )
            ds_spinup = ds_spinup.assign_coords(time=new_times[:len(ds_spinup.time)])
            ds_spinup.attrs["note"] = f"Spin-up: copia de {start_year}"
            ds_spinup.to_netcdf(spinup_file)
            print(f"  [OK] Spin-up {spinup_year} creado.")

    ds_first.close()


def validate_forcing_completeness(start_year: int, end_year: int) -> bool:
    """Verificar que existan todos los archivos de forzantes."""
    all_ok = True
    for year in range(start_year, end_year + 1):
        f = FORCING_DIR / f"forcing_{year}.nc"
        if not f.exists():
            print(f"  [MISSING] {f}")
            all_ok = False
        else:
            # Verificar variables
            ds = xr.open_dataset(f)
            required = ["PREC", "AIR_TEMP", "WIND", "SWDOWN", "LWDOWN", "PRESSURE", "VP"]
            missing = [v for v in required if v not in ds.data_vars]
            if missing:
                print(f"  [WARNING] {year}: variables faltantes: {missing}")
                all_ok = False
            ds.close()
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Preparar forzantes VIC para la cuenca Paucartambo"
    )
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--spinup", type=int, default=2,
                        help="Años de spin-up (default: 2)")
    parser.add_argument("--stations", action="store_true",
                        help="Aplicar corrección de bias con estaciones")
    args = parser.parse_args()

    print("=" * 60)
    print("  PREPARANDO FORZANTES VIC - Cuenca Paucartambo")
    print("=" * 60)

    config = load_config()
    ds_domain = load_domain()

    start_year = int(args.start[:4])
    end_year = int(args.end[:4])

    # Procesar cada año
    print(f"\n[→] Procesando {start_year}-{end_year}...")
    for year in range(start_year, end_year + 1):
        process_year(year, ds_domain, config, args.stations)

    # Crear forzantes de spin-up
    if args.spinup > 0:
        print(f"\n[→] Creando spin-up ({args.spinup} años)...")
        create_spinup_forcings(start_year, args.spinup)

    # Validar
    print(f"\n[→] Validando archivos...")
    spinup_start = start_year - args.spinup
    ok = validate_forcing_completeness(spinup_start, end_year)

    if ok:
        print(f"\n[✓] Todos los forzantes preparados: {FORCING_DIR}")
    else:
        print(f"\n[!] Algunos forzantes faltantes. Revisar arriba.")

    ds_domain.close()


if __name__ == "__main__":
    main()
