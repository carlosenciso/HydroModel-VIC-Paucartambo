"""
Utilidades para preparación de archivos VIC
"""

import numpy as np
import xarray as xr
from pathlib import Path


# ─────────────────────────────────────────────────────────
# Pedotransfer Functions (Cosby et al., 1984 + Saxton & Rawls, 2006)
# ─────────────────────────────────────────────────────────

def cosby_hydraulics(sand: np.ndarray, clay: np.ndarray) -> dict:
    """
    Derivar parámetros hidráulicos del suelo usando Cosby et al. (1984).

    Args:
        sand: Fracción de arena (0-1)
        clay: Fracción de arcilla (0-1)

    Returns:
        Diccionario con parámetros hidráulicos
    """
    silt = np.clip(1.0 - sand - clay, 0.01, 0.99)

    # Conductividad hidráulica saturada (m/s)
    Ksat = 7.0556e-6 * np.exp(3.52 * sand - 2.54 * clay)

    # Parámetro b de Campbell (exponente de retención)
    b = 3.10 + 15.7 * clay - 0.3 * sand

    # Presión de burbuja (m) - bubbling pressure
    psis = 0.01 * np.exp(1.54 - 0.0095 * (sand * 100) + 0.0063 * (silt * 100))

    # Porosidad (humedad saturada)
    theta_s = 0.489 - 0.00126 * (sand * 100)

    # Humedad en punto crítico (field capacity ~pF 2.0)
    theta_fc = theta_s * (psis / 3.36) ** (1.0 / b)

    # Humedad en punto de marchitez (pF 4.2)
    theta_wp = theta_s * (psis / 150.0) ** (1.0 / b)

    return {
        "Ksat": Ksat,
        "b": b,
        "psis": psis,
        "theta_s": theta_s,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "expt": 3.0 + 2.0 * b,
    }


def estimate_vic_soil_params(
    sand: np.ndarray,
    clay: np.ndarray,
    bulk_density: np.ndarray,
    organic: np.ndarray = None,
    elev: np.ndarray = None,
) -> dict:
    """
    Estimar parámetros VIC de suelo desde textura y densidad aparente.

    Args:
        sand: Fracción de arena (0-1), shape (nlayer, ny, nx)
        clay: Fracción de arcilla (0-1), shape (nlayer, ny, nx)
        bulk_density: Densidad aparente (g/cm3), shape (nlayer, ny, nx)
        organic: Fracción de materia orgánica (0-1)
        elev: Elevación del terreno (m), shape (ny, nx)

    Returns:
        Diccionario con arrays de parámetros VIC
    """
    nlayer, ny, nx = sand.shape

    if organic is None:
        organic = np.zeros_like(sand)

    hydraulics = cosby_hydraulics(sand, clay)

    # Densidad de partículas del suelo (g/cm3)
    soil_density = 2.65 * np.ones_like(bulk_density)

    # Porosidad efectiva
    porosity = 1.0 - bulk_density / soil_density

    # Parámetros de infiltración VIC (Bi/VIC - puede calibrarse)
    # Valor típico para cuencas húmedas tropicales
    infilt = np.full((ny, nx), 0.10)

    # Dsmax: velocidad máxima de flujo base (mm/day)
    Ds_max = hydraulics["Ksat"][0, :, :] * 86400 * 1000  # m/s -> mm/day
    Ds_max = np.clip(Ds_max, 1.0, 100.0)

    # Ds: fracción de Dsmax donde inicia flujo no-lineal
    Ds = np.full((ny, nx), 0.01)

    # Ws: fracción de humedad máxima donde inicia flujo no-lineal
    Ws = np.full((ny, nx), 0.90)

    # c: exponente de la curva de flujo base
    c = np.full((ny, nx), 2.0)

    # Profundidad de capas (m) - por defecto 0.1, 0.3, 0.6
    depth = np.array([0.10, 0.30, 0.60])[..., np.newaxis, np.newaxis]
    depth = np.broadcast_to(depth, (3, ny, nx)).copy()

    # Humedad inicial (fracción de capacidad de campo)
    init_moist = hydraulics["theta_fc"] * depth * 1000  # mm

    # Temperatura promedio del suelo (C) - estimación por elevación
    if elev is not None:
        avg_T = 25.0 - 0.0065 * elev
    else:
        avg_T = np.full((ny, nx), 20.0)

    # Profundidad de amortiguamiento térmico (m)
    dp = np.full((ny, nx), 4.0)

    # Contenido de cuarzo por capa (fracción)
    quartz = 0.5 * sand + 0.2 * (1 - sand - clay)

    return {
        "infilt": infilt,
        "Ds": Ds,
        "Dsmax": Ds_max,
        "Ws": Ws,
        "c": c,
        "depth": depth,
        "bulk_density": bulk_density,
        "soil_density": soil_density,
        "Ksat": hydraulics["Ksat"] * 86400 * 1000,  # m/s -> mm/day
        "expt": hydraulics["expt"],
        "bubble": hydraulics["psis"] * 1000,  # m -> mm
        "quartz": quartz,
        "Wcr_FRACT": hydraulics["theta_fc"] / hydraulics["theta_s"],
        "Wpwp_FRACT": hydraulics["theta_wp"] / hydraulics["theta_s"],
        "resid_moist": np.maximum(hydraulics["theta_wp"] * 0.5, 0.01),
        "init_moist": init_moist,
        "avg_T": avg_T,
        "dp": dp,
        "off_gmt": np.full((ny, nx), -5.0),  # Perú: UTC-5
        "rough": np.full((ny, nx), 0.01),
        "snow_rough": np.full((ny, nx), 0.0005),
        "fs_active": np.zeros((ny, nx), dtype=np.int32),
    }


# ─────────────────────────────────────────────────────────
# Utilidades NetCDF
# ─────────────────────────────────────────────────────────

def add_cf_attributes(ds: xr.Dataset, varname: str, units: str, long_name: str) -> xr.Dataset:
    """Agregar atributos CF conventions a una variable."""
    ds[varname].attrs.update({
        "units": units,
        "long_name": long_name,
        "_FillValue": -9999.0,
        "missing_value": -9999.0,
    })
    return ds


def create_grid(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                resolution: float) -> tuple:
    """
    Crear arrays de latitud y longitud para una grilla regular.

    Returns:
        (lats, lons): Arrays 1D de coordenadas
    """
    lats = np.arange(lat_min + resolution / 2, lat_max, resolution)
    lons = np.arange(lon_min + resolution / 2, lon_max, resolution)
    return lats[::-1], lons  # latitud de norte a sur


def compute_cell_area(lat: np.ndarray, resolution_deg: float) -> np.ndarray:
    """
    Calcular área de cada celda en m² (varía con latitud).

    Args:
        lat: Array 1D de latitudes
        resolution_deg: Resolución en grados

    Returns:
        Array 1D de áreas en m²
    """
    R_earth = 6371000.0  # Radio de la Tierra en metros
    lat_rad = np.radians(lat)
    res_rad = np.radians(resolution_deg)
    area = (R_earth ** 2) * np.abs(np.sin(lat_rad + res_rad / 2)
                                   - np.sin(lat_rad - res_rad / 2)) * res_rad
    return area


# ─────────────────────────────────────────────────────────
# Funciones de validación
# ─────────────────────────────────────────────────────────

def validate_forcing_file(fpath: str) -> bool:
    """Verificar que un archivo de forzantes tiene el formato correcto."""
    try:
        ds = xr.open_dataset(fpath)
        required = ["PRCP", "AIR_TEMP", "WIND", "SHORTWAVE", "LONGWAVE", "PRESSURE", "VP"]
        missing = [v for v in required if v not in ds.data_vars]
        if missing:
            print(f"[WARNING] Variables faltantes en {fpath}: {missing}")
            return False
        ds.close()
        return True
    except Exception as e:
        print(f"[ERROR] No se puede abrir {fpath}: {e}")
        return False


def kelvin_to_celsius(temp_k: np.ndarray) -> np.ndarray:
    return temp_k - 273.15


def pa_to_kpa(pressure_pa: np.ndarray) -> np.ndarray:
    return pressure_pa / 1000.0


def joules_to_watts(energy_j: np.ndarray, seconds_per_day: float = 86400.0) -> np.ndarray:
    """Convertir energía acumulada (J/m2/day) a potencia media (W/m2)."""
    return energy_j / seconds_per_day


def dewpoint_to_vp(td_k: np.ndarray) -> np.ndarray:
    """
    Calcular presión de vapor (kPa) desde temperatura de punto de rocío (K).
    Usa ecuación de Magnus.
    """
    td_c = kelvin_to_celsius(td_k)
    vp = 0.61078 * np.exp(17.27 * td_c / (td_c + 237.3))
    return vp


def wind_components_to_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Calcular velocidad del viento desde componentes u y v."""
    return np.sqrt(u ** 2 + v ** 2)
