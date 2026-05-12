#!/usr/bin/env python3
"""
Script 03: Preparar archivo de parámetros VIC
=============================================
Genera el archivo params.nc requerido por VIC5 Image Driver.

El archivo de parámetros contiene:
  Parámetros de Suelo:
    - infilt, Ds, Ds_max, Ws, c (parámetros de Arno/TOPMODEL)
    - depth (3 capas), bulk_density, soil_density, Ksat
    - expt (b de Campbell), bubble, quartz
    - Wcr_FRACT, Wpwp_FRACT, resid_moist
    - init_moist, avg_T, dp, off_gmt
    - rough, snow_rough

  Parámetros de Vegetación (por tipo):
    - Cv: fracción de cobertura por clase (nveg tipos)
    - LAI[12 meses], albedo[12 meses]
    - root_depth, root_fract (nlayer capas)
    - overstory, rarc, rmin

Uso:
    python scripts/03_prepare_parameters.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils.vic_utils import estimate_vic_soil_params

# ─────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent.parent
CONFIG_FILE  = PROJECT_DIR / "config" / "basin_config.yml"
DATA_DIR     = PROJECT_DIR / "data"
GEE_DIR      = DATA_DIR / "gee_raw"
DOMAIN_DIR   = DATA_DIR / "domain"
PARAMS_DIR   = DATA_DIR / "parameters"
PARAMS_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def load_domain() -> xr.Dataset:
    domain_file = DOMAIN_DIR / "domain.nc"
    if not domain_file.exists():
        raise FileNotFoundError(
            "domain.nc no encontrado. Ejecuta primero 02_prepare_domain.py"
        )
    return xr.open_dataset(domain_file)


def rename_xy(ds: xr.Dataset) -> xr.Dataset:
    """Rename x/y dims (wxee output) to lon/lat expected by interpolation calls."""
    rename_map = {}
    if "x" in ds.dims:
        rename_map["x"] = "lon"
    if "y" in ds.dims:
        rename_map["y"] = "lat"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


# ─────────────────────────────────────────────────────────
# Parámetros de Suelo
# ─────────────────────────────────────────────────────────

# Mapeo WorldCover → clase VIC (1-9)
WORLDCOVER_TO_VIC = {
    10: 1,   # Bosque
    20: 5,   # Matorral
    30: 3,   # Pastizal
    40: 4,   # Agricultura
    50: 7,   # Urbano
    60: 7,   # Suelo desnudo
    70: 8,   # Nieve
    80: 9,   # Agua
    90: 1,   # Humedal
    95: 1,   # Manglar
    100: 6,  # Musgo
}

# Parámetros de vegetación por clase VIC
# Formato: (overstory, rarc, rmin, [LAI_12meses], [albedo_12meses],
#           root_depth1, root_frac1, root_depth2, root_frac2, root_depth3, root_frac3)
VEG_PARAMS = {
    1: {  # Bosque tropical
        "overstory": 1, "rarc": 100.0, "rmin": 200.0,
        "LAI": [6.0]*12,
        "albedo": [0.12]*12,
        "veg_rough": 2.0, "displacement": 15.0,  # ~20m tall forest, d≈0.75*h
        "root_depth": [0.10, 0.30, 0.60],
        "root_frac":  [0.30, 0.40, 0.30],
    },
    2: {  # Bosque de neblina
        "overstory": 1, "rarc": 100.0, "rmin": 250.0,
        "LAI": [5.0, 5.0, 5.0, 5.2, 5.5, 5.8, 5.5, 5.2, 5.0, 4.8, 4.8, 5.0],
        "albedo": [0.13]*12,
        "veg_rough": 1.5, "displacement": 11.0,  # ~15m tall cloud forest, d≈0.75*h
        "root_depth": [0.10, 0.30, 0.60],
        "root_frac":  [0.40, 0.35, 0.25],
    },
    3: {  # Pastizal Puna
        "overstory": 0, "rarc": 60.0, "rmin": 150.0,
        "LAI": [2.5, 2.5, 2.0, 1.5, 1.2, 1.0, 1.0, 1.2, 1.5, 2.0, 2.5, 2.5],
        "albedo": [0.20, 0.19, 0.18, 0.17, 0.17, 0.18, 0.18, 0.17, 0.18, 0.19, 0.20, 0.20],
        "veg_rough": 0.15, "displacement": 0.10,
        "root_depth": [0.10, 0.30, 0.60],
        "root_frac":  [0.50, 0.35, 0.15],
    },
    4: {  # Agricultura
        "overstory": 0, "rarc": 70.0, "rmin": 120.0,
        "LAI": [2.0, 2.5, 3.5, 4.0, 3.5, 2.0, 1.5, 2.0, 3.0, 3.5, 2.5, 2.0],
        "albedo": [0.18, 0.17, 0.16, 0.15, 0.15, 0.15, 0.15, 0.15, 0.16, 0.17, 0.18, 0.18],
        "veg_rough": 0.10, "displacement": 0.07,
        "root_depth": [0.10, 0.30, 0.60],
        "root_frac":  [0.60, 0.30, 0.10],
    },
    5: {  # Arbustos andinos
        "overstory": 0, "rarc": 80.0, "rmin": 150.0,
        "LAI": [2.5, 2.5, 2.2, 2.0, 1.8, 1.5, 1.5, 1.8, 2.0, 2.2, 2.5, 2.5],
        "albedo": [0.17]*12,
        "veg_rough": 0.20, "displacement": 0.15,
        "root_depth": [0.10, 0.30, 0.60],
        "root_frac":  [0.45, 0.35, 0.20],
    },
    6: {  # Páramo/Jalca
        "overstory": 0, "rarc": 60.0, "rmin": 200.0,
        "LAI": [1.5, 1.5, 1.2, 1.0, 0.8, 0.6, 0.6, 0.8, 1.0, 1.2, 1.5, 1.5],
        "albedo": [0.22, 0.21, 0.20, 0.19, 0.19, 0.20, 0.20, 0.19, 0.20, 0.21, 0.22, 0.22],
        "veg_rough": 0.10, "displacement": 0.07,
        "root_depth": [0.10, 0.30, 0.60],
        "root_frac":  [0.60, 0.30, 0.10],
    },
    7: {  # Suelo desnudo
        "overstory": 0, "rarc": 999.0, "rmin": 999.0,
        "LAI": [0.0]*12,
        "albedo": [0.30]*12,
        "veg_rough": 0.001, "displacement": 0.001,
        "root_depth": [0.10, 0.30, 0.60],
        "root_frac":  [1.00, 0.00, 0.00],
    },
    8: {  # Nieve/Glaciar
        "overstory": 0, "rarc": 999.0, "rmin": 999.0,
        "LAI": [0.0]*12,
        "albedo": [0.70]*12,
        "veg_rough": 0.001, "displacement": 0.001,
        "root_depth": [0.10, 0.30, 0.60],
        "root_frac":  [1.00, 0.00, 0.00],
    },
    9: {  # Agua
        "overstory": 0, "rarc": 999.0, "rmin": 50.0,
        "LAI": [0.0]*12,
        "albedo": [0.07]*12,
        "veg_rough": 0.001, "displacement": 0.001,
        "root_depth": [0.10, 0.30, 0.60],
        "root_frac":  [1.00, 0.00, 0.00],
    },
}

NVEG = len(VEG_PARAMS)
NLAYER = 3
NMONTH = 12
NROOT_ZONES = 3

# Vegetation library parameters per VIC class
# wind_h: wind measurement height (m)
# RGL: minimum solar radiation for ET (W/m²)
# rad_atten, wind_atten: attenuation coefficients (fraction)
# trunk_ratio: fraction of plant height that is trunk (fraction)
VEG_LIB_PARAMS = {
    1: dict(wind_h=30.0, RGL=30.0,  rad_atten=0.5, wind_atten=0.5, trunk_ratio=0.2),  # Bosque tropical (wind measured at 30m)
    2: dict(wind_h=22.0, RGL=30.0,  rad_atten=0.5, wind_atten=0.5, trunk_ratio=0.2),  # Bosque neblina (wind measured at 22m)
    3: dict(wind_h= 2.0, RGL=100.0, rad_atten=0.5, wind_atten=0.5, trunk_ratio=0.2),  # Pastizal Puna
    4: dict(wind_h= 2.0, RGL=65.0,  rad_atten=0.5, wind_atten=0.5, trunk_ratio=0.2),  # Agricultura
    5: dict(wind_h= 5.0, RGL=65.0,  rad_atten=0.5, wind_atten=0.5, trunk_ratio=0.2),  # Arbustos andinos
    6: dict(wind_h= 2.0, RGL=100.0, rad_atten=0.5, wind_atten=0.5, trunk_ratio=0.2),  # Páramo/Jalca
    7: dict(wind_h= 2.0, RGL=999.0, rad_atten=0.5, wind_atten=0.5, trunk_ratio=0.2),  # Suelo desnudo
    8: dict(wind_h= 2.0, RGL=999.0, rad_atten=0.5, wind_atten=0.5, trunk_ratio=0.2),  # Nieve/Glaciar
    9: dict(wind_h= 2.0, RGL=999.0, rad_atten=0.5, wind_atten=0.5, trunk_ratio=0.2),  # Agua
}


def prepare_soil_params(ds_domain: xr.Dataset) -> dict:
    """
    Preparar parámetros de suelo desde SoilGrids.

    Args:
        ds_domain: Dataset del dominio VIC

    Returns:
        Diccionario con arrays de parámetros de suelo
    """
    lats = ds_domain.lat.values
    lons = ds_domain.lon.values
    mask = ds_domain["mask"].values
    nlat, nlon = len(lats), len(lons)

    soil_file = GEE_DIR / "soil.nc"

    if soil_file.exists():
        print("[PARAMS] Cargando datos de suelo desde SoilGrids...")
        ds_soil = rename_xy(xr.open_dataset(soil_file))

        # Interpolar al grid del dominio
        ds_soil_interp = ds_soil.interp(lat=lats, lon=lons, method="linear")

        # Extraer y organizar por capas VIC
        # Capas SoilGrids: 0=0-5cm, 1=5-15cm, 2=15-30cm, 3=30-60cm, 4=60-100cm, 5=100-200cm
        # Capa VIC 1 (0-10cm): media de sg[0] y sg[1]
        # Capa VIC 2 (10-40cm): media de sg[2] y sg[3]
        # Capa VIC 3 (40-100cm): media de sg[4] y sg[5]

        def get_vic_layer(var_name, sg_layers):
            arrays = []
            for i in sg_layers:
                if "depth" in ds_soil_interp[var_name].dims:
                    arr = ds_soil_interp[var_name].isel(depth=i).values
                else:
                    arr = ds_soil_interp[var_name].values
                arrays.append(arr)
            return np.nanmean(arrays, axis=0)

        sand = np.stack([
            get_vic_layer("sand", [0, 1]),
            get_vic_layer("sand", [2, 3]),
            get_vic_layer("sand", [4, 5]),
        ])  # (3, nlat, nlon)

        clay = np.stack([
            get_vic_layer("clay", [0, 1]),
            get_vic_layer("clay", [2, 3]),
            get_vic_layer("clay", [4, 5]),
        ])

        bulk_density = np.stack([
            get_vic_layer("bdod", [0, 1]),
            get_vic_layer("bdod", [2, 3]),
            get_vic_layer("bdod", [4, 5]),
        ])

        ds_soil.close()

    else:
        print("[WARNING] SoilGrids no encontrado. Usando valores típicos para suelos andinos.")

        # Valores típicos para la cuenca Paucartambo
        # Zona alta (Puna): más arcilla
        # Zona baja (selva): más arena
        elev = ds_domain.get("elev", xr.DataArray(
            np.full((nlat, nlon), 2000.0), dims=["lat", "lon"]
        )).values

        # Gradiente textural con elevación
        clay_frac = np.clip(0.20 + (elev - 1000) / 10000, 0.15, 0.45)
        sand_frac = np.clip(0.50 - (elev - 1000) / 8000, 0.20, 0.60)
        bulk_den = np.full_like(elev, 1.3)  # g/cm3

        sand = np.stack([sand_frac] * 3)
        clay = np.stack([clay_frac] * 3)
        bulk_density = np.stack([bulk_den] * 3)

    # Reemplazar NaN con valores típicos
    sand = np.where(np.isnan(sand), 0.45, sand)
    clay = np.where(np.isnan(clay), 0.25, clay)
    bulk_density = np.where(np.isnan(bulk_density), 1.3, bulk_density)

    # Normalizar texturas (suma = 1)
    total = sand + clay + np.clip(1.0 - sand - clay, 0.01, 1.0)
    sand = sand / total
    clay = clay / total

    # Obtener elevación del dominio
    elev = ds_domain.get("elev", None)
    elev_arr = elev.values if elev is not None else None

    # Calcular parámetros VIC
    params = estimate_vic_soil_params(
        sand=sand,
        clay=clay,
        bulk_density=bulk_density,
        elev=elev_arr,
    )

    return params


def prepare_vegetation_params(
    ds_domain: xr.Dataset,
    config: dict,
) -> dict:
    """
    Preparar parámetros de vegetación desde cobertura ESA WorldCover.

    Returns:
        Diccionario con arrays de vegetación
    """
    lats = ds_domain.lat.values
    lons = ds_domain.lon.values
    nlat, nlon = len(lats), len(lons)

    lc_file = GEE_DIR / "landcover.nc"
    lai_file = GEE_DIR / "lai_monthly.nc"

    # ── Cobertura vegetal ──────────────────────────────
    if lc_file.exists():
        print("[PARAMS] Cargando cobertura vegetal (ESA WorldCover)...")
        ds_lc = rename_xy(xr.open_dataset(lc_file))
        ds_lc = ds_lc.interp(lat=lats, lon=lons, method="nearest")
    else:
        print("[WARNING] WorldCover no encontrado. Usando vegetación típica.")
        ds_lc = None

    # ── LAI mensual ────────────────────────────────────
    if lai_file.exists():
        print("[PARAMS] Cargando LAI mensual (MODIS)...")
        ds_lai = rename_xy(xr.open_dataset(lai_file))
        ds_lai = ds_lai.interp(lat=lats, lon=lons, method="linear")
    else:
        ds_lai = None

    # Inicializar arrays de vegetación
    # Cv[nveg, nlat, nlon]: fracción por clase
    Cv = np.zeros((NVEG, nlat, nlon), dtype=np.float32)
    LAI = np.zeros((NVEG, NMONTH, nlat, nlon), dtype=np.float32)
    albedo = np.zeros((NVEG, NMONTH, nlat, nlon), dtype=np.float32)
    veg_rough = np.zeros((NVEG, nlat, nlon), dtype=np.float32)
    displacement = np.zeros((NVEG, nlat, nlon), dtype=np.float32)
    root_depth = np.zeros((NVEG, NROOT_ZONES, nlat, nlon), dtype=np.float32)
    root_frac = np.zeros((NVEG, NROOT_ZONES, nlat, nlon), dtype=np.float32)
    overstory = np.zeros((NVEG, nlat, nlon), dtype=np.int32)
    rarc = np.zeros((NVEG, nlat, nlon), dtype=np.float32)
    rmin = np.zeros((NVEG, nlat, nlon), dtype=np.float32)

    # Rellenar valores por defecto desde VEG_PARAMS
    for vic_class_idx, (vic_class, vp) in enumerate(VEG_PARAMS.items()):
        for m in range(12):
            LAI[vic_class_idx, m, :, :] = vp["LAI"][m]
            albedo[vic_class_idx, m, :, :] = vp["albedo"][m]
        veg_rough[vic_class_idx] = vp["veg_rough"]
        displacement[vic_class_idx] = vp["displacement"]
        for rz in range(NROOT_ZONES):
            root_depth[vic_class_idx, rz] = vp["root_depth"][rz]
            root_frac[vic_class_idx, rz] = vp["root_frac"][rz]
        overstory[vic_class_idx] = vp["overstory"]
        rarc[vic_class_idx] = vp["rarc"]
        rmin[vic_class_idx] = vp["rmin"]

    # Distribución de Cv desde WorldCover
    if ds_lc is not None:
        for wc_class, vic_class in WORLDCOVER_TO_VIC.items():
            frac_var = f"frac_{wc_class}"
            if frac_var in ds_lc:
                vic_idx = vic_class - 1  # 0-indexed
                frac_data = ds_lc[frac_var].values
                frac_data = np.where(np.isnan(frac_data), 0, frac_data)
                Cv[vic_idx] += frac_data
        ds_lc.close()
    else:
        # Distribución típica para cuenca Paucartambo
        # Basada en elevación
        elev = ds_domain.get("elev", None)
        if elev is not None:
            e = elev.values
            Cv[0] = np.where(e < 1500, 0.8, 0)           # Bosque tropical bajo
            Cv[1] = np.where((e >= 1500) & (e < 3000), 0.7, 0)  # Bosque neblina
            Cv[2] = np.where((e >= 3000) & (e < 4200), 0.7, 0)  # Puna
            Cv[3] = np.where((e >= 1000) & (e < 3000), 0.15, 0)  # Agricultura
            Cv[4] = np.where((e >= 2000) & (e < 3500), 0.1, 0)  # Arbustos
            Cv[5] = np.where(e >= 4200, 0.5, 0)           # Páramo
            Cv[6] = np.where(e >= 4800, 0.3, 0)           # Suelo desnudo
        else:
            Cv[0] = 0.5   # Bosque por defecto
            Cv[1] = 0.2
            Cv[2] = 0.2
            Cv[4] = 0.1

    # Normalizar Cv (suma = 1 en cada celda)
    cv_sum = Cv.sum(axis=0, keepdims=True)
    cv_sum = np.where(cv_sum == 0, 1, cv_sum)
    Cv = Cv / cv_sum

    # Reemplazar LAI con MODIS si disponible
    if ds_lai is not None:
        print("[PARAMS] Usando LAI de MODIS...")
        for vic_class_idx in range(NVEG):
            for m in range(12):
                lai_var = f"LAI_month_{m+1:02d}"
                if lai_var in ds_lai:
                    lai_data = ds_lai[lai_var].values
                    lai_data = np.where(np.isnan(lai_data), LAI[vic_class_idx, m], lai_data)
                    LAI[vic_class_idx, m] = np.maximum(lai_data * Cv[vic_class_idx], 0.01)
        ds_lai.close()

    return {
        "Cv": Cv,
        "LAI": LAI,
        "albedo": albedo,
        "veg_rough": veg_rough,
        "displacement": displacement,
        "root_depth": root_depth,
        "root_frac": root_frac,
        "overstory": overstory,
        "rarc": rarc,
        "rmin": rmin,
    }


# ─────────────────────────────────────────────────────────
# Construir NetCDF de parámetros
# ─────────────────────────────────────────────────────────

def build_parameter_netcdf(
    ds_domain: xr.Dataset,
    soil_params: dict,
    veg_params: dict,
) -> xr.Dataset:
    """
    Construir Dataset xarray con todos los parámetros VIC.
    """
    lats = ds_domain.lat.values
    lons = ds_domain.lon.values
    mask = ds_domain["mask"].values
    nlat, nlon = len(lats), len(lons)

    coords = {
        "lat": (["lat"], lats, {"units": "degrees_north", "axis": "Y"}),
        "lon": (["lon"], lons, {"units": "degrees_east", "axis": "X"}),
        "nlayer": (["nlayer"], np.arange(NLAYER, dtype=np.int32)),
        "veg_class": (["veg_class"], np.arange(1, NVEG + 1, dtype=np.int32)),
        "month": (["month"], np.arange(1, 13, dtype=np.int32)),
        "root_zone": (["root_zone"], np.arange(NROOT_ZONES, dtype=np.int32)),
    }

    # Aplicar máscara: NaN fuera de la cuenca
    def mask_var(arr):
        arr = arr.copy().astype(np.float32)
        if arr.ndim == 2:
            arr[mask == 0] = np.nan
        elif arr.ndim == 3:
            arr[:, mask == 0] = np.nan
        elif arr.ndim == 4:
            arr[:, :, mask == 0] = np.nan
        return arr

    data_vars = {}

    # ── Parámetros de suelo ──────────────────────────────
    # Parámetros 2D (scalar por celda)
    scalar_params = ["infilt", "Ds", "Dsmax", "Ws", "c", "avg_T", "dp",
                     "off_gmt", "rough", "snow_rough", "fs_active"]
    for p in scalar_params:
        if p in soil_params:
            data_vars[p] = (
                ["lat", "lon"],
                mask_var(soil_params[p]).astype(np.float32),
                {"_FillValue": np.float32(-9999.0), "long_name": p}
            )

    # Parámetros por capa (3D: nlayer, lat, lon)
    layer_params = ["depth", "bulk_density", "soil_density", "Ksat",
                    "expt", "bubble", "quartz", "Wcr_FRACT", "Wpwp_FRACT",
                    "resid_moist", "init_moist"]
    for p in layer_params:
        if p in soil_params:
            data_vars[p] = (
                ["nlayer", "lat", "lon"],
                mask_var(soil_params[p]).astype(np.float32),
                {"_FillValue": np.float32(-9999.0), "long_name": p}
            )

    # Elevación
    if "elev" in ds_domain:
        data_vars["elev"] = (
            ["lat", "lon"],
            mask_var(ds_domain["elev"].values).astype(np.float32),
            {"units": "m", "_FillValue": np.float32(-9999.0), "long_name": "elevation"}
        )

    # ── Parámetros de vegetación ─────────────────────────
    # Cv: (nveg, nlat, nlon)
    # VIC Image Driver requires Cv > 0 for ALL Nveg tiles in every active cell.
    # Apply tiny floor (1e-6) to all active cells so VIC's check passes,
    # then renormalize so Cv still sums to 1.
    cv_data = veg_params["Cv"].copy()
    active_2d = (mask == 1)
    for v in range(NVEG):
        cv_data[v] = np.where(active_2d & (cv_data[v] < 1e-6), 1e-6, cv_data[v])
    cv_sum = cv_data.sum(axis=0, keepdims=True)
    cv_sum = np.where(cv_sum == 0, 1.0, cv_sum)
    cv_data = cv_data / cv_sum
    data_vars["Cv"] = (
        ["veg_class", "lat", "lon"],
        mask_var(cv_data).astype(np.float32),
        {"_FillValue": np.float32(-9999.0),
         "long_name": "Fractional coverage of vegetation class",
         "units": "1"}
    )

    # LAI: (nveg, 12, nlat, nlon)
    data_vars["LAI"] = (
        ["veg_class", "month", "lat", "lon"],
        veg_params["LAI"].astype(np.float32),
        {"_FillValue": np.float32(-9999.0),
         "long_name": "Monthly leaf area index",
         "units": "m2/m2"}
    )

    # albedo: (nveg, 12, nlat, nlon)
    data_vars["albedo"] = (
        ["veg_class", "month", "lat", "lon"],
        veg_params["albedo"].astype(np.float32),
        {"_FillValue": np.float32(-9999.0),
         "long_name": "Monthly albedo",
         "units": "1"}
    )

    # root_depth, root_frac: (nveg, nroot_zone, nlat, nlon)
    data_vars["root_depth"] = (
        ["veg_class", "root_zone", "lat", "lon"],
        veg_params["root_depth"].astype(np.float32),
        {"_FillValue": np.float32(-9999.0),
         "long_name": "Root zone depth",
         "units": "m"}
    )
    data_vars["root_fract"] = (
        ["veg_class", "root_zone", "lat", "lon"],
        veg_params["root_frac"].astype(np.float32),
        {"_FillValue": np.float32(-9999.0),
         "long_name": "Root zone fraction",
         "units": "1"}
    )

    # Parámetros de vegetación 2D por clase
    for p in ["overstory", "rarc", "rmin", "veg_rough", "displacement"]:
        if p in veg_params:
            data_vars[p] = (
                ["veg_class", "lat", "lon"],
                veg_params[p].astype(np.float32),
                {"_FillValue": np.float32(-9999.0), "long_name": p}
            )

    # ── run_cell: requerido por VIC Image Driver ─────────
    # 1 = celda activa, 0 = inactiva (igual a la máscara del dominio)
    data_vars["run_cell"] = (
        ["lat", "lon"],
        mask.astype(np.int32),
        {"long_name": "Run grid cell", "units": "1",
         "description": "1 = active cell, 0 = inactive cell"}
    )

    # ── gridcel: ID único por celda (nombre exacto que usa VIC Image Driver) ─
    gridcell_id = np.arange(1, nlat * nlon + 1, dtype=np.int32).reshape(nlat, nlon)
    data_vars["gridcel"] = (
        ["lat", "lon"],
        gridcell_id,
        {"long_name": "Grid cell number"}
    )

    # ── Nveg: número de clases de vegetación por celda ─────
    nveg_arr = np.where(mask == 1, NVEG, 0).astype(np.int32)
    data_vars["Nveg"] = (
        ["lat", "lon"],
        nveg_arr,
        {"long_name": "Nveg", "units": "N/A",
         "description": "Number of vegetation tiles in the grid cell"}
    )

    # ── Vegetation library params (per veg_class, lat, lon) ─────────────
    veg_lib_units = {
        "wind_h":     "m",
        "RGL":        "W/m2",
        "rad_atten":  "1",
        "wind_atten": "1",
        "trunk_ratio": "1",
    }
    for fname in ["wind_h", "RGL", "rad_atten", "wind_atten", "trunk_ratio"]:
        arr = np.zeros((NVEG, nlat, nlon), dtype=np.float32)
        for vic_class_idx, vic_class in enumerate(sorted(VEG_LIB_PARAMS.keys())):
            arr[vic_class_idx, :, :] = VEG_LIB_PARAMS[vic_class][fname]
        data_vars[fname] = (
            ["veg_class", "lat", "lon"],
            arr,
            {"_FillValue": np.float32(-9999.0),
             "units": veg_lib_units[fname],
             "long_name": fname}
        )

    # ── annual_prec: precipitación media anual (mm) ───────────────────
    # ~1500 mm/yr estimated for Paucartambo basin (tropical Andes)
    annual_prec = np.where(mask == 1, 1500.0, np.nan).astype(np.float32)
    data_vars["annual_prec"] = (
        ["lat", "lon"],
        annual_prec,
        {"_FillValue": np.float32(-9999.0),
         "units": "mm",
         "long_name": "Mean annual precipitation"}
    )

    # ── phi_s: parámetro de potencial mátrico del suelo (por capa) ────
    # Typical value -1.5 m for tropical Andean soils
    phi_s = np.full((NLAYER, nlat, nlon), -1.5, dtype=np.float32)
    phi_s[:, mask == 0] = np.nan
    data_vars["phi_s"] = (
        ["nlayer", "lat", "lon"],
        phi_s,
        {"_FillValue": np.float32(-9999.0),
         "units": "m",
         "long_name": "Soil matric potential parameter"}
    )

    # ── Construir Dataset ────────────────────────────────
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs = {
        "title": "VIC Parameter File - Cuenca Paucartambo, Pasco, Peru",
        "history": "Generado por 03_prepare_parameters.py",
        "conventions": "CF-1.6",
        "nlayer": NLAYER,
        "nveg": NVEG,
    }

    return ds


def main():
    print("=" * 60)
    print("  PREPARANDO PARÁMETROS VIC - Cuenca Paucartambo")
    print("=" * 60)

    config = load_config()
    ds_domain = load_domain()

    # 1. Parámetros de suelo
    print("\n[1/3] Parámetros de suelo...")
    soil_params = prepare_soil_params(ds_domain)

    # 2. Parámetros de vegetación
    print("\n[2/3] Parámetros de vegetación...")
    veg_params = prepare_vegetation_params(ds_domain, config)

    # 3. Construir NetCDF
    print("\n[3/3] Construyendo archivo de parámetros...")
    ds_params = build_parameter_netcdf(ds_domain, soil_params, veg_params)

    # Guardar — strip _FillValue from attrs to avoid conflict with encoding
    for var in ds_params.data_vars:
        ds_params[var].attrs.pop("_FillValue", None)
    output_path = PARAMS_DIR / "params.nc"
    encoding = {
        var: {"zlib": True, "complevel": 4, "_FillValue": -9999.0}
        for var in ds_params.data_vars
        if ds_params[var].dtype in [np.float32, np.float64]
    }
    ds_params.to_netcdf(output_path, encoding=encoding)

    print(f"\n[✓] Parámetros guardados: {output_path}")
    print(f"    Variables: {list(ds_params.data_vars)}")
    print(f"    Dimensiones: {dict(ds_params.dims)}")

    ds_domain.close()


if __name__ == "__main__":
    main()
