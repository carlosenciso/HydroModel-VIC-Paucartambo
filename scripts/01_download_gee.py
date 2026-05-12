#!/usr/bin/env python3
"""
Script 01: Descarga de datos desde Google Earth Engine
=======================================================
Descarga todos los datos necesarios para la simulación VIC:
  - DEM (SRTM 30m → 1km)
  - Textura de suelo (SoilGrids 250m → 1km)
  - Cobertura vegetal (ESA WorldCover 10m → 1km)
  - LAI mensual (MODIS 500m → 1km)
  - Forzantes meteorológicos (ERA5-Land + CHIRPS diario)

Uso:
    export EE_SERVICE_ACCOUNT_JSON_B64=$(cat /path/to/key_b64.txt)
    python scripts/01_download_gee.py --start 2010-01-01 --end 2020-12-31
"""

import argparse
import sys
import os
from pathlib import Path

import ee
import wxee
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

# Agregar utils al path
sys.path.insert(0, str(Path(__file__).parent))
from utils.gee_auth import authenticate, get_basin_geometry
from utils.vic_utils import kelvin_to_celsius, pa_to_kpa, joules_to_watts, \
    dewpoint_to_vp, wind_components_to_speed


# ─────────────────────────────────────────────────────────
# Configuración de rutas
# ─────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_DIR / "config" / "basin_config.yml"
DATA_DIR = PROJECT_DIR / "data"
GEE_DIR = DATA_DIR / "gee_raw"
GEE_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────
# 1. DEM (SRTM 30m)
# ─────────────────────────────────────────────────────────

def static_col(img: ee.Image) -> ee.ImageCollection:
    """
    Envuelve una imagen estática en un ImageCollection con system:time_start,
    que wxee requiere para descargar como xarray.
    """
    return ee.ImageCollection([
        img.set("system:time_start", ee.Date("2000-01-01").millis())
    ])


def download_static(
    img: ee.Image,
    geometry: ee.Geometry,
    scale: int,
    band_name: str,
) -> np.ndarray:
    """
    Descarga una imagen estática de GEE como array numpy.
    Usa wxee con system:time_start forzado.
    """
    col = static_col(img.rename(band_name))
    ts = wxee.TimeSeries(col)
    ds = ts.wx.to_xarray(region=geometry, scale=scale, crs="EPSG:4326", progress=False)
    return ds[band_name].squeeze(drop=True)


def download_dem(geometry: ee.Geometry, scale: int = 1000) -> xr.Dataset:
    """
    Descarga DEM de SRTM a resolución objetivo.

    Args:
        geometry: Geometría de la cuenca
        scale: Resolución en metros (1000 = ~1km)

    Returns:
        xarray.Dataset con variable 'elevation'
    """
    print("[GEE] Descargando DEM (SRTM 30m)...")

    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    terrain = ee.Algorithms.Terrain(dem)

    elev  = download_static(dem.resample("bilinear"),          geometry, scale, "elevation")
    slope = download_static(terrain.select("slope"),           geometry, scale, "slope")
    aspect= download_static(terrain.select("aspect"),          geometry, scale, "aspect")

    result = xr.Dataset({
        "elevation": elev,
        "slope":     slope,
        "aspect":    aspect,
    })

    output_path = GEE_DIR / "dem.nc"
    result.to_netcdf(output_path)
    print(f"[OK] DEM guardado en {output_path}")
    return result


# ─────────────────────────────────────────────────────────
# 2. Suelo - SoilGrids
# ─────────────────────────────────────────────────────────

def download_soil(geometry: ee.Geometry, scale: int = 1000) -> xr.Dataset:
    """
    Descarga propiedades del suelo desde OpenLandMap SOL (250m).
    Fuente alternativa a SoilGrids con acceso público en GEE.

    Bandas por profundidad: b0=0cm, b10=10cm, b30=30cm, b60=60cm, b100=100cm, b200=200cm
    Capas VIC:
      Capa 1 (0-10cm):  promedio b0 + b10
      Capa 2 (10-40cm): promedio b30
      Capa 3 (40-100cm): promedio b60 + b100
    """
    print("[GEE] Descargando propiedades de suelo (OpenLandMap SOL 250m)...")

    # OpenLandMap - datasets de suelo con acceso público garantizado en GEE
    soil_assets = {
        "sand": "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02",
        "clay": "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02",
        "bdod": "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02",
    }
    # Bandas disponibles en OpenLandMap SOL (profundidades en cm)
    depth_bands = ["b0", "b10", "b30", "b60", "b100", "b200"]

    datasets = {}
    for var_name, asset_id in soil_assets.items():
        print(f"  Descargando {var_name}...")
        layers = []
        img_full = ee.Image(asset_id)
        for band in depth_bands:
            try:
                da = download_static(
                    img_full.select(band).rename(var_name),
                    geometry, scale, var_name
                )
                layers.append(da)
            except Exception as e:
                print(f"  [WARNING] {var_name}/{band}: {e}")
                layers.append(None)

        valid = [l for l in layers if l is not None]
        if valid:
            stacked = xr.concat(valid, dim="depth")
            stacked["depth"] = range(len(valid))
            datasets[var_name] = stacked

    ds_soil = xr.Dataset(datasets)

    # OpenLandMap: sand/clay en g/kg → fracción (dividir por 1000)
    for v in ["sand", "clay"]:
        if v in ds_soil:
            ds_soil[v] = ds_soil[v] / 1000.0

    # bdod: kg/dm3 → g/cm3 (ya en unidades correctas aprox.)
    if "bdod" in ds_soil:
        ds_soil["bdod"] = ds_soil["bdod"] / 100.0

    # Derivar silt = 1 - sand - clay (valores entre 0 y 1)
    if "sand" in ds_soil and "clay" in ds_soil:
        ds_soil["silt"] = (1.0 - ds_soil["sand"] - ds_soil["clay"]).clip(0, 1)

    output_path = GEE_DIR / "soil.nc"
    ds_soil.to_netcdf(output_path)
    print(f"[OK] Suelo guardado en {output_path}")
    return ds_soil


# ─────────────────────────────────────────────────────────
# 3. Cobertura Vegetal (ESA WorldCover)
# ─────────────────────────────────────────────────────────

def download_landcover(geometry: ee.Geometry, scale: int = 1000) -> xr.Dataset:
    """
    Descarga cobertura terrestre desde MODIS MCD12Q1 (500m, anual).
    Usa clasificación IGBP (Type1). Más estable y accesible que ESA WorldCover.
    Mapeo IGBP → clases VIC incluido en config/basin_config.yml.
    """
    print("[GEE] Descargando cobertura vegetal (MODIS MCD12Q1 2015)...")

    lc = (
        ee.ImageCollection("MODIS/061/MCD12Q1")
        .filterDate("2015-01-01", "2015-12-31")
        .first()
        .select("LC_Type1")   # Clasificación IGBP
    )

    # Descargar clase dominante (bilinear a 1km)
    lc_class_da = download_static(lc.rename("lc_class"), geometry, scale, "lc_class")

    # Fracciones por clase IGBP (1-17)
    igbp_classes = list(range(1, 18))
    fractions = {"lc_class": lc_class_da}

    for cls in igbp_classes:
        try:
            mask = lc.eq(cls).rename(f"frac_{cls}")
            fractions[f"frac_{cls}"] = download_static(
                mask, geometry, scale, f"frac_{cls}"
            )
        except Exception as e:
            print(f"  [WARNING] Clase IGBP {cls}: {e}")

    ds_lc = xr.Dataset(fractions)

    output_path = GEE_DIR / "landcover.nc"
    ds_lc.to_netcdf(output_path)
    print(f"[OK] Cobertura vegetal guardada en {output_path}")
    return ds_lc


# ─────────────────────────────────────────────────────────
# 4. LAI mensual (MODIS MOD15A2H)
# ─────────────────────────────────────────────────────────

def download_lai(geometry: ee.Geometry, scale: int = 1000) -> xr.Dataset:
    """
    Descarga LAI mensual de MODIS MOD15A2H (500m, 8-day).
    Calcula climatología mensual (2010-2020).
    """
    print("[GEE] Descargando LAI MODIS (climatología 2010-2020)...")

    lai_col = (
        ee.ImageCollection("MODIS/061/MOD15A2H")
        .filterDate("2010-01-01", "2020-12-31")
        .filterBounds(geometry)
        .select("Lai_500m")
    )

    # Climatología mensual — se agrega cada mes por separado con download_static
    # para evitar perder system:time_start al usar .mean()
    monthly_arrays = {}
    for month in range(1, 13):
        band = f"LAI_month_{month:02d}"
        monthly_img = (
            lai_col.filter(ee.Filter.calendarRange(month, month, "month"))
            .mean()
            .multiply(0.1)
            .rename(band)
        )
        try:
            da = download_static(monthly_img, geometry, scale, band)
            monthly_arrays[band] = da
        except Exception as e:
            print(f"  [WARNING] LAI mes {month}: {e}")

    ds = xr.Dataset(monthly_arrays)

    output_path = GEE_DIR / "lai_monthly.nc"
    ds.to_netcdf(output_path)
    print(f"[OK] LAI guardado en {output_path}")
    return ds


# ─────────────────────────────────────────────────────────
# 5. Forzantes Meteorológicos
# ─────────────────────────────────────────────────────────

def download_forcings(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    scale: int = 1000,
    chunk_months: int = 3,
) -> None:
    """
    Descarga forzantes meteorológicos diarios desde GEE.

    Fuentes:
        - ERA5-Land: Temperatura, viento, radiación, presión, punto de rocío
        - CHIRPS: Precipitación

    Descarga en chunks trimestrales para respetar el límite de 50 MB de GEE.
    Cada variable de ERA5 se descarga por separado para reducir el tamaño de
    cada solicitud. Los chunks trimestrales se concatenan y se guarda un único
    archivo anual por año.
    """
    import pandas as pd

    print(f"[GEE] Descargando forzantes: {start_date} a {end_date}...")
    forcing_dir = DATA_DIR / "gee_raw" / "forcings"
    forcing_dir.mkdir(parents=True, exist_ok=True)

    # Variables ERA5 a descargar — agrupadas para minimizar el nº de requests
    # sin exceder 50 MB por request. Cada grupo se descarga en una llamada wxee.
    era5_groups = {
        "temp":   ["temperature_2m_max", "temperature_2m_min"],
        "wind":   ["u_component_of_wind_10m", "v_component_of_wind_10m"],
        "rad":    ["surface_solar_radiation_downwards_sum",
                   "surface_thermal_radiation_downwards_sum"],
        "pres":   ["surface_pressure", "dewpoint_temperature_2m"],
    }

    def _download_quarter(q_start: str, q_end: str) -> xr.Dataset:
        """
        Descarga un trimestre de ERA5 + CHIRPS y devuelve un xr.Dataset procesado.
        Cada grupo de ERA5 se descarga por separado y luego se fusionan.
        """
        datasets_era5 = []
        for group_name, band_list in era5_groups.items():
            era5_q = (
                ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
                .filterDate(q_start, q_end)
                .filterBounds(geometry)
                .select(band_list)
            )
            ts = wxee.TimeSeries(era5_q)
            ds_g = ts.wx.to_xarray(
                region=geometry, scale=scale, crs="EPSG:4326", progress=False
            )
            datasets_era5.append(ds_g)

        ds_era5 = xr.merge(datasets_era5)

        # CHIRPS
        chirps_q = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(q_start, q_end)
            .filterBounds(geometry)
        )
        ds_chirps = wxee.TimeSeries(chirps_q).wx.to_xarray(
            region=geometry, scale=scale, crs="EPSG:4326", progress=False
        )
        if "precipitation" in ds_chirps:
            ds_chirps = ds_chirps.rename({"precipitation": "PREC"})

        # Construir dataset de salida con unidades VIC
        ds_out = xr.Dataset()
        prcp_var = "PREC" if "PREC" in ds_chirps else list(ds_chirps.data_vars)[0]
        ds_out["PREC"] = ds_chirps[prcp_var]

        if "temperature_2m_max" in ds_era5:
            ds_out["TMAX"] = kelvin_to_celsius(ds_era5["temperature_2m_max"])
            ds_out["TMIN"] = kelvin_to_celsius(ds_era5["temperature_2m_min"])
            ds_out["AIR_TEMP"] = (ds_out["TMAX"] + ds_out["TMIN"]) / 2.0

        if "u_component_of_wind_10m" in ds_era5:
            ds_out["WIND"] = wind_components_to_speed(
                ds_era5["u_component_of_wind_10m"],
                ds_era5["v_component_of_wind_10m"],
            )

        if "surface_solar_radiation_downwards_sum" in ds_era5:
            ds_out["SHORTWAVE"] = joules_to_watts(
                ds_era5["surface_solar_radiation_downwards_sum"]
            )
            ds_out["LONGWAVE"] = joules_to_watts(
                ds_era5["surface_thermal_radiation_downwards_sum"]
            )

        if "surface_pressure" in ds_era5:
            ds_out["PRESSURE"] = pa_to_kpa(ds_era5["surface_pressure"])

        if "dewpoint_temperature_2m" in ds_era5:
            ds_out["VP"] = dewpoint_to_vp(ds_era5["dewpoint_temperature_2m"])

        # Clip físico
        ds_out["PREC"]      = ds_out["PREC"].clip(min=0)
        ds_out["WIND"]      = ds_out["WIND"].clip(min=0.1, max=50)
        ds_out["SHORTWAVE"] = ds_out["SHORTWAVE"].clip(min=0)
        ds_out["LONGWAVE"]  = ds_out["LONGWAVE"].clip(min=0)
        ds_out["PRESSURE"]  = ds_out["PRESSURE"].clip(min=30, max=110)
        ds_out["VP"]        = ds_out["VP"].clip(min=0.01, max=10)

        return ds_out

    def download_yearly_chunk(year: int) -> None:
        output_file = forcing_dir / f"era5_chirps_{year}.nc"
        if output_file.exists():
            print(f"  [SKIP] {year} ya existe.")
            return

        print(f"  Descargando año {year} en chunks trimestrales...")

        # Dividir el año en trimestres
        quarters = [
            (f"{year}-01-01", f"{year}-03-31"),
            (f"{year}-04-01", f"{year}-06-30"),
            (f"{year}-07-01", f"{year}-09-30"),
            (f"{year}-10-01", f"{year}-12-31"),
        ]

        quarterly_ds = []
        for q_idx, (q_start, q_end) in enumerate(quarters, 1):
            print(f"    Q{q_idx}: {q_start} → {q_end}")
            try:
                ds_q = _download_quarter(q_start, q_end)
                quarterly_ds.append(ds_q)
            except Exception as e:
                print(f"    [ERROR] Q{q_idx} ({q_start}–{q_end}): {e}")
                return

        if not quarterly_ds:
            print(f"  [ERROR] No se pudo descargar ningún trimestre de {year}.")
            return

        # Concatenar trimestres a lo largo del tiempo
        try:
            ds_year = xr.concat(quarterly_ds, dim="time")
            ds_year = ds_year.sortby("time")
            ds_year.to_netcdf(output_file)
            print(f"  [OK] {year} guardado: {output_file.name}")
        except Exception as e:
            print(f"  [ERROR] Concatenando {year}: {e}")

    years = range(int(start_date[:4]), int(end_date[:4]) + 1)
    for year in tqdm(years, desc="Descargando forzantes anuales"):
        download_yearly_chunk(year)

    print(f"[OK] Forzantes guardados en {forcing_dir}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Descargar datos de GEE para simulación VIC - Cuenca Paucartambo"
    )
    parser.add_argument("--start", default="2010-01-01", help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", default="2020-12-31", help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument("--scale", type=int, default=1000, help="Resolución en metros")
    parser.add_argument("--only", nargs="+",
                        choices=["dem", "soil", "landcover", "lai", "forcings", "all"],
                        default=["all"],
                        help="Qué datos descargar")
    args = parser.parse_args()

    # 1. Autenticar GEE
    authenticate()

    # 2. Cargar configuración
    config = load_config()
    geometry = get_basin_geometry(config)
    scale = args.scale

    download_all = "all" in args.only

    # 3. Descargar datos estáticos
    if download_all or "dem" in args.only:
        download_dem(geometry, scale=scale)

    if download_all or "soil" in args.only:
        download_soil(geometry, scale=scale)

    if download_all or "landcover" in args.only:
        download_landcover(geometry, scale=scale)

    if download_all or "lai" in args.only:
        download_lai(geometry, scale=scale)

    # 4. Descargar forzantes temporales
    if download_all or "forcings" in args.only:
        download_forcings(
            geometry=geometry,
            start_date=args.start,
            end_date=args.end,
            scale=scale,
        )

    print("\n[✓] Descarga de datos GEE completada.")
    print(f"    Datos guardados en: {GEE_DIR}")


if __name__ == "__main__":
    main()
