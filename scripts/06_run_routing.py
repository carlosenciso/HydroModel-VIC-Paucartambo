#!/usr/bin/env python3
"""
Script 06: Enrutamiento hidrológico con RVIC
============================================
Usa RVIC (Routing for VIC) para enrutar el escurrimiento superficial
y el flujo base de VIC y obtener caudales en la red de drenaje.

RVIC utiliza convolución de hidrógrafas unitarias basada en la
dirección de flujo D8 (codificación ARCMAP) y el tiempo de viaje.

Pasos:
  1. Generar flow_direction.nc en la misma grilla que domain.nc
  2. Generar archivos de forzantes anuales para RVIC
  3. Configurar y correr rvic.parameters (calcula hidrógrafas unitarias)
  4. Configurar y correr rvic.convolution (enruta el caudal)
  5. Guardar serie temporal en CSV y NetCDF

Uso:
    python scripts/06_run_routing.py
    python scripts/06_run_routing.py --start 2014 --end 2015
    python scripts/06_run_routing.py --method linear  # fallback simple
"""

import argparse
import configparser
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, label

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent.parent
CONFIG_FILE  = PROJECT_DIR / "config" / "basin_config.yml"
DATA_DIR     = PROJECT_DIR / "data"
DOMAIN_DIR   = DATA_DIR / "domain"
GEE_DIR      = DATA_DIR / "gee_raw"
OUTPUT_DIR   = DATA_DIR / "outputs"
ROUTING_DIR  = DATA_DIR / "routing"
ROUTING_DIR.mkdir(parents=True, exist_ok=True)

# ARCMAP D8 encoding (ESRI) used by RVIC
# {code: (dy, dx)} where dy: row offset (N=-1, S=+1), dx: col offset (W=-1, E=+1)
ARCMAP_D8 = {
    64:  (-1,  0),   # N
    128: (-1,  1),   # NE
    1:   ( 0,  1),   # E
    2:   ( 1,  1),   # SE
    4:   ( 1,  0),   # S
    8:   ( 1, -1),   # SW
    16:  ( 0, -1),   # W
    32:  (-1, -1),   # NW
}
ARCMAP_CODES = sorted(ARCMAP_D8.keys())


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def load_domain() -> xr.Dataset:
    return xr.open_dataset(DOMAIN_DIR / "domain.nc")


def load_vic_output() -> xr.Dataset:
    flux_files = sorted(OUTPUT_DIR.glob("fluxes*.nc"))
    if not flux_files:
        raise FileNotFoundError(f"No VIC outputs in {OUTPUT_DIR}. Run 05_run_vic.py first.")
    print(f"[ROUTING] Loading VIC outputs: {len(flux_files)} files")
    return xr.open_mfdataset(flux_files, combine="by_coords")


# ─────────────────────────────────────────────────────────
# Flow direction computation
# ─────────────────────────────────────────────────────────

def priority_flood_fill(elev: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Priority Flood pit-filling algorithm (Wang & Liu 2006).
    Ensures every active cell has at least one downslope neighbor.
    Returns filled DEM.
    """
    import heapq

    nlat, nlon = elev.shape
    filled  = elev.copy().astype(np.float64)
    filled[mask == 0] = 1e9   # non-active cells act as barriers
    in_open = np.zeros((nlat, nlon), dtype=bool)

    heap = []
    # Seed: all active boundary cells
    for r in range(nlat):
        for c in range(nlon):
            if mask[r, c] == 1:
                is_border = (r == 0 or r == nlat - 1 or
                             c == 0 or c == nlon - 1 or
                             any(mask[r+dr, c+dc] == 0
                                 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                 if 0 <= r+dr < nlat and 0 <= c+dc < nlon))
                if is_border:
                    heapq.heappush(heap, (filled[r, c], r, c))
                    in_open[r, c] = True

    eps = 1e-4  # small gradient to break ties
    while heap:
        elv, r, c = heapq.heappop(heap)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < nlat and 0 <= nc < nlon):
                continue
            if mask[nr, nc] == 0 or in_open[nr, nc]:
                continue
            in_open[nr, nc] = True
            # Raise cell if it's lower than current cell (fills pit)
            new_elv = max(filled[nr, nc], elv + eps)
            filled[nr, nc] = new_elv
            heapq.heappush(heap, (new_elv, nr, nc))

    return filled


def compute_d8_flow_direction(elev: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                               mask: np.ndarray) -> tuple:
    """
    Compute D8 flow direction (ARCMAP encoding) from pit-filled DEM on domain grid.

    Returns
    -------
    fdr : (nlat, nlon) int32 with ARCMAP codes (64/128/1/2/4/8/16/32)
    flow_dist : (nlat, nlon) float32, distance to next downstream cell (m)
    """
    nlat, nlon = elev.shape

    # Cell size in meters (approximate at basin center)
    lat_center = np.mean(lats)
    res_lat_m = abs(lats[0] - lats[1]) * 111_000.0
    res_lon_m = abs(lons[0] - lons[1]) * 111_000.0 * np.cos(np.radians(lat_center))

    # Distance to each neighbour
    dist = {
        64:  res_lat_m,                                   # N
        128: np.sqrt(res_lat_m**2 + res_lon_m**2),        # NE
        1:   res_lon_m,                                    # E
        2:   np.sqrt(res_lat_m**2 + res_lon_m**2),        # SE
        4:   res_lat_m,                                    # S
        8:   np.sqrt(res_lat_m**2 + res_lon_m**2),        # SW
        16:  res_lon_m,                                    # W
        32:  np.sqrt(res_lat_m**2 + res_lon_m**2),        # NW
    }

    # Fill pits using Priority Flood
    print("[FDR] Priority Flood pit-filling...")
    elev_filled = priority_flood_fill(elev.astype(np.float64), mask)

    fdr       = np.full((nlat, nlon), 4, dtype=np.int32)   # default: S
    flow_dist = np.zeros((nlat, nlon), dtype=np.float32)

    best_slope = np.full((nlat, nlon), -np.inf)
    best_code  = np.full((nlat, nlon), 4, dtype=np.int32)
    best_dist  = np.full((nlat, nlon), dist[4], dtype=np.float32)

    for code, (dy, dx) in ARCMAP_D8.items():
        # Rows/cols of source and neighbour
        r0, r1 = max(0, -dy), min(nlat, nlat - dy)
        c0, c1 = max(0,  -dx), min(nlon, nlon - dx)
        nr0, nr1 = r0 + dy, r1 + dy
        nc0, nc1 = c0 + dx, c1 + dx

        slope = (elev_filled[r0:r1, c0:c1] - elev_filled[nr0:nr1, nc0:nc1]) / dist[code]

        region = (slice(r0, r1), slice(c0, c1))
        mask_better = (slope > best_slope[region]) & (mask[r0:r1, c0:c1] == 1)
        best_code[region]  = np.where(mask_better, code,       best_code[region])
        best_slope[region] = np.where(mask_better, slope,      best_slope[region])
        best_dist[region]  = np.where(mask_better, dist[code], best_dist[region])

    fdr[mask == 1]       = best_code[mask == 1]
    flow_dist[mask == 1] = best_dist[mask == 1]

    return fdr, flow_dist, elev_filled


def compute_flow_accumulation(fdr: np.ndarray, mask: np.ndarray,
                               elev_filled: np.ndarray) -> np.ndarray:
    """Compute flow accumulation by routing cells in descending elevation order."""
    nlat, nlon = fdr.shape
    flow_acc = np.where(mask == 1, 1.0, 0.0).astype(np.float32)

    # Sort by descending elevation (process ridges first)
    sort_order = np.argsort(-elev_filled.ravel())
    rows, cols = np.unravel_index(sort_order, (nlat, nlon))
    for r, c in zip(rows, cols):
        if mask[r, c] == 0:
            continue
        code = int(fdr[r, c])
        dy, dx = ARCMAP_D8.get(code, (0, 0))
        nr, nc = r + dy, c + dx
        if 0 <= nr < nlat and 0 <= nc < nlon and mask[nr, nc] == 1:
            flow_acc[nr, nc] += flow_acc[r, c]

    return flow_acc


def compute_basin_ids_vectorized(fdr: np.ndarray, mask: np.ndarray,
                                  outlet_row: int, outlet_col: int) -> np.ndarray:
    """
    Label cells draining to outlet using vectorised upstream tracing.
    All cells in the upstream catchment get basin_id=1.
    Other connected components of active cells get unique IDs ≥ 2.
    """
    nlat, nlon = fdr.shape

    # Build downstream lookup: which (r,c) does each active cell drain to?
    # Build inflow array: for each cell, which cells flow into it?
    # Represent as sparse: inflow_count + iterative upstream marker.

    # Step 1: build "flows_to" arrays (vectorised)
    drain_r = np.full((nlat, nlon), -1, dtype=np.int32)
    drain_c = np.full((nlat, nlon), -1, dtype=np.int32)
    active = mask == 1
    for code, (dy, dx) in ARCMAP_D8.items():
        sel = active & (fdr == code)
        r_idx, c_idx = np.where(sel)
        nr = r_idx + dy
        nc = c_idx + dx
        valid = (nr >= 0) & (nr < nlat) & (nc >= 0) & (nc < nlon)
        drain_r[r_idx[valid], c_idx[valid]] = nr[valid]
        drain_c[r_idx[valid], c_idx[valid]] = nc[valid]

    # Step 2: Build inflow list using numpy
    # For each cell, accumulate list of cells that flow into it
    inflow_r = [[] for _ in range(nlat * nlon)]
    ar, ac = np.where(active)
    for r, c in zip(ar, ac):
        dr, dc = drain_r[r, c], drain_c[r, c]
        if dr >= 0 and dc >= 0:
            inflow_r[dr * nlon + dc].append(r * nlon + c)

    # Step 3: BFS from outlet
    basin_id = np.zeros(nlat * nlon, dtype=np.int32)
    outlet_flat = outlet_row * nlon + outlet_col
    basin_id[outlet_flat] = 1
    queue = [outlet_flat]
    while queue:
        next_queue = []
        for flat_idx in queue:
            for upstream_flat in inflow_r[flat_idx]:
                r2, c2 = divmod(upstream_flat, nlon)
                if basin_id[upstream_flat] == 0 and mask[r2, c2] == 1:
                    basin_id[upstream_flat] = 1
                    next_queue.append(upstream_flat)
        queue = next_queue

    basin_id_2d = basin_id.reshape(nlat, nlon)

    # Step 4: label remaining connected components
    labeled, n_comps = label(mask == 1)
    next_id = 2
    for lb in range(1, n_comps + 1):
        cells_mask = labeled == lb
        if not np.any(basin_id_2d[cells_mask] == 1):
            basin_id_2d[cells_mask] = next_id
            next_id += 1

    return basin_id_2d


def prepare_fdr_netcdf(ds_domain: xr.Dataset) -> Path:
    """
    Build and write flow_direction.nc on the domain grid.
    Returns the path to the written file.
    """
    fdr_nc = DOMAIN_DIR / "flow_direction.nc"

    lats = ds_domain.lat.values   # decreasing (N→S)
    lons = ds_domain.lon.values   # increasing (W→E)
    mask = ds_domain["mask"].values
    area = ds_domain["area"].values
    nlat, nlon = len(lats), len(lons)

    # ── 1. Load and interpolate DEM to domain grid ──────────────────
    dem_nc = GEE_DIR / "dem.nc"
    if not dem_nc.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_nc}")

    ds_dem = xr.open_dataset(dem_nc)
    dem_y   = ds_dem.y.values  # decreasing (N→S), shape 179
    dem_x   = ds_dem.x.values  # increasing (W→E), shape 224
    elev_raw = ds_dem["elevation"].values  # (179, 224)

    # RegularGridInterpolator requires strictly monotonic (either direction)
    # dem_y is decreasing → flip to increasing for the interpolator
    dem_y_inc = dem_y[::-1]
    elev_inc  = elev_raw[::-1, :]   # flip rows

    interp = RegularGridInterpolator(
        (dem_y_inc, dem_x),
        elev_inc,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    pts = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
    elev_domain = interp(pts).reshape(nlat, nlon)
    print(f"[FDR] DEM interpolated to domain grid: {elev_domain.shape}, "
          f"min={np.nanmin(elev_domain):.0f} m, max={np.nanmax(elev_domain):.0f} m")

    # ── 2. Compute D8 flow direction ─────────────────────────────────
    print("[FDR] Computing D8 flow direction (ARCMAP encoding)...")
    fdr, flow_dist, elev_filled = compute_d8_flow_direction(elev_domain, lats, lons, mask)

    # ── 3. Flow accumulation ─────────────────────────────────────────
    print("[FDR] Computing flow accumulation...")
    flow_acc = compute_flow_accumulation(fdr, mask, elev_filled)
    print(f"    Max accumulation: {flow_acc.max():.0f} cells")

    # ── 4. Snap outlet to high-accumulation cell ─────────────────────
    config = load_config()
    outlet = config["basin"]["outlet"]
    outlet_row = np.argmin(np.abs(lats - outlet["lat"]))
    outlet_col = np.argmin(np.abs(lons - outlet["lon"]))

    # Search 30-cell radius (~0.27°) for the highest accumulation (channel cell).
    # The Yuncan intake coordinates may be on a tributary; the main Paucartambo
    # river channel with the expected basin area is ~0.2° further east.
    search_r = 30
    best_acc = flow_acc[outlet_row, outlet_col]
    best_r, best_c = outlet_row, outlet_col
    for r in range(max(0, outlet_row-search_r), min(nlat, outlet_row+search_r+1)):
        for c in range(max(0, outlet_col-search_r), min(nlon, outlet_col+search_r+1)):
            if mask[r, c] == 1 and flow_acc[r, c] > best_acc:
                best_acc = flow_acc[r, c]
                best_r, best_c = r, c
    outlet_row, outlet_col = best_r, best_c
    print(f"    Outlet snapped to lat={lats[outlet_row]:.4f}, lon={lons[outlet_col]:.4f}, "
          f"acc={flow_acc[outlet_row, outlet_col]:.0f}")

    # ── 5. Compute basin IDs ─────────────────────────────────────────
    print("[FDR] Computing basin IDs (BFS from outlet)...")
    basin_id = compute_basin_ids_vectorized(fdr, mask, outlet_row, outlet_col)
    n_basin_cells = np.sum(basin_id == 1)
    print(f"    Basin cells draining to Yuncan: {n_basin_cells}"
          f" (~{n_basin_cells:.0f} km²)")

    # ── 4. Write NetCDF ──────────────────────────────────────────────
    ds_fdr = xr.Dataset(
        {
            "flow_direction": (["lat", "lon"], fdr.astype(np.int32),
                               {"long_name": "D8 flow direction (ARCMAP encoding)",
                                "units": "1",
                                "VIC_convention": "ARCMAP",
                                "_FillValue": np.int32(0)}),
            "flow_distance":  (["lat", "lon"], flow_dist,
                               {"long_name": "Distance to downstream cell",
                                "units": "m",
                                "_FillValue": np.float32(-9999.0)}),
            "basin_id":       (["lat", "lon"], basin_id.astype(np.int32),
                               {"long_name": "Basin identifier",
                                "units": "1",
                                "_FillValue": np.int32(0)}),
            "flow_accumulation": (["lat", "lon"], flow_acc,
                               {"long_name": "Flow accumulation (upstream cell count)",
                                "units": "1",
                                "_FillValue": np.float32(-9999.0)}),
            "elevation":      (["lat", "lon"], elev_domain.astype(np.float32),
                               {"long_name": "Surface elevation",
                                "units": "m",
                                "_FillValue": np.float32(-9999.0)}),
        },
        coords={"lat": lats, "lon": lons},
        attrs={
            "title": "Flow direction file for RVIC routing",
            "basin": "Paucartambo, Pasco, Peru",
            "outlet": f"Yuncan, lat={outlet['lat']}, lon={outlet['lon']}",
        }
    )
    ds_fdr.to_netcdf(fdr_nc)
    print(f"[FDR] Saved: {fdr_nc}")
    return fdr_nc


# ─────────────────────────────────────────────────────────
# RVIC forcing files
# ─────────────────────────────────────────────────────────

def prepare_rvic_forcings(ds_vic: xr.Dataset, year_start: int, year_end: int) -> list:
    """
    Create yearly NetCDF forcing files for RVIC from VIC outputs.
    File names: routing/rvic_input.<YYYY>.nc
    Variable: total_runoff (mm/day) = OUT_RUNOFF + OUT_BASEFLOW
    """
    files = []
    total_runoff = (ds_vic["OUT_RUNOFF"] + ds_vic["OUT_BASEFLOW"])
    total_runoff.name = "total_runoff"

    for year in range(year_start, year_end + 1):
        out_file = ROUTING_DIR / f"rvic_input.{year}.nc"
        yr_data = total_runoff.sel(time=str(year))
        if yr_data.time.size == 0:
            print(f"[WARNING] No VIC data for year {year}, skipping.")
            continue

        # Build Dataset with explicit time units + calendar (required by RVIC)
        from netCDF4 import Dataset as NC4Dataset, date2num
        from datetime import datetime

        times_np = yr_data.time.values
        times_dt = pd.to_datetime(times_np)

        # Use a fixed common epoch for ALL years so RVIC's units-match check passes
        time_units = f"days since {year_start}-01-01 00:00:00"
        calendar   = "proleptic_gregorian"

        time_vals = np.array([
            date2num(dt.to_pydatetime(), units=time_units, calendar=calendar)
            for dt in times_dt
        ], dtype=np.float64)

        lats = yr_data.lat.values
        lons = yr_data.lon.values
        runoff = yr_data.values.astype(np.float32)  # (time, lat, lon)

        with NC4Dataset(str(out_file), "w") as nc:
            nc.createDimension("time", None)
            nc.createDimension("lat", len(lats))
            nc.createDimension("lon", len(lons))

            # time variable
            tvar = nc.createVariable("time", "f8", ("time",))
            tvar.units    = time_units
            tvar.calendar = calendar
            tvar[:] = time_vals

            # lat / lon
            latv = nc.createVariable("lat", "f4", ("lat",))
            latv.units = "degrees_north"
            latv[:] = lats

            lonv = nc.createVariable("lon", "f4", ("lon",))
            lonv.units = "degrees_east"
            lonv[:] = lons

            # total runoff
            rvar = nc.createVariable("total_runoff", "f4", ("time", "lat", "lon"),
                                     fill_value=-9999.0)
            rvar.units     = "mm"
            rvar.long_name = "Total liquid runoff (surface + baseflow)"
            rvar[:] = runoff

        files.append(str(out_file))
        print(f"[RVIC] Forcing file: {out_file} "
              f"({times_dt[0].date()} – {times_dt[-1].date()}, "
              f"mean={float(np.nanmean(runoff)):.2f} mm/day)")

    return files


# ─────────────────────────────────────────────────────────
# RVIC config file writers
# ─────────────────────────────────────────────────────────

def write_rvic_params_config(
    fdr_nc: Path,
    pour_points_csv: Path,
    uh_box_csv: Path,
    rvic_config: dict,
) -> Path:
    """Write RVIC parameters config file (.cfg)."""
    params_case_dir = ROUTING_DIR / "rvic_params"
    temp_dir        = ROUTING_DIR / "rvic_temp"
    params_case_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # preserve case

    cfg["OPTIONS"] = {
        "LOG_LEVEL":          "INFO",
        "VERBOSE":            "False",
        "REMAP":              "False",
        "AGGREGATE":          "False",
        "SEARCH_FOR_CHANNEL": "False",
        "SUBSET_DAYS":        "0",           # 0 = no subsetting
        "CONSTRAIN_FRACTIONS": "False",
        "CLEAN":              "True",
        "GRIDID":             "paucartambo",
        "NETCDF_FORMAT":      "NETCDF4_CLASSIC",
        "CASE_DIR":           str(params_case_dir),
        "TEMP_DIR":           str(temp_dir),
        "CASEID":             "paucartambo",
        "CASESTR":            "Paucartambo basin",
        "CALENDAR":           "PROLEPTIC_GREGORIAN",
    }
    cfg["POUR_POINTS"] = {
        "FILE_NAME": str(pour_points_csv),
    }
    cfg["UH_BOX"] = {
        "FILE_NAME":    str(uh_box_csv),
        "HEADER_LINES": "1",
    }
    cfg["ROUTING"] = {
        "FILE_NAME":           str(fdr_nc),
        "FLOW_DIRECTION_VAR":  "flow_direction",
        "LATITUDE_VAR":        "lat",
        "LONGITUDE_VAR":       "lon",
        "BASIN_ID_VAR":        "basin_id",
        "FLOW_DISTANCE_VAR":   "flow_distance",
        "SOURCE_AREA_VAR":     "flow_accumulation",
        "VELOCITY":            str(rvic_config.get("velocity", 1.5)),
        "DIFFUSION":           str(rvic_config.get("diffusion", 2000.0)),
        "OUTPUT_INTERVAL":     "86400",
        "BASIN_FLOWDAYS":      str(rvic_config.get("max_day_uh", 50)),
        "CELL_FLOWDAYS":       "5",
    }
    cfg["DOMAIN"] = {
        "FILE_NAME":       str(DOMAIN_DIR / "domain.nc"),
        "LONGITUDE_VAR":   "lon",
        "LATITUDE_VAR":    "lat",
        "LAND_MASK_VAR":   "mask",
        "FRACTION_VAR":    "frac",   # param_file.py uses domain['FRACTION_VAR']
        "AREA_VAR":        "area",
    }

    cfg_path = ROUTING_DIR / "rvic_parameters.cfg"
    with open(cfg_path, "w") as f:
        cfg.write(f)
    print(f"[RVIC] Parameters config: {cfg_path}")
    return cfg_path


def write_rvic_convolution_config(
    params_file: Path,
    year_start: int,
    year_end: int,
) -> Path:
    """Write RVIC convolution config file (.cfg)."""
    conv_case_dir = ROUTING_DIR / "rvic_conv"
    conv_case_dir.mkdir(parents=True, exist_ok=True)

    cfg = configparser.ConfigParser()
    cfg.optionxform = str

    cfg["OPTIONS"] = {
        "LOG_LEVEL":      "INFO",
        "VERBOSE":        "False",
        "CASE_DIR":       str(conv_case_dir),
        "CASEID":         "paucartambo",
        "CASESTR":        "Paucartambo routing",
        "CALENDAR":       "PROLEPTIC_GREGORIAN",
        "RUN_TYPE":       "drystart",
        "RUN_STARTDATE":  f"{year_start}-01-01-00",  # TIMESTAMPFORM='%Y-%m-%d-%H'
        "STOP_OPTION":    "date",
        "STOP_N":         "-999",
        "STOP_DATE":      f"{year_end}-12-31",
        "REST_OPTION":    "date",
        "REST_N":         "-999",
        "REST_DATE":      f"{year_end}-12-31",
        "REST_NCFORM":    "NETCDF4_CLASSIC",
    }
    cfg["DOMAIN"] = {
        "FILE_NAME":     str(DOMAIN_DIR / "domain.nc"),
        "LONGITUDE_VAR": "lon",
        "LATITUDE_VAR":  "lat",
        "LAND_MASK_VAR": "mask",
        "FRACTION_VAR":  "frac",
        "AREA_VAR":      "area",
    }
    cfg["PARAM_FILE"] = {
        "FILE_NAME": str(params_file),
    }
    cfg["INPUT_FORCINGS"] = {
        "DATL_PATH":     str(ROUTING_DIR) + "/",
        "DATL_FILE":     "rvic_input.$YYYY.nc",
        "TIME_VAR":      "time",
        "LATITUDE_VAR":  "lat",
        "DATL_LIQ_FLDS": "total_runoff",
        "START":         str(year_start),
        "END":           str(year_end),
    }
    cfg["HISTORY"] = {
        "RVICHIST_NTAPES":   "1",
        "RVICHIST_MFILT":    str(365 * (year_end - year_start + 1) + 2),
        "RVICHIST_NDENS":    "1",
        "RVICHIST_NHTFRQ":   "-24",
        "RVICHIST_AVGFLAG":  "A",
        "RVICHIST_OUTTYPE":  "array",
        "RVICHIST_NCFORM":   "NETCDF4_CLASSIC",
        "RVICHIST_UNITS":    "m3/s",
    }

    cfg_path = ROUTING_DIR / "rvic_convolution.cfg"
    with open(cfg_path, "w") as f:
        cfg.write(f)
    print(f"[RVIC] Convolution config: {cfg_path}")
    return cfg_path


def create_uh_box(output_file: Path, dt_hours: int = 24) -> None:
    """UH box function: uniform input pulse at single 24h timestep.
    NOTE: RVIC expects time in seconds (find_ts computes input_interval in s).
    t_cell = CELL_FLOWDAYS * 86400 / input_interval, so if input_interval=86400
    and CELL_FLOWDAYS=5 → t_cell=5 (manageable). If hours were used instead,
    t_cell=18000 → memory explosion (~56 GB arrays).
    """
    dt_seconds = float(dt_hours) * 3600.0
    t = np.array([0.0, dt_seconds])
    uh = np.array([0.5, 0.5])
    df = pd.DataFrame({"time": t, "uh": uh})
    df.to_csv(output_file, index=False)


def create_pour_points(config: dict, fdr_nc: Path = None) -> Path:
    """Write pour_points.csv.

    If fdr_nc is provided, snap the pour point to the cell with the
    MAXIMUM flow_accumulation among cells with basin_id == 1.  This is
    always the delineated basin outlet, which is guaranteed to be valid
    for RVIC's search_catchment step.  The original config coordinates
    are used only if fdr_nc is not available.
    """
    outlet = config["basin"]["outlet"]
    gauges = config.get("routing", {}).get("gauges", [])
    points = []
    for g in gauges:
        points.append({"lons": g["lon"], "lats": g["lat"],
                       "names": g["name"].replace(" ", "_")})
    if not any(abs(p["lats"] - outlet["lat"]) < 0.01 and
               abs(p["lons"] - outlet["lon"]) < 0.01 for p in points):
        points.insert(0, {"lons": outlet["lon"], "lats": outlet["lat"],
                           "names": outlet["name"].replace(" ", "_")})

    if fdr_nc is not None and fdr_nc.exists():
        import xarray as xr
        fdr = xr.open_dataset(fdr_nc)
        facc = fdr["flow_accumulation"].values
        bid  = fdr["basin_id"].values
        lat_arr = fdr.lat.values
        lon_arr = fdr.lon.values

        snapped = []
        for p in points:
            # Find the cell with max accumulation within basin_id == 1.
            # This is the FDR outlet itself (the basin pour point is always
            # the highest-accumulation cell in the labeled basin).
            basin_mask = (bid == 1)
            if basin_mask.any():
                best_idx = int(np.nanargmax(np.where(basin_mask, facc, -1)))
                bi, bj = np.unravel_index(best_idx, facc.shape)
                slat, slon, acc = float(lat_arr[bi]), float(lon_arr[bj]), int(facc[bi, bj])
                print(f"[RVIC] Snapped '{p['names']}' "
                      f"({p['lats']:.4f},{p['lons']:.4f}) → "
                      f"({slat:.4f},{slon:.4f}) basin_outlet acc={acc}")
            else:
                slat, slon = p["lats"], p["lons"]
                print(f"[RVIC] No basin_id=1 cells found, using original coordinates")
            snapped.append({"lons": slon, "lats": slat, "names": p["names"]})
        points = snapped

    df = pd.DataFrame(points, columns=["lons", "lats", "names"])
    csv_path = ROUTING_DIR / "pour_points.csv"
    df.to_csv(csv_path, index=False)
    print(f"[RVIC] Pour points: {csv_path}")
    print(df.to_string(index=False))
    return csv_path


# ─────────────────────────────────────────────────────────
# Run RVIC
# ─────────────────────────────────────────────────────────

def run_rvic_parameters(cfg_path: Path) -> Path:
    """Run rvic.parameters, return path to generated params file."""
    from rvic.parameters import parameters
    print("[RVIC] Running rvic.parameters …")
    parameters(str(cfg_path), numofproc=1)

    # Find generated params file
    params_dir = ROUTING_DIR / "rvic_params" / "params"
    nc_files = sorted(params_dir.glob("*.rvic.prm.*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"RVIC params file not found in {params_dir}")
    params_file = nc_files[-1]
    print(f"[RVIC] Params file: {params_file}")
    return params_file


def run_rvic_convolution(cfg_path: Path) -> Path:
    """Run rvic.convolution, return path to history output."""
    from rvic.convolution import convolution
    print("[RVIC] Running rvic.convolution …")
    convolution(str(cfg_path))

    hist_dir = ROUTING_DIR / "rvic_conv" / "hist"
    nc_files = sorted(hist_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"RVIC convolution output not found in {hist_dir}")
    hist_file = nc_files[-1]
    print(f"[RVIC] History file: {hist_file}")
    return hist_file


def extract_streamflow_from_rvic(hist_file: Path, config: dict) -> pd.Series:
    """Extract streamflow time series from RVIC history output."""
    ds = xr.open_dataset(hist_file)
    print(f"[RVIC] History variables: {list(ds.data_vars)}")
    print(f"[RVIC] History dimensions: {dict(ds.dims)}")

    outlet = config["basin"]["outlet"]

    # RVIC history may have outlet_name dimension
    if "outlet_name" in ds.dims or "outlet" in ds.dims:
        sf = ds["streamflow"].isel(outlet=0) if "outlet" in ds.dims else ds["streamflow"].isel(outlet_name=0)
    elif "streamflow" in ds.data_vars:
        sf = ds["streamflow"]
        # If 2D (time, outlet), take first outlet
        if sf.ndim > 1:
            sf = sf.isel({d: 0 for d in sf.dims if d != "time"})
    else:
        raise KeyError("No 'streamflow' variable in RVIC output")

    times = pd.to_datetime(ds.time.values)
    series = pd.Series(sf.values, index=times, name="streamflow_m3s")
    return series


# ─────────────────────────────────────────────────────────
# Linear fallback
# ─────────────────────────────────────────────────────────

def linear_routing(ds_vic: xr.Dataset, ds_domain: xr.Dataset, config: dict,
                   velocity: float = 1.5) -> pd.Series:
    """Simple travel-time routing (fallback if RVIC fails)."""
    print("[ROUTING] Linear travel-time routing (fallback)…")
    outlet = config["basin"]["outlet"]
    mask   = ds_domain["mask"].values
    lats   = ds_domain.lat.values
    lons   = ds_domain.lon.values
    area   = ds_domain["area"].values

    outlet_idx = (np.argmin(np.abs(lats - outlet["lat"])),
                  np.argmin(np.abs(lons - outlet["lon"])))
    outlet_mask = np.zeros_like(mask, dtype=bool)
    outlet_mask[outlet_idx] = True
    dist_cells   = distance_transform_edt(~outlet_mask)
    res_deg      = abs(lats[1] - lats[0])
    cell_size_m  = res_deg * 111_000
    dist_m       = dist_cells * cell_size_m * 1.5  # tortuosity
    travel_days  = np.maximum(0, dist_m / (velocity * 86_400))

    total_mm = (ds_vic["OUT_RUNOFF"] + ds_vic["OUT_BASEFLOW"]).values
    times    = pd.to_datetime(ds_vic.time.values)
    ntime, nlat, nlon = total_mm.shape
    q_m3s    = total_mm * area[None] / (1000.0 * 86400.0)
    q_m3s    = np.where(mask[None] == 0, 0.0, q_m3s)

    max_lag  = int(travel_days[mask == 1].max()) + 2
    q_routed = np.zeros(ntime + max_lag)
    ai, aj   = np.where(mask == 1)
    for i, j in zip(ai, aj):
        lag = int(round(travel_days[i, j]))
        if lag < ntime + max_lag:
            q_routed[lag:lag + ntime] += q_m3s[:, i, j]

    streamflow = pd.Series(q_routed[:ntime], index=times, name="streamflow_m3s")
    return streamflow


# ─────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────

def save_routing_results(streamflow: pd.Series, config: dict,
                         method: str = "rvic") -> None:
    outlet = config["basin"]["outlet"]

    csv_file = ROUTING_DIR / "streamflow_yuncan.csv"
    df = streamflow.to_frame(name="streamflow_m3s")
    df.index.name = "date"
    df.to_csv(csv_file)
    print(f"[OK] CSV: {csv_file}")

    print(f"\n[ROUTING] {outlet['name']} statistics ({method}):")
    print(f"  Mean:       {streamflow.mean():.1f} m³/s")
    print(f"  Median:     {streamflow.median():.1f} m³/s")
    print(f"  Maximum:    {streamflow.max():.1f} m³/s  ({streamflow.idxmax().date()})")
    print(f"  Minimum:    {streamflow.min():.3f} m³/s  ({streamflow.idxmin().date()})")
    print(f"  Q95:        {streamflow.quantile(0.95):.1f} m³/s")
    print(f"  Q05:        {streamflow.quantile(0.05):.1f} m³/s")

    ds_out = xr.Dataset(
        {"streamflow": (["time"], streamflow.values.astype(np.float32),
                        {"units": "m3 s-1",
                         "long_name": f"Streamflow at {outlet['name']}",
                         "_FillValue": np.float32(-9999.0)})},
        coords={"time": streamflow.index},
        attrs={"routing_method": method,
               "outlet_lat": outlet["lat"],
               "outlet_lon": outlet["lon"]},
    )
    nc_file = ROUTING_DIR / "streamflow_yuncan.nc"
    ds_out.to_netcdf(nc_file)
    print(f"[OK] NetCDF: {nc_file}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def patch_rvic_compatibility():
    """
    Patch RVIC source files for compatibility with:
    - pandas >= 1.0: `DataFrame.ix` removed → use `DataFrame.loc`
    - NumPy >= 1.24: `np.float` removed → use `float`
    """
    import importlib
    patches = {
        "rvic.parameters": [
            ("pour_points.ix[",          "pour_points.loc["),
            ("np.finfo(np.float).",       "np.finfo(float)."),
        ],
        "rvic.core.param_file": [
            ("np.finfo(np.float).",       "np.finfo(float)."),
            # outlet_name dtype='S...' (bytes) -> 'U...' (unicode) for netCDF4 stringtochar
            ("dtype='S{0}'.format(MAX_NC_CHARS)",
             "dtype='U{0}'.format(MAX_NC_CHARS)"),
        ],
        # valid_range='0, 86400' (string) → numeric list for NC_DOUBLE variables
        "rvic.core.share": [
            ("valid_range='0, 86400'", "valid_range=[0.0, 86400.0]"),
        ],
        # variables.py: stringtochar expects Unicode in new netCDF4 but bytes in old.
        # Use np.frombuffer to get exact MAX_NC_CHARS byte chars directly.
        "rvic.core.variables": [
            ("stringtochar(np.array(b_string.ljust(MAX_NC_CHARS)))",
             "np.frombuffer(b_string[:MAX_NC_CHARS].ljust(MAX_NC_CHARS), dtype='S1')"),
        ],
        # history.py: _outlet_name is already a char array from params file;
        # newer netCDF4's stringtochar can't handle MaskedArray of bytes.
        "rvic.core.history": [
            ("char_names = stringtochar(self._outlet_name)",
             "char_names = np.ma.filled(self._outlet_name, fill_value=b' ') "
             "if hasattr(self._outlet_name, 'filled') else self._outlet_name"),
        ],
    }
    for mod_name, replacements in patches.items():
        try:
            mod = importlib.import_module(mod_name)
            rvic_file = mod.__file__
            with open(rvic_file, "r") as f:
                src = f.read()
            changed = False
            for old, new in replacements:
                if old in src:
                    src = src.replace(old, new)
                    changed = True
                    print(f"[RVIC] Patched '{old}' in {mod_name}")
            if changed:
                with open(rvic_file, "w") as f:
                    f.write(src)
                importlib.reload(mod)
        except Exception as e:
            print(f"[WARNING] Could not patch {mod_name}: {e}")

    # Reload higher-level modules that import patched sub-modules,
    # so they pick up the new class definitions.
    for top_mod in ["rvic.convolution", "rvic.parameters"]:
        try:
            importlib.reload(importlib.import_module(top_mod))
        except Exception as e:
            print(f"[WARNING] Could not reload {top_mod}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="RVIC routing – Cuenca Paucartambo"
    )
    parser.add_argument("--method", choices=["rvic", "linear"], default="rvic")
    parser.add_argument("--start", type=int, default=2003)
    parser.add_argument("--end",   type=int, default=2025)
    parser.add_argument("--velocity", type=float, default=1.5)
    parser.add_argument("--skip-fdr", action="store_true",
                        help="Reuse existing flow_direction.nc")
    parser.add_argument("--skip-params", action="store_true",
                        help="Reuse existing RVIC params file")
    args = parser.parse_args()

    print("=" * 60)
    print("  RVIC ROUTING – Cuenca Paucartambo")
    print("=" * 60)

    config    = load_config()
    ds_domain = load_domain()
    ds_vic    = load_vic_output()
    print(f"\n  VIC period: {pd.to_datetime(ds_vic.time.values[0]).date()} "
          f"→ {pd.to_datetime(ds_vic.time.values[-1]).date()}")
    print(f"  VIC variables: {list(ds_vic.data_vars)}")

    rvic_config = config.get("routing", {}).get("rvic", {})

    if args.method == "linear":
        print("\n[→] Linear routing…")
        sf = linear_routing(ds_vic, ds_domain, config, args.velocity)
        save_routing_results(sf, config, method="linear")
        print(f"\n[✓] Done. Results in {ROUTING_DIR}")
        return

    # ── RVIC path ────────────────────────────────────────────────────
    try:
        import rvic
        print(f"\n[RVIC] version {rvic.__version__}")
        patch_rvic_compatibility()
    except ImportError:
        print("[WARNING] RVIC not installed, falling back to linear routing.")
        sf = linear_routing(ds_vic, ds_domain, config, args.velocity)
        save_routing_results(sf, config, method="linear_fallback")
        return

    # 1. Flow direction NetCDF
    fdr_nc = DOMAIN_DIR / "flow_direction.nc"
    if args.skip_fdr and fdr_nc.exists():
        print(f"[→] Reusing existing FDR file: {fdr_nc}")
    else:
        print("\n[→] Building flow direction NetCDF…")
        fdr_nc = prepare_fdr_netcdf(ds_domain)

    # 2. Pour points & UH box (snap to nearest high-accumulation cell)
    pour_points_csv = create_pour_points(config, fdr_nc=fdr_nc)
    uh_box_csv      = ROUTING_DIR / "uh_box.csv"
    create_uh_box(uh_box_csv, dt_hours=int(rvic_config.get("unit_hydrograph_dt", 6)))

    # 3. RVIC forcing files
    print("\n[→] Preparing RVIC forcing files…")
    prepare_rvic_forcings(ds_vic, args.start, args.end)

    # 4. RVIC parameters
    params_dir = ROUTING_DIR / "rvic_params" / "params"
    existing_params = sorted(params_dir.glob("*.rvic.prm.*.nc")) if params_dir.exists() else []
    if args.skip_params and existing_params:
        params_file = existing_params[-1]
        print(f"[→] Reusing existing params file: {params_file}")
    else:
        print("\n[→] Running RVIC parameters…")
        params_cfg = write_rvic_params_config(
            fdr_nc, pour_points_csv, uh_box_csv, rvic_config
        )
        try:
            params_file = run_rvic_parameters(params_cfg)
        except Exception as e:
            print(f"[ERROR] RVIC parameters failed: {e}")
            import traceback; traceback.print_exc()
            print("[FALLBACK] Using linear routing.")
            sf = linear_routing(ds_vic, ds_domain, config, args.velocity)
            save_routing_results(sf, config, method="linear_fallback")
            return

    # 5. RVIC convolution
    print("\n[→] Running RVIC convolution…")
    conv_cfg = write_rvic_convolution_config(params_file, args.start, args.end)
    try:
        hist_file = run_rvic_convolution(conv_cfg)
        sf = extract_streamflow_from_rvic(hist_file, config)
    except Exception as e:
        print(f"[ERROR] RVIC convolution failed: {e}")
        import traceback; traceback.print_exc()
        print("[FALLBACK] Using linear routing.")
        sf = linear_routing(ds_vic, ds_domain, config, args.velocity)
        save_routing_results(sf, config, method="linear_fallback")
        return

    save_routing_results(sf, config, method="rvic")
    ds_domain.close()
    ds_vic.close()
    print(f"\n[✓] Routing completed. Results in {ROUTING_DIR}")


if __name__ == "__main__":
    main()
