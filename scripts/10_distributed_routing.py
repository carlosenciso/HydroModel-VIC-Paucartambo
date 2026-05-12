#!/usr/bin/env python3
"""
Script 10: Routing distribuido sobre toda la grilla VIC
========================================================
Calcula el caudal acumulado (m³/s) en cada celda activa del dominio
usando la dirección de flujo D8 y el runoff de VIC.

Algoritmo:
  1. Ordenar celdas topológicamente (cabeceras → outlet) con Kahn
  2. Inicializar discharge[t, r, c] = (OUT_RUNOFF + OUT_BASEFLOW) × area / 1000 / 86400
  3. Propagar aguas abajo: discharge[t, dr, dc] += discharge[t, r, c]
  Un solo pase sobre las ~39k celdas por chunk temporal.

Output: data/routing/discharge_grid.nc
  - Variable: discharge (m³/s), dims (time, lat, lon)
  - Celdas inactivas: NaN

Uso:
    python scripts/10_distributed_routing.py
    python scripts/10_distributed_routing.py --start 2003 --end 2025
    python scripts/10_distributed_routing.py --start 2003 --end 2025 --chunk-years 5
"""

import argparse
import sys
import warnings
from pathlib import Path
from collections import deque

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"
DOMAIN_DIR  = DATA_DIR / "domain"
OUTPUT_DIR  = DATA_DIR / "outputs"
ROUTING_DIR = DATA_DIR / "routing"
ROUTING_DIR.mkdir(parents=True, exist_ok=True)

# ARCMAP D8: {code: (dy, dx)}  dy>0 = sur, dx>0 = este
ARCMAP_D8 = {
    64:  (-1,  0),
    128: (-1,  1),
    1:   ( 0,  1),
    2:   ( 1,  1),
    4:   ( 1,  0),
    8:   ( 1, -1),
    16:  ( 0, -1),
    32:  (-1, -1),
}


# ─────────────────────────────────────────────────────────
# Topological sort (Kahn's algorithm)
# ─────────────────────────────────────────────────────────

def build_topo_order(fdr: np.ndarray, mask: np.ndarray):
    """
    Devuelve la lista de (row, col) de celdas activas ordenadas
    de cabeceras (sin afluentes) al outlet (máxima acumulación).

    Usa Kahn's algorithm sobre el grafo D8 del dominio.
    """
    nlat, nlon = fdr.shape

    # Para cada celda activa, calcular (drain_r, drain_c) = celda downstream
    drain_r = np.full((nlat, nlon), -1, dtype=np.int32)
    drain_c = np.full((nlat, nlon), -1, dtype=np.int32)
    in_degree = np.zeros((nlat, nlon), dtype=np.int32)

    active_r, active_c = np.where(mask == 1)
    for r, c in zip(active_r, active_c):
        code = int(fdr[r, c])
        if code not in ARCMAP_D8:
            continue
        dy, dx = ARCMAP_D8[code]
        dr, dc = r + dy, c + dx
        if 0 <= dr < nlat and 0 <= dc < nlon and mask[dr, dc] == 1:
            drain_r[r, c] = dr
            drain_c[r, c] = dc
            in_degree[dr, dc] += 1

    # Kahn: iniciar con celdas sin afluentes (cabeceras)
    queue = deque()
    for r, c in zip(active_r, active_c):
        if in_degree[r, c] == 0:
            queue.append((r, c))

    topo_order = []
    while queue:
        r, c = queue.popleft()
        topo_order.append((r, c))
        dr, dc = drain_r[r, c], drain_c[r, c]
        if dr >= 0:
            in_degree[dr, dc] -= 1
            if in_degree[dr, dc] == 0:
                queue.append((dr, dc))

    n_active = int(mask.sum())
    if len(topo_order) != n_active:
        print(f"  [WARN] Topo sort incompleto: {len(topo_order)}/{n_active} celdas "
              f"(posibles ciclos en el D8 en {n_active - len(topo_order)} celdas)")

    return topo_order, drain_r, drain_c


# ─────────────────────────────────────────────────────────
# Routing distribuido
# ─────────────────────────────────────────────────────────

def distributed_routing_chunk(
    runoff_mm: np.ndarray,        # (ntime, nlat, nlon) mm/day
    area_m2: np.ndarray,          # (nlat, nlon) m²
    mask: np.ndarray,             # (nlat, nlon) 0/1
    topo_order: list,
    drain_r: np.ndarray,
    drain_c: np.ndarray,
) -> np.ndarray:
    """
    Acumula runoff aguas abajo y devuelve discharge (m³/s) para cada celda.

    Conversión: mm/day × m² / (1000 mm/m × 86400 s/day) = m³/s
    """
    # Local contribution de cada celda (m³/s)
    discharge = runoff_mm * area_m2[np.newaxis, :, :] / (1000.0 * 86400.0)
    discharge = np.where(mask[np.newaxis, :, :] == 1, discharge, 0.0)

    # Propagar en orden topológico: cabeceras → outlet
    for r, c in topo_order:
        dr, dc = drain_r[r, c], drain_c[r, c]
        if dr >= 0:
            discharge[:, dr, dc] += discharge[:, r, c]

    # Enmascarar celdas inactivas
    discharge[:, mask == 0] = np.nan
    return discharge.astype(np.float32)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Routing distribuido VIC — discharge gridded NetCDF"
    )
    parser.add_argument("--start", type=int, default=2003,
                        help="Año inicio del análisis (excluye warm-up)")
    parser.add_argument("--end",   type=int, default=2025,
                        help="Año fin")
    parser.add_argument("--chunk-years", type=int, default=5,
                        help="Años por chunk para controlar memoria (default: 5)")
    parser.add_argument("--output", type=str, default="discharge_grid.nc",
                        help="Nombre del archivo de salida en data/routing/")
    args = parser.parse_args()

    print("=" * 60)
    print("  ROUTING DISTRIBUIDO VIC — Cuenca Paucartambo")
    print("=" * 60)

    # ── 1. Cargar dominio y dirección de flujo ────────────
    print("\n[→] Cargando dominio y flow direction...")
    ds_domain = xr.open_dataset(DOMAIN_DIR / "domain.nc")
    ds_fdr    = xr.open_dataset(DOMAIN_DIR / "flow_direction.nc")

    mask   = ds_domain["mask"].values.astype(np.int8)
    area   = ds_domain["area"].values.astype(np.float64)   # m²
    lats   = ds_domain["lat"].values
    lons   = ds_domain["lon"].values
    nlat, nlon = mask.shape

    fdr = ds_fdr["flow_direction"].values.astype(np.int32)

    n_active = int(mask.sum())
    print(f"  Grilla: {nlat} × {nlon} | Celdas activas: {n_active:,}")
    print(f"  Período: {args.start}–{args.end} | Chunk: {args.chunk_years} años")

    # ── 2. Ordenamiento topológico ────────────────────────
    print("\n[→] Ordenamiento topológico D8 (Kahn)...")
    topo_order, drain_r, drain_c = build_topo_order(fdr, mask)
    print(f"  Celdas ordenadas: {len(topo_order):,} / {n_active:,}")

    # ── 3. Cargar outputs VIC en el rango pedido ──────────
    print(f"\n[→] Cargando outputs VIC {args.start}–{args.end}...")
    flux_files = sorted(OUTPUT_DIR.glob("fluxes.*.nc"))
    flux_files = [
        f for f in flux_files
        if args.start <= int(f.stem.split(".")[1][:4]) <= args.end
    ]
    if not flux_files:
        print(f"[ERROR] No se encontraron archivos flux para {args.start}–{args.end}")
        sys.exit(1)
    print(f"  {len(flux_files)} archivos flux encontrados")

    # ── 4. Preparar archivo de salida (modo append por chunks) ─
    out_file = ROUTING_DIR / args.output
    if out_file.exists():
        out_file.unlink()
        print(f"  [INFO] Archivo previo eliminado: {out_file.name}")

    encoding = {
        "discharge": {
            "zlib": True, "complevel": 4, "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }

    # ── 5. Procesar por chunks de años ────────────────────
    years = list(range(args.start, args.end + 1))
    year_chunks = [
        years[i:i + args.chunk_years]
        for i in range(0, len(years), args.chunk_years)
    ]

    all_chunks = []
    for chunk_idx, chunk_years in enumerate(year_chunks):
        y0, y1 = chunk_years[0], chunk_years[-1]
        print(f"\n[→] Chunk {chunk_idx + 1}/{len(year_chunks)}: {y0}–{y1}...")

        chunk_files = [f for f in flux_files
                       if y0 <= int(f.stem.split(".")[1][:4]) <= y1]
        ds_vic = xr.open_mfdataset(chunk_files, combine="by_coords")

        runoff_mm = (ds_vic["OUT_RUNOFF"] + ds_vic["OUT_BASEFLOW"]).values
        times = ds_vic["time"].values
        ds_vic.close()

        ntime = runoff_mm.shape[0]
        print(f"  Timesteps: {ntime} | Acumulando runoff D8...")

        discharge = distributed_routing_chunk(
            runoff_mm, area, mask, topo_order, drain_r, drain_c
        )

        ds_chunk = xr.Dataset(
            {
                "discharge": (
                    ["time", "lat", "lon"],
                    discharge,
                    {
                        "units": "m3 s-1",
                        "long_name": "Accumulated streamflow (D8 upstream accumulation)",
                        "standard_name": "river_discharge",
                    },
                )
            },
            coords={
                "time": times,
                "lat":  (["lat"],  lats, {"units": "degrees_north"}),
                "lon":  (["lon"],  lons, {"units": "degrees_east"}),
            },
            attrs={
                "title":       "VIC Distributed Streamflow — Cuenca Paucartambo",
                "routing":     "D8 upstream accumulation (instantaneous, no lag)",
                "spinup_note": "Years 2000-2002 excluded (warm-up)",
                "conventions": "CF-1.6",
            },
        )
        all_chunks.append(ds_chunk)
        print(f"  [OK] Chunk {chunk_idx + 1} procesado")

    # ── 6. Concatenar y guardar ───────────────────────────
    print(f"\n[→] Concatenando {len(all_chunks)} chunks y guardando...")
    ds_out = xr.concat(all_chunks, dim="time")

    # Estadísticas en el outlet (celda con mayor acumulación)
    outlet_discharge = ds_out["discharge"].where(
        ds_fdr["flow_accumulation"] == float(ds_fdr["flow_accumulation"].max())
    ).max(dim=["lat", "lon"])
    print(f"  Caudal outlet — media: {float(outlet_discharge.mean()):.1f} m³/s  "
          f"máx: {float(outlet_discharge.max()):.1f} m³/s")

    ds_out.to_netcdf(out_file, encoding=encoding)
    size_mb = out_file.stat().st_size / 1e6
    print(f"\n[✓] Guardado: {out_file}  ({size_mb:.0f} MB)")
    print(f"    Dims: time={len(ds_out.time)}, lat={nlat}, lon={nlon}")


if __name__ == "__main__":
    main()
