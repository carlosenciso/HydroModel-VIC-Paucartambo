#!/usr/bin/env python3
"""
Script 11: Tiempo de desplazamiento desde lagunas hasta Yuncan
==============================================================
Calcula el tiempo de viaje del agua desde las principales lagunas
naturales de la cuenca Paucartambo hasta la central de Yuncan,
siguiendo la red de flujo D8.

Metodología:
  1. Detección automática de lagunas: celdas con alta acumulación
     de flujo en zonas relativamente planas (criterio topográfico).
  2. Trazado del camino D8 desde cada laguna hasta el outlet (Yuncan).
  3. Cálculo de longitud de trayecto (suma de flow_distance).
  4. Estimación del tiempo de viaje con la ecuación de Manning:
       v = (1/n) * R^(2/3) * S^(1/2)
     donde n=0.04 (río andino), R=2.0 m (radio hidráulico típico),
     S = ΔZ / L (pendiente media del trayecto).

Uso:
    python scripts/11_travel_time.py
    python scripts/11_travel_time.py --acc-threshold 200 --elev-min 3500
    python scripts/11_travel_time.py --n-manning 0.05 --radius 1.5
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pandas as pd
from scipy.ndimage import label, maximum_filter, uniform_filter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"
DOMAIN_DIR  = DATA_DIR / "domain"
ROUTING_DIR = DATA_DIR / "routing"
PLOTS_DIR   = PROJECT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ARCMAP D8 encoding → (dy, dx)
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

# Outlet: central Yuncan
YUNCAN_LON = -75.3195
YUNCAN_LAT = -10.9965

# Manning roughness y radio hidráulico por defecto
DEFAULT_N_MANNING = 0.04   # río andino de montaña
DEFAULT_R_HYD     = 2.0    # m — radio hidráulico representativo


# ─────────────────────────────────────────────────────────
def load_domain():
    ds = xr.open_dataset(DOMAIN_DIR / "flow_direction.nc")
    fdir  = ds["flow_direction"].values.astype(float)
    fdist = ds["flow_distance"].values.astype(float)
    facc  = ds["flow_accumulation"].values.astype(float)
    elev  = ds["elevation"].values.astype(float)
    lats  = ds["lat"].values
    lons  = ds["lon"].values
    ds.close()
    return fdir, fdist, facc, elev, lats, lons


def find_outlet_cell(lats, lons, target_lon, target_lat):
    """Return (row, col) of the grid cell closest to (target_lon, target_lat)."""
    j = int(np.argmin(np.abs(lons - target_lon)))
    i = int(np.argmin(np.abs(lats - target_lat)))
    return i, j


def trace_d8_path(start_i, start_j, fdir, fdist, elev, outlet_ij, max_steps=5000):
    """
    Follow D8 flow path from (start_i, start_j) to outlet_ij.
    Returns:
        path      : list of (i, j) cells (empty if path doesn't reach outlet)
        length_m  : total path length in metres
        elev_start: elevation at source cell
        elev_end  : elevation at outlet cell
        reached   : True if path successfully reached the outlet
    """
    nrows, ncols = fdir.shape
    i, j = start_i, start_j
    outlet_i, outlet_j = outlet_ij
    path = [(i, j)]
    length_m = 0.0
    visited = set()
    visited.add((i, j))

    for _ in range(max_steps):
        if i == outlet_i and j == outlet_j:
            elev_start = float(elev[start_i, start_j])
            elev_end   = float(elev[outlet_i, outlet_j])
            return path, length_m, elev_start, elev_end, True
        code = fdir[i, j]
        if np.isnan(code) or int(code) not in ARCMAP_D8:
            break
        dy, dx = ARCMAP_D8[int(code)]
        ni, nj = i + dy, j + dx
        if not (0 <= ni < nrows and 0 <= nj < ncols):
            break
        if (ni, nj) in visited:
            break   # cycle detected
        length_m += fdist[i, j]
        i, j = ni, nj
        visited.add((i, j))
        path.append((i, j))

    return [], 0.0, 0.0, 0.0, False   # did not reach outlet


def estimate_travel_time(length_m, elev_start, elev_end, n=DEFAULT_N_MANNING, R=DEFAULT_R_HYD):
    """
    Estimate travel time (hours) using Manning equation:
        v = (1/n) * R^(2/3) * S^(1/2)
    Falls back to v_min=0.3 m/s if slope is negligible.
    """
    if length_m < 1:
        return 0.0
    delta_z = max(elev_start - elev_end, 0.0)
    S = delta_z / length_m
    v_min = 0.3   # m/s floor (slow meandering)
    v_max = 4.0   # m/s ceiling (steep torrent)
    if S < 1e-6:
        v = v_min
    else:
        v = (1.0 / n) * (R ** (2.0 / 3.0)) * (S ** 0.5)
        v = np.clip(v, v_min, v_max)
    hours = (length_m / v) / 3600.0
    return hours


def detect_lagunas(facc, elev, fdir, lats, lons,
                   acc_threshold=200, elev_min=3500,
                   flat_window=5, flat_std_max=30.0,
                   min_cluster_dist_cells=10):
    """
    Detect natural laguna candidates:
      - flow_accumulation > acc_threshold  (drains enough area)
      - elevation > elev_min               (Andean high-altitude lakes)
      - flat neighbourhood (std of elev in flat_window x flat_window < flat_std_max)
      - local maximum of flow_accumulation (outlet of the lake)

    Returns list of (i, j, name, facc_value, elev_value).
    """
    nrows, ncols = facc.shape
    valid_mask = ~np.isnan(fdir)

    # 1. Criteria masks
    mask_acc  = facc > acc_threshold
    mask_elev = elev > elev_min
    mask_valid = valid_mask

    # 2. Flatness: local std of elevation
    elev_safe = np.where(valid_mask, elev, np.nan)
    elev_filled = np.where(valid_mask, elev, 0.0)
    e2 = uniform_filter(elev_filled ** 2, size=flat_window)
    e1 = uniform_filter(elev_filled,      size=flat_window)
    local_std = np.sqrt(np.maximum(e2 - e1 ** 2, 0.0))
    mask_flat = local_std < flat_std_max

    # 3. Combined candidate mask
    candidates = mask_acc & mask_elev & mask_flat & mask_valid

    # 4. Local maxima of flow_accumulation within candidates
    facc_cand = np.where(candidates, facc, 0.0)
    facc_local_max = maximum_filter(facc_cand, size=flat_window)
    local_maxima = (facc_cand == facc_local_max) & (facc_cand > 0)

    # 5. Cluster nearby maxima and pick the cell with highest accumulation per cluster
    labeled, n_clusters = label(local_maxima)
    lagunas = []
    for cluster_id in range(1, n_clusters + 1):
        idxs = np.argwhere(labeled == cluster_id)
        accs = [facc[r, c] for r, c in idxs]
        best = idxs[int(np.argmax(accs))]
        ri, ci = int(best[0]), int(best[1])
        lagunas.append((ri, ci, float(facc[ri, ci]), float(elev[ri, ci])))

    # 6. Remove clusters too close to each other (keep highest-acc one)
    lagunas.sort(key=lambda x: -x[2])
    filtered = []
    for lag in lagunas:
        ri, ci, acc_v, elev_v = lag
        too_close = False
        for fr, fc, _, _ in filtered:
            if abs(ri - fr) < min_cluster_dist_cells and abs(ci - fc) < min_cluster_dist_cells:
                too_close = True
                break
        if not too_close:
            filtered.append(lag)

    # 7. Name them by approximate position (Norte/Sur/Este/Oeste)
    named = []
    lat_c = np.mean(lats)
    lon_c = np.mean(lons)
    for k, (ri, ci, acc_v, elev_v) in enumerate(filtered):
        lat_pt = lats[ri]
        lon_pt = lons[ci]
        pos_v = "N" if lat_pt < lat_c else "S"   # lat decreases northward
        pos_h = "E" if lon_pt > lon_c else "O"
        name = f"Laguna-{k+1:02d}_{pos_v}{pos_h}"
        named.append((ri, ci, name, acc_v, elev_v))

    return named


def run(acc_threshold=200, elev_min=3500, n_manning=DEFAULT_N_MANNING,
        r_hyd=DEFAULT_R_HYD, flat_std_max=30.0, flat_window=5):

    print("[TRAVEL TIME] Cargando red de flujo...")
    fdir, fdist, facc, elev, lats, lons = load_domain()

    outlet_ij = find_outlet_cell(lats, lons, YUNCAN_LON, YUNCAN_LAT)
    print(f"[TRAVEL TIME] Outlet Yuncan: celda ({outlet_ij[0]}, {outlet_ij[1]}), "
          f"elev={elev[outlet_ij]:.0f} m, "
          f"lat={lats[outlet_ij[0]]:.4f}, lon={lons[outlet_ij[1]]:.4f}")

    print(f"[TRAVEL TIME] Detectando lagunas (acc>{acc_threshold} km², elev>{elev_min} m)...")
    lagunas = detect_lagunas(facc, elev, fdir, lats, lons,
                              acc_threshold=acc_threshold,
                              elev_min=elev_min,
                              flat_std_max=flat_std_max,
                              flat_window=flat_window)
    print(f"[TRAVEL TIME] Lagunas detectadas: {len(lagunas)}")

    results = []
    paths_all = []
    for ri, ci, name, acc_v, elev_v in lagunas:
        path, length_m, e_src, e_out, reached = trace_d8_path(
            ri, ci, fdir, fdist, elev, outlet_ij)
        if not reached:
            print(f"  {name:25s}  [NO LLEGA AL OUTLET — descartada]")
            continue
        hours = estimate_travel_time(length_m, e_src, e_out, n=n_manning, R=r_hyd)
        delta_z = max(e_src - e_out, 0.0)
        slope_pm = (delta_z / length_m) * 1000 if length_m > 0 else 0  # m/km
        results.append({
            "Laguna":           name,
            "Lat":              round(float(lats[ri]), 4),
            "Lon":              round(float(lons[ci]), 4),
            "Elevacion_m":      round(e_src, 0),
            "AreaDrenaje_km2":  round(acc_v * 1.0, 0),  # 1 km2/cell
            "LongitudVia_km":   round(length_m / 1000, 1),
            "DeltaZ_m":         round(delta_z, 0),
            "Pendiente_mxkm":   round(slope_pm, 2),
            "TiempoViaje_h":    round(hours, 1),
            "TiempoViaje_dias": round(hours / 24, 2),
        })
        paths_all.append((path, name, hours))
        print(f"  {name:25s}  L={length_m/1000:6.1f} km  ΔZ={delta_z:.0f} m  "
              f"t={hours:.1f} h ({hours/24:.2f} d)")

    df = pd.DataFrame(results).sort_values("TiempoViaje_h")
    csv_out = ROUTING_DIR / "travel_time_lagunas.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[TRAVEL TIME] Tabla guardada en {csv_out}")
    print(df.to_string(index=False))

    # ── Mapa ──────────────────────────────────────────────────────────────────
    print("\n[TRAVEL TIME] Generando mapa...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Hillshade del DEM
    from scipy.ndimage import gaussian_filter
    elev_sm = gaussian_filter(np.where(~np.isnan(fdir), elev, np.nan), sigma=1)
    dx_elev = np.gradient(elev_sm, axis=1)
    dy_elev = np.gradient(elev_sm, axis=0)
    hillshade = -dx_elev * np.cos(np.radians(45)) - dy_elev * np.sin(np.radians(45))
    hs_valid = hillshade[~np.isnan(hillshade)]
    hillshade = (hillshade - hs_valid.min()) / \
                (hs_valid.max() - hs_valid.min() + 1e-9)

    lon2d, lat2d = np.meshgrid(lons, lats)
    basin_mask = ~np.isnan(fdir)
    elev_plot = np.where(basin_mask, elev, np.nan)

    im = ax.pcolormesh(lon2d, lat2d, elev_plot, cmap="terrain",
                       vmin=500, vmax=5500, transform=ccrs.PlateCarree(), alpha=0.8)
    ax.pcolormesh(lon2d, lat2d, np.where(basin_mask, hillshade, np.nan),
                  cmap="gray", vmin=0, vmax=1, alpha=0.25,
                  transform=ccrs.PlateCarree())

    # Red de ríos (facc > threshold)
    river_mask = (facc > acc_threshold) & basin_mask
    ax.scatter(lon2d[river_mask], lat2d[river_mask],
               c="royalblue", s=0.3, alpha=0.5, transform=ccrs.PlateCarree(),
               label="Red fluvial")

    # Caminos trazados — coloreados por tiempo de viaje
    cmap_path = plt.cm.plasma
    t_max = max((r["TiempoViaje_h"] for r in results), default=1)
    for path, name, hours in paths_all:
        if len(path) < 2:
            continue
        path_lons = [lons[c] for _, c in path]
        path_lats = [lats[r] for r, _ in path]
        color = cmap_path(hours / t_max)
        ax.plot(path_lons, path_lats, color=color, lw=1.5, alpha=0.85,
                transform=ccrs.PlateCarree())

    # Colorbar para tiempo de viaje
    sm = plt.cm.ScalarMappable(cmap=cmap_path, norm=mcolors.Normalize(0, t_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, shrink=0.6)
    cbar.set_label("Tiempo de viaje (h)", fontsize=10)

    # Lagunas
    for ri, ci, name, acc_v, elev_v in lagunas:
        matched = [r for r in results if r["Laguna"] == name]
        if not matched:
            continue
        t_h = matched[0]["TiempoViaje_h"]
        color = cmap_path(t_h / t_max)
        ax.scatter(lons[ci], lats[ri], s=80, c=[color], edgecolors="white",
                   linewidths=0.8, zorder=5, transform=ccrs.PlateCarree())
        ax.annotate(f"{name}\n{t_h:.0f}h",
                    xy=(lons[ci], lats[ri]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=6.5, color="white",
                    path_effects=[
                        matplotlib.patheffects.withStroke(linewidth=1.5, foreground="black")
                    ],
                    transform=ccrs.PlateCarree())

    # Outlet Yuncan
    ax.scatter(YUNCAN_LON, YUNCAN_LAT, s=150, c="red", marker="^",
               zorder=10, transform=ccrs.PlateCarree(), label="Yuncan (outlet)")
    ax.annotate("Yuncan", xy=(YUNCAN_LON, YUNCAN_LAT),
                xytext=(6, -12), textcoords="offset points",
                fontsize=9, fontweight="bold", color="red",
                transform=ccrs.PlateCarree())

    # Colorbar de elevación
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.5,
                 label="Elevación (m s.n.m.)")

    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")
    ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", color="gray")
    ax.set_title(f"Tiempo de desplazamiento desde lagunas hasta Yuncan\n"
                 f"(Manning n={n_manning}, R={r_hyd} m — {len(results)} lagunas detectadas)",
                 fontsize=12)

    legend_elements = [
        Line2D([0], [0], color="royalblue", lw=1, label="Red fluvial"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
               markersize=10, label="Yuncan"),
        mpatches.Patch(facecolor="none", edgecolor="none",
                       label=f"n Manning = {n_manning}"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8)

    map_out = PLOTS_DIR / "travel_time_lagunas.png"
    plt.tight_layout()
    plt.savefig(map_out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[TRAVEL TIME] Mapa guardado en {map_out}")

    return df


# ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Tiempo de viaje lagunas → Yuncan")
    p.add_argument("--acc-threshold", type=float, default=200,
                   help="Acumulación mínima (km²) para considerar una laguna [200]")
    p.add_argument("--elev-min", type=float, default=3500,
                   help="Elevación mínima (m) para filtrar lagunas andinas [3500]")
    p.add_argument("--flat-std-max", type=float, default=30.0,
                   help="Std máxima de elevación local para criterio de planicie [30 m]")
    p.add_argument("--n-manning", type=float, default=DEFAULT_N_MANNING,
                   help=f"Coeficiente de rugosidad de Manning [{DEFAULT_N_MANNING}]")
    p.add_argument("--radius", type=float, default=DEFAULT_R_HYD,
                   help=f"Radio hidráulico representativo (m) [{DEFAULT_R_HYD}]")
    return p.parse_args()


if __name__ == "__main__":
    import matplotlib
    args = parse_args()
    df = run(
        acc_threshold=args.acc_threshold,
        elev_min=args.elev_min,
        n_manning=args.n_manning,
        r_hyd=args.radius,
        flat_std_max=args.flat_std_max,
    )
