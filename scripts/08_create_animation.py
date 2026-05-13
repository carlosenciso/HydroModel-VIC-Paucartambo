#!/usr/bin/env python3
"""
Script 08: Crear animación de la dinámica hidrológica
=====================================================
Genera un archivo animado (GIF/MP4) mostrando la evolución
espacial y temporal de variables hidrológicas en la cuenca.

Variables disponibles para animar:
  - Precipitación (PRCP)
  - Humedad del suelo (SOIL_MOIST)
  - Escorrentía superficial (RUNOFF)
  - SWE (Equivalente agua en nieve)
  - Caudal enrutado (si disponible)

La animación incluye:
  1. Mapa espacial de la variable en cada paso de tiempo
  2. Serie temporal con cursor moviéndose
  3. Barra de escala y leyenda

Uso:
    python scripts/08_create_animation.py
    python scripts/08_create_animation.py --variable SOIL_MOIST --year 2015
    python scripts/08_create_animation.py --format gif --fps 8
    python scripts/08_create_animation.py --streamflow --year-start 2024 --year-end 2025
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import yaml
import cmocean

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
PROJECT_DIR  = Path(__file__).parent.parent
CONFIG_FILE  = PROJECT_DIR / "config" / "basin_config.yml"
DATA_DIR     = PROJECT_DIR / "data"
OUTPUT_DIR   = DATA_DIR / "outputs"
ROUTING_DIR  = DATA_DIR / "routing"
DOMAIN_DIR   = DATA_DIR / "domain"
PLOTS_DIR    = PROJECT_DIR / "plots"
ANIMS_DIR    = PROJECT_DIR / "animations"
ANIMS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuración por variable
VAR_CONFIG = {
    "OUT_PREC": {
        "label": "Precipitación",
        "units": "mm/día",
        "cmap": cmocean.cm.rain,
        "vmin": 0,
        "vmax_percentile": 99,
        "norm": "linear",
    },
    "OUT_SOIL_MOIST": {
        "label": "Humedad del Suelo",
        "units": "mm",
        "cmap": cmocean.cm.matter,
        "vmin": 0,
        "vmax_percentile": 95,
        "norm": "linear",
    },
    "OUT_RUNOFF": {
        "label": "Escorrentía Superficial",
        "units": "mm/día",
        "cmap": cmocean.cm.rain,
        "vmin": 0,
        "vmax_percentile": 98,
        "norm": "linear",
    },
    "OUT_BASEFLOW": {
        "label": "Flujo Base",
        "units": "mm/día",
        "cmap": cmocean.cm.haline,
        "vmin": 0,
        "vmax_percentile": 98,
        "norm": "linear",
    },
    "OUT_EVAP": {
        "label": "Evapotranspiración",
        "units": "mm/día",
        "cmap": cmocean.cm.turbid,
        "vmin": 0,
        "vmax_percentile": 98,
        "norm": "linear",
    },
    "OUT_SWE": {
        "label": "Equivalente Agua en Nieve (SWE)",
        "units": "mm",
        "cmap": cmocean.cm.ice_r,
        "vmin": 0,
        "vmax_percentile": 99,
        "norm": "linear",
    },
}


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def load_data(variable: str, year: int = None) -> tuple:
    """
    Cargar datos para animación.

    Returns:
        (da, times, lats, lons, mask)
    """
    # Cargar dominio
    ds_domain = xr.open_dataset(DOMAIN_DIR / "domain.nc")
    mask = ds_domain["mask"].values
    lats = ds_domain.lat.values
    lons = ds_domain.lon.values
    ds_domain.close()

    # Cargar fluxes VIC
    flux_files = sorted(OUTPUT_DIR.glob("fluxes*.nc"))
    if not flux_files:
        raise FileNotFoundError("No se encontraron salidas VIC.")

    ds = xr.open_mfdataset(flux_files, combine="by_coords")

    if variable not in ds:
        available = list(ds.data_vars)
        raise ValueError(f"Variable '{variable}' no disponible. Opciones: {available}")

    da = ds[variable]

    # Filtrar por año si se especifica
    if year is not None:
        da = da.sel(time=da.time.dt.year == year)
        if len(da.time) == 0:
            raise ValueError(f"No hay datos para el año {year}")

    # Aplicar máscara
    da = da.where(mask)

    # Para SOIL_MOIST, sumar capas si tiene dimensión nlayer
    if "nlayer" in da.dims:
        da = da.sum(dim="nlayer")

    times = pd.to_datetime(da.time.values)

    return da, times, lats, lons, mask


def create_animation(
    variable: str,
    year: int = None,
    freq: str = "7D",
    output_format: str = "gif",
    fps: int = 6,
    dpi: int = 120,
    config: dict = None,
) -> Path:
    """
    Crear animación de una variable hidrológica.

    Args:
        variable: Variable a animar (e.g., 'OUT_PREC')
        year: Año específico (None = todos los años)
        freq: Frecuencia de muestreo ('7D' = semanal, '1D' = diario)
        output_format: 'gif' o 'mp4'
        fps: Frames por segundo
        dpi: Resolución de la animación
        config: Configuración del proyecto

    Returns:
        Path al archivo animado
    """
    print(f"[ANIM] Creando animación: {variable}" +
          (f" - {year}" if year else "") + f" ({output_format.upper()})")

    # Cargar datos
    da, times, lats, lons, mask = load_data(variable, year)

    # Resamplear si frecuencia especificada
    if freq != "1D":
        da_resampled = da.resample(time=freq).mean()
        times_plot = pd.to_datetime(da_resampled.time.values)
        data = da_resampled.values
    else:
        times_plot = times
        data = da.values

    nframes = len(times_plot)
    print(f"  Frames: {nframes} | Período: {times_plot[0].date()} → {times_plot[-1].date()}")

    # Configuración de la variable
    var_cfg = VAR_CONFIG.get(variable, {
        "label": variable,
        "units": "",
        "cmap": plt.cm.viridis,
        "vmin": 0,
        "vmax_percentile": 95,
        "norm": "linear",
    })

    vmax = np.nanpercentile(data[np.isfinite(data)], var_cfg["vmax_percentile"])
    vmin = var_cfg["vmin"]

    # Usar escala logarítmica para precipitación y escorrentía si hay mucha variabilidad
    use_log = (variable in ["OUT_PREC", "OUT_RUNOFF"] and vmax > 50)
    if use_log:
        norm = mcolors.LogNorm(vmin=max(vmin, 0.1), vmax=max(vmax, 1))
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Outlet para marcador
    outlet = config["basin"]["outlet"]

    # -- Crear figura ------------------------------------------
    fig = plt.figure(figsize=(14, 8), facecolor="black")
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[3, 1],
        height_ratios=[4, 1],
        hspace=0.05, wspace=0.1,
    )

    ax_map  = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_ts   = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[:, 1])

    # -- Mapa base ---------------------------------------------
    ax_map.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                      crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="white", alpha=0.5)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="white", alpha=0.3)
    ax_map.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="cyan", alpha=0.4)
    ax_map.set_facecolor("black")

    # -- Serie temporal (fondo) --------------------------------
    basin_mean_ts = np.nanmean(data.reshape(nframes, -1), axis=1)
    ax_ts.set_facecolor("#1a1a1a")
    ax_ts.plot(range(nframes), basin_mean_ts,
               color="gray", linewidth=1, alpha=0.8)
    ax_ts.set_xlim(0, nframes - 1)
    ax_ts.tick_params(colors="white", labelsize=7)
    ax_ts.set_ylabel(var_cfg["units"], color="white", fontsize=7)
    ax_ts.spines[:].set_color("gray")

    # -- Panel de info -----------------------------------------
    ax_info.set_facecolor("black")
    ax_info.axis("off")

    # Colorbar en ax_info
    sm = plt.cm.ScalarMappable(cmap=var_cfg["cmap"], norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.76, 0.25, 0.02, 0.5])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"{var_cfg['label']}\n({var_cfg['units']})",
                   color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white", fontsize=7)
    cbar.outline.set_edgecolor("white")

    # Título en ax_info
    title_text = ax_info.text(
        0.5, 0.95,
        f"Cuenca {config['basin']['name']}\n{config['basin']['department']}, Peru",
        transform=ax_info.transAxes,
        color="white", fontsize=9, ha="center", va="top", fontweight="bold"
    )

    # -- Elementos de animación --------------------------------
    # Frame inicial
    frame0 = np.where(mask == 0, np.nan, data[0])
    im = ax_map.pcolormesh(
        lons, lats, frame0,
        transform=ccrs.PlateCarree(),
        cmap=var_cfg["cmap"],
        norm=norm,
        rasterized=True,
    )

    # Punto de outlet
    ax_map.plot(outlet["lon"], outlet["lat"],
                "^", color="yellow", markersize=12,
                transform=ccrs.PlateCarree(), zorder=10,
                label=outlet["name"])
    ax_map.legend(loc="lower right", fontsize=7,
                  facecolor="black", edgecolor="white",
                  labelcolor="white")

    # Título del mapa
    date_text = ax_map.set_title("", color="white", fontsize=10, pad=5)

    # Cursor de tiempo en la serie temporal
    vline = ax_ts.axvline(x=0, color="red", linewidth=1.5)
    time_dot = ax_ts.plot([0], [basin_mean_ts[0]],
                          "o", color="red", markersize=5)[0]

    # Texto de estadísticas en ax_info
    stats_text = ax_info.text(
        0.5, 0.55,
        "",
        transform=ax_info.transAxes,
        color="lightgray", fontsize=8,
        ha="center", va="center",
        family="monospace",
    )

    # Fondo general
    fig.patch.set_facecolor("black")

    # -- Función de actualización ------------------------------
    def update(frame_idx):
        frame_data = np.where(mask == 0, np.nan, data[frame_idx])
        im.set_array(frame_data.ravel())

        # Actualizar fecha
        date_str = times_plot[frame_idx].strftime("%d %b %Y")
        date_text.set_text(f"{var_cfg['label']} — {date_str}")

        # Cursor tiempo
        vline.set_xdata([frame_idx, frame_idx])
        time_dot.set_data([frame_idx], [basin_mean_ts[frame_idx]])

        # Estadísticas
        valid = frame_data[np.isfinite(frame_data)]
        if len(valid) > 0:
            stats_str = (
                f"Media: {np.nanmean(valid):.2f} {var_cfg['units']}\n"
                f"Máx:   {np.nanmax(valid):.2f}\n"
                f"Mín:   {np.nanmin(valid[valid > 0] if valid[valid > 0].size > 0 else valid):.3f}\n"
                f"P95:   {np.nanpercentile(valid, 95):.2f}"
            )
            stats_text.set_text(stats_str)

        return [im, date_text, vline, time_dot, stats_text]

    # -- Crear animación ---------------------------------------
    print(f"  Renderizando {nframes} frames...")
    anim = FuncAnimation(
        fig,
        update,
        frames=nframes,
        blit=True,
        interval=1000 / fps,
    )

    # Guardar
    year_str = str(year) if year else "all"
    output_name = f"animation_{variable}_{year_str}.{output_format}"
    output_path = ANIMS_DIR / output_name

    print(f"  Guardando {output_path}...")

    if output_format == "mp4":
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2000,
                                  metadata={"title": f"VIC {variable}"})
            anim.save(str(output_path), writer=writer, dpi=dpi)
        except Exception as e:
            print(f"  [WARNING] MP4 falló ({e}). Usando GIF como fallback...")
            output_path = ANIMS_DIR / output_name.replace(".mp4", ".gif")
            writer = PillowWriter(fps=fps)
            anim.save(str(output_path), writer=writer, dpi=dpi)
    else:
        writer = PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"[OK] Animación guardada: {output_path}")
    return output_path


def create_snapshot_grid(
    variable: str,
    year: int,
    n_snapshots: int = 12,
    config: dict = None,
) -> Path:
    """
    Crear una figura estática con múltiples snapshots (cuadrícula mensual).

    Útil como alternativa a la animación cuando no hay soporte para video.
    """
    print(f"[SNAP] Creando grid mensual: {variable} - {year}")

    da, times, lats, lons, mask = load_data(variable, year)

    # Seleccionar un día representativo por mes (día 15)
    monthly_snapshots = []
    for m in range(1, 13):
        month_data = da.sel(time=(da.time.dt.month == m) & (da.time.dt.year == year))
        if len(month_data.time) > 0:
            monthly_snapshots.append(month_data.mean(dim="time").values)
        else:
            monthly_snapshots.append(np.full((len(lats), len(lons)), np.nan))

    var_cfg = VAR_CONFIG.get(variable, {
        "label": variable, "units": "",
        "cmap": plt.cm.viridis, "vmin": 0, "vmax_percentile": 95
    })

    all_data = np.stack(monthly_snapshots)
    vmax = np.nanpercentile(all_data[np.isfinite(all_data)], var_cfg["vmax_percentile"])
    vmin = var_cfg["vmin"]

    months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
              "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             facecolor="black")
    fig.suptitle(
        f"{var_cfg['label']} — {year} | Cuenca {config['basin']['name']}, Peru",
        color="white", fontsize=13, fontweight="bold"
    )

    for i, (ax, month_data, month_name) in enumerate(
            zip(axes.flat, monthly_snapshots, months)):

        frame = np.where(mask == 0, np.nan, month_data)
        ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])
        ax.set_facecolor("black")

        im = ax.pcolormesh(
            lons, lats, frame,
            cmap=var_cfg["cmap"],
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="cyan", alpha=0.4)
        ax.set_title(month_name, color="white", fontsize=9, pad=3)

        # Outlet
        outlet = config["basin"]["outlet"]
        ax.plot(outlet["lon"], outlet["lat"],
                "^", color="yellow", markersize=6,
                transform=ccrs.PlateCarree())

    # Colorbar
    plt.subplots_adjust(bottom=0.1, hspace=0.1, wspace=0.05)
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.025])
    sm = plt.cm.ScalarMappable(
        cmap=var_cfg["cmap"],
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"{var_cfg['label']} ({var_cfg['units']})",
                   color="white", fontsize=9)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "xticklabels"), color="white")

    output_path = PLOTS_DIR / f"monthly_snapshots_{variable}_{year}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()
    print(f"[OK] Grid mensual guardado: {output_path}")
    return output_path


def create_streamflow_animation(
    year_start: int = None,
    year_end: int = None,
    freq: str = "7D",
    output_format: str = "gif",
    fps: int = 6,
    dpi: int = 120,
    config: dict = None,
) -> Path:
    """
    Animacion con:
      - Mapa: red fluvial D8 coloreada por caudal estimado en cada segmento
      - Panel inferior: serie temporal de caudal construyendose progresivamente
      - Panel derecho: estadisticas y caudal actual en el outlet

    El caudal en cada segmento se estima escalando por acumulacion de flujo:
        Q_seg(t) = (flow_acc_seg / max_acc) * Q_outlet(t)

    Args:
        year_start: Primer ano del periodo (None = desde el inicio)
        year_end:   Ultimo ano del periodo  (None = hasta el final)
        freq:       Frecuencia de frames ('7D' semanal, '1D' diario)
    """
    print("[ANIM] Creando animación de caudal enrutado (STREAMFLOW)...")

    # ── Cargar caudal enrutado ────────────────────────────────────
    sf_csv = ROUTING_DIR / "streamflow_yuncan.csv"
    sf_nc  = ROUTING_DIR / "streamflow_yuncan.nc"

    if sf_nc.exists():
        ds_sf  = xr.open_dataset(sf_nc)
        sf_var = next((v for v in ds_sf.data_vars
                       if ds_sf[v].dtype in [np.float32, np.float64]), None)
        if sf_var is None:
            raise ValueError(f"No se encontró variable numérica en {sf_nc}")
        sf_times  = pd.to_datetime(ds_sf.time.values)
        sf_values = ds_sf[sf_var].values.ravel()
        ds_sf.close()
        print(f"  Caudal desde NetCDF: {len(sf_times)} pasos | "
              f"Media={np.nanmean(sf_values):.1f} m3/s | "
              f"Max={np.nanmax(sf_values):.1f} m3/s")
    elif sf_csv.exists():
        df_sf     = pd.read_csv(sf_csv, parse_dates=["date"])
        sf_times  = pd.to_datetime(df_sf["date"].values)
        sf_values = df_sf.iloc[:, 1].values.ravel().astype(float)
        print(f"  Caudal desde CSV: {len(sf_times)} pasos | "
              f"Media={np.nanmean(sf_values):.1f} m3/s | "
              f"Max={np.nanmax(sf_values):.1f} m3/s")
    else:
        raise FileNotFoundError(
            f"No se encontró caudal enrutado en {sf_csv} ni {sf_nc}. "
            "Ejecuta primero el script 06_run_routing.py."
        )

    # Filtrar por rango de años
    if year_start is not None:
        m = pd.DatetimeIndex(sf_times).year >= year_start
        sf_times = sf_times[m]; sf_values = sf_values[m]
    if year_end is not None:
        m = pd.DatetimeIndex(sf_times).year <= year_end
        sf_times = sf_times[m]; sf_values = sf_values[m]
    if len(sf_times) == 0:
        raise ValueError("No hay datos de caudal para el periodo solicitado")

    # ── Resamplear caudal a la frecuencia de frames ───────────────
    sf_series       = pd.Series(sf_values, index=sf_times)
    sf_resampled    = sf_series.resample(freq).mean().dropna()
    times_plot      = sf_resampled.index
    sf_frame_values = sf_resampled.values.astype(np.float64)
    nframes         = len(times_plot)
    print(f"  Frames ({freq}): {nframes} | "
          f"Periodo: {times_plot[0].date()} -> {times_plot[-1].date()}")

    # ── Estadisticas globales ─────────────────────────────────────
    sf_mean = float(np.nanmean(sf_values))
    sf_max  = float(np.nanmax(sf_values))
    sf_min  = float(np.nanmin(sf_values))
    sf_q95  = float(np.nanpercentile(sf_values, 95))
    sf_q05  = float(np.nanpercentile(sf_values, 5))

    # ── Cargar red fluvial desde flow_direction.nc ────────────────
    from matplotlib.collections import LineCollection as LC

    ARCMAP_D8 = {
        64: (-1, 0), 128: (-1, 1), 1: (0, 1),  2: (1, 1),
         4: ( 1, 0),   8: ( 1,-1), 16: (0,-1), 32: (-1,-1),
    }
    fdr_nc = DOMAIN_DIR / "flow_direction.nc"
    ds_fdr   = xr.open_dataset(fdr_nc)
    fdr      = ds_fdr["flow_direction"].values.astype(int)
    flow_acc = ds_fdr["flow_accumulation"].values.astype(float)
    basin_id = ds_fdr["basin_id"].values
    lats     = ds_fdr.lat.values
    lons     = ds_fdr.lon.values
    ds_fdr.close()

    nlat, nlon = fdr.shape
    basin_mask = basin_id == 1
    acc_thresh = float(np.nanpercentile(flow_acc[basin_mask], 70))
    river_mask = basin_mask & (flow_acc >= acc_thresh)
    max_acc    = float(flow_acc[river_mask].max())

    # Construir segmentos (celda -> celda downstream)
    segments  = []
    seg_acc_n = []    # flow_acc normalizado [0..1] por segmento
    ri, ci    = np.where(river_mask)
    for r, c in zip(ri, ci):
        dy, dx = ARCMAP_D8.get(int(fdr[r, c]), (0, 0))
        nr, nc = r + dy, c + dx
        if 0 <= nr < nlat and 0 <= nc < nlon and river_mask[nr, nc]:
            segments.append([(lons[c], lats[r]), (lons[nc], lats[nr])])
            seg_acc_n.append(flow_acc[r, c] / max_acc)

    segments  = segments
    seg_acc_n = np.array(seg_acc_n, dtype=np.float64)
    linewidths = 0.4 + 2.8 * (seg_acc_n ** 0.4)
    print(f"  Segmentos de rio: {len(segments)} (umbral acc>={acc_thresh:.0f})")

    # Plasma: purpura -> magenta -> naranja -> amarillo (visible en fondo negro)
    river_cmap = plt.cm.plasma
    river_norm = mcolors.LogNorm(vmin=1.0, vmax=max(sf_max, 2.0))
    outlet     = config["basin"]["outlet"]

    # ── Crear figura ──────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor="black")
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[3, 1.2],
        height_ratios=[3.5, 1.5],
        hspace=0.08, wspace=0.12,
    )
    ax_map  = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_ts   = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[:, 1])

    # ── Mapa base ─────────────────────────────────────────────────
    ax_map.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                      crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.BORDERS,   linewidth=0.5, edgecolor="#555555", alpha=0.6)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555555", alpha=0.4)
    ax_map.set_facecolor("#010810")

    # Red fluvial como LineCollection coloreada por caudal estimado
    # Q_seg(t) = clip((acc_seg / max_acc) * Q_outlet(t), 1, inf)
    init_flow = np.clip(seg_acc_n * sf_frame_values[0], 1.0, None)
    lc = LC(segments, cmap=river_cmap, norm=river_norm,
            linewidths=linewidths, zorder=5,
            transform=ccrs.PlateCarree())
    lc.set_array(init_flow)
    ax_map.add_collection(lc)

    # Outlet marker
    ax_map.plot(outlet["lon"], outlet["lat"],
                "^", color="yellow", markersize=11,
                transform=ccrs.PlateCarree(), zorder=10,
                markeredgecolor="white", markeredgewidth=0.5)
    ax_map.text(outlet["lon"] + 0.04, outlet["lat"] + 0.04,
                outlet["name"], color="yellow", fontsize=7,
                transform=ccrs.PlateCarree(), zorder=11)
    date_text = ax_map.set_title("", color="white", fontsize=10, pad=5, fontweight="bold")

    # Colorbar de la red fluvial
    sm_r    = plt.cm.ScalarMappable(cmap=river_cmap, norm=river_norm)
    sm_r.set_array([])
    cbar_ax = fig.add_axes([0.77, 0.20, 0.02, 0.30])
    cbar    = plt.colorbar(sm_r, cax=cbar_ax)
    cbar.set_label("Caudal estimado\n(m3/s)", color="white", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white", fontsize=6)
    cbar.outline.set_edgecolor("#555555")

    # ── Serie temporal progresiva (empieza vacia) ─────────────────
    ax_ts.set_facecolor("#0a0a1a")
    ax_ts.axhline(sf_mean, color="gold", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_ts.set_xlim(0, nframes - 1)
    ax_ts.set_ylim(bottom=0, top=sf_max * 1.15)
    ax_ts.tick_params(colors="white", labelsize=7)
    ax_ts.set_ylabel("Caudal (m3/s)", color="white", fontsize=8)
    ax_ts.set_xlabel("Tiempo",        color="white", fontsize=7)
    ax_ts.spines[:].set_color("#444444")
    ax_ts.text(nframes * 0.01, sf_mean * 1.05,
               f"Media: {sf_mean:.1f}", color="gold", fontsize=6, alpha=0.8)

    tick_positions = np.linspace(0, nframes - 1, min(6, nframes)).astype(int)
    tick_labels    = [pd.Timestamp(times_plot[i]).strftime("%b'%y")
                      for i in tick_positions]
    ax_ts.set_xticks(tick_positions)
    ax_ts.set_xticklabels(tick_labels, color="white", fontsize=6)

    prog_line, = ax_ts.plot([], [], color="#00b4d8", linewidth=1.3, zorder=4)
    tip_dot,   = ax_ts.plot([], [], "o", color="white", markersize=5, zorder=5)

    # ── Panel de info (estatico) ──────────────────────────────────
    ax_info.set_facecolor("black")
    ax_info.axis("off")

    ax_info.text(
        0.5, 0.97,
        f"Cuenca {config['basin']['name']}\n{config['basin']['department']}, Peru",
        transform=ax_info.transAxes,
        color="white", fontsize=9, ha="center", va="top", fontweight="bold"
    )
    ax_info.text(
        0.5, 0.88, f"Central H. {outlet['name']}",
        transform=ax_info.transAxes,
        color="#00b4d8", fontsize=8, ha="center", va="top"
    )
    ax_info.plot([0, 1], [0.85, 0.85], color="#333333", linewidth=0.5,
                 transform=ax_info.transAxes, clip_on=False)

    ax_info.text(
        0.05, 0.79,
        (f"  --- Estadisticas ---\n"
         f"  Media:   {sf_mean:6.1f} m3/s\n"
         f"  Maximo:  {sf_max:6.1f} m3/s\n"
         f"  Minimo:  {sf_min:6.1f} m3/s\n"
         f"  Q95:     {sf_q95:6.1f} m3/s\n"
         f"  Q05:     {sf_q05:6.1f} m3/s"),
        transform=ax_info.transAxes,
        color="#aaaaaa", fontsize=7.5, ha="left", va="top", family="monospace"
    )
    ax_info.plot([0, 1], [0.52, 0.52], color="#333333", linewidth=0.5,
                 transform=ax_info.transAxes, clip_on=False)

    # Caudal actual (dinamico)
    current_q_text = ax_info.text(
        0.5, 0.47, "",
        transform=ax_info.transAxes,
        color="white", fontsize=22, ha="center", va="top", fontweight="bold"
    )
    ax_info.text(0.5, 0.35, "m3/s",
                 transform=ax_info.transAxes,
                 color="#aaaaaa", fontsize=10, ha="center", va="top")
    ax_info.text(0.5, 0.30, "Caudal enrutado",
                 transform=ax_info.transAxes,
                 color="#666666", fontsize=7.5, ha="center", va="top")

    ax_bar = ax_info.inset_axes([0.05, 0.10, 0.90, 0.06])
    ax_bar.set_xlim(0, 1); ax_bar.set_ylim(0, 1)
    ax_bar.set_facecolor("#111111")
    ax_bar.set_xticks([]); ax_bar.set_yticks([])
    ax_bar.spines[:].set_visible(False)
    bar_patch = plt.Rectangle((0, 0), 0.0, 1.0, color="#00b4d8")
    ax_bar.add_patch(bar_patch)
    fig.patch.set_facecolor("black")

    # ── Funcion de actualizacion ──────────────────────────────────
    def update(frame_idx):
        # Actualizar colores de la red fluvial
        q_outlet = float(sf_frame_values[frame_idx])
        lc.set_array(np.clip(seg_acc_n * q_outlet, 1.0, None))

        date_str = pd.Timestamp(times_plot[frame_idx]).strftime("%d %b %Y")
        date_text.set_text(f"Red fluvial — Caudal   {date_str}")

        # Serie temporal progresiva
        prog_line.set_data(range(frame_idx + 1), sf_frame_values[:frame_idx + 1])
        tip_dot.set_data([frame_idx], [sf_frame_values[frame_idx]])

        # Caudal actual en panel info
        current_q_text.set_text(f"{q_outlet:.1f}")
        if q_outlet > sf_q95:
            current_q_text.set_color("#ff4444")
        elif q_outlet > sf_mean:
            current_q_text.set_color("#00b4d8")
        else:
            current_q_text.set_color("#aaddff")

        frac = min(max((q_outlet - sf_min) / max(sf_max - sf_min, 0.01), 0), 1)
        bar_patch.set_width(frac)

        return [lc, date_text, prog_line, tip_dot, current_q_text, bar_patch]

    # ── Crear y guardar animacion ─────────────────────────────────
    print(f"  Renderizando {nframes} frames...")
    anim = FuncAnimation(
        fig, update, frames=nframes,
        blit=True, interval=1000 / fps,
    )

    year_str    = (f"{year_start}-{year_end}" if year_start and year_end else "all")
    output_name = f"animation_STREAMFLOW_{year_str}.{output_format}"
    output_path = ANIMS_DIR / output_name

    print(f"  Guardando {output_path}...")
    if output_format == "mp4":
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2500,
                                  metadata={"title": "Caudal Yuncan - Paucartambo"})
            anim.save(str(output_path), writer=writer, dpi=dpi)
        except Exception as e:
            print(f"  [WARNING] MP4 fallo ({e}). Usando GIF como fallback...")
            output_path = ANIMS_DIR / output_name.replace(".mp4", ".gif")
            writer = PillowWriter(fps=fps)
            anim.save(str(output_path), writer=writer, dpi=dpi)
    else:
        writer = PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"[OK] Animacion de caudal guardada: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Crear animación de dinámica hidrológica - Paucartambo"
    )
    parser.add_argument(
        "--variable",
        choices=list(VAR_CONFIG.keys()),
        default="OUT_SOIL_MOIST",
        help="Variable a animar",
    )
    parser.add_argument("--year", type=int, default=None,
                        help="Año único a animar (para --all-vars y --snapshots)")
    parser.add_argument("--year-start", type=int, default=None, dest="year_start",
                        help="Primer año del período (para --streamflow)")
    parser.add_argument("--year-end",   type=int, default=None, dest="year_end",
                        help="Último año del período (para --streamflow)")
    parser.add_argument("--freq", default="7D",
                        help="Frecuencia de frames (7D=semanal, 1D=diario)")
    parser.add_argument("--format", dest="output_format",
                        choices=["gif", "mp4"], default="gif",
                        help="Formato de salida")
    parser.add_argument("--fps", type=int, default=6,
                        help="Frames por segundo")
    parser.add_argument("--dpi", type=int, default=100,
                        help="Resolución de la animación")
    parser.add_argument("--all-vars", action="store_true",
                        help="Animar todas las variables disponibles")
    parser.add_argument("--snapshots", action="store_true",
                        help="Crear grid de snapshots mensuales en lugar de animación")
    parser.add_argument("--streamflow", action="store_true",
                        help="Crear animación de caudal enrutado (RVIC/routing)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ANIMACIÓN DE DINÁMICA HIDROLÓGICA - Paucartambo")
    print("=" * 60)

    config = load_config()

    if args.streamflow:
        create_streamflow_animation(
            year_start=args.year_start,
            year_end=args.year_end,
            freq=args.freq,
            output_format=args.output_format,
            fps=args.fps,
            dpi=args.dpi,
            config=config,
        )
    elif args.all_vars:
        # Verificar qué variables están disponibles
        flux_files = sorted(OUTPUT_DIR.glob("fluxes*.nc"))
        if flux_files:
            ds_check = xr.open_mfdataset(flux_files, combine="by_coords")
            available = [v for v in VAR_CONFIG if v in ds_check.data_vars]
            ds_check.close()
        else:
            available = [args.variable]

        for var in available:
            try:
                if args.snapshots:
                    year = args.year or 2015
                    create_snapshot_grid(var, year, config=config)
                else:
                    create_animation(
                        var, args.year, args.freq,
                        args.output_format, args.fps, args.dpi, config
                    )
            except Exception as e:
                print(f"  [ERROR] {var}: {e}")
    else:
        if args.snapshots:
            year = args.year or 2015
            create_snapshot_grid(args.variable, year, config=config)
        else:
            create_animation(
                args.variable, args.year, args.freq,
                args.output_format, args.fps, args.dpi, config
            )

    print(f"\n[OK] Animaciones guardadas en: {ANIMS_DIR}")


if __name__ == "__main__":
    main()
