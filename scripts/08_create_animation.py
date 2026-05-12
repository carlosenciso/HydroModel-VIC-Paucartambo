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

# ─────────────────────────────────────────────────────────
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

    # ── Crear figura ──────────────────────────────────────
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

    # ── Mapa base ─────────────────────────────────────────
    ax_map.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                      crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="white", alpha=0.5)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="white", alpha=0.3)
    ax_map.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="cyan", alpha=0.4)
    ax_map.set_facecolor("black")

    # ── Serie temporal (fondo) ────────────────────────────
    basin_mean_ts = np.nanmean(data.reshape(nframes, -1), axis=1)
    ax_ts.set_facecolor("#1a1a1a")
    ax_ts.plot(range(nframes), basin_mean_ts,
               color="gray", linewidth=1, alpha=0.8)
    ax_ts.set_xlim(0, nframes - 1)
    ax_ts.tick_params(colors="white", labelsize=7)
    ax_ts.set_ylabel(var_cfg["units"], color="white", fontsize=7)
    ax_ts.spines[:].set_color("gray")

    # ── Panel de info ─────────────────────────────────────
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

    # ── Elementos de animación ────────────────────────────
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

    # ── Función de actualización ──────────────────────────
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

    # ── Crear animación ───────────────────────────────────
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
    year: int = None,
    freq: str = "7D",
    output_format: str = "gif",
    fps: int = 6,
    dpi: int = 120,
    config: dict = None,
) -> Path:
    """
    Crear animación que muestra:
      - Mapa espacial: escorrentía total (OUT_RUNOFF + OUT_BASEFLOW)
      - Panel inferior: caudal enrutado en Yuncan con cursor temporal
      - Panel derecho: estadísticas y valor actual del caudal

    Args:
        year: Año específico (None = todos los años)
        freq: Frecuencia de muestreo de la variable espacial ('7D', '1D')
        output_format: 'gif' o 'mp4'
        fps: Frames por segundo
        dpi: Resolución de la animación
        config: Configuración del proyecto

    Returns:
        Path al archivo animado
    """
    print("[ANIM] Creando animación de caudal enrutado (STREAMFLOW)...")

    # ── Cargar caudal enrutado ────────────────────────────────
    sf_csv = ROUTING_DIR / "streamflow_yuncan.csv"
    sf_nc  = ROUTING_DIR / "streamflow_yuncan.nc"

    if sf_nc.exists():
        ds_sf = xr.open_dataset(sf_nc)
        # Buscar variable de caudal (primera variable numérica)
        sf_var = None
        for v in ds_sf.data_vars:
            if ds_sf[v].dtype in [np.float32, np.float64]:
                sf_var = v
                break
        if sf_var is None:
            raise ValueError(f"No se encontró variable numérica en {sf_nc}")
        sf_times = pd.to_datetime(ds_sf.time.values)
        sf_values = ds_sf[sf_var].values.ravel()
        ds_sf.close()
        print(f"  Caudal desde NetCDF: {len(sf_times)} pasos | "
              f"Media={np.nanmean(sf_values):.1f} m³/s | "
              f"Máx={np.nanmax(sf_values):.1f} m³/s")
    elif sf_csv.exists():
        df_sf = pd.read_csv(sf_csv, parse_dates=["time"])
        sf_times  = df_sf["time"].values
        sf_times  = pd.to_datetime(sf_times)
        sf_values = df_sf.iloc[:, 1].values.ravel().astype(float)
        print(f"  Caudal desde CSV: {len(sf_times)} pasos | "
              f"Media={np.nanmean(sf_values):.1f} m³/s | "
              f"Máx={np.nanmax(sf_values):.1f} m³/s")
    else:
        raise FileNotFoundError(
            f"No se encontró caudal enrutado en {sf_csv} ni {sf_nc}. "
            "Ejecuta primero el script 06_run_routing.py."
        )

    # Filtrar por año si se especifica
    if year is not None:
        mask_year = pd.DatetimeIndex(sf_times).year == year
        sf_times  = sf_times[mask_year]
        sf_values = sf_values[mask_year]
        if len(sf_times) == 0:
            raise ValueError(f"No hay datos de caudal para el año {year}")

    # ── Cargar escorrentía total VIC ──────────────────────────
    ds_domain = xr.open_dataset(DOMAIN_DIR / "domain.nc")
    land_mask = ds_domain["mask"].values
    lats = ds_domain.lat.values
    lons = ds_domain.lon.values
    ds_domain.close()

    flux_files = sorted(OUTPUT_DIR.glob("fluxes*.nc"))
    if not flux_files:
        raise FileNotFoundError("No se encontraron salidas VIC en data/outputs/")

    ds_vic = xr.open_mfdataset(flux_files, combine="by_coords")

    # Seleccionar período coincidente con streamflow
    t0, t1 = sf_times[0], sf_times[-1]
    ds_vic = ds_vic.sel(time=slice(str(t0.date()), str(t1.date())))

    # Calcular escorrentía total = superficial + flujo base
    if "OUT_RUNOFF" in ds_vic and "OUT_BASEFLOW" in ds_vic:
        da_runoff = (ds_vic["OUT_RUNOFF"] + ds_vic["OUT_BASEFLOW"]).where(land_mask)
        runoff_label = "Escorrentía Total (Sup. + Base)"
    elif "OUT_RUNOFF" in ds_vic:
        da_runoff = ds_vic["OUT_RUNOFF"].where(land_mask)
        runoff_label = "Escorrentía Superficial"
    else:
        raise ValueError("No se encontró OUT_RUNOFF en las salidas VIC.")

    # Resamplear si frecuencia especificada
    if freq != "1D":
        da_resampled = da_runoff.resample(time=freq).mean()
    else:
        da_resampled = da_runoff

    times_plot = pd.to_datetime(da_resampled.time.values)
    runoff_data = da_resampled.values  # shape: (nframes, nlat, nlon)
    nframes = len(times_plot)
    ds_vic.close()

    print(f"  Frames (escorrentía {freq}): {nframes} | "
          f"Período: {times_plot[0].date()} → {times_plot[-1].date()}")

    # ── Mapear caudal a los mismos timesteps de la animación ──
    # Para cada frame, encontrar el caudal más cercano en tiempo
    sf_times_idx = pd.DatetimeIndex(sf_times)
    sf_frame_values = np.array([
        sf_values[np.argmin(np.abs(sf_times_idx - t))]
        for t in times_plot
    ])

    # ── Escala de colores ─────────────────────────────────────
    valid_r = runoff_data[np.isfinite(runoff_data)]
    vmax_r  = np.nanpercentile(valid_r, 98) if len(valid_r) > 0 else 10.0
    vmin_r  = 0.0
    cmap_r  = cmocean.cm.rain
    norm_r  = mcolors.Normalize(vmin=vmin_r, vmax=max(vmax_r, 0.1))

    outlet = config["basin"]["outlet"]

    # Estadísticas globales del caudal
    sf_mean  = np.nanmean(sf_values)
    sf_max   = np.nanmax(sf_values)
    sf_min   = np.nanmin(sf_values)
    sf_q95   = np.nanpercentile(sf_values, 95)
    sf_q05   = np.nanpercentile(sf_values, 5)

    # ── Crear figura ──────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor="black")
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[3, 1.2],
        height_ratios=[3.5, 1.5],
        hspace=0.08, wspace=0.12,
    )

    ax_map  = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_ts   = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[:, 1])

    # ── Mapa base ─────────────────────────────────────────────
    ax_map.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                      crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="white", alpha=0.5)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="white", alpha=0.3)
    ax_map.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor="deepskyblue", alpha=0.5)
    ax_map.set_facecolor("#0a0a1a")

    # ── Serie temporal de caudal (fondo completo) ─────────────
    ax_ts.set_facecolor("#0a0a1a")
    # Dibujar el período completo de caudal como referencia
    sf_x_full = np.linspace(0, nframes - 1, len(sf_times))
    ax_ts.fill_between(sf_x_full, sf_values, alpha=0.15, color="deepskyblue")
    ax_ts.plot(sf_x_full, sf_values, color="steelblue", linewidth=0.8, alpha=0.7)
    # Línea de la media
    ax_ts.axhline(sf_mean, color="gold", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_ts.set_xlim(0, nframes - 1)
    ax_ts.set_ylim(bottom=0, top=sf_max * 1.15)
    ax_ts.tick_params(colors="white", labelsize=7)
    ax_ts.set_ylabel("Caudal (m³/s)", color="white", fontsize=8)
    ax_ts.set_xlabel("Tiempo", color="white", fontsize=7)
    ax_ts.spines[:].set_color("#444444")
    ax_ts.text(nframes * 0.01, sf_mean * 1.05,
               f"Media: {sf_mean:.1f}", color="gold", fontsize=6, alpha=0.8)

    # Etiquetas del eje X (fechas)
    tick_positions = np.linspace(0, nframes - 1, min(6, nframes)).astype(int)
    tick_labels = [times_plot[i].strftime("%b'%y") for i in tick_positions]
    ax_ts.set_xticks(tick_positions)
    ax_ts.set_xticklabels(tick_labels, color="white", fontsize=6)

    # ── Panel de info (estático) ──────────────────────────────
    ax_info.set_facecolor("black")
    ax_info.axis("off")

    # Título de la cuenca
    ax_info.text(
        0.5, 0.97,
        f"Cuenca {config['basin']['name']}\n{config['basin']['department']}, Perú",
        transform=ax_info.transAxes,
        color="white", fontsize=9, ha="center", va="top", fontweight="bold"
    )
    ax_info.text(
        0.5, 0.88,
        f"Central H. {outlet['name']}",
        transform=ax_info.transAxes,
        color="deepskyblue", fontsize=8, ha="center", va="top"
    )

    # Separador
    ax_info.plot([0, 1], [0.85, 0.85], color="#444444", linewidth=0.5,
                 transform=ax_info.transAxes, clip_on=False)

    # Estadísticas globales (estáticas)
    stats_global = (
        f"  ─── Estadísticas ───\n"
        f"  Media:   {sf_mean:6.1f} m³/s\n"
        f"  Máximo:  {sf_max:6.1f} m³/s\n"
        f"  Mínimo:  {sf_min:6.1f} m³/s\n"
        f"  Q95:     {sf_q95:6.1f} m³/s\n"
        f"  Q05:     {sf_q05:6.1f} m³/s"
    )
    ax_info.text(
        0.05, 0.79,
        stats_global,
        transform=ax_info.transAxes,
        color="#aaaaaa", fontsize=7.5,
        ha="left", va="top", family="monospace"
    )

    # Separador
    ax_info.plot([0, 1], [0.52, 0.52], color="#444444", linewidth=0.5,
                 transform=ax_info.transAxes, clip_on=False)

    # Colorbar de escorrentía
    sm = plt.cm.ScalarMappable(cmap=cmap_r, norm=norm_r)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.77, 0.20, 0.02, 0.30])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"{runoff_label}\n(mm/día)", color="white", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white", fontsize=6)
    cbar.outline.set_edgecolor("#555555")

    # ── Elementos dinámicos ───────────────────────────────────
    frame0 = np.where(land_mask == 0, np.nan, runoff_data[0])
    im = ax_map.pcolormesh(
        lons, lats, frame0,
        transform=ccrs.PlateCarree(),
        cmap=cmap_r,
        norm=norm_r,
        rasterized=True,
    )

    # Marcador del outlet
    ax_map.plot(outlet["lon"], outlet["lat"],
                "^", color="yellow", markersize=12,
                transform=ccrs.PlateCarree(), zorder=10,
                markeredgecolor="white", markeredgewidth=0.5)
    ax_map.text(outlet["lon"] + 0.05, outlet["lat"] + 0.05,
                outlet["name"], color="yellow", fontsize=7,
                transform=ccrs.PlateCarree(), zorder=11)

    # Título del mapa (dinámico)
    date_text = ax_map.set_title(
        "", color="white", fontsize=10, pad=5,
        fontweight="bold"
    )

    # Cursor temporal en la serie de caudal
    vline = ax_ts.axvline(x=0, color="red", linewidth=1.5, zorder=5)
    time_dot = ax_ts.plot(
        [0], [sf_frame_values[0]],
        "o", color="red", markersize=6, zorder=6
    )[0]

    # Caudal actual (panel dinámico en ax_info)
    current_q_text = ax_info.text(
        0.5, 0.47,
        "",
        transform=ax_info.transAxes,
        color="white", fontsize=22,
        ha="center", va="top", fontweight="bold"
    )
    current_unit_text = ax_info.text(
        0.5, 0.35,
        "m³/s",
        transform=ax_info.transAxes,
        color="#aaaaaa", fontsize=10,
        ha="center", va="top"
    )
    current_label = ax_info.text(
        0.5, 0.30,
        "Caudal enrutado",
        transform=ax_info.transAxes,
        color="#888888", fontsize=7.5,
        ha="center", va="top"
    )

    # Barra de progreso del caudal usando un axes secundario en ax_info
    ax_bar = ax_info.inset_axes([0.05, 0.10, 0.90, 0.06])
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_facecolor("#222222")
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.spines[:].set_visible(False)
    bar_patch = plt.Rectangle((0, 0), 0.0, 1.0, color="deepskyblue")
    ax_bar.add_patch(bar_patch)

    fig.patch.set_facecolor("black")

    # ── Función de actualización ──────────────────────────────
    def update(frame_idx):
        # Actualizar mapa de escorrentía
        frame_data = np.where(land_mask == 0, np.nan, runoff_data[frame_idx])
        im.set_array(frame_data.ravel())

        # Actualizar fecha en el título del mapa
        date_str = times_plot[frame_idx].strftime("%d %b %Y")
        date_text.set_text(f"Escorrentía → Caudal   {date_str}")

        # Actualizar cursor temporal
        vline.set_xdata([frame_idx, frame_idx])
        q_now = float(sf_frame_values[frame_idx])
        time_dot.set_data([frame_idx], [q_now])

        # Actualizar caudal actual en el panel de info
        current_q_text.set_text(f"{q_now:.1f}")

        # Color del texto según comparación con la media
        if q_now > sf_q95:
            current_q_text.set_color("#ff4444")
        elif q_now > sf_mean:
            current_q_text.set_color("deepskyblue")
        else:
            current_q_text.set_color("#aaddff")

        # Actualizar barra de progreso
        frac = min(max((q_now - sf_min) / max(sf_max - sf_min, 0.01), 0), 1)
        bar_patch.set_width(frac)

        return [im, date_text, vline, time_dot, current_q_text, bar_patch]

    # ── Crear animación ───────────────────────────────────────
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
    output_name = f"animation_STREAMFLOW_{year_str}.{output_format}"
    output_path = ANIMS_DIR / output_name

    print(f"  Guardando {output_path}...")
    if output_format == "mp4":
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2500,
                                  metadata={"title": "Caudal Yuncan - Paucartambo"})
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
    print(f"[OK] Animación de caudal guardada: {output_path}")
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
                        help="Año a animar (None = todos)")
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
            args.year, args.freq,
            args.output_format, args.fps, args.dpi, config
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

    print(f"\n[✓] Animaciones guardadas en: {ANIMS_DIR}")


if __name__ == "__main__":
    main()
