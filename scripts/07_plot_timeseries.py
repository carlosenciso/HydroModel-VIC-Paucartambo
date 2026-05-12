#!/usr/bin/env python3
"""
Script 07: Visualización de series temporales
=============================================
Genera gráficos de series temporales de:
  1. Caudal en la Central Hidroeléctrica Yuncan
  2. Balance hídrico de la cuenca (P, ET, Q, ΔS)
  3. Componentes del escurrimiento
  4. Análisis de frecuencias (curva de duración de caudales)
  5. Comparación mensual/estacional

Uso:
    python scripts/07_plot_timeseries.py
    python scripts/07_plot_timeseries.py --output-format pdf
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
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import xarray as xr
import yaml

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent.parent
CONFIG_FILE  = PROJECT_DIR / "config" / "basin_config.yml"
DATA_DIR     = PROJECT_DIR / "data"
OUTPUT_DIR   = DATA_DIR / "outputs"
ROUTING_DIR  = DATA_DIR / "routing"
DOMAIN_DIR   = DATA_DIR / "domain"
PLOTS_DIR    = PROJECT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Paleta de colores
COLORS = {
    "precip":    "#1565C0",
    "et":        "#2E7D32",
    "runoff":    "#FF6F00",
    "baseflow":  "#6A1B9A",
    "swe":       "#00838F",
    "streamflow": "#D32F2F",
    "soil_moist": "#795548",
}


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def load_streamflow() -> pd.Series:
    """Cargar serie de caudal en Yuncan."""
    nc_file = ROUTING_DIR / "streamflow_yuncan.nc"
    csv_file = ROUTING_DIR / "streamflow_yuncan.csv"

    if csv_file.exists():
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        return df["streamflow_m3s"]
    elif nc_file.exists():
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        return df["streamflow_m3s"]
    else:
        raise FileNotFoundError(
            "No se encontraron resultados de enrutamiento.\n"
            "Ejecuta primero: python scripts/06_run_routing.py"
        )


def load_vic_fluxes() -> xr.Dataset:
    """Cargar fluxes de VIC (medias espaciales)."""
    flux_files = sorted(OUTPUT_DIR.glob("fluxes*.nc"))
    if not flux_files:
        raise FileNotFoundError("No se encontraron salidas VIC.")
    return xr.open_mfdataset(flux_files, combine="by_coords")


def basin_mean(ds: xr.Dataset, var: str, mask: np.ndarray = None) -> pd.Series:
    """Calcular media espacial de una variable sobre la cuenca."""
    if var not in ds:
        return None

    da = ds[var]
    if mask is not None:
        da = da.where(mask)

    # Average over spatial dims; sum over any layer dim so result is 1D (time,)
    spatial_mean = da.mean(dim=["lat", "lon"])
    for extra_dim in ["nlayer", "frost_area_fract", "snow_band"]:
        if extra_dim in spatial_mean.dims:
            spatial_mean = spatial_mean.sum(dim=extra_dim)

    series = spatial_mean.to_series()
    series.index = pd.to_datetime(series.index)
    return series


def set_axis_style(ax, xlabel="", ylabel="", title="", fontsize=11):
    """Estilo consistente para los ejes."""
    ax.set_xlabel(xlabel, fontsize=fontsize - 1)
    ax.set_ylabel(ylabel, fontsize=fontsize - 1)
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)
    ax.tick_params(labelsize=fontsize - 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")
    return ax


# ─────────────────────────────────────────────────────────
# 1. Caudal en Yuncan
# ─────────────────────────────────────────────────────────

def plot_streamflow_timeseries(
    streamflow: pd.Series,
    config: dict,
    output_fmt: str = "png",
) -> None:
    """Plot de serie temporal de caudal en Yuncan."""
    outlet = config["basin"]["outlet"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        f"Caudal Simulado - {outlet['name']}\n"
        f"Cuenca {config['basin']['name']}, {config['basin']['department']} - Peru",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Panel 1: Serie completa
    ax1 = axes[0]
    ax1.fill_between(streamflow.index, streamflow.values,
                     alpha=0.3, color=COLORS["streamflow"])
    ax1.plot(streamflow.index, streamflow.values,
             color=COLORS["streamflow"], linewidth=0.8, label="Caudal diario")

    # Media móvil 30 días
    q_smooth = streamflow.rolling(30, center=True).mean()
    ax1.plot(streamflow.index, q_smooth,
             color="darkred", linewidth=2, label="Media móvil 30d")

    set_axis_style(ax1, ylabel="Caudal (m³/s)",
                   title="Serie Temporal de Caudal")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # Panel 2: Caudal mensual promedio
    ax2 = axes[1]
    q_monthly = streamflow.resample("ME").mean()
    ax2.bar(q_monthly.index, q_monthly.values,
            width=25, color=COLORS["streamflow"], alpha=0.7)
    ax2.plot(q_monthly.index, q_monthly.values,
             "o-", color="darkred", linewidth=1.5, markersize=4)

    set_axis_style(ax2, ylabel="Caudal (m³/s)",
                   title="Caudal Mensual Promedio")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Panel 3: Ciclo estacional
    ax3 = axes[2]
    q_seasonal = streamflow.groupby(streamflow.index.month).mean()
    q_std = streamflow.groupby(streamflow.index.month).std()
    months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
              "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    x = np.arange(1, 13)

    ax3.fill_between(x,
                     q_seasonal.values - q_std.values,
                     q_seasonal.values + q_std.values,
                     alpha=0.2, color=COLORS["streamflow"])
    ax3.plot(x, q_seasonal.values, "o-",
             color=COLORS["streamflow"], linewidth=2,
             markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax3.set_xticks(x)
    ax3.set_xticklabels(months)

    set_axis_style(ax3, ylabel="Caudal (m³/s)",
                   title="Régimen Estacional (promedio ± 1σ)")

    # Añadir estadísticas
    stats_text = (
        f"Media: {streamflow.mean():.0f} m³/s | "
        f"Máx: {streamflow.max():.0f} m³/s | "
        f"Mín: {streamflow.min():.1f} m³/s"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    output_file = PLOTS_DIR / f"streamflow_timeseries.{output_fmt}"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {output_file}")


# ─────────────────────────────────────────────────────────
# 2. Balance hídrico
# ─────────────────────────────────────────────────────────

def plot_water_balance(
    ds_vic: xr.Dataset,
    config: dict,
    output_fmt: str = "png",
) -> None:
    """Plot del balance hídrico de la cuenca."""
    # Medias espaciales
    mask_file = DOMAIN_DIR / "domain.nc"
    mask = None
    if mask_file.exists():
        ds_domain = xr.open_dataset(mask_file)
        mask = ds_domain["mask"].values == 1
        ds_domain.close()

    prec = basin_mean(ds_vic, "OUT_PREC", mask)
    et   = basin_mean(ds_vic, "OUT_EVAP", mask)
    ro   = basin_mean(ds_vic, "OUT_RUNOFF", mask)
    bf   = basin_mean(ds_vic, "OUT_BASEFLOW", mask)
    swe  = basin_mean(ds_vic, "OUT_SWE", mask)
    sm   = basin_mean(ds_vic, "OUT_SOIL_MOIST", mask)

    if prec is None:
        print("[WARNING] OUT_PREC no encontrado. Saltando plot de balance.")
        return

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.35)

    fig.suptitle(
        f"Balance Hídrico - Cuenca {config['basin']['name']}, Peru\n",
        fontsize=14, fontweight="bold"
    )

    # Subplots
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    # ── Panel 1: Precipitación y ET ──────────────────────
    if prec is not None:
        prec_monthly = prec.resample("ME").sum()
        et_monthly = et.resample("ME").sum() if et is not None else None

        ax1.bar(prec_monthly.index, prec_monthly.values,
                width=25, color=COLORS["precip"], alpha=0.7, label="Precipitación")
        if et_monthly is not None:
            ax1.bar(et_monthly.index, et_monthly.values,
                    width=25, color=COLORS["et"], alpha=0.7, label="ET")

        set_axis_style(ax1, ylabel="mm/mes",
                       title="Precipitación y Evapotranspiración Mensual")
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.xaxis.set_major_locator(mdates.YearLocator())

    # ── Panel 2: Componentes del escurrimiento ─────────
    if ro is not None and bf is not None:
        ro_monthly = ro.resample("ME").sum()
        bf_monthly = bf.resample("ME").sum()

        ax2.stackplot(
            ro_monthly.index,
            ro_monthly.values,
            bf_monthly.values,
            labels=["Escorrentía directa", "Flujo base"],
            colors=[COLORS["runoff"], COLORS["baseflow"]],
            alpha=0.7,
        )
        set_axis_style(ax2, ylabel="mm/mes",
                       title="Componentes del Escurrimiento")
        ax2.legend(fontsize=8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Panel 3: Ciclo estacional de P y ET ────────────
    if prec is not None:
        months = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        prec_seas = prec.groupby(prec.index.month).mean() * 30
        et_seas = et.groupby(et.index.month).mean() * 30 if et is not None else None

        x = np.arange(1, 13)
        ax3.bar(x - 0.2, prec_seas.values, width=0.35,
                color=COLORS["precip"], label="Precipitación")
        if et_seas is not None:
            ax3.bar(x + 0.2, et_seas.values, width=0.35,
                    color=COLORS["et"], label="ET")
        ax3.set_xticks(x)
        ax3.set_xticklabels(months, fontsize=9)
        set_axis_style(ax3, ylabel="mm/mes",
                       title="Ciclo Estacional (promedio)")
        ax3.legend(fontsize=8)

    # ── Panel 4: Humedad del suelo ─────────────────────
    if sm is not None:
        ax4.plot(sm.index, sm.values, color=COLORS["soil_moist"],
                 linewidth=1, alpha=0.8)
        sm_smooth = sm.rolling(30).mean()
        ax4.plot(sm_smooth.index, sm_smooth.values,
                 color="black", linewidth=2, label="Media móvil 30d")
        set_axis_style(ax4, ylabel="mm",
                       title="Humedad del Suelo (total)")
        ax4.legend(fontsize=8)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Panel 5: SWE ────────────────────────────────────
    if swe is not None and swe.max() > 0.1:
        ax5.fill_between(swe.index, swe.values,
                         color=COLORS["swe"], alpha=0.5)
        ax5.plot(swe.index, swe.values, color=COLORS["swe"], linewidth=1)
        set_axis_style(ax5, ylabel="mm SWE",
                       title="Equivalente Agua en Nieve (SWE)")
        ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax5.text(0.5, 0.5, "SWE ≈ 0\n(sin cobertura de nieve significativa)",
                 ha="center", va="center", transform=ax5.transAxes,
                 fontsize=11, color="gray")
        set_axis_style(ax5, title="Equivalente Agua en Nieve (SWE)")

    output_file = PLOTS_DIR / f"water_balance.{output_fmt}"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {output_file}")


# ─────────────────────────────────────────────────────────
# 3. Curva de Duración de Caudales
# ─────────────────────────────────────────────────────────

def plot_flow_duration_curve(
    streamflow: pd.Series,
    config: dict,
    output_fmt: str = "png",
) -> None:
    """Curva de duración de caudales (CDC)."""
    outlet = config["basin"]["outlet"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Curva de Duración de Caudales - {outlet['name']}",
        fontsize=13, fontweight="bold"
    )

    # Calcular excedencias
    sorted_q = np.sort(streamflow.dropna().values)[::-1]
    exceedance = np.arange(1, len(sorted_q) + 1) / len(sorted_q) * 100

    # Panel 1: Escala lineal
    ax1 = axes[0]
    ax1.plot(exceedance, sorted_q, color=COLORS["streamflow"], linewidth=2)
    ax1.fill_between(exceedance, sorted_q, alpha=0.2, color=COLORS["streamflow"])
    set_axis_style(ax1,
                   xlabel="Probabilidad de excedencia (%)",
                   ylabel="Caudal (m³/s)",
                   title="Escala lineal")

    # Caudales característicos
    for prob, label in [(10, "Q10"), (50, "Q50"), (90, "Q90"), (95, "Q95")]:
        q_val = np.percentile(sorted_q, 100 - prob)
        ax1.axvline(prob, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax1.annotate(f"{label}={q_val:.0f}",
                    xy=(prob, q_val), fontsize=8, color="gray",
                    xytext=(5, 0), textcoords="offset points")

    # Panel 2: Escala log-log
    ax2 = axes[1]
    ax2.semilogy(exceedance, sorted_q, color=COLORS["streamflow"], linewidth=2)
    ax2.fill_between(exceedance, sorted_q, 1e-3, alpha=0.2, color=COLORS["streamflow"])
    set_axis_style(ax2,
                   xlabel="Probabilidad de excedencia (%)",
                   ylabel="Caudal (m³/s) - escala log",
                   title="Escala semilogarítmica")

    plt.tight_layout()
    output_file = PLOTS_DIR / f"flow_duration_curve.{output_fmt}"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {output_file}")


# ─────────────────────────────────────────────────────────
# 4. Estadísticas de resumen (tabla)
# ─────────────────────────────────────────────────────────

def save_summary_stats(
    streamflow: pd.Series,
    ds_vic: xr.Dataset,
    config: dict,
) -> None:
    """Guardar tabla de estadísticas de resumen."""
    outlet = config["basin"]["outlet"]

    stats = {
        "Variable": [],
        "Unidad": [],
        "Media": [],
        "Std": [],
        "Min": [],
        "Max": [],
        "P5": [],
        "P50": [],
        "P95": [],
    }

    # Caudal
    q = streamflow.dropna()
    stats["Variable"].append(f"Caudal ({outlet['name']})")
    stats["Unidad"].append("m³/s")
    stats["Media"].append(f"{q.mean():.2f}")
    stats["Std"].append(f"{q.std():.2f}")
    stats["Min"].append(f"{q.min():.3f}")
    stats["Max"].append(f"{q.max():.2f}")
    stats["P5"].append(f"{q.quantile(0.05):.2f}")
    stats["P50"].append(f"{q.median():.2f}")
    stats["P95"].append(f"{q.quantile(0.95):.2f}")

    # Variables VIC (medias espaciales)
    vic_vars = {
        "OUT_PREC": ("Precipitación", "mm/day"),
        "OUT_EVAP": ("ET", "mm/day"),
        "OUT_RUNOFF": ("Escorrentía", "mm/day"),
        "OUT_BASEFLOW": ("Flujo base", "mm/day"),
        "OUT_SWE": ("SWE", "mm"),
    }

    for var, (name, unit) in vic_vars.items():
        if var in ds_vic:
            v = ds_vic[var].mean(dim=["lat", "lon"]).values
            stats["Variable"].append(name)
            stats["Unidad"].append(unit)
            stats["Media"].append(f"{np.nanmean(v):.3f}")
            stats["Std"].append(f"{np.nanstd(v):.3f}")
            stats["Min"].append(f"{np.nanmin(v):.3f}")
            stats["Max"].append(f"{np.nanmax(v):.3f}")
            stats["P5"].append(f"{np.nanpercentile(v, 5):.3f}")
            stats["P50"].append(f"{np.nanpercentile(v, 50):.3f}")
            stats["P95"].append(f"{np.nanpercentile(v, 95):.3f}")

    import pandas as pd
    df_stats = pd.DataFrame(stats)
    csv_file = PLOTS_DIR / "summary_statistics.csv"
    df_stats.to_csv(csv_file, index=False)
    print(f"[OK] Estadísticas: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualización de series temporales VIC - Paucartambo"
    )
    parser.add_argument("--output-format", choices=["png", "pdf", "svg"],
                        default="png", help="Formato de salida (default: png)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    print("=" * 60)
    print("  VISUALIZACIÓN DE SERIES TEMPORALES - Paucartambo")
    print("=" * 60)

    config = load_config()
    fmt = args.output_format
    matplotlib.rcParams["figure.dpi"] = args.dpi

    # Cargar datos
    print("\n[→] Cargando datos...")
    streamflow = load_streamflow()
    print(f"  Caudal: {len(streamflow)} días")

    try:
        ds_vic = load_vic_fluxes()
        print(f"  Fluxes VIC: {list(ds_vic.data_vars)}")
    except FileNotFoundError as e:
        print(f"  [WARNING] {e}")
        ds_vic = None

    # Generar plots
    print("\n[→] Generando gráficos...")

    print("  [1/4] Serie temporal de caudal...")
    plot_streamflow_timeseries(streamflow, config, fmt)

    if ds_vic is not None:
        print("  [2/4] Balance hídrico...")
        plot_water_balance(ds_vic, config, fmt)

    print("  [3/4] Curva de duración de caudales...")
    plot_flow_duration_curve(streamflow, config, fmt)

    if ds_vic is not None:
        print("  [4/4] Estadísticas de resumen...")
        save_summary_stats(streamflow, ds_vic, config)

    print(f"\n[✓] Gráficos guardados en: {PLOTS_DIR}")
    if ds_vic is not None:
        ds_vic.close()


if __name__ == "__main__":
    main()
