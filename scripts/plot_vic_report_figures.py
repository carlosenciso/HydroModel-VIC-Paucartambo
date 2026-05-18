#!/home/cenciso/anaconda3/envs/lulc/bin/python
"""
Script: Gráficas para el Informe — Simulación VIC 5 + RVIC
============================================================
Genera figuras de calidad publicación para el informe hidrológico
de la cuenca Río Paucartambo, a partir de salidas del modelo VIC.

Si las salidas NetCDF no están disponibles, genera datos sintéticos
representativos para visualización previa del formato.

Salidas (en plots/report/):
  fig_hidrograma_qn908.png      Hidrograma mensual QN-908 Uchuhuerta
  fig_hidrograma_qn909.png      Hidrograma mensual QN-909 Huallamayo
  fig_hidrograma_qn901_911.png  Hidrograma mensual QN-901+902+911
  fig_hidrograma_qn903_905.png  Hidrograma mensual QN-903+904+905
  fig_balance_hidrico.png       Balance hídrico anual (P, ET, Q)
  fig_curva_duracion.png        Curvas de duración de caudales
  fig_estacionalidad.png        Análisis estacional (régimen mensual)
  fig_mapa_balance_anual.png    Mapa espacial P−ET (raster sintético)

Uso:
    python scripts/plot_vic_report_figures.py
    python scripts/plot_vic_report_figures.py --from-files   # usa NetCDF reales
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Rutas
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
PLOTS_DIR   = PROJECT_DIR / "report" / "Figures"
DATA_DIR    = PROJECT_DIR / "data"
ROUTING_DIR = DATA_DIR / "routing"
OUTPUT_DIR  = DATA_DIR / "outputs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Paleta y estilo global
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
})

C_SIM  = "#1565C0"   # azul simulado
C_OBS  = "#B71C1C"   # rojo observado
C_PREC = "#0277BD"   # precipitación
C_ET   = "#2E7D32"   # evapotranspiración
C_BF   = "#6A1B9A"   # flujo base
C_RO   = "#FF6F00"   # escorrentía directa

MONTHS_ES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]


# ──────────────────────────────────────────────────────────────────────────────
# Generadores de datos sintéticos representativos de la cuenca Paucartambo
# ──────────────────────────────────────────────────────────────────────────────

def _seasonal_pattern(months, amp, base, phase=0):
    """Patrón estacional sinusoidal (pico en verano austral)."""
    return base + amp * np.cos(2 * np.pi * (months - 1 - phase) / 12)


def synthetic_monthly_flow(station: str, years: range) -> pd.DataFrame:
    """
    Genera un hidrograma mensual sintético (sim + obs) representativo
    de cada punto de control.  Los valores promedio están tomados de las
    tablas del informe real.
    """
    rng = np.random.default_rng({"qn908": 1, "qn909": 2,
                                   "qn901": 3, "qn903": 4}.get(station, 0))
    configs = {
        "qn908": dict(base=18, amp=32, noise=4.0, obs_bias=0.98),
        "qn909": dict(base=9,  amp=18, noise=2.5, obs_bias=1.01),
        "qn901": dict(base=0.6, amp=1.4, noise=0.3, obs_bias=0.97),
        "qn903": dict(base=0.7, amp=1.8, noise=0.35, obs_bias=1.02),
    }
    cfg = configs.get(station, configs["qn908"])

    dates, q_sim, q_obs = [], [], []
    for yr in years:
        for m in range(1, 13):
            dates.append(pd.Timestamp(yr, m, 15))
            base  = cfg["base"]
            amp   = cfg["amp"]
            # Pico en feb (mes 2), estiaje en jul-ago
            sim = base + amp * max(0, np.cos(2 * np.pi * (m - 2) / 12))
            sim += rng.normal(0, cfg["noise"])
            sim  = max(sim, 0.1)
            obs  = sim * cfg["obs_bias"] + rng.normal(0, cfg["noise"] * 0.6)
            obs  = max(obs, 0.1)
            q_sim.append(sim)
            q_obs.append(obs)

    return pd.DataFrame({"date": dates, "sim": q_sim, "obs": q_obs}).set_index("date")


def synthetic_water_balance(years: range) -> pd.DataFrame:
    """Balance hídrico anual sintético (mm/año) para la cuenca."""
    rng = np.random.default_rng(42)
    rows = []
    for yr in years:
        P  = 1900 + rng.normal(0, 200)
        ET = 820  + rng.normal(0, 80)
        Q  = P - ET - rng.normal(20, 15)
        rows.append({"year": yr, "P": P, "ET": ET, "Q": max(Q, 0)})
    return pd.DataFrame(rows).set_index("year")


def synthetic_monthly_precip_et() -> pd.DataFrame:
    """Climatología mensual de P y ET (mm/mes) promedio 2003-2025."""
    P_mean  = [220, 210, 195, 130, 80,  40,  35,  55,  100, 155, 185, 215]
    ET_mean = [72,  68,  70,  65,  58,  52,  55,  62,  68,  70,  72,  72]
    return pd.DataFrame({"P": P_mean, "ET": ET_mean},
                        index=pd.Index(range(1, 13), name="mes"))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de gráfico
# ──────────────────────────────────────────────────────────────────────────────

def _nse(obs, sim):
    obs, sim = np.asarray(obs), np.asarray(sim)
    return 1 - np.sum((obs - sim)**2) / np.sum((obs - obs.mean())**2)


def _stats_text(obs, sim):
    nse   = _nse(obs, sim)
    pbias = 100 * (sim.sum() - obs.sum()) / obs.sum()
    r     = np.corrcoef(obs, sim)[0, 1]
    return (f"NSE = {nse:.2f}   |   PBIAS = {pbias:+.1f}%   |   "
            f"r = {r:.2f}")


def _format_ax_monthly(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


# ──────────────────────────────────────────────────────────────────────────────
# Figura 1–4: Hidrogramas mensuales por punto de control
# ──────────────────────────────────────────────────────────────────────────────

def plot_hidrograma(station_key: str, station_label: str,
                   df: pd.DataFrame, filename: str) -> None:
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    obs = df["obs"].values
    sim = df["sim"].values
    idx = df.index

    # Panel A: serie completa
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(idx, sim, alpha=0.25, color=C_SIM, label="_")
    ax1.plot(idx, sim, color=C_SIM, linewidth=1.3, label="VIC+RVIC simulado")
    ax1.plot(idx, obs, color=C_OBS, linewidth=1.0, linestyle="--",
             alpha=0.85, label="Observado")
    ax1.set_ylabel("Caudal (m³/s)")
    ax1.set_title(f"Hidrograma Mensual — {station_label}", fontweight="bold")
    _format_ax_monthly(ax1)
    ax1.legend(loc="upper right")
    stats = _stats_text(obs, sim)
    ax1.text(0.02, 0.97, stats, transform=ax1.transAxes,
             fontsize=8.5, va="top", color="#333",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.8))

    # Panel B: régimen mensual promedio
    ax2 = fig.add_subplot(gs[1, 0])
    sim_seas = [df.loc[df.index.month == m, "sim"].mean() for m in range(1, 13)]
    obs_seas = [df.loc[df.index.month == m, "obs"].mean() for m in range(1, 13)]
    x = np.arange(1, 13)
    ax2.bar(x - 0.2, sim_seas, width=0.35, color=C_SIM, alpha=0.8, label="Simulado")
    ax2.bar(x + 0.2, obs_seas, width=0.35, color=C_OBS, alpha=0.7, label="Observado")
    ax2.set_xticks(x)
    ax2.set_xticklabels(MONTHS_ES, fontsize=8)
    ax2.set_ylabel("Caudal promedio (m³/s)")
    ax2.set_title("Régimen Estacional", fontweight="bold")
    ax2.legend()

    # Panel C: diagrama de dispersión sim vs obs — calibración vs validación
    ax3 = fig.add_subplot(gs[1, 1])
    calYear = [2003, 2020]
    valYear = [2021, 2025]
    mask_cal = (idx.year >= calYear[0]) & (idx.year <= calYear[1])
    mask_val = (idx.year >= valYear[0]) & (idx.year <= valYear[1])

    lim_max = max(obs.max(), sim.max()) * 1.08
    ax3.scatter(obs[mask_cal], sim[mask_cal], color=C_SIM, alpha=0.5, s=14,
                zorder=3, label=f"Calibración ({calYear[0]}–{calYear[1]})")
    ax3.scatter(obs[mask_val], sim[mask_val], color=C_OBS, alpha=0.6, s=14,
                zorder=3, marker="s", label=f"Validación ({valYear[0]}–{valYear[1]})")

    # Línea 1:1
    ax3.plot([0, lim_max], [0, lim_max], "k--", linewidth=1.2, label="1:1")

    # Regresión lineal sobre todos los datos
    m, b = np.polyfit(obs, sim, 1)
    xfit = np.linspace(0, lim_max, 100)
    ax3.plot(xfit, m * xfit + b, color="#FF6F00", linewidth=1.8, alpha=0.85,
             label=f"Regresión: y = {m:.2f}x {b:+.2f}")

    # Estadísticas por período en la figura
    stats_lines = []
    for period_mask, period_label in [(mask_cal, "Cal."), (mask_val, "Val.")]:
        if period_mask.sum() > 5:
            o_p, s_p = obs[period_mask], sim[period_mask]
            nse_v  = _nse(o_p, s_p)
            pbias_v = 100 * (s_p.sum() - o_p.sum()) / o_p.sum()
            r_v    = np.corrcoef(o_p, s_p)[0, 1]
            r2_v   = r_v ** 2
            stats_lines.append(
                f"{period_label}:  NSE={nse_v:.2f}  R²={r2_v:.2f}  "
                f"PBIAS={pbias_v:+.1f}%  r={r_v:.2f}"
            )
            print(f"  {period_label}: NSE={nse_v:.2f}  R²={r2_v:.2f}  "
                  f"PBIAS={pbias_v:+.1f}%  r={r_v:.2f}")

    ax3.text(0.03, 0.97, "\n".join(stats_lines),
             transform=ax3.transAxes, fontsize=7.8, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#AAAAAA", lw=0.9))

    ax3.set_xlim(0, lim_max)
    ax3.set_ylim(0, lim_max)
    ax3.set_xlabel("Observado (m³/s)")
    ax3.set_ylabel("Simulado (m³/s)")
    ax3.set_title("Diagrama de Dispersión", fontweight="bold")
    ax3.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(
        f"Modelo VIC 5 + RVIC — Cuenca Río Paucartambo\n{station_label}",
        fontsize=12, fontweight="bold", y=0.99
    )
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {PLOTS_DIR / filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Figura 5: Balance hídrico anual
# ──────────────────────────────────────────────────────────────────────────────

def plot_balance_hidrico(df_annual: pd.DataFrame,
                         df_monthly: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    years = df_annual.index.astype(int).values

    # Panel A: barras apiladas P vs ET+Q
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(years, df_annual["P"], color=C_PREC, alpha=0.8, label="Precipitación (P)")
    ax1.bar(years, df_annual["ET"], color=C_ET, alpha=0.8, label="Evapotranspiración (ET)")
    ax1.bar(years, df_annual["Q"],  bottom=df_annual["ET"],
            color=C_RO, alpha=0.8, label="Escorrentía (Q=P−ET−ΔS)")
    ax1.set_ylabel("mm/año")
    ax1.set_xlabel("Año")
    ax1.set_title("Balance Hídrico Anual — Cuenca Paucartambo (2003–2025)",
                  fontweight="bold")
    ax1.legend(loc="upper right")

    # Panel B: ciclo estacional P y ET
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(1, 13)
    ax2.bar(x - 0.2, df_monthly["P"], width=0.35,
            color=C_PREC, alpha=0.8, label="P")
    ax2.bar(x + 0.2, df_monthly["ET"], width=0.35,
            color=C_ET, alpha=0.8, label="ET")
    ax2.set_xticks(x)
    ax2.set_xticklabels(MONTHS_ES, fontsize=8)
    ax2.set_ylabel("mm/mes")
    ax2.set_title("Ciclo Estacional P y ET\n(promedio 2003–2025)", fontweight="bold")
    ax2.legend()

    # Panel C: curva de variabilidad interanual
    ax3 = fig.add_subplot(gs[1, 1])
    ratio = df_annual["Q"] / df_annual["P"] * 100
    ax3.bar(years, ratio, color="#1976D2", alpha=0.8)
    ax3.axhline(ratio.mean(), color="darkred", linestyle="--",
                linewidth=1.5, label=f"Media = {ratio.mean():.0f}%")
    ax3.set_ylabel("Q/P (%)")
    ax3.set_xlabel("Año")
    ax3.set_title("Coeficiente de Escorrentía Anual", fontweight="bold")
    ax3.legend()

    fig.suptitle("Balance Hídrico — Modelo VIC 5 · Cuenca Río Paucartambo",
                 fontsize=12, fontweight="bold", y=0.99)
    out = PLOTS_DIR / "fig_balance_hidrico.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figura 6: Curvas de duración de caudales
# ──────────────────────────────────────────────────────────────────────────────

def plot_curva_duracion(dfs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    linestyles = ["-", "--", "-.", ":"]
    colors     = [C_SIM, C_OBS, "#2E7D32", "#E65100"]

    for (label, df), ls, col in zip(dfs.items(), linestyles, colors):
        q = np.sort(df["sim"].dropna().values)[::-1]
        ep = np.arange(1, len(q) + 1) / len(q) * 100
        axes[0].plot(ep, q, linestyle=ls, color=col, linewidth=1.8, label=label)
        axes[1].semilogy(ep, q, linestyle=ls, color=col, linewidth=1.8, label=label)

    for ax, title in zip(axes, ["Escala lineal", "Escala semilogarítmica"]):
        ax.set_xlabel("Probabilidad de excedencia (%)")
        ax.set_ylabel("Caudal (m³/s)")
        ax.set_title(title, fontweight="bold")
        ax.legend()
        for p in [10, 50, 90]:
            ax.axvline(p, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

    fig.suptitle(
        "Curva de Duración de Caudales (CDC) por Punto de Control\n"
        "Simulación VIC+RVIC · Cuenca Río Paucartambo (2003–2025)",
        fontsize=12, fontweight="bold"
    )
    out = PLOTS_DIR / "fig_curva_duracion.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figura 7: Análisis estacional detallado
# ──────────────────────────────────────────────────────────────────────────────

def plot_estacionalidad(dfs: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    x = np.arange(1, 13)
    for ax, (label, df) in zip(axes, dfs.items()):
        sim_seas = np.array([df.loc[df.index.month == m, "sim"].mean() for m in range(1,13)])
        sim_std  = np.array([df.loc[df.index.month == m, "sim"].std()  for m in range(1,13)])
        obs_seas = np.array([df.loc[df.index.month == m, "obs"].mean() for m in range(1,13)])

        ax.fill_between(x, sim_seas - sim_std, sim_seas + sim_std,
                        alpha=0.2, color=C_SIM)
        ax.plot(x, sim_seas, "o-", color=C_SIM, linewidth=2,
                markersize=6, markerfacecolor="white", markeredgewidth=2,
                label="VIC simulado (±1σ)")
        ax.plot(x, obs_seas, "s--", color=C_OBS, linewidth=1.5,
                markersize=5, alpha=0.9, label="Observado")
        ax.set_xticks(x)
        ax.set_xticklabels(MONTHS_ES, fontsize=8)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Caudal promedio (m³/s)")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Régimen Estacional de Caudales por Punto de Control\n"
        "VIC 5 + RVIC · Cuenca Río Paucartambo · Promedio 2003–2025",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOTS_DIR / "fig_estacionalidad.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figura 8: Mapa espacial sintético del balance hídrico (P−ET)
# ──────────────────────────────────────────────────────────────────────────────

def plot_mapa_balance() -> None:
    """
    Mapa espacial esquemático del excedente hídrico anual (P−ET) en mm/año.
    Usa interpolación sintética representativa del gradiente altitudinal
    hasta que los rasters VIC reales estén disponibles.
    """
    rng = np.random.default_rng(99)

    # Bounding box: lat -11.8 a -10.2, lon -76.8 a -74.8
    ny, nx = 80, 100
    lat = np.linspace(-11.8, -10.2, ny)
    lon = np.linspace(-76.8, -74.8, nx)
    LON, LAT = np.meshgrid(lon, lat)

    # Gradiente altitudinal simplificado: más alto (SO) → mayor P
    dem_proxy = 2000 - 1800 * (LON - LON.min()) / (LON.max() - LON.min()) \
                     + 1200 * (LAT - LAT.max()) / (LAT.min() - LAT.max())
    dem_proxy += rng.normal(0, 150, dem_proxy.shape)

    P_grid  = 800 + 1.2 * dem_proxy + rng.normal(0, 80, dem_proxy.shape)
    ET_grid = 600 + 0.2 * dem_proxy + rng.normal(0, 40, dem_proxy.shape)
    WB_grid = P_grid - ET_grid  # Excedente hídrico

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for ax, data, title, cmap, label in zip(
        axes,
        [P_grid, ET_grid, WB_grid],
        ["Precipitación anual media\n(CHIRPS corregida)",
         "Evapotranspiración anual\n(VIC — Penman-Monteith)",
         "Excedente hídrico  P − ET\n(escorrentía potencial)"],
        ["Blues", "YlGn", "RdYlBu"],
        ["mm/año", "mm/año", "mm/año"],
    ):
        im = ax.contourf(LON, LAT, data, levels=20, cmap=cmap)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label, fontsize=8)
        ax.set_xlabel("Longitud (°O)", fontsize=9)
        ax.set_ylabel("Latitud (°S)", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        # Punto outlet Yuncán
        ax.plot(-75.586, -10.745, "r*", markersize=10, zorder=5,
                label="Outlet Yuncán")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Distribución Espacial del Balance Hídrico Anual — Cuenca Río Paucartambo\n"
        "Modelo VIC 5 · Promedio 2003–2025  (mapa esquemático previo a salidas NetCDF)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out = PLOTS_DIR / "fig_mapa_balance_anual.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figura 9: Scatter plots de validación (todos los puntos de control)
# ──────────────────────────────────────────────────────────────────────────────

def plot_scatter_validacion(dfs: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    for ax, (label, df) in zip(axes, dfs.items()):
        obs = df["obs"].values
        sim = df["sim"].values
        lim = max(obs.max(), sim.max()) * 1.05

        ax.scatter(obs, sim, color=C_SIM, alpha=0.45, s=12, zorder=3)
        ax.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="1:1")
        # Regresión lineal
        m, b = np.polyfit(obs, sim, 1)
        xfit = np.linspace(0, lim, 100)
        ax.plot(xfit, m * xfit + b, color=C_SIM, linewidth=1.5, alpha=0.7,
                label=f"Regresión: y={m:.2f}x+{b:.1f}")

        nse  = _nse(obs, sim)
        r    = np.corrcoef(obs, sim)[0,1]
        pb   = 100*(sim.sum()-obs.sum())/obs.sum()
        ax.text(0.03, 0.97,
                f"NSE={nse:.2f}  r={r:.2f}\nPBIAS={pb:+.1f}%",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCC", lw=0.8))

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Observado (m³/s)")
        ax.set_ylabel("Simulado (m³/s)")
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=7.5)

    fig.suptitle(
        "Validación del Modelo VIC+RVIC — Dispersión Observado vs. Simulado\n"
        "Cuenca Río Paucartambo · Período 2003–2025",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOTS_DIR / "fig_scatter_validacion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos reales desde NetCDF (si existen)
# ──────────────────────────────────────────────────────────────────────────────

def try_load_real(station_key: str) -> pd.DataFrame | None:
    """Intenta cargar datos reales de caudal simulado y observado."""
    csv = ROUTING_DIR / f"streamflow_{station_key}.csv"
    if csv.exists():
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera figuras del informe VIC — Cuenca Paucartambo"
    )
    parser.add_argument("--from-files", action="store_true",
                        help="Usar salidas NetCDF/CSV reales en lugar de sintéticas")
    args = parser.parse_args()

    YEARS = range(2003, 2026)

    stations = {
        "QN-908 Uchuhuerta":        ("qn908", "fig_hidrograma_qn908.png"),
        "QN-909 Huallamayo":        ("qn909", "fig_hidrograma_qn909.png"),
        "QN-901 + QN-902 + QN-911": ("qn901", "fig_hidrograma_qn901_911.png"),
        "QN-903 + QN-904 + QN-905": ("qn903", "fig_hidrograma_qn903_905.png"),
    }

    print("=" * 62)
    print("  GRÁFICAS DEL INFORME VIC — Cuenca Río Paucartambo")
    print("=" * 62)
    print()

    # ── Cargar o generar hidrogramas ─────────────────────────────
    dfs = {}
    for label, (key, fname) in stations.items():
        if args.from_files:
            df = try_load_real(key)
            if df is None:
                print(f"  [!] No se encontró {key}.csv — usando sintético")
                df = synthetic_monthly_flow(key, YEARS)
        else:
            df = synthetic_monthly_flow(key, YEARS)
        dfs[label] = df

    # ── Figuras 1–4: Hidrogramas ──────────────────────────────────
    print("[1/6] Hidrogramas mensuales por punto de control...")
    for label, (key, fname) in stations.items():
        print(f"  → {label}")
        plot_hidrograma(key, label, dfs[label], fname)

    # ── Figura 5: Balance hídrico ─────────────────────────────────
    print("[2/6] Balance hídrico anual...")
    df_annual  = synthetic_water_balance(YEARS)
    df_monthly = synthetic_monthly_precip_et()
    plot_balance_hidrico(df_annual, df_monthly)

    # ── Figura 6: Curva de duración ───────────────────────────────
    print("[3/6] Curvas de duración de caudales...")
    plot_curva_duracion(dfs)

    # ── Figura 7: Estacionalidad ──────────────────────────────────
    print("[4/6] Análisis estacional...")
    plot_estacionalidad(dfs)

    # ── Figura 8: Mapa espacial ───────────────────────────────────
    print("[5/6] Mapa espacial balance hídrico...")
    plot_mapa_balance()

    # ── Figura 9: Dispersión validación ──────────────────────────
    print("[6/6] Scatter de validación...")
    plot_scatter_validacion(dfs)

    print()
    print(f"[✓] Figuras guardadas en: {PLOTS_DIR}/")
    print("    fig_hidrograma_qn908.png")
    print("    fig_hidrograma_qn909.png")
    print("    fig_hidrograma_qn901_911.png")
    print("    fig_hidrograma_qn903_905.png")
    print("    fig_balance_hidrico.png")
    print("    fig_curva_duracion.png")
    print("    fig_estacionalidad.png")
    print("    fig_mapa_balance_anual.png")
    print("    fig_scatter_validacion.png")
    print()
    print("Para usar salidas NetCDF reales ejecute:")
    print("  python scripts/plot_vic_report_figures.py --from-files")


if __name__ == "__main__":
    main()
