#!/usr/bin/env python3
"""
Script 09: Mapa de la red de drenaje (routing network)
=======================================================
Genera un mapa estático de alta calidad mostrando:

  1. Red de ríos derivada de la acumulación de flujo D8
  2. Cuenca hidrográfica de Paucartambo
  3. Mapa de elevación (hillshade)
  4. Punto de control Yuncan
  5. Direcciones de flujo (submuestreadas)

Este mapa sirve para VALIDAR que la red de drenaje
es coherente con la topografía real de la cuenca.

Uso:
    python scripts/09_plot_routing_network.py
    python scripts/09_plot_routing_network.py --threshold 200 --arrows
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
import yaml
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_DIR / "config" / "basin_config.yml"
DATA_DIR    = PROJECT_DIR / "data"
DOMAIN_DIR  = DATA_DIR / "domain"
ROUTING_DIR = DATA_DIR / "routing"
PLOTS_DIR   = PROJECT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ARCMAP D8 encoding → (dy, dx) unit vectors
D8_VECTORS = {
    1:   ( 0,  1),   # E
    2:   ( 1,  1),   # SE
    4:   ( 1,  0),   # S
    8:   ( 1, -1),   # SW
    16:  ( 0, -1),   # W
    32:  (-1, -1),   # NW
    64:  (-1,  0),   # N
    128: (-1,  1),   # NE
}


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def hillshade(elevation: np.ndarray, azimuth: float = 315,
              altitude: float = 45) -> np.ndarray:
    """Calcular hillshade desde un array de elevación."""
    az  = np.deg2rad(360 - azimuth)
    alt = np.deg2rad(altitude)
    dy, dx = np.gradient(elevation)
    slope  = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0, 1)


def extract_river_network(facc: np.ndarray, mask: np.ndarray,
                          threshold: int = 100) -> np.ndarray:
    """
    Extraer red de ríos como máscara binaria donde
    flow_accumulation >= threshold dentro de la cuenca.
    """
    river = (facc >= threshold) & (mask == 1)
    return river


def plot_routing_network(
    threshold: int = 100,
    show_arrows: bool = False,
    arrow_stride: int = 15,
    dpi: int = 180,
    config: dict = None,
) -> Path:
    """
    Crear mapa de la red de drenaje y acumulación de flujo.

    Args:
        threshold: Mínimo de celdas upstream para considerarse río
        show_arrows: Mostrar vectores de dirección de flujo
        arrow_stride: Espaciado entre flechas (en celdas)
        dpi: Resolución del mapa
        config: Configuración del proyecto

    Returns:
        Path al PNG guardado
    """
    print("[NET] Cargando datos de dominio y routing...")

    # ── Cargar datos ──────────────────────────────────────────
    ds_fdr = xr.open_dataset(DOMAIN_DIR / "flow_direction.nc")
    fdr    = ds_fdr["flow_direction"].values
    facc   = ds_fdr["flow_accumulation"].values
    elev   = ds_fdr["elevation"].values
    basin  = ds_fdr["basin_id"].values
    lats   = ds_fdr.lat.values
    lons   = ds_fdr.lon.values
    ds_fdr.close()

    ds_dom = xr.open_dataset(DOMAIN_DIR / "domain.nc")
    mask   = ds_dom["mask"].values
    ds_dom.close()

    print(f"  Grid: {fdr.shape} | Res: {abs(lats[1]-lats[0]):.4f}°")
    print(f"  Elevación: {np.nanmin(elev):.0f}–{np.nanmax(elev):.0f} m")
    print(f"  Acumulación máx: {int(np.nanmax(facc))} celdas")

    outlet = config["basin"]["outlet"]

    # ── Hillshade ─────────────────────────────────────────────
    elev_sm = gaussian_filter(np.where(mask == 1, elev, np.nan), sigma=1)
    elev_sm = np.where(np.isnan(elev_sm), np.nanmean(elev_sm), elev_sm)
    hs = hillshade(elev_sm, azimuth=315, altitude=35)

    # ── Red de ríos (escala log de acumulación) ───────────────
    facc_masked = np.where(mask == 1, facc, np.nan)
    facc_log    = np.log10(np.clip(facc_masked, 1, None))

    river_mask  = extract_river_network(facc, mask, threshold)
    n_river_cells = int(np.sum(river_mask))
    print(f"  Celdas de río (acc ≥ {threshold}): {n_river_cells}")

    # ── Figura ────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11), facecolor="white")

    # Panel principal: mapa de la cuenca
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lons.min() - 0.02, lons.max() + 0.02,
                   lats.min() - 0.02, lats.max() + 0.02],
                  crs=ccrs.PlateCarree())

    # 1. Hillshade de fondo (solo dentro de la cuenca)
    hs_plot = np.where(mask == 1, hs, np.nan)
    ax.pcolormesh(
        lons, lats, hs_plot,
        cmap="gray", vmin=0, vmax=1,
        transform=ccrs.PlateCarree(),
        rasterized=True, zorder=1,
    )

    # 2. Elevación tenue sobre hillshade (topo colormap)
    elev_plot = np.where(mask == 1, elev, np.nan)
    ax.pcolormesh(
        lons, lats, elev_plot,
        cmap="terrain",
        vmin=200, vmax=5500,
        alpha=0.35,
        transform=ccrs.PlateCarree(),
        rasterized=True, zorder=2,
    )

    # 3. Fondo gris fuera de la cuenca
    outside = np.where(mask == 0, 1.0, np.nan)
    ax.pcolormesh(
        lons, lats, outside,
        cmap="gray_r", vmin=0, vmax=1,
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        rasterized=True, zorder=3,
    )

    # 4. Red de ríos coloreada por acumulación (log-scale)
    # Capas: ríos principales (mayor acumulación) en azul más intenso
    for acc_min, acc_max, color, lw, alpha, label in [
        (threshold,    500,   "#9ecae1", 0.6, 0.7, f"Quebradas (acc {threshold}–500)"),
        (500,         2000,   "#4292c6", 0.8, 0.8, "Ríos menores (500–2000)"),
        (2000,        8000,   "#2171b5", 1.0, 0.9, "Ríos principales (2000–8000)"),
        (8000,   999999999,   "#084594", 1.5, 1.0, "Río Paucartambo (>8000)"),
    ]:
        river_layer = np.where(
            (facc >= acc_min) & (facc < acc_max) & (mask == 1),
            1.0, np.nan
        )
        ax.pcolormesh(
            lons, lats, river_layer,
            cmap=mcolors.ListedColormap([color]),
            vmin=0, vmax=1, alpha=alpha,
            transform=ccrs.PlateCarree(),
            rasterized=True, zorder=4,
        )

    # 5. Flechas de dirección de flujo (submuestreadas)
    if show_arrows:
        rows = np.arange(0, fdr.shape[0], arrow_stride)
        cols = np.arange(0, fdr.shape[1], arrow_stride)
        dlat = abs(lats[1] - lats[0])
        dlon = abs(lons[1] - lons[0])
        for r in rows:
            for c in cols:
                if mask[r, c] != 1:
                    continue
                d = int(fdr[r, c])
                if d not in D8_VECTORS:
                    continue
                dy_d, dx_d = D8_VECTORS[d]
                scale = 0.6 * dlat
                ax.annotate(
                    "", fontsize=0,
                    xy=(lons[c] + dx_d * scale, lats[r] + dy_d * scale),
                    xytext=(lons[c], lats[r]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="black", lw=0.3,
                        mutation_scale=4,
                    ),
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )

    # 6. Contorno de la cuenca
    from matplotlib.contour import QuadContourSet
    import matplotlib.ticker as ticker
    mask_smooth = gaussian_filter(mask.astype(float), sigma=1)
    cs = ax.contour(
        lons, lats, mask_smooth,
        levels=[0.5], colors=["#1a1a1a"],
        linewidths=1.5,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

    # 7. Punto de outlet (Yuncan)
    ax.plot(
        outlet["lon"], outlet["lat"],
        "v", color="#d62728", markersize=14,
        markeredgecolor="white", markeredgewidth=1.0,
        transform=ccrs.PlateCarree(), zorder=10,
    )
    ax.text(
        outlet["lon"] + 0.04, outlet["lat"] - 0.06,
        f"◀ {outlet['name']}\n  ({outlet['lat']:.2f}°, {outlet['lon']:.2f}°)",
        color="white",
        fontsize=8, fontweight="bold",
        transform=ccrs.PlateCarree(), zorder=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#d62728",
                  edgecolor="white", alpha=0.85),
    )

    # 8. Gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.4,
        color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    # 9. Features geográficas
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray", alpha=0.7, zorder=7)

    # ── Colorbar de elevación ─────────────────────────────────
    sm_elev = plt.cm.ScalarMappable(
        cmap="terrain",
        norm=mcolors.Normalize(vmin=200, vmax=5500)
    )
    sm_elev.set_array([])
    cbar_ax = fig.add_axes([0.14, 0.08, 0.35, 0.018])
    cbar = plt.colorbar(sm_elev, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Elevación (m.s.n.m.)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Leyenda de la red de ríos ─────────────────────────────
    legend_elements = [
        Line2D([0], [0], color="#9ecae1", lw=2,
               label=f"Quebradas (acc {threshold}–500 celdas)"),
        Line2D([0], [0], color="#4292c6", lw=2,
               label="Ríos menores (500–2,000)"),
        Line2D([0], [0], color="#2171b5", lw=2.5,
               label="Ríos principales (2,000–8,000)"),
        Line2D([0], [0], color="#084594", lw=3,
               label="Río Paucartambo (>8,000)"),
        Line2D([0], [0], marker="v", color="w",
               markerfacecolor="#d62728", markersize=9,
               label=f"Control: {outlet['name']}"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=7.5,
        framealpha=0.92,
        edgecolor="gray",
        fancybox=True,
    )

    # ── Título ────────────────────────────────────────────────
    basin_name = config["basin"]["name"]
    dept = config["basin"]["department"]
    n_basin_cells = int(np.sum(mask == 1))
    area_km2 = n_basin_cells  # ~1 km² por celda a 0.009°

    ax.set_title(
        f"Red de Drenaje D8 — Cuenca {basin_name}, {dept}, Perú\n"
        f"Acumulación de flujo (D8 routing) · "
        f"Resolución 0.009° (~1 km) · "
        f"Área ≈ {area_km2:,} km²",
        fontsize=11, fontweight="bold", pad=10,
    )

    # ── Anotación estadística ─────────────────────────────────
    max_acc_km2 = int(np.nanmax(facc))
    stats_txt = (
        f"Cuenca Paucartambo\n"
        f"Celdas activas: {n_basin_cells:,}\n"
        f"Acc. máxima: {max_acc_km2:,} celdas\n"
        f"Umbral de río: {threshold} celdas\n"
        f"Celdas de río: {n_river_cells:,}"
    )
    ax.text(
        0.014, 0.98, stats_txt,
        transform=ax.transAxes,
        fontsize=7.5, va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="white", edgecolor="gray",
                  alpha=0.88),
        zorder=12,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    output_path = PLOTS_DIR / f"routing_network_acc{threshold}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] Mapa de routing guardado: {output_path}")
    return output_path


def plot_flow_accumulation_map(dpi: int = 180, config: dict = None) -> Path:
    """
    Mapa de acumulación de flujo en escala logarítmica.
    Muestra la continuidad del drenaje en toda la cuenca.
    """
    print("[NET] Mapa de acumulación de flujo (log-scale)...")

    ds_fdr = xr.open_dataset(DOMAIN_DIR / "flow_direction.nc")
    facc   = ds_fdr["flow_accumulation"].values
    elev   = ds_fdr["elevation"].values
    lats   = ds_fdr.lat.values
    lons   = ds_fdr.lon.values
    ds_fdr.close()

    ds_dom = xr.open_dataset(DOMAIN_DIR / "domain.nc")
    mask   = ds_dom["mask"].values
    ds_dom.close()

    outlet = config["basin"]["outlet"]

    facc_masked = np.where(mask == 1, facc, np.nan)
    facc_log    = np.log10(np.clip(facc_masked, 1, None))

    hs = hillshade(gaussian_filter(
        np.where(mask == 1, elev, np.nanmean(elev)), sigma=1
    ))

    fig, ax = plt.subplots(
        figsize=(12, 9),
        subplot_kw={"projection": ccrs.PlateCarree()},
        facecolor="white"
    )
    ax.set_extent([lons.min() - 0.01, lons.max() + 0.01,
                   lats.min() - 0.01, lats.max() + 0.01])

    # Hillshade de fondo
    ax.pcolormesh(lons, lats, np.where(mask == 1, hs, np.nan),
                  cmap="gray", vmin=0, vmax=1,
                  transform=ccrs.PlateCarree(), rasterized=True, zorder=1)

    # Acumulación de flujo (log scale) — toda la cuenca
    im = ax.pcolormesh(
        lons, lats, facc_log,
        cmap="Blues",
        vmin=0, vmax=np.log10(np.nanmax(facc)),
        alpha=0.75,
        transform=ccrs.PlateCarree(),
        rasterized=True, zorder=2,
    )

    # Contorno de cuenca
    mask_sm = gaussian_filter(mask.astype(float), sigma=1)
    ax.contour(lons, lats, mask_sm, levels=[0.5],
               colors=["#333333"], linewidths=1.5,
               transform=ccrs.PlateCarree(), zorder=3)

    # Outlet
    ax.plot(outlet["lon"], outlet["lat"], "v",
            color="#d62728", markersize=12,
            markeredgecolor="white", markeredgewidth=0.8,
            transform=ccrs.PlateCarree(), zorder=5)
    ax.text(outlet["lon"] + 0.04, outlet["lat"],
            outlet["name"], color="#d62728",
            fontsize=8, fontweight="bold",
            transform=ccrs.PlateCarree(), zorder=6,
            bbox=dict(facecolor="white", edgecolor="#d62728",
                      alpha=0.8, boxstyle="round,pad=0.2"))

    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color="gray", alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    # Colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.65])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label("log₁₀(Acumulación de flujo)\n[# celdas upstream]", fontsize=8)

    # Ticks legibles
    tick_vals = [0, 1, 2, 3, 4, int(np.log10(np.nanmax(facc)))]
    tick_lbls = ["1", "10", "100", "1,000", "10,000",
                 f"{int(np.nanmax(facc)):,}"]
    cbar.set_ticks(tick_vals[:len([v for v in tick_vals
                                   if v <= np.log10(np.nanmax(facc))])])
    cbar.set_ticklabels(tick_lbls[:cbar.get_ticks().size])
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(
        f"Acumulación de Flujo D8 — Cuenca {config['basin']['name']}\n"
        f"Áreas más azules = mayor área de captación upstream",
        fontsize=11, fontweight="bold", pad=8,
    )

    output_path = PLOTS_DIR / "flow_accumulation_map.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] Mapa de acumulación guardado: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Mapa de red de drenaje D8 - Cuenca Paucartambo"
    )
    parser.add_argument(
        "--threshold", type=int, default=100,
        help="Mínimo de celdas upstream para considerarse río (default: 100)"
    )
    parser.add_argument(
        "--arrows", action="store_true",
        help="Mostrar flechas de dirección de flujo (más lento)"
    )
    parser.add_argument(
        "--arrow-stride", type=int, default=15,
        help="Espaciado entre flechas (default: 15 celdas)"
    )
    parser.add_argument(
        "--dpi", type=int, default=180,
        help="Resolución del mapa (default: 180)"
    )
    parser.add_argument(
        "--accumulation-only", action="store_true",
        help="Solo generar mapa de acumulación de flujo"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  RED DE DRENAJE D8 — Cuenca Paucartambo")
    print("=" * 60)

    config = load_config()

    if not args.accumulation_only:
        plot_routing_network(
            threshold=args.threshold,
            show_arrows=args.arrows,
            arrow_stride=args.arrow_stride,
            dpi=args.dpi,
            config=config,
        )

    plot_flow_accumulation_map(dpi=args.dpi, config=config)

    print(f"\n[✓] Mapas guardados en: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
