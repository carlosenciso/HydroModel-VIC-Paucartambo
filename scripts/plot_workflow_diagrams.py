#!/home/cenciso/anaconda3/envs/lulc/bin/python
"""
Script: Diagramas de Flujo del Proceso de Simulación VIC
=========================================================
Genera diagramas de flujo que ilustran la metodología completa de
simulación hidrológica de la cuenca Paucartambo con VIC 5 + RVIC.

Salidas:
  plots/diagrama_pipeline_vic.png     -- Cadena metodológica completa
  plots/diagrama_celda_vic.png        -- Procesos internos por celda VIC
  plots/diagrama_rvic.png             -- Flujo del enrutamiento RVIC

Uso:
    python scripts/plot_workflow_diagrams.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
# Usar report/Figures como salida (directorio escribible)
PLOTS_DIR = PROJECT_DIR / "report" / "Figures"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def rounded_box(ax, x, y, w, h, text, color, fontsize=9, text_color="white",
                style="round,pad=0.05", lw=1.0, alpha=1.0, bold=False):
    """Dibuja una caja redondeada con texto centrado."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=style, linewidth=lw,
        edgecolor="white", facecolor=color, alpha=alpha, zorder=3
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight, zorder=4,
            wrap=True, multialignment="center")


def arrow(ax, x1, y1, x2, y2, color="#444444", lw=1.5, style="-|>"):
    """Flecha entre dos puntos."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw),
        zorder=2
    )


def section_label(ax, x, y, text, color, fontsize=9.5):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5),
            zorder=5)


# ──────────────────────────────────────────────────────────────────────────────
# DIAGRAMA 1: Cadena metodológica completa
# ──────────────────────────────────────────────────────────────────────────────

def plot_pipeline():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    fig.suptitle(
        "Cadena Metodológica de Simulación Hidrológica\n"
        "Cuenca Río Paucartambo — Modelo VIC 5 + RVIC",
        fontsize=14, fontweight="bold", y=0.97, color="#1A237E"
    )

    # ── Colores por etapa ───────────────────────────────────────
    C_DATA  = "#1565C0"   # azul oscuro — datos de entrada
    C_PREP  = "#00695C"   # verde oscuro — preparación
    C_VIC   = "#4527A0"   # violeta      — modelo VIC
    C_RVIC  = "#E65100"   # naranja      — enrutamiento
    C_OUT   = "#B71C1C"   # rojo oscuro  — salidas
    C_HEAD  = "#37474F"   # gris encabezado

    # ── FILA 0: Encabezados de etapa ────────────────────────────
    etapas = [
        (2.0,  9.2, "ETAPA 1\nDatos de Entrada",    C_DATA),
        (5.0,  9.2, "ETAPA 2\nPreparación",          C_PREP),
        (8.5,  9.2, "ETAPA 3\nModelo VIC 5",         C_VIC),
        (12.0, 9.2, "ETAPA 4\nEnrutamiento RVIC",   C_RVIC),
        (14.8, 9.2, "ETAPA 5\nSalidas",              C_OUT),
    ]
    for xc, yc, txt, col in etapas:
        rounded_box(ax, xc, yc, 2.8, 0.65, txt, col, fontsize=8.5, bold=True)

    # ── ETAPA 1: Datos satelitales / reanálisis ──────────────────
    datos = [
        (2.0, 8.0, "CHIRPS Daily\n(Precipitación)\n~5 km · 1981–2025"),
        (2.0, 6.8, "ERA5-Land\n(Tmax, Tmin, Viento\nRad. Solar, Vapor)"),
        (2.0, 5.6, "SoilGrids 250m\n(Arena, Arcilla, Limo)"),
        (2.0, 4.4, "ESA WorldCover\n(Cobertura vegetal)\n10 m → 1 km"),
        (2.0, 3.2, "MODIS MOD15A2H\n(LAI mensual)\n500 m · 2010–2020"),
        (2.0, 2.0, "SRTM DEM 30 m\n(Elevación / Red\nde drenaje D8)"),
    ]
    for xc, yc, txt in datos:
        rounded_box(ax, xc, yc, 2.8, 0.85, txt, C_DATA, fontsize=7.5)

    # ── ETAPA 2: Preparación de datos ───────────────────────────
    prep = [
        (5.0, 7.5, "Corrección de sesgo\nCHIRPS (Quantile\nMapping vs. SENAMHI)"),
        (5.0, 6.0, "Corrección topográfica\nERA5 → grilla 1 km\n(gradiente adiabático)"),
        (5.0, 4.5, "Funciones de\npedotransferencia\nCosby et al. (1984)"),
        (5.0, 3.0, "Parámetros vegetación\nVIC (9 clases +\nLAI mensual MODIS)"),
        (5.0, 1.8, "DEM SRTM → Máscara\ncuenca · domain.nc\n(pysheds, D8)"),
    ]
    for xc, yc, txt in prep:
        rounded_box(ax, xc, yc, 2.8, 0.9, txt, C_PREP, fontsize=7.5)

    # ── ETAPA 3: VIC 5 Image Driver ─────────────────────────────
    vic_boxes = [
        (8.5, 8.2, "Domain NetCDF\n(mask, area, frac)\n~4 000 celdas activas"),
        (8.5, 7.0, "Parámetros de Suelo\n(3 capas: Ksat, b,\nψs, θs, Dm, Ds, Ws)"),
        (8.5, 5.8, "Parámetros vegetación\n(fracción cobertura,\nLAI, resistencia)"),
        (8.5, 4.6, "Forzantes diarios\n(NetCDF · CHIRPS +\nERA5 · 2000–2025)"),
    ]
    for xc, yc, txt in vic_boxes:
        rounded_box(ax, xc, yc, 2.8, 0.85, txt, C_VIC, fontsize=7.5)

    # Caja central VIC RUN
    rounded_box(ax, 8.5, 3.2, 3.2, 0.85,
                "VIC 5 Image Driver\n(MPI paralelo · paso diario\nFULL_ENERGY=FALSE · SNOW_BAND)",
                "#6A1B9A", fontsize=8.0, bold=True, lw=1.8)

    # Salidas VIC
    vic_out = [
        (8.5, 2.0, "Escorrentía + flujo\nbase por celda\n(mm/día · NetCDF4)"),
        (8.5, 1.0, "ET, SWE, humedad\nsuelo (variables\nauxiliares)"),
    ]
    for xc, yc, txt in vic_out:
        rounded_box(ax, xc, yc, 2.8, 0.75, txt, "#7B1FA2", fontsize=7.5)

    # ── ETAPA 4: RVIC ───────────────────────────────────────────
    rvic_boxes = [
        (12.0, 7.5, "Red de drenaje D8\n(SRTM 30 m)\ndist. celda→outlet"),
        (12.0, 6.2, "Hidrógrafas unitarias\n(ec. difusión-advección\nv=1.5 m/s, D=2000 m²/s)"),
        (12.0, 5.0, "Convolución espacial\nQ(t)=Σ qj·UHj·Aj\n(Lohmann 1998)"),
        (12.0, 3.8, "Puntos de control:\nQN-908 Uchuhuerta\nQN-909 Huallamayo\nQN-901-911 / QN-903-905"),
    ]
    for xc, yc, txt in rvic_boxes:
        rounded_box(ax, xc, yc, 2.8, txt.count("\n") * 0.3 + 0.65, txt, C_RVIC, fontsize=7.5)

    # ── ETAPA 5: Salidas finales ─────────────────────────────────
    out_boxes = [
        (14.8, 6.8, "Caudal mensual\nnaturalizado\n(m³/s)"),
        (14.8, 5.5, "Hidrograma\ndiario en\noutlet Yuncán"),
        (14.8, 4.2, "Balance hídrico\ndistribuido\n(P, ET, Q, ΔS)"),
        (14.8, 3.0, "Validación NSE\nPBIAS, RSR\nWillmott d, r"),
    ]
    for xc, yc, txt in out_boxes:
        rounded_box(ax, xc, yc, 2.6, 0.8, txt, C_OUT, fontsize=7.5)

    # ── FLECHAS entre etapas ─────────────────────────────────────
    # Etapa 1 → Etapa 2
    for y in [8.0, 6.8, 5.6, 4.4, 3.2, 2.0]:
        arrow(ax, 3.4, y, 3.6, y, color=C_DATA)
    # Etapa 2 → Etapa 3
    for y in [7.5, 6.0, 4.5, 3.0, 1.8]:
        arrow(ax, 6.4, y, 6.6, y, color=C_PREP)
    # Dentro de VIC: boxes → RUN
    for y in [8.2, 7.0, 5.8, 4.6]:
        arrow(ax, 8.5, y - 0.43, 8.5, 3.63, color=C_VIC, lw=1.2)
    # VIC RUN → salidas VIC
    arrow(ax, 8.5, 2.78, 8.5, 2.38, color="#7B1FA2")
    arrow(ax, 8.5, 1.63, 8.5, 1.38, color="#7B1FA2")
    # VIC salidas → RVIC
    arrow(ax, 9.9, 2.0, 10.1, 5.0, color="#7B1FA2", lw=1.5)
    # RVIC interno
    arrow(ax, 12.0, 7.18, 12.0, 6.53, color=C_RVIC)
    arrow(ax, 12.0, 5.87, 12.0, 5.35, color=C_RVIC)
    arrow(ax, 12.0, 4.65, 12.0, 4.35, color=C_RVIC)
    # RVIC → salidas
    arrow(ax, 13.4, 5.0, 13.5, 5.5, color=C_RVIC, lw=1.5)
    arrow(ax, 13.4, 6.2, 13.5, 6.8, color=C_RVIC, lw=1.0)
    arrow(ax, 13.4, 3.8, 13.5, 3.0, color=C_RVIC, lw=1.0)

    # ── Leyenda de productos satelitales ────────────────────────
    ax.text(0.2, 0.35,
            "GEE = Google Earth Engine\nSEHANMI = Red meteorológica observada\n"
            "OSINERGMIN N°196-2016 / Proc. Téc. N°41",
            fontsize=7, color="#555555", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.8))

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = PLOTS_DIR / "diagrama_pipeline_vic.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# DIAGRAMA 2: Procesos internos por celda VIC
# ──────────────────────────────────────────────────────────────────────────────

def plot_vic_cell():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    fig.suptitle(
        "Procesos Físicos en cada Celda VIC 5 (~1 km²)\n"
        "Cuenca Río Paucartambo — Balance Hídrico Diario",
        fontsize=13, fontweight="bold", y=0.98, color="#1A237E"
    )

    # ── Forzantes de entrada (arriba) ──────────────────────────
    forzantes = [
        (1.5, 8.2, "Precipitación\n(CHIRPS·QM)\nmm/día"),
        (3.5, 8.2, "Tmax / Tmin\n(ERA5+lapse)\n°C"),
        (5.5, 8.2, "Radiación solar\n(ERA5→W/m²)"),
        (7.5, 8.2, "Viento\n(ERA5 u,v)\nm/s"),
        (9.5, 8.2, "Presión vapor\n(ERA5 Td)\nkPa"),
        (11.5, 8.2, "Presión atm\n(z-dependiente)\nkPa"),
    ]
    for xc, yc, txt in forzantes:
        rounded_box(ax, xc, yc, 1.7, 0.75, txt, "#1565C0", fontsize=7.5)
        arrow(ax, xc, yc - 0.38, xc, 7.35, color="#1565C0", lw=1.2)

    # ── Dosel vegetal ───────────────────────────────────────────
    rounded_box(ax, 6.5, 7.0, 12.0, 0.55,
                "DOSEL VEGETAL  ·  Intercepción de precipitación  ·  Fracción de cobertura por clase VIC (ESA WorldCover + MODIS LAI)",
                "#2E7D32", fontsize=8.0, bold=True)

    # ── Penman-Monteith ─────────────────────────────────────────
    rounded_box(ax, 6.5, 6.1, 5.5, 0.7,
                "Evapotranspiración — Penman-Monteith\n"
                "ETveg (transpiración) + ETcanopy (reevaporación) + ETbare",
                "#388E3C", fontsize=8.0)
    arrow(ax, 6.5, 6.73, 6.5, 6.45, color="#2E7D32")

    # ── Infiltración (curva de Arno) ────────────────────────────
    rounded_box(ax, 2.8, 6.1, 4.2, 0.7,
                "Infiltración Variable (curva de Arno)\n"
                "i(x)=imax·[1−(1−As)^(1/binfilt)]  →  Qs superficial",
                "#E65100", fontsize=8.0)
    arrow(ax, 2.8, 6.73, 2.8, 6.45, color="#E65100")

    # ── Balance de nieve (sobre 3800 m) ─────────────────────────
    rounded_box(ax, 11.0, 6.1, 3.5, 0.7,
                "Balance de Energía — Nieve\n"
                "dUs/dt = Rn,s − Hs − λf·M + Gs  →  SWE",
                "#00838F", fontsize=8.0)
    arrow(ax, 11.0, 6.73, 11.0, 6.45, color="#00838F")

    # ── Suelo 3 capas ───────────────────────────────────────────
    layers = [
        (6.5, 5.0, "CAPA 1  (0–10 cm)\nHumedad rápida — θ₁\nKsat₁, b₁, ψs₁, θs₁"),
        (6.5, 3.9, "CAPA 2  (10–40 cm)\nHumedad de mediano plazo — θ₂\nKsat₂, b₂, ψs₂, θs₂"),
        (6.5, 2.8, "CAPA 3  (40–100 cm)\nReserva lenta — θ₃  →  Flujo base Qb\n(Francini-Pacciani ARNO: Dm, Ds, Wm, Ws)"),
    ]
    layer_colors = ["#5D4037", "#4E342E", "#3E2723"]
    for (xc, yc, txt), col in zip(layers, layer_colors):
        rounded_box(ax, xc, yc, 11.5, 0.7, txt, col, fontsize=8.0)
        arrow(ax, xc, yc + 0.35, xc, yc + 0.70, color=col, lw=0.8)

    # Flechas ET → capa 1
    arrow(ax, 6.5, 5.72, 6.5, 5.35, color="#388E3C")
    # Qs → fuera
    arrow(ax, 2.8, 5.72, 2.8, 5.0, color="#E65100")
    ax.text(1.5, 4.7, "Qs\n(escorrentía\ndirecta)", fontsize=7.5,
            color="#E65100", ha="center", weight="bold")
    # SWE → capa 1 (deshielo)
    arrow(ax, 11.0, 5.72, 11.0, 5.35, color="#00838F")
    ax.text(11.8, 5.5, "Fusión\nnival\n→ Qs", fontsize=7.5,
            color="#00838F", ha="center")

    # Qb desde capa 3
    arrow(ax, 11.5, 2.45, 12.5, 2.1, color="#3E2723")
    ax.text(12.5, 1.85, "Qb\n(flujo base)", fontsize=7.5,
            color="#3E2723", ha="center", weight="bold")

    # ── Salidas por celda ───────────────────────────────────────
    rounded_box(ax, 6.5, 1.5, 11.5, 0.65,
                "SALIDAS POR CELDA  →  Qs + Qb [mm/día · NetCDF4]  ·  ET  ·  SWE  ·  θ₁, θ₂, θ₃",
                "#880E4F", fontsize=8.5, bold=True)
    arrow(ax, 6.5, 2.45, 6.5, 1.83, color="#880E4F", lw=1.8)

    # ── Transferencia a RVIC ────────────────────────────────────
    arrow(ax, 6.5, 1.18, 6.5, 0.68, color="#880E4F", lw=2.0)
    rounded_box(ax, 6.5, 0.45, 6.0, 0.55,
                "→ RVIC: Convolución con Hidrógrafa Unitaria → Q(t) en outlet",
                "#B71C1C", fontsize=8.0, bold=True)

    # ── Nota sobre parámetros ───────────────────────────────────
    ax.text(0.15, 0.55,
            "Parámetros de suelo:\nCosby et al. (1984) + SoilGrids 250m\n"
            "Calibración: binfilt, Ds, Ws, Dm",
            fontsize=7, color="#555", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#AAAAAA", lw=0.8))

    plt.tight_layout(rect=[0, 0.0, 1, 0.96])
    out = PLOTS_DIR / "diagrama_celda_vic.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# DIAGRAMA 3: Flujo del enrutamiento RVIC
# ──────────────────────────────────────────────────────────────────────────────

def plot_rvic_routing():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("#FFF8F0")

    fig.suptitle(
        "Módulo de Enrutamiento RVIC — Cuenca Río Paucartambo\n"
        "Convolución de Hidrógrafas Unitarias (Lohmann et al., 1998)",
        fontsize=13, fontweight="bold", y=0.97, color="#BF360C"
    )

    C = "#E65100"
    CL = "#FF8A50"
    CD = "#B71C1C"

    # ── Paso 1: Escorrentía distribuida ─────────────────────────
    rounded_box(ax, 2.5, 7.0, 4.5, 0.75,
                "Escorrentía por celda VIC: qj(τ) [mm/día]\n"
                "~4 000 celdas activas · grilla 1 km\n"
                "Qs(celda) + Qb(celda) → mm/día",
                "#4527A0", fontsize=8.5)

    # ── Paso 2: Red de drenaje D8 ────────────────────────────────
    rounded_box(ax, 8.0, 7.0, 4.5, 0.75,
                "Red de drenaje D8 — SRTM 30 m\n"
                "Dirección de flujo en 8 vecinos\n"
                "Distancia acumulada dj (celda→outlet)",
                "#1565C0", fontsize=8.5)

    arrow(ax, 4.75, 7.0, 5.75, 7.0, color="#555", lw=1.5)
    arrow(ax, 10.25, 7.0, 11.25, 7.0, color="#555", lw=1.5)

    # ── Paso 3: Hidrógrafa unitaria ──────────────────────────────
    rounded_box(ax, 7.0, 5.5, 8.0, 1.1,
                "Hidrógrafa Unitaria por celda (ecuación difusión-advección):\n\n"
                "UH(t) = d / √(4π D t³) · exp[−(d−vt)² / (4Dt)]\n\n"
                "v = 1.5 m/s  ·  D = 2000 m²/s  ·  Δt_UH = 3 h  ·  T_max = 100 días",
                C, fontsize=8.5, bold=False)

    arrow(ax, 2.5, 6.63, 2.5, 6.05, color="#4527A0", lw=1.3)
    arrow(ax, 8.0, 6.63, 8.0, 6.05, color="#1565C0", lw=1.3)
    # convergencia al box UH
    arrow(ax, 4.75, 5.5, 3.01, 5.5, color="#555", lw=1.2)

    # ── Paso 4: Convolución ──────────────────────────────────────
    rounded_box(ax, 7.0, 4.0, 10.0, 1.0,
                "CONVOLUCIÓN ESPACIOTEMPORAL:\n\n"
                "Q(t) = Σⱼ Σ_τ  qⱼ(τ) · UHⱼ(t−τ) · Aⱼ\n\n"
                "Suma sobre N celdas y horizonte temporal T_max",
                CD, fontsize=9.0, bold=True, lw=2.0)

    arrow(ax, 7.0, 4.95, 7.0, 4.50, color=C, lw=2.0)

    # ── Paso 5: Puntos de control ────────────────────────────────
    puntos = [
        (2.0, 2.5, "QN-908\nUchuhuerta\n889 km²"),
        (4.5, 2.5, "QN-909\nHuallamayo\n445 km²"),
        (7.0, 2.5, "QN-910\nVictoria I"),
        (9.5, 2.5, "QN-901+902+911\n(afluentes\nadicionales)"),
        (12.0, 2.5, "QN-906\nYuncán\n(outlet)\n1583 km²"),
    ]
    colors_p = ["#0D47A1", "#1565C0", "#1976D2", "#1E88E5", CD]
    for (xc, yc, txt), col in zip(puntos, colors_p):
        rounded_box(ax, xc, yc, 2.2, 0.95, txt, col, fontsize=8.0)
        arrow(ax, xc, 3.5, xc, 2.98, color=col, lw=1.3)

    ax.text(7.0, 3.6, "Q(t)  en cada punto de control  [m³/s · paso diario]",
            ha="center", va="center", fontsize=9, color="#333",
            bbox=dict(boxstyle="round,pad=0.2", fc="#FFF3E0", ec=C, lw=1))

    # ── Paso 6: Agregación y validación ─────────────────────────
    rounded_box(ax, 4.5, 1.1, 5.5, 0.75,
                "Agregación mensual  →  Caudal promedio mensual naturalizado [m³/s]\n"
                "Comparación con observados SENAMHI · NSE, PBIAS, RSR, Willmott d, r",
                "#B71C1C", fontsize=8.0, bold=True)

    rounded_box(ax, 11.5, 1.1, 4.0, 0.75,
                "Informe OSINERGMIN\nResolución N°196-2016-OS/CD\nProcedimiento Técnico N°41",
                "#37474F", fontsize=8.0, bold=True)

    for xc in [2.0, 4.5, 7.0]:
        arrow(ax, xc, 2.02, 4.5, 1.48, color="#B71C1C", lw=1.0)
    arrow(ax, 12.0, 2.02, 11.5, 1.48, color="#37474F", lw=1.0)

    plt.tight_layout(rect=[0, 0.0, 1, 0.94])
    out = PLOTS_DIR / "diagrama_rvic.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DIAGRAMAS DE FLUJO — Simulación VIC 5 + RVIC")
    print("  Cuenca Río Paucartambo · C.H. Yuncán")
    print("=" * 60)
    print()
    print("[1/3] Pipeline metodológico completo...")
    plot_pipeline()
    print("[2/3] Procesos físicos por celda VIC...")
    plot_vic_cell()
    print("[3/3] Enrutamiento RVIC...")
    plot_rvic_routing()
    print()
    print(f"[✓] Diagramas guardados en: {PLOTS_DIR}/")
    print("    diagrama_pipeline_vic.png  → \\includegraphics{Figures/diagrama_pipeline_vic}")
    print("    diagrama_celda_vic.png     → \\includegraphics{Figures/diagrama_celda_vic}")
    print("    diagrama_rvic.png          → \\includegraphics{Figures/diagrama_rvic}")
