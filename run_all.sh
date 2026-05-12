#!/bin/bash
# ============================================================
# Pipeline Completo - VIC Cuenca Paucartambo
# ============================================================
# Ejecuta todos los pasos del proyecto VIC en secuencia.
#
# Uso:
#   1. Desde el host (con Docker):
#      docker-compose run --rm vic bash /app/run_all.sh
#
#   2. Dentro del contenedor:
#      bash run_all.sh
#
#   3. Con fechas específicas:
#      START=2015-01-01 END=2018-12-31 bash run_all.sh
#
# Variables de entorno requeridas:
#   EE_SERVICE_ACCOUNT_JSON_B64 - Token GEE (service account)

set -euo pipefail

# ─────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────
START_DATE="${START:-2010-01-01}"
END_DATE="${END:-2020-12-31}"
SCALE="${SCALE:-1000}"       # Resolución en metros
SPINUP_YEARS="${SPINUP:-2}"  # Años de spin-up

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)"
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1" | tee -a "$MAIN_LOG"
}

success() {
    echo -e "${GREEN}[✓] $1${NC}" | tee -a "$MAIN_LOG"
}

warn() {
    echo -e "${YELLOW}[!] $1${NC}" | tee -a "$MAIN_LOG"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$MAIN_LOG"
    exit 1
}

run_step() {
    local step_num="$1"
    local step_name="$2"
    local cmd="$3"
    local step_log="$LOG_DIR/step${step_num}_${TIMESTAMP}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "PASO ${step_num}: ${step_name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local start_time=$(date +%s)

    if eval "$cmd" 2>&1 | tee -a "$step_log" "$MAIN_LOG"; then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        success "Paso ${step_num} completado en ${elapsed}s"
        return 0
    else
        error "Paso ${step_num} falló. Ver log: ${step_log}"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────
# Verificar autenticación GEE
# ─────────────────────────────────────────────────────────
check_gee_auth() {
    if [ -z "${EE_SERVICE_ACCOUNT_JSON_B64:-}" ]; then
        # Intentar cargar desde archivo
        KEY_FILE="/home/cenciso/Documents/WORK/ENGIE/DashBoard/windShortTermForecast/key_b64.txt"
        if [ -f "$KEY_FILE" ]; then
            export EE_SERVICE_ACCOUNT_JSON_B64=$(cat "$KEY_FILE")
            log "Token GEE cargado desde: $KEY_FILE"
        else
            warn "EE_SERVICE_ACCOUNT_JSON_B64 no configurado."
            warn "Algunos pasos que requieren GEE podrían fallar."
            warn "Para configurar:"
            warn "  export EE_SERVICE_ACCOUNT_JSON_B64=\$(cat $KEY_FILE)"
        fi
    else
        log "Token GEE encontrado en variable de entorno."
    fi
}

# ─────────────────────────────────────────────────────────
# Encabezado
# ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   VIC HYDROLOGICAL MODEL - CUENCA PAUCARTAMBO, PERU     ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║   Período:    ${START_DATE} → ${END_DATE}              ║"
echo "║   Resolución: ${SCALE}m (~1km)                          ║"
echo "║   Spin-up:    ${SPINUP_YEARS} años                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Log principal: $MAIN_LOG"
echo ""

check_gee_auth

# ─────────────────────────────────────────────────────────
# Pipeline de ejecución
# ─────────────────────────────────────────────────────────

# PASO 1: Descarga de datos GEE
run_step "1" "Descarga de datos Google Earth Engine" \
    "python ${SCRIPT_DIR}/01_download_gee.py \
        --start ${START_DATE} \
        --end ${END_DATE} \
        --scale ${SCALE}"

# PASO 2: Preparar dominio
run_step "2" "Preparación del dominio VIC" \
    "python ${SCRIPT_DIR}/02_prepare_domain.py"

# PASO 3: Preparar parámetros
run_step "3" "Preparación de parámetros de suelo y vegetación" \
    "python ${SCRIPT_DIR}/03_prepare_parameters.py"

# PASO 4: Preparar forzantes
run_step "4" "Preparación de forzantes meteorológicos" \
    "python ${SCRIPT_DIR}/04_prepare_forcings.py \
        --start ${START_DATE} \
        --end ${END_DATE} \
        --spinup ${SPINUP_YEARS}"

# PASO 5: Ejecutar VIC
run_step "5" "Ejecución del modelo VIC (Image Driver)" \
    "python ${SCRIPT_DIR}/05_run_vic.py \
        --start ${START_DATE} \
        --end ${END_DATE} \
        --spinup"

# PASO 6: Enrutamiento
run_step "6" "Enrutamiento hidrológico → Central Yuncan" \
    "python ${SCRIPT_DIR}/06_run_routing.py \
        --method linear \
        --travel-time"

# PASO 7: Visualizaciones
run_step "7" "Generando gráficos y series temporales" \
    "python ${SCRIPT_DIR}/07_plot_timeseries.py"

# PASO 8: Animaciones
log "PASO 8: Generando animaciones de dinámica hidrológica"
echo ""

# Animación humedad del suelo
python "${SCRIPT_DIR}/08_create_animation.py" \
    --variable OUT_SOIL_MOIST \
    --year 2015 \
    --freq 7D \
    --format gif \
    --fps 8 \
    2>&1 | tee -a "$MAIN_LOG" || warn "Animación SOIL_MOIST falló (no crítico)"

# Animación precipitación
python "${SCRIPT_DIR}/08_create_animation.py" \
    --variable OUT_PREC \
    --year 2015 \
    --freq 7D \
    --format gif \
    --fps 8 \
    2>&1 | tee -a "$MAIN_LOG" || warn "Animación PREC falló (no crítico)"

# Snapshots mensuales (alternativa estática)
python "${SCRIPT_DIR}/08_create_animation.py" \
    --variable OUT_RUNOFF \
    --year 2015 \
    --snapshots \
    2>&1 | tee -a "$MAIN_LOG" || warn "Snapshots RUNOFF fallaron (no crítico)"

success "Paso 8 completado"

# ─────────────────────────────────────────────────────────
# Resumen final
# ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  PIPELINE COMPLETADO                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Resultados disponibles en:"
echo "  ├── data/outputs/     → Salidas VIC (NetCDF)"
echo "  ├── data/routing/     → Caudales enrutados"
echo "  ├── plots/            → Gráficos y series temporales"
echo "  └── animations/       → Animaciones GIF/MP4"
echo ""
echo "  Gráficos clave:"
echo "  ├── plots/streamflow_timeseries.png"
echo "  ├── plots/water_balance.png"
echo "  ├── plots/flow_duration_curve.png"
echo "  └── animations/animation_OUT_SOIL_MOIST_2015.gif"
echo ""
echo "  Log completo: $MAIN_LOG"
echo ""
success "Simulación VIC completada exitosamente para la cuenca Paucartambo"
