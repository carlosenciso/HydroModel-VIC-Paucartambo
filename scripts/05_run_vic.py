#!/usr/bin/env python3
"""
Script 05: Ejecutar modelo VIC (Image Driver)
=============================================
Genera el archivo de parámetros globales y ejecuta VIC5 Image Driver.

El ejecutable `vic_image.exe` debe estar en el PATH o en /opt/VIC/build/.
Dentro del contenedor Docker, está disponible en el PATH.

Uso:
    python scripts/05_run_vic.py --start 2010-01-01 --end 2020-12-31
    python scripts/05_run_vic.py --start 2010-01-01 --end 2020-12-31 --spinup
"""

import argparse
import subprocess
import sys
import shutil
import os
from pathlib import Path
from datetime import datetime

import yaml

# ─────────────────────────────────────────────────────────
PROJECT_DIR    = Path(__file__).parent.parent
CONFIG_FILE    = PROJECT_DIR / "config" / "basin_config.yml"
TEMPLATE_FILE  = PROJECT_DIR / "config" / "global_params_template.txt"
DATA_DIR       = PROJECT_DIR / "data"
DOMAIN_DIR     = DATA_DIR / "domain"
PARAMS_DIR     = DATA_DIR / "parameters"
FORCING_DIR    = DATA_DIR / "forcings"
OUTPUT_DIR     = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIC_EXE_CANDIDATES = [
    "vic_image.exe",
    "/opt/VIC/build/vic_image.exe",
    "/usr/local/bin/vic_image.exe",
    str(PROJECT_DIR / "bin" / "vic_image.exe"),
]


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def find_vic_executable() -> str:
    """Buscar el ejecutable VIC en las rutas conocidas."""
    # Primero buscar en PATH
    vic_in_path = shutil.which("vic_image.exe")
    if vic_in_path:
        return vic_in_path

    for candidate in VIC_EXE_CANDIDATES:
        if Path(candidate).exists():
            return candidate

    raise FileNotFoundError(
        "vic_image.exe no encontrado.\n"
        "Opciones:\n"
        "  1. Usar Docker: docker-compose run vic python scripts/05_run_vic.py\n"
        "  2. Compilar VIC: cd /opt/VIC && cmake -DVIC_DRIVER=IMAGE . && make\n"
        "  3. Agregar vic_image.exe al PATH"
    )


STATE_DIR = DATA_DIR / "states"


def find_state_file(year: int) -> Path | None:
    """Buscar el archivo de estado guardado al final del año dado."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # VIC guarda como: state.{YYYYMMDD}_{HH}{MM}{SS}.nc
    # STATESEC=64800 → 18:00:00 → suffix _180000.nc
    pattern = f"state.{year}1231_*.nc"
    candidates = sorted(STATE_DIR.glob(pattern))
    return candidates[-1] if candidates else None


def generate_global_params(
    start_date: str,
    end_date: str,
    init_state: Path | None = None,
) -> Path:
    """
    Generar el archivo de parámetros globales de VIC.

    Args:
        start_date:  Fecha inicio (YYYY-MM-DD)
        end_date:    Fecha fin (YYYY-MM-DD)
        init_state:  Path al state file inicial (None = arrancar desde cero)
    """
    with open(TEMPLATE_FILE) as f:
        template = f.read()

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    init_state_line = (
        f"INIT_STATE      {init_state}" if init_state else ""
    )

    global_params = template.format(
        start_month=start.month,
        start_day=start.day,
        start_year=start.year,
        end_month=end.month,
        end_day=end.day,
        end_year=end.year,
        domain_file=str(DOMAIN_DIR / "domain.nc"),
        forcing_dir=str(FORCING_DIR) + "/",
        parameters_file=str(PARAMS_DIR / "params.nc"),
        output_dir=str(OUTPUT_DIR) + "/",
        init_state_line=init_state_line,
        state_dir=str(STATE_DIR),
    )

    output_path = DATA_DIR / "global_params.txt"
    with open(output_path, "w") as f:
        f.write(global_params)

    print(f"[VIC] Parámetros globales: {output_path}")
    return output_path


def validate_inputs() -> bool:
    """Verificar que todos los archivos de entrada existen."""
    required_files = [
        DOMAIN_DIR / "domain.nc",
        PARAMS_DIR / "params.nc",
    ]

    all_ok = True
    for f in required_files:
        if not f.exists():
            print(f"[ERROR] No encontrado: {f}")
            all_ok = False
        else:
            print(f"[OK] {f.name}")

    # Verificar que hay al menos un archivo de forzantes
    forcing_files = list(FORCING_DIR.glob("forcing_*.nc"))
    if not forcing_files:
        print(f"[ERROR] No hay archivos de forzantes en {FORCING_DIR}")
        all_ok = False
    else:
        print(f"[OK] Forzantes: {len(forcing_files)} archivos encontrados")

    return all_ok


def run_vic(global_params_file: Path, vic_exe: str) -> int:
    """
    Ejecutar VIC5 Image Driver.

    Args:
        global_params_file: Path al archivo de parámetros globales
        vic_exe: Path al ejecutable VIC

    Returns:
        Código de retorno del proceso (0 = éxito)
    """
    cmd = [vic_exe, "-g", str(global_params_file)]

    print(f"\n[VIC] Ejecutando: {' '.join(cmd)}")
    print("=" * 60)

    # Crear directorio de logs
    log_dir = PROJECT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"vic_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with open(log_file, "w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env={**os.environ, "OMP_NUM_THREADS": "4"},  # OpenMP parallelism
        )

        # Mostrar output en tiempo real y guardar log
        # Filter out high-volume WARN lines to prevent log flooding
        _warn_count = 0
        for line in process.stdout:
            if "[WARN]" in line:
                _warn_count += 1
                if _warn_count <= 10:
                    print(line, end="")
                    log.write(line)
                elif _warn_count == 11:
                    msg = "[INFO] Suppressing repeated [WARN] lines (too many to log)...\n"
                    print(msg, end="")
                    log.write(msg)
                # else: skip writing to avoid disk flooding
            else:
                print(line, end="")
                log.write(line)
        if _warn_count > 10:
            msg = f"[INFO] Total suppressed [WARN] lines: {_warn_count - 10}\n"
            print(msg, end="")
            log.write(msg)

        process.wait()

    print("=" * 60)
    print(f"[VIC] Log guardado: {log_file}")

    return process.returncode


def check_outputs() -> None:
    """Verificar que VIC generó las salidas esperadas."""
    output_files = list(OUTPUT_DIR.glob("*.nc"))

    if not output_files:
        print("[ERROR] No se generaron archivos de salida.")
        return

    print(f"\n[VIC] Salidas generadas: {len(output_files)} archivos")
    for f in sorted(output_files):
        import xarray as xr
        try:
            ds = xr.open_dataset(f)
            vars_str = ", ".join(list(ds.data_vars)[:5])
            print(f"  {f.name}: [{ds.dims}] {vars_str}...")
            ds.close()
        except Exception as e:
            print(f"  {f.name}: [ERROR] {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar VIC5 Image Driver - Cuenca Paucartambo"
    )
    parser.add_argument("--start", default="2000-01-01",
                        help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31",
                        help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo generar global_params.txt sin ejecutar VIC")
    parser.add_argument("--vic-exe", type=str, default=None,
                        help="Ruta al ejecutable vic_image.exe")
    parser.add_argument("--resume", action="store_true",
                        help="Saltar años que ya tienen flux output y state file")
    # Argumento legacy (no-op, mantenido por compatibilidad)
    parser.add_argument("--spinup", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    start_year = int(args.start[:4])
    end_year   = int(args.end[:4])

    print("=" * 60)
    print("  EJECUTANDO VIC5 IMAGE DRIVER - Cuenca Paucartambo")
    print(f"  Modo: año por año ({start_year}–{end_year}), restart cada año")
    print("=" * 60)

    # 1. Encontrar ejecutable VIC
    if args.dry_run:
        vic_exe = "vic_image.exe (dry-run)"
    else:
        if args.vic_exe:
            vic_exe = args.vic_exe
        else:
            try:
                vic_exe = find_vic_executable()
            except FileNotFoundError as e:
                print(f"\n[ERROR] {e}")
                sys.exit(1)
        print(f"[VIC] Ejecutable: {vic_exe}")

    # 2. Validar entradas
    print("\n[→] Validando archivos de entrada...")
    if not validate_inputs():
        print("\n[ERROR] Archivos de entrada faltantes. Ejecuta scripts 01-04 primero.")
        sys.exit(1)

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Correr año por año
    failed_years = []
    for year in range(start_year, end_year + 1):
        flux_file   = OUTPUT_DIR / f"fluxes.{year}-01-01.nc"
        state_file  = find_state_file(year)

        # --resume: saltar si ya existe output completo Y state file guardado
        if args.resume and flux_file.exists() and state_file is not None:
            size_mb = flux_file.stat().st_size / 1e6
            print(f"[SKIP] {year}: flux ({size_mb:.0f} MB) + state OK — saltando")
            continue

        # Buscar state del año anterior como init
        init_state = find_state_file(year - 1) if year > start_year else None
        if init_state:
            print(f"\n[→] {year}: iniciando desde state {init_state.name}")
        else:
            print(f"\n[→] {year}: sin state previo (arranque desde cero)")

        start_date = f"{year}-01-01"
        end_date   = f"{year}-12-31"

        global_params_file = generate_global_params(
            start_date=start_date,
            end_date=end_date,
            init_state=init_state,
        )

        if args.dry_run:
            print(f"  [DRY-RUN] {year}: global_params.txt generado, no ejecutado.")
            continue

        print(f"[VIC] Corriendo {year}...")
        ret = run_vic(global_params_file, vic_exe)

        if ret != 0:
            print(f"\n[ERROR] VIC falló en año {year} (código {ret})")
            print(f"  State del año anterior preservado: {init_state}")
            print(f"  Para reintentar: python scripts/05_run_vic.py "
                  f"--start {year}-01-01 --end {end_year}-12-31 --resume")
            failed_years.append(year)
            break  # No continuar — el estado de suelos sería incorrecto

        # Verificar que el state file fue guardado
        new_state = find_state_file(year)
        if new_state:
            print(f"[VIC] State guardado: {new_state.name}")
        else:
            print(f"[WARN] No se encontró state file para {year} en {STATE_DIR}")

    # 4. Resumen
    print("\n" + "=" * 60)
    if not failed_years:
        print(f"[✓] VIC completado: {start_year}–{end_year}")
        check_outputs()
    else:
        print(f"[!] Falló en año {failed_years[0]}. Años anteriores completos y con state.")
        sys.exit(1)


if __name__ == "__main__":
    main()
