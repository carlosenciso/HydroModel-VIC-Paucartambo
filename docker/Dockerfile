# ============================================================
# Dockerfile - VIC Hydrological Model
# Cuenca Paucartambo, Pasco - Peru
# ============================================================
# Build: docker build -t vic-paucartambo -f docker/Dockerfile .
# Run:   docker-compose run --rm vic python scripts/05_run_vic.py

FROM ubuntu:22.04

LABEL maintainer="ENGIE Peru - Hidrología"
LABEL description="VIC5 Image Driver + Python stack para cuenca Paucartambo"

ARG DEBIAN_FRONTEND=noninteractive

# ─────────────────────────────────────────────────────────
# 1. Dependencias del sistema
# ─────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Compilación
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # NetCDF4 (requerido por VIC image driver)
    libnetcdf-dev \
    libnetcdf-c++4-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    # ZLIB (compresión)
    zlib1g-dev \
    # Python
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # Geoespacial
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libgdal-dev \
    gdal-bin \
    # Otros
    libopenmpi-dev \
    openmpi-bin \
    nco \
    cdo \
    libeccodes-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────
# 2. Compilar VIC5 Image Driver desde fuente
# ─────────────────────────────────────────────────────────
WORKDIR /opt

RUN git clone --depth=1 --branch VIC.5.1.0 \
    https://github.com/UW-Hydro/VIC.git VIC \
    || git clone --depth=1 https://github.com/UW-Hydro/VIC.git VIC

WORKDIR /opt/VIC

# Verificar nc-config para diagnóstico
RUN nc-config --libs && nc-config --cflags

# Compilar VIC5 Image Driver
# Se fuerza NC_LIBS con todas las dependencias de HDF5/zlib para evitar
# errores de linker en Ubuntu 22.04 (netCDF linkea transitivamante HDF5)
# VIC 5.1.0 tiene variables globales en headers que fallan con GCC>=10 (flag -fno-common por defecto).
# Fix: agregar -fcommon al Makefile para compatibilidad con GCC 11 (Ubuntu 22.04).
RUN cd vic/drivers/image && \
    sed -i 's/-std=c99/-std=c99 -fcommon/' Makefile && \
    NC_LIBS="$(nc-config --libs) -lhdf5_serial -lhdf5_serial_hl -lz -ldl" \
    NC_CFLAGS="$(nc-config --cflags)" \
    make -j$(nproc) 2>&1 && \
    cp vic_image.exe /usr/local/bin/ && \
    chmod +x /usr/local/bin/vic_image.exe

# Verificar compilación
RUN vic_image.exe --help 2>&1 | head -5 || echo "[INFO] VIC compilado"

# ─────────────────────────────────────────────────────────
# 3. Entorno Python
# ─────────────────────────────────────────────────────────
WORKDIR /app

# Copiar requirements primero (cache de Docker)
COPY requirements.txt /app/requirements.txt

# Instalar dependencias Python
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        # Geoespacial y datos
        numpy>=1.24.0 \
        scipy>=1.11.0 \
        pandas>=2.0.0 \
        xarray>=2023.6.0 \
        netCDF4>=1.6.4 \
        h5netcdf>=1.2.0 \
        dask[complete]>=2023.6.0 \
        # Geoespacial
        geopandas>=0.13.0 \
        rasterio>=1.3.8 \
        pyproj>=3.6.0 \
        shapely>=2.0.0 \
        fiona>=1.9.4 \
        pysheds>=0.3.5 \
        # GEE
        earthengine-api>=0.1.390 \
        wxee>=0.4.0 \
        # Visualización
        matplotlib>=3.7.0 \
        cartopy>=0.22.0 \
        seaborn>=0.12.2 \
        imageio>=2.31.0 \
        imageio-ffmpeg>=0.4.9 \
        Pillow>=10.0.0 \
        cmocean>=3.0.3 \
        # Routing
        rvic>=1.1.0 \
        # Utilidades
        tqdm>=4.65.0 \
        pyyaml>=6.0 \
        click>=8.1.0 \
        requests>=2.31.0

# ─────────────────────────────────────────────────────────
# 4. Copiar código del proyecto
# ─────────────────────────────────────────────────────────
COPY config/ /app/config/
COPY scripts/ /app/scripts/

# Crear directorios de datos (serán montados como volumes)
RUN mkdir -p /data/forcings /data/parameters /data/domain \
             /data/routing /data/outputs /data/gee_raw \
             /app/plots /app/animations /app/logs

# ─────────────────────────────────────────────────────────
# 5. Variables de entorno
# ─────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=4
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj
ENV PATH="/usr/local/bin:${PATH}"

# ─────────────────────────────────────────────────────────
# 6. Entrypoint
# ─────────────────────────────────────────────────────────
WORKDIR /app

CMD ["python3", "scripts/05_run_vic.py", "--help"]
