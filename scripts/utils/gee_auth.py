"""
Google Earth Engine Authentication
Autenticación via Service Account (sin login interactivo)
"""

import os
import base64
import json
import ee


def authenticate() -> None:
    """
    Autenticar con GEE usando service account desde variable de entorno.

    Requiere:
        export EE_SERVICE_ACCOUNT_JSON_B64=$(cat /path/to/key_b64.txt)
    """
    json_b64 = os.environ.get("EE_SERVICE_ACCOUNT_JSON_B64")

    if not json_b64:
        raise EnvironmentError(
            "Variable EE_SERVICE_ACCOUNT_JSON_B64 no encontrada.\n"
            "Ejecuta: export EE_SERVICE_ACCOUNT_JSON_B64=$(cat "
            "/home/cenciso/Documents/WORK/ENGIE/DashBoard/"
            "windShortTermForecast/key_b64.txt)"
        )

    try:
        json_str = base64.b64decode(json_b64).decode("utf-8")
        service_account_info = json.loads(json_str)

        credentials = ee.ServiceAccountCredentials(
            email=service_account_info["client_email"],
            key_data=json_str,
        )
        ee.Initialize(credentials)
        print(f"[GEE] Autenticado como: {service_account_info['client_email']}")

    except Exception as e:
        raise RuntimeError(f"Error al autenticar con GEE: {e}") from e


def get_basin_geometry(config: dict) -> ee.Geometry:
    """
    Construir geometría de la cuenca desde bounding box.

    Args:
        config: Diccionario con configuración de basin_config.yml

    Returns:
        ee.Geometry.Rectangle de la cuenca
    """
    bbox = config["basin"]["bbox"]
    return ee.Geometry.Rectangle(
        [bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]]
    )
