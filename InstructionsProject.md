1. Realizar un proyecto usando el modelo hidrologico VIC
2. Usar el compilador image.exe (existen tres compiladores classic, image, cesm)
3. Usar generar todos los predictores usando de forma eficiente Google Earth Engine (GEE)
para los forzantes, he guardado el token para que puedas realizar la
autenticacion, sin necesidad de logearse, es decir ya no es necesario que pegue
el token, sino que esa llave esta en este sistema por el momento:
export EE_SERVICE_ACCOUNT_JSON_B64=$(cat
/home/cenciso/Documents/WORK/ENGIE/DashBoard/windShortTermForecast/key_b64.txt)
4. El proyecto debe usar docker.
5. El proyecto debe usar la cuenca de paucartambo. 
6. Debe ser a paso diario 
7. debes usar librerias de python, xarray, wxee (permite descargar xarray de
   gee), o puedes generar points aleatorios densos en la cuenca descargar de
   forma puntual y luego grillarlo, ve la manera mas optima. 
8. Debes usar un routing complementario al modelo vic, creo que VIC maneja su
   propio routing.
9. debes hacer una simulacion que nos permita luego generar un archivo animado
   de como es la dinamica en la cuenca, propondria usar (por el tamano de la
   cuenca), datos de 1km o lo mas fino posible.
10. Realizar un plot timeseries de un punto de caudal de la cuenca. 
11. Permitir mejorar la simulacion, agregando estaciones de monitoreo de
    precipitacion, gestiona esto, pero como son pocas, dejalo como un paso opcional.
11. Opcional: De ser posible calcular, bajo alguna metodologia simple, el rango de tiempo
    de desplazamiento del agua desde cada embalse a la central de yuncan.
