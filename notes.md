# TODOS:
- Make gifs of evolution over wavelength
- Make plots with arcseconds on axes
- Improve data plots: surface plots (only for magnetic field?) is there better ways to represent
- revisa la parte de la aproximación del campo débil (en Jennerholm Hammar por ejemplo).
- el perfil de I en absorción (lo que debes tener mayoritariamente) es esencialmente una función que decrece llega a un mínimo y vuelve a crecer, ¿qué aspecto esperas que tenga su primera derivada?, ¿y su segunda derivada?


# Notes on work

For calibration: 
Selected region of the image which does not have total polarization, this indicates there is not a very active magnetic field.

Normalization:
With quiet sun it works well for I, but not for the other variabes:
--> Have applied usual normalization formula, data_n = (data - data_min) / data_range
--> TODO: check this is OK

----------------------------
## Notas reunión 01/07/2025

Para calibrar:
1. Seleccionamos una región de sol en calma, que no presente el efecto Doppler y que no presente un campo magnético fuerte.
   --> TODO: enviar cuadrado con este campo para confirmar que el campo seleccionado está OK
2. Sacamos el perfil de intensidades medio para cada longitud de onda. Esto se compara con los perfiles de referencia (archivo de calibración).
3. Calibramos con este comparación todas las longitudes de onda --> esta es la parte que no sé hacer.
4. Dividimos todas las imágenes por la intensidad promedio
   ?--> la misma para todas las longitudes de onda?
El resultado de este ejercicio será el mismo cubo de datos, pero calibrado en longitud de onda (los datos iniciales sólo tienen un índice) y normalizado. Tendremos valores ~1 en el sol en calma, y valores <<1 en la mancha.

Luego, miraremos la polarización circular en V (? confirmar esto con las transparencias de teoría).

Campo magnético: utilizar las ecuaciones más sencillas para el campo debil (Ana lo enviará por correo). Comparar con bibliografía.

## NOTAS TEORÍA:
Poros: concentración magnética fuerte, los puntos oscuros sin penumbra.
I, V, U son independientes. Q incluye parte de I. -> me van a enviar artículo.


## NOTAS INSTRUMENTALES:
La escala X e Y nos permite determinar el modo del Spectropolarimeter, en este caso es el modo normal.  --> this could also have been a clue I had the axes wrong
--> TODO: Plotear con segundos de arco en los ejes, en lugar de pixeles. Ojo que posiblemente no sean pixeles cuadrados.
Los datos proporcionados son de "nivel 1", es decir están listos para usar. Han sido procesados por el equipo (modulación polarimetrica, se han juntado las longitudes de arco, etc.) y por Sara (ha procesado todos los FITS individuales para generar este cubo).
En los 83 minutos que tarda en hacerse la foto, asumimos que no hay cambios grandes en la estructura. Suelen ser cambios más pequeños en filamentos o puentes de luz.
--> TODO: sacar videos del ljhelioviewer de este periodo de tiempo específico para añadir a la presentación.
Hinode tiene resolución espectral alta, pero no tiene resolución espacial tán óptima.
--> TODO: buscar ejemplos de otros instrumentos que aportan información distinta.
--> TODO: mirar los datos de HMI en intensidad pueden mostrar cuanto evoluciona esta mancha, ver si hay una flare (sí), si se ha notado en la fotosfera.



-------------------------------------------

- los datos del TFM: AR_12665_133153_0.h5 (en breve subiré un fichero para decirte cómo abrirlo)

- vídeos mostrando la evolución de la región activa mientras estaba en el disco visible del Sol: archivos mp4

- listado bibliografía con los enlaces a los documentos, se irá ampliando a medida que se necesite

- header: cabecera de uno de los slit images de SOT, te da información sobre diferentes parámetros, si quieres ver qué significa cada abreviatura puedes consultarlo en el documento sobre el instrumento que se incluye en el listado de la bibliografía
-> where did this header come from? I can't find the headersin the .hdf5 file

- un fichero quick_view_data, donde puedes ver una vista preliminar de diferentes mapas de los datos

- dos pdfs con el material de las dos sesiones que hicimos por Zoom




Para leer los datos en Python se necesita la librería "h5py". En particular, se pueden leer ejecutando la siguiente orden:

import h5py
data = h5py.File('AR_12665_133153_0.h5', 'r')

Para ver el contenido del fichero (lo que hay en la variable definida como "data") se hace:

list(data.keys())

esto muestra que hay un array dentro de "data" llamado "stokes". Para ver qué hay dentro de ese array, hacemos:

data['stokes'].shape 

El resultado te mostrará que es un array de 4 dimensiones. La primera dimensión es el número de parámetros de Stokes, la segunda es la dimensión espacial en el eje X, la tercera la dimensión espacial en el eje Y y la cuarta el número de longitudes de onda medidas.



He subido al directorio de cloud que compartimos:

- archivo fts_calibration.npz necesario para hacer la calibración en longitud de onda. Cuando lo abras, las keys son x (longitud de onda), y (intensidad), c (continuo en unidades CGS). Si representas x frente a y, verás el perfil de intensidad con el que comparar el perfil de intensidad promedio que sacarás de la región de Sol en calma que elijas. El perfil de intensidad del archivo de configuración cubre entre las longitudes de onda 6300 y 6304 Å, tendrás que ver cómo compararlo con tus datos porque el paso en longitud de onda y las longitudes iniciales y finales en tus datos no es el mismo que en los de calibración. Además, el perfil de intensidad para calibrar muestra: dos líneas más fuertes sobre 6301.5 y 6302.5 (esas son las que se han medido en Hinode), otras dos líneas menos marcadas las cuales son telúricas (como tus datos son medidos por un satélite no aparecen) y algunas más débiles. Puedes encontrar información sobre la calibración en longitud de onda en datos solares en https://www.aanda.org/articles/aa/full_html/2011/04/aa15664-10/aa15664-10.html

- publicación Lites & Ichimoto (2013) sobre la reducción y calibración de datos de Hinode/SOT. Es el archivo s11207-012-0205-4.pdf

Por otra parte:

- Las ecuaciones que puse en la presentación de espectropolarimetría se podrían utilizar pero hay otras más directas (son a las que me refería en la reunión de hoy). Estas últimas puedes encontrarlas en algunas publicaciones de la lista de la bibliografía. Concretamente, están en la publicación de Jennerholm Hammar, sección 3, ecuaciones 3.15, 3.23, 3.28, 3.30.

- Puedes ver el software para visualizar la evolución de estructuras solares observadas mediante diferente instrumentación en https://www.jhelioviewer.org/
