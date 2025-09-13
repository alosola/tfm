# TODOS:

- Make plots with arcseconds on axes
- Improve data plots:
- - surface plots for magnetic field
- - "stream line" style plots for the angles, see Figure 5.2b of Jennerholm Hammar
- Make calculation of Gbar following Sara's email, fix values in magnetic field calculations
- Either fix derivatives functions, or remove in favour of numpy gradient
- Total magnetic field! is missing

# Questions
- Equation 3.23 in JH is "valid in the line core", does this apply to our case?
- delta lambda B -> absolute value?

- where do the factors from the magnetic field calculations come from? (Jennerholm Hammar)
-- f == filling factor, fraction of the magnetic field covering each resolution element
-- C1 is  4.6686 × 10−13 * lambda0^2 * gbar
-- C2 is 5.4490 × 10−26 * lambda0^4 * Gbar
-- gbar denotes the effective Landé factor, calculated using Equation 3.5 (Lozitsky)
-- Gbar effective Landé factor for linear polarization
-- lambda0 is the central wavelength of the spectral line
-- vmic is the mictroturbulent velocity, describing the impact of thermal motions to the line
-- Tk is the kinetic temperature
-- kB is the Boltzmann constant = 1.3806488e-16 [erg/K]
-- M = mass of the atomic element
-- me = Electron rest mass, 9.10938291e−28 [g]
-- e0 = Eelectron charge, 4.80320451 × 10−19 [statC]
-- c = Speed of light in vacuum, 2.99792458 × 108 [m · s−1]
All values of the constants are according to the 2010 CODATA recommendations and recomputed in units of the CGS metric system (see e.g. http://physics. nist.gov/cuu/Constants/index.html).  (Table A.1)

# Notes on work

For calibration:
Selected region of the image which does not have total polarization, this indicates there is not a very active magnetic field.

Normalization:
With mean intensity of quiet sun region continuum (average of first 5 wavelegth measurements)


----------------------------
## Notas reunión 20/08/2025
All data is normalized to the quiet sun intensity value--this is assumed to be the same in all cases. Therefore the resulting magnetic field strength is proportional to the Landé factor--specific to each spectral line.

Filling factor: for each pixel, represents how much of it is magnetic. for example, if a pixel catches part of the penumbra and part of the quiet sun, that pixel is not 100% magnetic--but determining the precise fraction is not trivial. Assume =1 for the whole field for this work.

Derivatives of I can be compared to Q, U (first) and V (second). See notes from last meeting. This is key to the weak field approximation.

Azimuth: can only be determined between +/- 90 degrees (not +/-180) because a single PoV causes degeneration in data--we can't determine the polarity of the vector direction (arrow on vector "stick").
For sunspots, we know the polarity of the spot and can determine the polarity of all the values in the spot, but not those of the quiet sun (at least not easily).

Next steps:
- Finish calculations of magnetic field
- Calculate velocities
- Compare with images from AIA, chromosphere, different altitudes, evolution, etc.
- Write!


----------------------------
## Notas reunión 06/08/2025
Q, U, V reflect the behaviour of the rate of change--therefore they are highest contrast when the rate of change of I is highest, about halfway down the spectral line.
V is related to the first derivative, and U to the second derivative.

The quiet sun can present magnetization regions, due to polarization from scattering or other sources that are outside the scope of this work. This work is focoused on the Zeeman effect, which dominates the polarisation around the sunspot.

Inverse lobes in the profile graphs are due to inverse magnetic fields (probable indicates a change in azimuth, pointing inwards and outwards from sun core). In the Magnetogram (longitudinal) this is represented by white/black spots.

This is related to the unfolding (desdoblamiento) of the spectral line due to the Zeeman effect. This is seen clearly in the umbra, where the lobe change through the core of the spectral line has a small meseta, instead of a sharp inclination. This is representative of a very intense magnetic field. The energy of the three atomic transitions (sigma-, pi, sigma+) is similar, but different enough that three lobes apear in the spectral line.

Negative values in Q, U, V represent magnetic field polarity, be careful not to lose this data.

NIST: database of spectral lines. Possibly find information on the slight dip which can be seen in the umbra in I and V about halfway between the 301 and 302 spectral lines. Find the calibrated wavelength value for this line, and match it to the data in NIST.
https://www.nist.gov/pml/atomic-spectra-database
https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=&output_type=0&low_w=6301.4&upp_w=6302.6&unit=0&de=0&plot_out=0&I_scale_type=1&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&g_out=on&submit=Retrieve+Data



----------------------------
## Notas reunión 23/07/2025
Líneas brillantes que aparecen en la umbra el en mapa de polarización circular--estructuras convectivas? Porqué hay zonas oscuras alrededor? Tiene que ver con la línea espectral? Aparecen en algúnas de las imágenes de I?

Spot (superior derecho): se ve en jhelioviewer que es una porción de la umbra que se separa y pasa por la penumbra para salir de la mancha. También se ve que antes la umbra estaba dividida en 2 por un punente de luz.

Puntos de polarización circular fuera de la mancha: aparecerán en el magnetograma (campo longitudinal). Puede ser por el calentamiento coronal, por la supergranulación, o por la formación de campos magnéticos en forma de burbuja... vemos que aparecen algunos como pequeños poros en el I del contínuo, otros serán puntos brillantes en I pero que requieren mas resolución angular. Este efecto es disperso, pero no despreciable.
Cerca de la mancha se podría deber a céldas superconvectivas.

Al acercarnos al núcleo de la línea espectral, subimos en la fotosfera, porque esta es más opaca.

Un gránulo es una burbuja de materia caliente que sube. Mientras mas sube, mas densa, más opaca--aparece como I menor. Ocurre algo parecido con los núcleos de lso filamentos, que son formados por apilamiento del plasma.



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
