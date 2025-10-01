Questions:
- citations with [1] are ok? or should I expand
- source for temp wien/black body applicable to continuum


1) Temperatura:
La pregunta que nos haciamos es si podiamos aplicar la aproximacion de Wien tanto en el continuo como en el nucleo de la linea. La respuesta es que hay que tener mucho cuidado en el nucleo, ya que cuando el plasma que estamos observando atraviesa un campo magnetico fuerte, la linea espectral, si es muy sensible al campo magnetico (nuestro caso), sufre un desdoblamiento. Si el campo es muy fuerte y el campo es totalmente longitudinal (inclinacion 0 o 180 grados), se puede llegar a separar tanto las componentes sigma, que la parte central (donde tendriamos el nucleo de la linea) alcanza valores de continuo.

Para entender mejor esto, te adjunto una imagen. La linea azul es la linea 6173 totalmente en reposo, la de laboratorio. La linea naranja es esa misma linea pero con campo magnetico de 3000G e inclinacion 0 grados. Puedes ver que si cogemos la linea naranja y calculamos las temperaturas en la longitud de onda marcada con la linea azul y en la marcada con la linea roja, obtendremos dos valores muy parecidos, lo cual sabemos que no es cierto, porque la temperatura en el continuo no es igual que la temperatura en la linea espectral.

Asi que yo pondria en el informe los mapas que has calculado para el continuo, me olvidaria de las graficas que hiciste de la temperatura para cada longitud de onda para diferentes pixeles y comentaria que esta aproximacion es valida en el continuo, ya que no se ve afectado por el campo magnetico mientras que las lineas espectrales sensibles a este (como son las nuestras) si estan afectadas.


2) Velocidad a lo largo de la linea de vision:
He estado dandole vueltas y me da que el hecho de que se te vayan las velocidades al azul podria ser debido a lo que llamamos "convective blueshift". Puedes echarle un ojo a estos articulos:
https://ui.adsabs.harvard.edu/abs/2023A%26A...680A..62E/abstract
https://ui.adsabs.harvard.edu/abs/2018A%26A...611A...4L/abstract
https://ui.adsabs.harvard.edu/abs/2019A%26A...624A..57L/abstract

Si que es verdad, que en base a estos trabajos, el valor deberia rondar los 100 m/s para las lineas 6301 y 6302 y porque la mancha esta localizada a un angulo heliocentrico de unos 0.98 y a ti te salian valores un poco mas altos. Puedes calcular el valor promedio en la zona de Sol en calma que comentamos?

Se me ha olvidado comentarte en el apartado de la velocidad, que veras que en algun caso revisan el blueshift segun la resolucion de tu instrument/telescopio. Decirte que para Hinode es R=~30000, ya que se define como

R = lambda / Delta_lambda (R_hinode = 6301 / 0.21 =~ 30000)

Un saludo, Ana


3) Calculo de la inclinacion y el azimut:
SFA:
Echale un ojo a la seccion 3.5 de este articulo:
https://ui.adsabs.harvard.edu/abs/2003A%26A...408.1115K/abstract
Te suenan de algo la ecuacion 5 y la 12 dividida entre la 11?

WFA:
Echale un ojo a la seccion 3 de este articulo:
https://ui.adsabs.harvard.edu/abs/2012MNRAS.419..153M/abstract


4) Picos claros en el perfil de Q del perfil promedio del Sol en calma
Le he estado dando vueltas y me da la impresion que es mas problema de calibraciones que de otra cosa. En muchos casos, las calibraciones que se aplican no son precisos pero si suficientes para tener una buena calidad de datos. No le daria mayor importancia a esta parte.


Ya me dices si esto aclara todas las dudas que nos surgieron ayer y permiteme darte las gracias por plantearme dudas como estas, ya que, como te dije, me hace pensar en cosas basicas que de otra manera no me plantearia.



Un saludo, Ana




En cuanto a deltas negativos o positivos segun la posicion de los lobulos de V (positivo-negativo o negativo-positivo), efectivamente el signo del campo magnetico te lo dara el hecho de que el lobulo positivo este en la parte roja o azul del perfil, pero para la comparacion usa el valor absoluto. Por convenio, cuando el lobulo azul es negativo y el rojo es positivo, el campo magnetico longitudinal es positivo y viceversa, con lo cual creo que la asuncion que haces es correcta.




1) The Sun: Structure and Phenomena

     1.1) Intro

     1.2) Solar Structure

     1.3) Active Regions and Sunspots

     1.4) Zeemann Effect and Stokes Parameters

     1.5) Weak Field Approximation and Strong Field Approximation

     1.5) Motivation and Objectives



2) Instrumentation and Data

     2.1) Hinode Mission

     2.2) Instruments Onboard

     2.3) Data Overview and Calibrations



3) Methodology and Results (las subsecciones ya dependeria de como los quieras mostrar)

     3.1) Magnetic Field Strength

     3.2) Magnetic Field Inclination

     3.3) Magnetic Field Azimuth

     3.4) Temperature

     3.5) Line-of-sight Velocity



4) Discussion and Conclusions










Aunque tal vez seria interesante analizar por que en unas zonas se aplica una aproximacion y por que en otras zonas la otra aproximacion para ponerlo en el informe. (desdoblamiento Zeeman es mayor/menor que la separacion por Doppler)





Sobre los puntos que están en la penumbra externa y que coinciden en lugares que pasan de ser positivos a negativos. Efectivamente, son aquellos parches donde el escenario físico cambia con respecto a lo que se tiene alrededor. No te preocupes si la inclinación del campo magnético, cuando la calcules, te sale rara. Estos píxeles tienen perfiles más complejos y la aproximación que estamos usando es demasiado sencilla para poder caracterizarlos bien. Cuando calcules la velocidad en la línea de visión de todo el mapa volveremos sobre estos parches y los explicaremos, porque en ese momento ya tendrás más herramientas para poderlo hacer.


Figuras:
- poner el 0 en los ejes x e y en la esquina inferior izquierda
- poner los ejes en segundos de arco (esto es multiplicar cada eje por 0.16, ya que tu resolución es de 0.16"/pixel tanto en x como en y) o en km (multiplicar cada eje por 0.16 y por 725 km, en la superficie del Sol 1" equivale aprox. a 725 km),
- usar paletas de color que ayuden a visualizar bien lo que quieres mostrar, poner unidades en ejes y barra de color (si la hubiere).
- Consejo para los pies de figura: empezar con una frase que haga de titulo de esta y luego describirla
- Se suele poner el cero del eje y abajo a la izquierda en lugar de arriba a la izquierda.

Sobre los ángulos. A esto me gustaría que le dieras una vuelta. En realidad, las ecuaciones que te puse en el pdf están evaluadas para cada lambda. Pero claro, tú quieres tener un sólo valor para la inclinación y el azimut que te dé información sobre cómo es el campo magnético. Pensando en cómo son las formas de Q, U y V, ¿tú qué harías?
