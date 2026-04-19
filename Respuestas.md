# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
En esta fase inicial se ha realizado una auditoría completa del dataset NYC Airbnb Open Data con el fin de garantizar la integridad de los datos antes de proceder al modelado predictivo. El análisis se ha estructurado en cuatro dimensiones clave:

1. Calidad y Limpieza de los Datos
El dataset presentaba un volumen de datos considerable (~49,000 registros), pero con problemas de ruido y valores ausentes que requerían intervención:

Tratamiento de Nulos: Se detectó que la columna reviews_per_month tenía un 20.5% de valores nulos. Tras analizar la relación con otras variables, se determinó que estos nulos no eran errores, sino alojamientos sin actividad, por lo que se imputaron con el valor 0.

Reducción de Dimensionalidad: Se eliminaron columnas de texto e identificadores (id, name, host_name, last_review) que no aportan valor estadístico para una regresión, optimizando así el uso de memoria y la claridad del modelo.

2. Comportamiento de la Variable Objetivo (price)
El análisis de la variable price reveló una distribución con un fuerte sesgo positivo. La presencia de alojamientos con precios extremos (hasta 10,000$) generaba una asimetría y curtosis muy elevadas, lo que invalidaría los supuestos de una regresión lineal estándar.

Decisión Crítica: Se aplicó un filtrado por el método de Rango Intercuartílico (IQR), estableciendo un límite superior de 334$. Esta limpieza redujo la varianza del target y permitió trabajar con un subconjunto de datos mucho más representativo del mercado inmobiliario común de Nueva York.

3. Perfil de los Alojamientos (Variables Categóricas)
Se observa un claro desbalanceo geográfico y de producto:

Localización: El dataset está dominado por Manhattan y Brooklyn, que concentran la oferta de mayor precio.

Tipología: Los "Pisos completos" y las "Habitaciones privadas" representan la gran mayoría del mercado. Esto indica que el futuro modelo de regresión tendrá un rendimiento óptimo en estos segmentos, pero podría ser menos preciso en categorías minoritarias como "Habitaciones compartidas".

4. Relaciones Lineales y Multicolinealidad
El análisis de correlación de Pearson arrojó una conclusión fundamental: la relación entre las variables numéricas individuales y el precio es muy débil (todas < 0.1).

Esto sugiere que el precio no se explica de forma aislada por la disponibilidad o el número de reseñas, sino que es un fenómeno multivariante donde la ubicación y el tipo de inmueble juegan un papel crucial.

Ausencia de Multicolinealidad: Al no detectarse pares de variables predictoras con una correlación superior a 0.9, confirmamos que no existe redundancia de información, lo que garantiza la estabilidad matemática de los coeficientes del modelo de regresión lineal.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset proviene de la plataforma Kaggle, recopilando datos abiertos de Inside Airbnb sobre la actividad de alquileres en la ciudad de Nueva York.

La variable objetivo (target) es price. Tiene sentido realizar una regresión sobre ella porque es una variable cuantitativa continua. Nuestro objetivo es predecir un valor numérico exacto (el precio por noche) a partir de un conjunto de características (ubicación, disponibilidad, tipo de habitación), lo cual es la definición fundamental de un problema de regresión.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> Tras analizar los histogramas, la mayoría de las variables numéricas (como price, minimum_nights y number_of_reviews) muestran una distribución con sesgo positivo (cola larga hacia la derecha), lo que indica una alta concentración de valores bajos y pocos valores extremadamente altos.Outliers: Se han detectado valores atípicos críticos en la variable price, con precios que alcanzan los 10.000$, distorsionando la media y el análisis.Tratamiento: He aplicado el método del Rango Intercuartílico (IQR) para filtrar price. He eliminado los registros por encima del límite superior (334$). Para el límite inferior, dado que el cálculo matemático resultaba negativo -90, (lo cual no tiene sentido), se ha ajustado al mínimo real positivo del dataset. Esto garantiza un modelo más estable y realista.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

>Tras analizar la matriz de correlación de Pearson, las tres variables con mayor impacto (ordenadas por su valor absoluto) son:

availability_365: Valor absoluto de 0.0818 (Coeficiente real: 0.0818).
calculated_host_listings_count: Valor absoluto de 0.0575 (Coeficiente real: 0.0575).
reviews_per_month: Valor absoluto de 0.0506 (Coeficiente real: -0.0506).

Interpretación crítica: Es importante destacar que estas correlaciones son muy bajas (todas inferiores a 0.1). En la escala de Pearson, valores tan cercanos a cero indican una relación lineal muy débil o despreciable entre estas variables individuales y el precio. Esto sugiere que el precio por noche en Nueva York está influido por una combinación multivariable compleja y que factores categóricos (como la ubicación exacta o el tipo de alojamiento) o variables no presentes en el dataset podrían tener un peso mucho mayor que estas métricas numéricas por sí solas.

Multicolinealidad: No se han detectado pares de variables predictoras con $|r| > 0.9$, lo que confirma que no hay redundancia severa entre los datos numéricos de entrada.
**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

>Sí, existen valores nulos significativos, especialmente en la columna reviews_per_month (20.5%) y la fecha last_review (20.5%).

Tratamiento:

reviews_per_month: Se han imputado con 0, ya que la falta de datos coincide con alojamientos que tienen 0 reseñas totales.

last_review: Se ha optado por eliminar la columna, ya que el análisis de fechas excede el objetivo de esta regresión lineal.

name y host_name: Presentaban nulos residuales (<0.1%) y, al ser identificadores sin valor estadístico, se han eliminado para limpiar el dataset.

Tras estos pasos, se aplicó dropna() para eliminar cualquier fila con nulos restantes y asegurar un entrenamiento limpio.
---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
Añade aqui tu descripción y analisis:

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

>MAE (Error Absoluto Medio): 71.0791
RMSE (Raíz del Error Cuadrático Medio): 197.7370
R² (Coeficiente de Determinación): 0.1162

No, el modelo funciona de manera deficiente.
Bajo poder explicativo ($R^2$): Un valor de $0.1162$ indica que el modelo solo es capaz de explicar el 11.6% de la variabilidad del precio. 
Esto significa que casi el 90% del precio depende de factores que el modelo no está capturando o que la relación no es lineal.

Error elevado (MAE): Un error medio de 71$ por noche es sumamente alto, considerando que gran parte de los alojamientos (tras la limpieza de outliers) se mueven en rangos de entre 50$ y 200$. Errar por 71$ supone un margen de error relativo muy grande.

Presencia de valores extremos (RMSE): El hecho de que el RMSE (197.73) sea casi el triple que el MAE confirma que el modelo comete errores muy grandes en ciertos registros (penalizados doblemente por el RMSE). Como se ve en el gráfico de residuos, hay fallos de predicción de miles de dólares.

Underfitting: El modelo sufre de un claro infra-ajuste. La realidad de los precios en Nueva York es demasiado compleja para ser modelada con una simple línea recta basándose únicamente en las variables disponibles.
---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
Añade aqui tu descripción y analisis:

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> Es una fórmula que busca el equilibro. 
La parte (XᵀX) mide la forma en que se relacionan las variables entre sí (varianza y covarianza). Cuando se realiza la inversiónn buscamos aislar el efecto de cada variable.
Xᵀ mide como se relaciona cada variable con la variable objetivo
El resultado es el peso, que mide la importancia que le damos a cada característica

Es necesario añadir una columna de unos a la matriz X, porque si no lo hiciéramos obligas matemáticamente a que la recta pase por el punto (0,0).
En álgebra lineal, añadir esa columna de unos permite tratar el intercepto como una variable más, convirtiendo una operación de suma ($y = ax + b$) en un producto matricial limpio, que es lo que la fórmula β = (XᵀX)⁻¹ Xᵀ sabe resolver.

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       |    4.864995    |
| β₁        | 2.0       |    2.063618    |
| β₂        | -1.0      |   -1.117038    |
| β₃        | 0.5       |    0.438517    |

> Los coeficientes calculados por la función de Regresión Lineal Múltiple implementada mediante la Ecuación Normal son sumamente próximos a los valores reales definidos en el generador sintético.

La ligera desviación observada (en el orden de centésimas o décimas) es completamente normal y esperada. Esto se debe a que la variable objetivo $y$ fue generada introduciendo un ruido gaussiano (sigma=1.5). En cualquier modelo de regresión real, el algoritmo intenta encontrar la tendencia subyacente filtrando el ruido aleatorio; el hecho de que los valores ajustados sigan la dirección y magnitud de los reales (por ejemplo, manteniendo el signo negativo en beta_2) confirma que la implementación matemática es correcta y robusta.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> Los valores obtenidos se aproximan satisfactoriamente a los de referencia, cumpliendo con los rangos de tolerancia esperados para una implementación de Mínimos Cuadrados Ordinarios (OLS).

MAE y RMSE: Mis resultados (1.16 y 1.46) están incluso ligeramente por debajo de la referencia, lo que indica que el error medio de las predicciones es bajo y que el modelo ha capturado bien la estructura de los datos sintéticos. La cercanía entre MAE y RMSE sugiere que no hay errores atípicos (outliers) extremos en el conjunto de test que distorsionen la evaluación.

R² (Coeficiente de Determinación): He obtenido un 0.6897. Aunque es algo inferior al 0.80 ideal mencionado, sigue siendo un resultado sólido que indica que el modelo explica casi el 69% de la varianza de la variable objetivo.

**Pregunta 3.4** — Compara los resultados con la reacción logística anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido. 

>Al comparar los resultados de la Regresión Lineal Múltiple del Ejercicio 3 con la Regresión Lineal del Ejercicio 2, observamos una diferencia drástica en el rendimiento:
| Aspecto | Ejercicio 2 (Airbnb NYC) | Ejercicio 3 (NumPy Sintético) |
| :--- | :--- | :--- |
| **Origen de los datos** | Datos reales del mercado de NYC | Datos generados matemáticamente |
| **Relación de variables** | Compleja, ruidosa y no lineal | Lineal pura (definida por fórmula) |
| **Coeficiente $R^2$** | **0.1162** (Bajo poder explicativo) | **0.6897** (Alto poder explicativo) |
| **MAE (Error medio)** | **71.08$** (Error significativo) | **1.16** (Error mínimo) |
| **Conclusión** | Underfitting por complejidad de la realidad | Ajuste preciso de una función conocida |
¿Qué ha sucedido? Explicación técnica:Naturaleza de los datos: En el Ejercicio 3, los datos se generaron artificialmente mediante una función matemática lineal conocida (y = 5 + 2x_1 - 1x_2 + 0.5x_3). Al haber una relación lineal "de fábrica", es normal que el modelo NumPy logre un R^2 alto; simplemente está redescubriendo la fórmula que nosotros mismos pusimos.Complejidad de la realidad vs. Simulación: El dataset de Airbnb (Ejercicio 2) es "ruido real". El precio de una vivienda no depende solo de 3 o 4 factores numéricos, sino de miles de variables (subida de demanda estacional, decoración, opiniones subjetivas, etc.). La regresión lineal es demasiado simple para capturar esa complejidad, lo que explica por qué el R^2 es tan pobre (0.11) en comparación con el modelo sintético.Calidad del ajuste: Mientras que en el Ejercicio 3 los coeficientes ajustados son casi idénticos a los reales, en el Ejercicio 2 el modelo "da palos de ciego", llegando incluso a predecir precios negativos.

---

## Ejercicio 4 — Series Temporales
---
Añade aqui tu descripción y analisis:

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> Sí, la serie presenta una tendencia clara.

Es una tendencia de tipo lineal y aditiva.
Es creciente (positiva) a lo largo de todo el periodo analizado (2018-2023).Empieza en un valor aproximado de 60 en 2018 y termina cerca de los 160 a finales de 2023. Esto supone un incremento total de unos 100 unidades en 6 años, lo que equivale a una pendiente media de aproximadamente 0.05 unidades por día.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Sí, existe una estacionalidad muy marcada y regular. El periodo es anual (365 días). Se observa un ciclo completo que se repite exactamente cada año, con máximos hacia la mitad del año y mínimos al inicio/final del mismo.La amplitud del patrón estacional oscila aproximadamente entre -20 y +15 unidades. Es decir, el efecto estacional puede variar el valor de la serie en un rango de unas 35 unidades a lo largo del año. Al ser una descomposición aditiva, esta amplitud se mantiene constante a pesar de que la tendencia crezca.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> Sí, se aprecian ciclos de largo plazo. > En el gráfico de la serie original y en el de la tendencia, se observa que la línea no es una recta perfecta, sino que presenta una ondulación suave que tarda varios años en completarse (aproximadamente un ciclo de 4 años).

La tendencia es la dirección general a largo plazo (en este caso, un crecimiento constante y sostenido).

El ciclo representa oscilaciones o variaciones que suben y bajan alrededor de esa tendencia, pero que no tienen un periodo fijo y corto como la estacionalidad (anual). Mientras la tendencia es lineal o monótona, el ciclo es ondulatorio. En la descomposición, la tendencia a menudo "absorbe" estos ciclos si no se especifica un modelo de filtrado más complejo.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> Sí, el residuo se ajusta casi perfectamente a un ruido ideal (ruido blanco gaussiano).Basándome en los resultados del archivo ej4_analisis.txt y los gráficos generados, justifico esta afirmación con los siguientes datos:Media: 0.1271 (Muy cercana a 0, lo que indica que no hay un sesgo sistemático en los errores).Desviación típica (sigma): 3.22 (Muestra una dispersión constante, lo que sugiere homocedasticidad).Test de Normalidad (Jarque-Bera): El p-valor es 0.5766.Justificación: Dado que el p-valor (0.5766) es significativamente mayor que el nivel de significancia estándar (0.05), no podemos rechazar la hipótesis nula de normalidad. Esto, sumado a que el test ADF dio un p-valor de 0.0000 (confirmando que es estacionario) y que el histograma sigue la campana de Gauss, confirma que el residuo es ruido aleatorio puro y que el modelo de descomposición ha extraído toda la estructura útil de la serie.

---

*Fin del documento de respuestas*
