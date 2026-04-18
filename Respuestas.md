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

> _Escribe aquí tu respuesta_


---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
Añade aqui tu descripción y analisis:

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> _Escribe aquí tu respuesta_

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       |                |
| β₁        | 2.0       |                |
| β₂        | -1.0      |                |
| β₃        | 0.5       |                |

> _Escribe aquí tu respuesta_

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> _Escribe aquí tu respuesta_

**Pregunta 3.4* — Compara los resultados con la reacción logística anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido. 

> _Escribe aquí tu respuesta_

---

## Ejercicio 4 — Series Temporales
---
Añade aqui tu descripción y analisis:

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> _Escribe aquí tu respuesta_

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> _Escribe aquí tu respuesta_

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> _Escribe aquí tu respuesta_

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> _Escribe aquí tu respuesta_

---

*Fin del documento de respuestas*
