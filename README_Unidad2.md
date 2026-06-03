# Unidad 2: Modelos No Paramétricos

---

## Introducción

Esta unidad aborda una de las divisiones fundamentales en el aprendizaje automático: la diferencia entre los **modelos paramétricos** y los **modelos no paramétricos** para la estimación de densidad y la realización de predicciones (tanto clasificación como regresión).

Mientras que los modelos paramétricos asumen una forma funcional predefinida para los datos (por ejemplo, que siguen una distribución gaussiana o que su comportamiento puede representarse con un polinomio), los **modelos no paramétricos** no hacen ninguna suposición sobre la distribución subyacente. En cambio, dejan que los datos "hablen por sí mismos", adaptando la función de densidad directamente a partir de las observaciones disponibles.

Los objetivos principales de esta unidad son:

- Comprender la diferencia conceptual entre modelos paramétricos y no paramétricos.
- Estudiar el **método del histograma** como modelo no paramétrico básico.
- Estudiar el **método de k-vecinos más cercanos (k-NN)** para clasificación y regresión.
- Estudiar el **método de ventana de Parzen** (estimación de densidad por kernel).
- Entender la importancia de la **normalización de variables** en métodos basados en distancias.
- Analizar las ventajas, limitaciones y consideraciones prácticas de cada método.

---

## Contenido de la Unidad

La unidad se compone de un único notebook/documento en formato PDF exportado desde una página web interactiva (Jupyter Book), correspondiente a la **Clase 04** del curso *Introducción al Machine Learning* (repositorio `jdariasl/Intro_ML_2025`). El documento integra texto explicativo, código Python ejecutable, fórmulas matemáticas y visualizaciones 3D de los resultados.

---

## Estructura de Archivos

---

### Archivo: `Modelos_no_parámetricos___Introducción_al_Machine_Learning.pdf`

#### Propósito

Documento principal de la unidad. Corresponde a la exportación de un notebook interactivo de Jupyter Book que cubre de manera teórica y práctica los tres grandes métodos de aprendizaje no paramétrico: histograma, k-vecinos más cercanos y ventana de Parzen. Incluye código Python ejecutado, fórmulas matemáticas renderizadas en LaTeX y gráficos 3D resultantes de cada experimento.

---

#### Contenido y Funcionamiento Detallado

---

### Sección 1 – Contexto: Aprendizaje de Máquina como Estimación de Densidad

El documento abre estableciendo el marco conceptual general. El aprendizaje de máquina puede entenderse como el problema de ajustar un modelo que describa la densidad de probabilidad de un conjunto de datos:

- **Aprendizaje supervisado**: Se busca ajustar el modelo condicional `p(y/x)` (probabilidad de la salida dado el vector de entrada). Esto cubre regresión y clasificación.
- **Aprendizaje no supervisado**: Se busca ajustar el modelo de densidad marginal `p(x)` (estructura intrínseca de los datos sin etiquetas).

Este enfoque unificado, llamado **estimación de densidad**, es la base sobre la que se clasifican los modelos de aprendizaje.

---

### Sección 2 – Modelos Paramétricos vs. No Paramétricos

#### Modelos Paramétricos

Un modelo paramétrico **asume una forma funcional fija** para la densidad de probabilidad. Los datos se "ajustan" a esa forma estimando un número constante de parámetros durante el entrenamiento.

**Ejemplos presentados en el documento:**

| Modelo | Descripción | Parámetros |
|---|---|---|
| Distribución Gaussiana | Se asume que los datos siguen una campana de Gauss | 2 (media `μ` y varianza `σ²`) |
| Regresión polinomial de grado 2 | Se asume que la relación entre x e y es cuadrática | 3 (coeficientes del polinomio) |

**Ventajas:**
- Computacionalmente muy eficientes.
- El número de parámetros no crece con los datos.

**Desventajas:**
- Si la suposición sobre la forma de la función no se cumple en los datos reales, el modelo incurrirá en errores sistemáticos (sesgo alto).

#### Modelos No Paramétricos

Un modelo no paramétrico **no asume ninguna forma funcional** para la distribución. El modelo se adapta completamente a los datos disponibles, sin un número fijo de parámetros predefinido. En la práctica, el número efectivo de "parámetros" crece con el tamaño del conjunto de datos.

---

### Sección 3 – Método del Histograma

#### Concepto

El histograma es el ejemplo más básico de un modelo no paramétrico. Funciona dividiendo el espacio de las variables en **celdas (bins) de igual tamaño** y contando cuántos puntos caen en cada celda.

#### Fórmula en 1D

Para una variable unidimensional, la estimación de densidad para un punto `x` es:

```
p̂(x) = nj / (Σ(nj) * dx)
```

Donde:
- `nj`: número de muestras en la celda de ancho `dx` que contiene al punto `x`.
- `Nc`: número total de celdas.
- `dx`: ancho de cada celda.

#### Fórmula en múltiples dimensiones

Para el caso multivariado (2 o más variables), si todas las celdas son de igual tamaño:

```
p̂(x) = nj / N
```

Donde `N` es el número total de muestras y `nj` es el número de muestras en la celda correspondiente.

#### Código Python

```python
v = np.random.randn(100, 100)
data = np.r_[np.random.normal(10, 2, 1000), np.random.normal(-10, 2, 1000)]
hist(data, 50, (-20, 20))
show()
```

**Explicación paso a paso:**
1. Se generan 2000 muestras de dos distribuciones normales: una centrada en `+10` y otra en `-10`, ambas con desviación estándar `2`.
2. Se construye un histograma con 50 bins en el rango `[-20, 20]`.
3. El resultado muestra dos picos claramente separados, uno en torno a -10 y otro en torno a +10.

**Resultado visual:** El histograma muestra claramente la naturaleza bimodal de los datos, algo que un modelo paramétrico gaussiano simple no capturaría sin modificaciones.

#### Histograma en 2D

```python
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x, y = np.random.rand(2, 100) * 4
hist, xedges, yedges = np.histogram2d(x, y, bins=4)
elements = (len(xedges) - 1) * (len(yedges) - 1)
xpos, ypos = np.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(elements)
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b')
plt.show()
```

**Explicación paso a paso:**
1. Se generan 100 puntos aleatorios uniformes en 2D en el rango [0, 4].
2. Se computa el histograma 2D con `np.histogram2d` usando 4 bins por dimensión (total: 16 celdas).
3. Se construye una visualización 3D de barras donde la altura de cada barra representa la densidad (número de puntos) en esa celda.

#### Aplicación a Clasificación con Histograma

Para clasificación, el procedimiento es:
1. Construir un histograma separado para cada clase (usando los mismos intervalos y número de bins).
2. Para una nueva muestra, evaluar la probabilidad en cada histograma.
3. Asignar la muestra a la clase con mayor probabilidad.

#### Aplicación a Regresión con Histograma

Para regresión, el proceso es más elaborado:

El modelo de regresión se interpreta como el **valor esperado de la distribución condicional** `p(y|x)`:

```
y = f(x) = E[p(y|x)]
```

Usando la regla de Bayes (`p(y,x) = p(y|x) · p(x)`), se puede estimar:

```
E[p(y|x)] = Σ y · p(y|x) = Σ y · p(y,x) / p(x)
```

**Algoritmo para regresión con histograma:**
1. Construir un histograma conjunto incluyendo todas las variables de entrada `x` (de dimensión `d`) **más** la variable de salida `y`. El histograma tiene `d+1` dimensiones.
2. Para una nueva muestra `x*`, encontrar todas las celdas en el eje `y` donde la probabilidad de `x*` es distinta de cero. Si cada variable se divide en `L` celdas, existirán `L` celdas candidatas.
3. Para cada celda candidata, determinar el valor representativo `yk` (por ejemplo, el valor medio de la celda).
4. Calcular la predicción como:

```
y* = Σ(k=1 a L) [ yk · p(x*, y) / p(x*) ]
```

**Ejemplo práctico – Predicción del HPI (House Price Index):**

El documento ilustra este método con un ejemplo de predicción del índice de precios de vivienda (HPI) en función de dos variables: **edad del inmueble** y **tasa de impuesto (hipoteca)**. Se compara la función de regresión obtenida por histograma versus regresiones polinomiales de grado 1 y grado 2.

**Conclusión visual:** La función del histograma tiene forma arbitraria (no suave), pero se adapta a la distribución real de los datos. Sin embargo, con pocos datos el ajuste tampoco es bueno, ya que muchas celdas quedan vacías.

#### Ventajas y Limitaciones del Histograma

| Característica | Descripción |
|---|---|
| ✅ Simplicidad | Muy simple de implementar. No requiere almacenar las muestras de entrenamiento |
| ❌ Maldición de la dimensionalidad | El número de celdas crece **exponencialmente** con la dimensión: si hay `d` variables y `L` bins por variable, el total de celdas es `L^d` |
| ❌ Discontinuidades | Presenta cambios abruptos en las fronteras entre celdas y zonas con probabilidad cero |
| ❌ Requiere muchas muestras | Con alta dimensionalidad, se necesitan cantidades masivas de datos para poblar todas las celdas |

---

### Sección 4 – Método de k-Vecinos Más Cercanos (k-NN)

#### Concepto

El método de k-NN es un algoritmo de aprendizaje no paramétrico que **almacena todas las muestras de entrenamiento** y realiza predicciones basándose en los `k` puntos más cercanos a una nueva muestra.

**Suposición fundamental:** La función que describe el comportamiento de los datos es suave. Por lo tanto, el comportamiento de un punto desconocido puede estimarse como el promedio del comportamiento de sus `k` vecinos más cercanos.

#### Hiperparámetros del método

| Hiperparámetro | Descripción | Impacto |
|---|---|---|
| `k` (número de vecinos) | Cuántos vecinos se consideran | Un `k` pequeño captura patrones locales pero es sensible al ruido; un `k` grande suaviza pero puede perder precisión local |
| Medida de distancia | Cómo se define la "cercanía" | Afecta directamente qué puntos se consideran vecinos |

#### Medidas de Distancia

Sean `x` y `z` dos vectores de dimensión `d`:

**1. Distancia Euclidiana:**
```
d(x, z) = √[ Σ(j=1 a d) (xj - zj)² ]
```
La distancia más común. Corresponde a la distancia "en línea recta" en el espacio.

**2. Distancia Manhattan (o L1):**
```
d(x, z) = Σ(j=1 a d) |xj - zj|
```
Suma de diferencias absolutas por dimensión. Menos sensible a valores atípicos en dimensiones individuales.

**3. Distancia de Minkowski (generalización):**
```
d(x, z) = [ Σ(j=1 a d) |xj - zj|^q ]^(1/q)
```
Generaliza las anteriores: cuando `q=1` es la Manhattan, cuando `q=2` es la Euclidiana.

#### Algoritmo para Clasificación (k-NN)

1. Seleccionar el número de vecinos `k` y una medida de distancia.
2. Almacenar el conjunto de entrenamiento completo: los pares `(xi, yi)` para `i = 1, ..., N`.
3. Al llegar una nueva muestra `x*`, calcular la distancia entre `x*` y **cada una** de las muestras almacenadas: `vi = d(x*, xi)`.
4. Seleccionar los `k` puntos con los menores valores de `vi` (los k vecinos más cercanos).
5. Extraer las etiquetas `yi` de esos k vecinos.
6. Asignar a `x*` la clase que aparece con mayor frecuencia (la **moda**):

```
y* = moda({ yj^v }(j=1 a k))
```

**Ejemplo:** Si `k=5` y 3 vecinos son de la clase 1 y 2 vecinos de la clase 2, la muestra se clasifica como clase 1.

#### Algoritmo para Regresión (k-NN)

El procedimiento es idéntico al de clasificación hasta el paso 5. La diferencia está en la decisión final: en regresión se toma el **promedio** de los valores `y` de los k vecinos:

```
y* = (1/k) · Σ(j=1 a k) yj^v
```

#### Código Python – Clasificación con k-NN en el dataset Iris

```python
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X, y = iris.data, iris.target

x_min, x_max = X[:, 1].min() - .1, X[:, 1].max() + .1
y_min, y_max = X[:, 2].min() - .1, X[:, 2].max() + .1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

k = 7
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(X[:,1:3], y)

Z = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        Z[i,j] = neigh.predict([[xx[1,i], yy[j,1]]])

plt.figure()
plt.title(u'Clasificación k-vecinos, k = ' + str(k), fontsize=14)
plt.pcolormesh(xx, yy, Z.T, cmap=CMAP_LIGHT)
plt.scatter(X[:,1], X[:,2], c=y)
```

**Explicación paso a paso:**
1. Se carga el dataset **Iris** (150 muestras, 3 clases de flores, 4 características). Se usan las características 1 y 2 (índices) para visualización en 2D.
2. Se crea una malla de 100×100 puntos que cubre el espacio de las dos características seleccionadas.
3. Se instancia el clasificador k-NN de scikit-learn con `k=7` y se entrena con `fit`.
4. Para cada punto de la malla, se predice la clase usando `predict`.
5. Se visualiza la **frontera de decisión** coloreando cada región según la clase predicha, y se superponen los datos originales.

**Resultado:** Se obtiene una visualización de las regiones de decisión no lineales del clasificador k-NN. Con `k=7`, la frontera es relativamente suave; con k pequeño (k=1) sería muy irregular.

#### Problema con Variables de Escalas Diferentes

Al aplicar k-NN al problema HPI con las variables originales (edad: rango 20-60; hipoteca: rango 100,000-300,000), la variable **hipoteca domina completamente** el cálculo de distancias. Esto hace que la variable `edad` sea ignorada en la práctica, produciendo funciones de regresión constantes a lo largo del eje de esa variable.

> "Como el método de k-vecinos está basado en una medida de distancia, es claro que la distancia de la variable Hipoteca será mucho más grande en magnitud, que la distancia de la variable edad. Por esa razón una variable opaca la otra..."

---

### Sección 5 – Normalización de Variables

Para solucionar el problema anterior, es imprescindible **escalar o normalizar** las variables de entrada antes de aplicar métodos basados en distancias.

#### Tipos de Normalización

**1. Normalización Max-Min (Min-Max Scaling):**
```
x_n = (x - min(X)) / (max(X) - min(X))
```
- Todas las variables quedan en el rango `[0, 1]`.
- Preserva la distribución relativa de los valores.

**2. Normalización Z-Score (Estandarización):**
```
x_n = (x - mean(X)) / std(X)
```
- Todas las variables quedan con **media 0** y **varianza 1**.
- Útil cuando los datos se aproximan a una distribución normal.

**Consideración importante:** Estas normalizaciones se aplican **de forma independiente para cada variable**. La letra mayúscula `X` representa el conjunto completo de muestras de entrenamiento. Los parámetros de normalización (min, max, media, std) se calculan **solo con los datos de entrenamiento** y luego se aplican a los datos de prueba.

**Efecto de la normalización:** Al normalizar las variables Hipoteca y Edad antes del k-NN, la función de regresión resultante toma en cuenta ambas variables de manera equilibrada, produciendo resultados mucho más significativos.

#### Efecto del hiperparámetro k en la regresión (con normalización)

- **k pequeño (ej. k=2):** La función de regresión es muy irregular, se ajusta casi exactamente a los datos de entrenamiento (sobreajuste / alta varianza).
- **k grande (ej. k=50):** La función se suaviza progresivamente, aproximándose a un plano casi constante (subajuste / alto sesgo).
- **k óptimo:** Existe un valor intermedio que equilibra sesgo y varianza. Su selección requiere técnicas como validación cruzada (tema de clases posteriores).

---

### Sección 6 – Método de Ventana de Parzen (Kernel Density Estimation)

#### Comparación conceptual con los métodos anteriores

| Método | Definición de Vecindad |
|---|---|
| Histograma | Celda fija de tamaño predefinido (estática) |
| k-NN | Área variable: se expande hasta incluir los k vecinos más cercanos |
| Ventana de Parzen | No define una vecindad: usa **todas las muestras** con pesos según la distancia |

#### Concepto

El método de ventana de Parzen asigna un **peso a cada muestra de entrenamiento** basado en su distancia a la muestra objetivo. Muestras cercanas tienen más influencia; muestras lejanas, menos. Este peso se calcula mediante una **función kernel**.

#### Función de Densidad (Estimación de Parzen)

```
f(x*) = (1 / (N · h^d)) · Σ(i=1 a N) K(ui),    ui = dist(x*, xi) / h
```

Donde:
- `K`: función kernel (peso asignado a cada muestra).
- `xi`: i-ésima muestra de entrenamiento.
- `x*`: nueva muestra a predecir.
- `h`: parámetro de ancho de banda (bandwidth), controla qué tan rápido decae el peso con la distancia.
- `N`: número total de muestras de entrenamiento.
- `d`: dimensión del espacio de entrada.

#### Función Kernel Gaussiana (la más utilizada)

```
K(u) = exp(-½ · u²)
```

- Asigna el mayor peso (`1`) cuando la distancia es `0`.
- El peso decae suavemente a medida que aumenta la distancia.
- El parámetro `h` actúa como la desviación estándar de la gaussiana: controla el ancho de la ventana.

**Si `h` es pequeño:** Solo los vecinos muy cercanos tienen peso significativo → función muy aguda, sensible al ruido.  
**Si `h` es grande:** Todos los puntos contribuyen de forma más uniforme → función más suave pero menos sensible a variaciones locales.

**Nota:** Si se usa una ventana cuadrada (rectangular) como kernel, el resultado es similar al del histograma.

#### Código Python – Estimación de densidad con ventana de Parzen

```python
h = 0.1
mu1, sigma1 = 0, 0.3
mu2, sigma2 = 3, 0.5
np.random.seed(1)
Normal1 = np.random.normal(mu1, sigma1, 200)
Normal2 = np.random.normal(mu2, sigma2, 200)
X = np.append(Normal1, Normal2)

x = sorted(np.random.randn(200) + 2)
pred = list()
for i in range(len(x)):
    pred.append(plothpi.parzenW(X, x[i], h))

ax2.plot(x, pred, '-o')
ax1.hist(Normal1, density=True, color='steelblue')
ax1.hist(Normal2, density=True, color='steelblue')
```

**Explicación paso a paso:**
1. Se generan 200 muestras de una gaussiana centrada en `0` (σ=0.3) y 200 de otra centrada en `3` (σ=0.5).
2. Se concatenan para crear un dataset bimodal.
3. Con `h=0.1`, se aplica la ventana de Parzen para estimar la densidad en cada punto de una malla de evaluación.
4. El panel izquierdo muestra el histograma de los datos originales; el derecho muestra la densidad estimada por Parzen.

**Resultado:** La estimación de Parzen recupera la forma bimodal de la distribución de manera suave, sin las discontinuidades del histograma.

#### Aplicación a Clasificación con Ventana de Parzen

Para clasificación, el procedimiento es análogo al del histograma pero usando la función de Parzen como estimador de densidad:

1. Para cada clase `c`, estimar `p(x* | clase=c)` usando la ecuación de Parzen con las muestras de entrenamiento de esa clase.
2. Asignar la nueva muestra a la clase que maximice la probabilidad estimada.

```python
h = 10
for i in range(100):
    for j in range(100):
        p1 = plothpi.parzenW(X[:50, 1:3], [xx[1,i], yy[j,1]], h)
        p2 = plothpi.parzenW(X[50:100, 1:3], [xx[1,i], yy[j,1]], h)
        p3 = plothpi.parzenW(X[100:150, 1:3], [xx[1,i], yy[j,1]], h)
        p = np.array([p1, p2, p3])
        Z[i,j] = p.argmax(axis=0)
```

**Explicación:** Se evalúa la densidad de Parzen por separado para cada una de las 3 clases del dataset Iris (50 muestras cada una). La clase asignada es la que produce la densidad estimada más alta.

#### Aplicación a Regresión con Ventana de Parzen – Estimador de Nadaraya-Watson

El estimador más utilizado para regresión con ventanas de Parzen es el **Estimador de Nadaraya-Watson**:

```
y* = [ Σ(i=1 a N) K(ui) · yi ] / [ Σ(i=1 a N) K(ui) ]

donde: ui = dist(x*, xi) / h
```

Este estimador calcula un **promedio ponderado** de todos los valores de salida `yi` del conjunto de entrenamiento, donde cada peso es proporcional a la similitud (kernel) entre la muestra nueva `x*` y cada muestra de entrenamiento `xi`.

**Propiedades:**
- A diferencia del histograma y k-NN, produce funciones de salida **suaves** porque la función kernel es continua.
- El parámetro `h` controla el balance entre ajuste local y suavidad global.

#### Código Python – Estimador de Nadaraya-Watson

```python
X_nadayara = np.sort(4 * np.random.rand(100, 1), axis=0)
y_nadayara = np.sin(X_nadayara).ravel()
y_nadayara += 0.5 * (0.5 - np.random.rand(y_nadayara.size))

h = 0.48859075319621886  # Silverman bandwidth

pred = list()
for i in range(len(X_nadayara)):
    pred.append(plothpi.nadaraya_watson(X_nadayara, X_nadayara[i], y_nadayara, h))

plt.scatter(X_nadayara, y_nadayara, c='k', label='Datos Originales')
plt.plot(X_nadayara, pred, '--', c='r', label='NadarayaWatson')
```

**Explicación paso a paso:**
1. Se generan 100 puntos de entrada en [0, 4] y valores de salida siguiendo una función seno con ruido gaussiano añadido.
2. El parámetro `h` se establece con el estimador de Silverman (ver abajo).
3. Para cada punto de entrenamiento se calcula la predicción usando la función `nadaraya_watson` de la librería local.
4. Se visualizan los datos originales y la curva ajustada.

**Resultado:** La curva roja se ajusta suavemente a los datos, capturando la forma sinusoidal sin sobreajustarse al ruido.

#### Efecto del parámetro h en la regresión HPI (con normalización)

| Valor de h | Comportamiento |
|---|---|
| `h` muy pequeño (0.1) | Función muy irregular: cada punto de entrenamiento forma un pico aislado |
| `h` intermedio (100) | Función suave: captura tendencias generales respetando variaciones locales |
| `h` muy grande (100,000) | Función casi constante: todos los puntos contribuyen por igual (similar a la media global) |

**Importancia de la normalización:** Al igual que con k-NN, el método de Parzen también es sensible a la escala de las variables. Sin normalización, la variable con mayor magnitud domina las distancias y, por ende, los pesos del kernel.

#### Estimador de Silverman (Bandwidth Selection)

El documento menciona el **Silverman bandwidth** como una heurística ampliamente utilizada para estimar un valor razonable del parámetro `h` automáticamente a partir de los datos. La regla de Silverman balancea el trade-off entre sesgo y varianza de la estimación kernel, seleccionando `h` en función de la desviación estándar de los datos y el número de muestras.

---

### Sección 7 – Consideraciones Computacionales y Escalabilidad

El documento cierra con una reflexión sobre las limitaciones computacionales de los modelos no paramétricos en general:

> "Los modelos no paramétricos son muy flexibles porque permiten construir fronteras de decisión o funciones de regresión muy variadas, pero son computacionalmente muy ineficientes para conjuntos de datos muy grandes..."

**Razón:** Para la mayoría de estos métodos (k-NN, Parzen) es necesario:
1. **Almacenar todo el conjunto de entrenamiento** en memoria.
2. **Calcular distancias** entre la nueva muestra y todas las muestras almacenadas en cada predicción.

Esto implica una complejidad computacional de predicción `O(N)` o superior, lo cual se vuelve prohibitivo para conjuntos de datos masivos.

**Por esta razón**, se han propuesto muchas variantes y estructuras de datos (como k-d trees, ball trees, aproximaciones de vecinos más cercanos) para hacer estos métodos más eficientes a gran escala.

---

## Conceptos Clave

| Concepto | Definición |
|---|---|
| **Estimación de densidad** | Problema de aproximar la función de densidad de probabilidad que genera los datos |
| **Modelo paramétrico** | Asume forma funcional fija para la densidad; número constante de parámetros |
| **Modelo no paramétrico** | No asume forma funcional; la complejidad crece con los datos |
| **Histograma** | Estimación de densidad por conteo en celdas discretas del espacio |
| **Maldición de la dimensionalidad** | El número de celdas del histograma crece exponencialmente con la dimensión |
| **k-NN** | Predicción basada en los k puntos más cercanos según una medida de distancia |
| **Distancia Euclidiana** | Raíz cuadrada de la suma de diferencias cuadradas por dimensión |
| **Distancia Manhattan** | Suma de valores absolutos de diferencias por dimensión |
| **Distancia de Minkowski** | Generalización paramétrica de Euclidiana y Manhattan |
| **Moda** | Valor más frecuente; se usa para clasificación en k-NN |
| **Normalización Max-Min** | Escala variables al rango [0, 1] |
| **Normalización Z-Score** | Transforma variables para tener media 0 y varianza 1 |
| **Ventana de Parzen** | Estimación de densidad usando pesos suavizados por función kernel |
| **Función kernel** | Función que asigna pesos según la distancia; define la forma de la ventana |
| **Kernel Gaussiano** | Kernel basado en la función exponencial; produce estimaciones suaves |
| **Parámetro h (bandwidth)** | Controla el ancho de la ventana en Parzen; afecta el suavizado |
| **Estimador Nadaraya-Watson** | Promedio ponderado por kernel para regresión no paramétrica |
| **Silverman bandwidth** | Heurística para selección automática del ancho de banda óptimo |
| **HPI (House Price Index)** | Ejemplo práctico del documento: predicción de índice de precios de vivienda |

---

## Flujo General de la Unidad

```
Contexto: Aprendizaje = Estimación de Densidad
            │
            ├─── Modelos Paramétricos (revisión)
            │         └─── Asumen forma funcional → eficientes pero rígidos
            │
            └─── Modelos No Paramétricos (foco de la unidad)
                      │
                      ├─── 1. Histograma
                      │         ├─── 1D y 2D+
                      │         ├─── Clasificación (una fdp por clase)
                      │         └─── Regresión (histograma conjunto x,y)
                      │
                      ├─── 2. k-Vecinos Más Cercanos
                      │         ├─── Clasificación (moda de vecinos)
                      │         ├─── Regresión (promedio de vecinos)
                      │         └─── ⚠ Necesidad de normalización
                      │                   ├─── Max-Min
                      │                   └─── Z-Score
                      │
                      └─── 3. Ventana de Parzen
                                ├─── Estimación de densidad (kernel gaussiano)
                                ├─── Clasificación (una fdp por clase)
                                └─── Regresión (Nadaraya-Watson)
```

Los tres métodos abordan el mismo problema desde perspectivas distintas de "vecindad":
- **Histograma**: vecindad estática y discreta.
- **k-NN**: vecindad adaptativa según los k puntos más cercanos.
- **Parzen**: sin vecindad; todos los puntos contribuyen con peso continuo.

---

## Conceptos Aprendidos

1. El aprendizaje automático puede unificarse bajo el paradigma de **estimación de densidad de probabilidad**.
2. Los modelos paramétricos son eficientes pero pueden fallar si sus suposiciones no se cumplen en los datos reales.
3. Los modelos no paramétricos son más flexibles pero generalmente más costosos computacionalmente.
4. El histograma es el modelo no paramétrico más simple: divide el espacio en celdas y cuenta muestras.
5. El histograma sufre la **maldición de la dimensionalidad**: el número de celdas crece exponencialmente.
6. El método k-NN no asume ninguna forma funcional y es aplicable tanto a clasificación como a regresión.
7. En clasificación con k-NN, la predicción es la **moda** de las etiquetas de los k vecinos.
8. En regresión con k-NN, la predicción es la **media** de los valores de los k vecinos.
9. Los métodos basados en distancias (k-NN, Parzen) son **sensibles a la escala** de las variables.
10. La normalización (Max-Min o Z-Score) es un paso de preprocesamiento esencial para estos métodos.
11. El parámetro `k` en k-NN controla el balance sesgo-varianza: k grande → más sesgo, menos varianza.
12. La ventana de Parzen asigna **pesos continuos** a todas las muestras de entrenamiento, no solo a las k más cercanas.
13. El kernel gaussiano produce estimaciones de densidad suaves y continuas.
14. El parámetro `h` (bandwidth) en Parzen controla el nivel de suavizado de la estimación.
15. El **estimador de Nadaraya-Watson** es el método estándar para regresión con ventana de Parzen.
16. A diferencia del histograma y k-NN, la ventana de Parzen produce funciones de regresión suaves.
17. Todos los métodos no paramétricos presentan el problema de **escalabilidad**: requieren almacenar y consultar todo el conjunto de entrenamiento en cada predicción.
18. El **Silverman bandwidth** es una estrategia heurística efectiva para seleccionar automáticamente el valor de `h`.

---

## Conclusiones

### Conclusiones Técnicas

- Ninguno de los tres métodos estudiados es universalmente superior. La elección depende del tamaño del dataset, la dimensionalidad del espacio y los recursos computacionales disponibles.
- El histograma es la opción más simple pero menos recomendable en alta dimensionalidad por la maldición de la dimensionalidad.
- k-NN es robusto y fácil de entender, pero su rendimiento predictivo depende críticamente de la correcta normalización de variables y la selección del valor de `k`.
- La ventana de Parzen ofrece las estimaciones más suaves y teóricamente mejor fundamentadas, a costa de calcular pesos para **todas** las muestras en cada predicción.
- La normalización de variables no es opcional en métodos basados en distancias: es un requisito para que el modelo funcione correctamente.

### Conclusiones Académicas

- Los modelos no paramétricos ilustran de manera poderosa el concepto de **aprendizaje desde los datos**, sin imponer estructura previa.
- El trade-off sesgo-varianza (controlado por `k` o `h`) es un concepto central en Machine Learning que se manifiesta claramente en estos métodos.
- Comprender estos modelos básicos es fundamental antes de abordar técnicas más avanzadas como Support Vector Machines, Gradient Boosting o Redes Neuronales, que también lidian con los mismos principios de generalización y ajuste.
- La necesidad de preprocesamiento (normalización) que emerge naturalmente en esta unidad sienta las bases para entender el **pipeline** completo de un proyecto de Machine Learning.

---

## Recursos y Referencias

### Librerías Python Utilizadas

| Librería | Uso en el documento |
|---|---|
| `numpy` | Operaciones matriciales, generación de datos aleatorios, histogramas |
| `matplotlib` | Visualizaciones 2D y 3D |
| `scipy` | Funciones matemáticas avanzadas |
| `sklearn.datasets` | Dataset Iris para ejemplos de clasificación |
| `sklearn.neighbors.KNeighborsClassifier` | Implementación de k-NN para clasificación |
| `sklearn.preprocessing` | Normalización de variables |
| `ipywidgets` | Widgets interactivos en Jupyter (para explorar parámetros) |
| `local.library` (plothpi) | Librería local del curso con funciones para graficar HPI, implementaciones de Parzen y Nadaraya-Watson |

### Herramientas y Frameworks

- **Jupyter Book** / **JupyterLab**: Entorno de ejecución del notebook.
- **scikit-learn**: Librería principal de Machine Learning en Python.

### Conceptos para Profundizar

- Silverman, B.W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall. (Referencia para el Silverman bandwidth)
- Nadaraya, E.A. (1964). *On estimating regression*. Theory of Probability and Its Applications.
- Watson, G.S. (1964). *Smooth regression analysis*. Sankhyā: The Indian Journal of Statistics.
- Parzen, E. (1962). *On estimation of a probability density function and mode*. The Annals of Mathematical Statistics.

### Repositorio del Curso

- **URL**: `https://jdariasl.github.io/Intro_ML_2025/`
- **Clase correspondiente**: `Clase 04 - Modelos no Paramétricos`

---

*Documentación generada para la Unidad 2 del curso Introducción al Machine Learning.*
