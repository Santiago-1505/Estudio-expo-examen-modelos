# Unidad 4: Modelos de Mezcla de Gaussianas y Clustering

---

## Introducción

Esta unidad aborda dos temas centrales del **aprendizaje no supervisado** en Machine Learning: los **Modelos de Mezcla de Funciones Gaussianas (GMM)** y los **métodos de Clustering**. Ambos temas están profundamente relacionados, ya que los GMM pueden emplearse tanto para clasificación generativa como para agrupamiento de datos.

El objetivo de la unidad es comprender cómo estructurar y modelar datos cuando no se dispone de etiquetas de salida, explorando técnicas que descubren patrones inherentes en los conjuntos de datos. Se estudian desde métodos jerárquicos clásicos hasta el popular algoritmo K-means y su conexión con el algoritmo de Esperanza-Maximización (EM).

---

## Contenido de la Unidad

La unidad está compuesta por dos notebooks/documentos principales exportados como PDF desde un entorno Jupyter:

| Archivo | Tema principal |
|---|---|
| `Modelos_de_Mezcla_de_Funciones_Gausianas.pdf` | Modelos generativos, GMM, Algoritmo EM |
| `Clustering___Introducción_al_Machine_Learning.pdf` | Clustering jerárquico, K-means, métricas de evaluación |

---

## Estructura de Archivos

---

### Archivo 1: `Modelos_de_Mezcla_de_Funciones_Gausianas.pdf`

#### Propósito

Introducir el concepto de **Modelos Generativos** y la necesidad de modelar distribuciones de probabilidad complejas (multimodales) a través de mezclas de Gaussianas. Se explica el **Algoritmo EM** como método de estimación de parámetros bajo máxima verosimilitud.

---

#### Contenido y Funcionamiento Detallado

##### 1. Modelos Discriminativos vs. Generativos

El documento inicia estableciendo una distinción fundamental en los enfoques de clasificación en Machine Learning:

- **Modelos Discriminativos**: Buscan la frontera de separación entre clases. El ajuste se realiza minimizando directamente el error de clasificación. Ejemplos: SVM, Regresión Logística.

- **Modelos Generativos**: Estiman la función de densidad de probabilidad (fdp) de cada clase por separado, usando criterios como máxima verosimilitud. Una vez estimada la fdp, es posible *generar* nuevas muestras de esa distribución. Ejemplos: Naive Bayes, GMM.

**Importancia pedagógica**: Esta distinción define dos paradigmas completamente diferentes de entrenamiento y predicción, con implicaciones distintas sobre cómo el modelo interpreta los datos.

---

##### 2. Motivación para los Modelos de Mezcla

Se presenta un ejemplo visual en el que los datos de cada clase **no forman un único conglomerado**, sino que están divididos en dos o más grupos separados en el espacio de características. Al intentar ajustar una única Gaussiana por clase (como haría un clasificador de función discriminante gaussiana estándar), los modelos se ajustan muy mal, ya que una sola elipse no puede representar fielmente una distribución multimodal.

**Código ilustrativo presentado:**

```python
x1 = np.random.rand(2,50)
x2 = np.random.rand(2,50) + 2
x3 = np.random.rand(2,50) + np.tile([[-1],[2]], (1, 50))
x4 = np.random.rand(2,50) + np.tile([[3],[1]], (1, 50))
XC1 = np.concatenate((x1,x3), axis=1)
XC2 = np.concatenate((x2,x4), axis=1)
```

- `x1`, `x3`: Sub-grupos de la clase 1 (azul), ubicados en diferentes regiones.
- `x2`, `x4`: Sub-grupos de la clase 2 (rojo), también en regiones separadas.
- `XC1`, `XC2`: Conjuntos completos de cada clase, cada uno con dos conglomerados internos.

El resultado visual demuestra que una Gaussiana única (representada por una elipse grande) **no encapsula** correctamente la distribución real de los datos.

---

##### 3. Definición del Modelo GMM

La fdp de una clase bajo un modelo GMM se define como:

$$p(\mathbf{x} | \Theta_i) = \sum_{j=1}^{M} \omega_{ij} \, \mathcal{N}(\mathbf{x} \mid \mu_{ij}, \Sigma_{ij})$$

**Parámetros del modelo:**

| Símbolo | Significado |
|---|---|
| `M` | Número de conglomerados (componentes Gaussianas) — hiperparámetro |
| `ω_ij` | Peso de cada componente; mide su representatividad relativa |
| `μ_ij` | Vector de medias del conglomerado `j` en la clase `i` |
| `Σ_ij` | Matriz de covarianza del conglomerado `j` en la clase `i` |

**Restricción:** La suma de los pesos debe ser 1: `Σ ω_ij = 1`.

**Hiperparámetro M**: El número de componentes `M` no se puede determinar visualmente cuando la dimensionalidad es alta. La práctica recomendada es usar **validación cruzada**, igual que se hace para el hiperparámetro `K` en K-NN.

---

##### 4. Implementación manual de la probabilidad GMM

```python
def GaussProb(X, medias, covars, pesos):
    M = len(covars)          # Número de componentes
    N, d = X.shape           # N muestras, d dimensiones
    Pprob = np.zeros(N).reshape(N,1)
    precision = []
    for i in range(M):
        precision.append(np.linalg.inv(covars[i]))  # Inversas de covarianzas
    for j in range(N):
        prob = 0
        for i in range(M):
            tem = (X[j,:] - medias[i])
            tem1 = np.dot(np.dot(tem, precision[i]), tem.T)  # Forma cuadrática
            tem2 = 1/((math.pi**(d/2)) * (np.linalg.det(covars[i]))**(0.5))
            prob += pesos[i] * tem2 * np.exp(-0.5 * tem1)
        Pprob[j] = prob
    return Pprob
```

**Lógica paso a paso:**
1. Se precalculan las matrices de precisión (inversas de covarianza) para eficiencia.
2. Para cada muestra `j`, se evalúa la contribución de cada componente `i`:
   - Se calcula la forma cuadrática `(x - μ)ᵀ Σ⁻¹ (x - μ)`.
   - Se calcula el factor normalizador de la Gaussiana multivariada.
   - Se pondera por el peso `ω_i` de la componente.
3. La probabilidad total es la suma ponderada de todas las componentes.

---

##### 5. Estimación de parámetros: Algoritmo EM

El problema de aprendizaje es estimar `Θ = {μ_ij, Σ_ij, ω_ij}` para cada clase, dado el conjunto de entrenamiento.

**Criterio de máxima verosimilitud:**

$$\arg\max_\Theta \sum_{k=1}^{N_i} \log \sum_{j=1}^{M} \omega_{ij} \, \mathcal{N}(\mathbf{x}_k \mid \mu_{ij}, \Sigma_{ij})$$

**El desafío**: No existe solución de forma cerrada porque la derivación de la log-verosimilitud produce ecuaciones que dependen circularmente de los propios parámetros a estimar.

**Solución — El concepto de Responsabilidad:**

$$\gamma_{kl} = \frac{\omega_{il} \, \mathcal{N}(\mathbf{x}_k \mid \mu_{il}, \Sigma_{il})}{\sum_j \omega_{ij} \, \mathcal{N}(\mathbf{x}_k \mid \mu_{ij}, \Sigma_{ij})}$$

`γ_kl` representa la **probabilidad de que la muestra `k` haya sido generada por el conglomerado `l`**. Es una matriz de dimensión `(N_i × M)`.

**Ecuaciones de actualización (Paso M):**

- **Media:**
$$\hat{\mu}_{il} = \frac{1}{n_l} \sum_{k=1}^{N_i} \gamma_{kl} \, \mathbf{x}_k$$

- **Covarianza:**
$$\hat{\Sigma}_{il} = \frac{1}{n_l} \sum_{k=1}^{N_i} \gamma_{kl} (\mathbf{x}_k - \mu_{il})(\mathbf{x}_k - \mu_{il})^T$$

- **Pesos:**
$$\hat{\omega}_{il} = \frac{n_l}{N_i}, \quad \text{donde } n_l = \sum_{k=1}^{N_i} \gamma_{kl}$$

`n_l` puede interpretarse como el **número efectivo de puntos asignados al conglomerado `l`**.

**Pasos del Algoritmo EM:**
1. Inicializar aleatoriamente los parámetros `{μ, Σ, ω}`.
2. **Paso E** (Expectation): Calcular `γ_kl` para todas las muestras y componentes.
3. **Paso M** (Maximization): Actualizar `μ`, `Σ` y `ω` usando las ecuaciones anteriores.
4. Repetir E y M hasta convergencia.

**Garantía del algoritmo:** En cada iteración la verosimilitud crece o se mantiene, pero **nunca decrece**. Sin embargo, **no garantiza convergencia al máximo global** (solo local).

---

##### 6. Ejemplo 1D con sklearn

```python
x1 = np.random.normal(0, 2, 1000)
x2 = np.random.normal(20, 4, 1000)
x3 = np.random.normal(-20, 6, 1000)
X = np.concatenate((x1,x2,x3), axis=0)
```

Se generan 3000 muestras de una mezcla de tres Gaussianas unidimensionales con medias `0`, `20`, `-20` y desviaciones `2`, `4`, `6` respectivamente.

**Ajuste con diferentes iteraciones:**

```python
gmm = GaussianMixture(n_components=3, covariance_type='full', max_iter=i)
gmm.fit(Y)
logprob = np.exp(gmm.score_samples(x1plt))
```

Se muestra visualmente cómo la fdp estimada evoluciona desde una forma inicial difusa (iter=0) hasta converger a la forma real de la distribución (iter=500). Los resultados finales del algoritmo son muy cercanos a los parámetros reales:

| Parámetro | Real | Estimado (aprox.) |
|---|---|---|
| μ₁ | 0 | 0.073 |
| μ₂ | 20 | 20.025 |
| μ₃ | -20 | -20.071 |

---

##### 7. Ejemplo 2D con diferentes tipos de covarianza

Se generan muestras bidimensionales a partir de un GMM con parámetros conocidos:

```python
mc = [0.4, 0.4, 0.2]                         # Pesos de mezcla
centroids = [array([0,0]), array([3,3]), array([0,4])]
ccov = [array([[1,0.4],[0.4,1]]), diag((1,2)), diag((0.4,0.1))]
```

Se prueban tres tipos de matriz de covarianza en sklearn:

| Tipo | Descripción | Parámetro sklearn |
|---|---|---|
| `full` | Covarianza completa (correlaciones entre dimensiones) | `covariance_type='full'` |
| `diag` | Solo varianzas por dimensión (sin correlaciones cruzadas) | `covariance_type='diag'` |
| `spherical` | Una única varianza por componente (esfera en todas las dimensiones) | `covariance_type='spherical'` |

Los resultados muestran que `full` recupera mejor los parámetros originales, mientras que `diag` y `spherical` son aproximaciones que funcionan bien cuando las correlaciones son bajas, con la ventaja de necesitar menos parámetros.

---

### Archivo 2: `Clustering___Introducción_al_Machine_Learning.pdf`

#### Propósito

Presentar el problema general de **Clustering (agrupamiento)** como tarea de aprendizaje no supervisado, explorar los métodos jerárquicos/aglomerativos y el algoritmo **K-means**, así como las métricas para evaluar y seleccionar el número óptimo de clusters.

---

#### Contenido y Funcionamiento Detallado

##### 1. Definición del Problema de Clustering

El clustering consiste en organizar un conjunto de objetos en subgrupos donde los elementos dentro de cada grupo son **más similares entre sí** que con los objetos de otros grupos. Características clave:

- Pertenece al **aprendizaje no supervisado**: las muestras solo tienen un vector de características `x_i`, sin etiqueta de salida objetivo.
- Es una **colección de métodos para exploración de datos**.
- Un mismo conjunto puede tener múltiples agrupaciones válidas; el método debe elegirse cuidadosamente.
- Frecuentemente se usa como **etapa previa** a técnicas supervisadas.

**Relación con GMM:** El algoritmo EM es en sí mismo una técnica de clustering no supervisada, pero existen otros métodos alternativos.

---

##### 2. Métodos Jerárquicos/Aglomerativos

###### Principio de funcionamiento

Los métodos aglomerativos son los más comunes para resumir la estructura de un conjunto de muestras. Su lógica es:

1. Cada muestra comienza como un nodo independiente (su propio cluster).
2. En cada paso se unen los **dos nodos más cercanos** formando un nuevo nodo.
3. El proceso continúa hasta que todo el conjunto forma un único nodo.

El resultado se visualiza mediante un **dendrograma**, una representación arbórea que muestra la jerarquía de fusiones y la distancia a la que se realizaron.

###### Métricas de distancia utilizadas

```python
# Distancia euclidiana
distxy = pdist(X, metric='euclidean')

# Distancia de Minkowski (p=1 equivale a distancia Manhattan)
distxy = pdist(X, metric='minkowski', p=1)
```

###### Estrategias de enlace (linkage)

Los distintos esquemas difieren en cómo se actualiza la medida de similitud entre clusters después de cada fusión:

| Método | Actualización `d(I∪J, K)` | Medida entre clusters A y B |
|---|---|---|
| `single` | `min(d(I,K), d(J,K))` | Distancia mínima entre cualquier par de puntos |
| `complete` | `max(d(I,K), d(J,K))` | Distancia máxima entre cualquier par de puntos |
| `average` | Promedio ponderado por tamaño | Promedio de todas las distancias inter-cluster |
| `weighted` | `(d(I,K) + d(J,K)) / 2` | Promedio sin ponderar por tamaño |
| `ward` | Minimiza la varianza total dentro del cluster | Basada en la suma de cuadrados |
| `centroid` | Distancia entre centroides ajustada | Basada en centroides |
| `median` | Punto medio de los centroides | Basada en medianas |

**Código de ejemplo:**

```python
from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(distxy, method='complete')  # o 'ward', 'weighted', etc.
dn = dendrogram(Z)
```

El parámetro `color_threshold` permite controlar visualmente el número de clusters identificados:

```python
dn = dendrogram(Z, color_threshold=0.5*max(Z[:,2]))
```

Reducir el umbral al 50% del máximo en lugar del 70% por defecto produce más clusters identificados en el dendrograma.

###### Ventajas de los métodos aglomerativos

- No requieren definir previamente el número de clusters.
- Son fáciles de implementar.
- Los dendrogramas son una herramienta intuitiva para interpretar resultados.

###### Desventajas de los métodos aglomerativos

- **Alto costo computacional**: requieren reestimación constante de distancias entre nodos; no escalan bien con conjuntos de datos grandes.
- **No permiten backtracking**: una fusión realizada no puede deshacerse.
- **Efecto cadena**: especialmente en el método `single`, puntos intermedios pueden "encadenar" clusters que deberían estar separados.

---

##### 3. Métodos de Clustering: Duros vs. Suaves

| Tipo | Descripción | Ejemplo |
|---|---|---|
| **Duro** | Cada muestra pertenece a exactamente un cluster | K-means |
| **Suave** | Cada muestra pertenece a múltiples clusters con distintos grados de pertenencia | GMM, Fuzzy C-means |

---

##### 4. K-means (Método Duro)

###### Concepto

K-means es el algoritmo de clustering más popular. Busca separar las muestras en `K` grupos de igual varianza, minimizando la **inercia** o dispersión intra-cluster:

$$\min_{\mu_k} \sum_{k=1}^K \sum_{\mathbf{x}_k \in C_k} \|\mathbf{x}_k - \mu_k\|^2$$

Donde `μ_k` es el centroide del cluster `k` y `C_k` es el conjunto de muestras del cluster `k`.

**Relación con EM:** K-means es una variante simplificada del algoritmo EM que asume:
- Todos los clusters son igualmente probables: `ω_i = 1/M`.
- Igual desviación en todos los clusters.
- Solo se optimizan las medias (centroides).

Todo punto en un cluster está más cercano a su centroide que al de cualquier otro cluster.

###### Algoritmo de Lloyd (implementación iterativa)

1. Inicializar aleatoriamente `K` centroides.
2. **Asignación**: Asignar cada punto al centroide más cercano.
3. **Actualización**: Recalcular cada centroide como la media de los puntos asignados.
4. Repetir 2 y 3 hasta convergencia (los centroides no cambian).

El resultado puede visualizarse mediante un **diagrama de Voronoi**, donde cada región corresponde al espacio más cercano a un centroide.

###### Implementación con sklearn

```python
from sklearn.cluster import KMeans

for M in range(3, 5):
    kmeans = KMeans(init='random', n_clusters=M, n_init=1)
    kmeans.fit(X2)
    Y2 = kmeans.predict(X2)
    plt.scatter(X2[:,0], X2[:,1], c=Y2)
```

**Parámetros importantes:**

| Parámetro | Descripción | Recomendación |
|---|---|---|
| `n_clusters` | Número de clusters `K` | Debe explorarse con métodos como Elbow o Silhouette |
| `init` | Método de inicialización | Usar `'k-means++'` en lugar de `'random'` |
| `n_init` | Número de inicializaciones | Usar `10` para obtener la solución de menor inercia |

###### Desventajas de K-means

- **Alta dependencia de la inicialización**: diferentes inicializaciones pueden converger a mínimos locales distintos.
- **Solución recomendada 1**: Usar `init='k-means++'`, que inicializa los centroides en puntos distantes entre sí, produciendo mejores resultados.
- **Solución recomendada 2**: Usar `n_init=10` para que el algoritmo se ejecute múltiples veces y se seleccione la solución de menor inercia.

---

##### 5. Selección del Número de Clusters

Escoger el número óptimo de clusters es una tarea no trivial cuando no se tiene información a priori. El documento presenta dos criterios:

###### 5.1 Curva del Codo (Elbow Method)

Mide el **porcentaje de varianza explicada** en función del número de clusters `K`. Se calcula como el cociente entre la varianza entre grupos y la varianza total:

```python
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist, pdist

KM = [kmeans(X, k) for k in range(1, K_MAX+1)]
centroids = [cent for (cent, var) in KM]
D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
dist = [np.min(D, axis=1) for D in D_k]

tot_withinss = [sum(d**2) for d in dist]      # Suma de cuadrados intra-cluster
totss = sum(pdist(X)**2) / X.shape[0]          # Varianza total
betweenss = totss - tot_withinss               # Varianza entre clusters

ax.plot(KK, betweenss/totss*100, 'b*-')
```

**Interpretación**: Se busca el punto de "codo" donde agregar más clusters deja de aportar ganancias significativas en varianza explicada. En el ejemplo del dataset Iris, el codo se identifica en `K=2`.

###### 5.2 Coeficiente Silhouette

Es una de las métricas más robustas para evaluar la calidad del clustering. Para cada muestra `i`:

$$S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad \text{si } |C_i| > 1$$

Donde:
- `a(i)`: Distancia media de la muestra `i` a todas las demás muestras **en su mismo cluster** (cohesión).
- `b(i)`: Mínima distancia media de la muestra `i` a muestras de **cualquier otro cluster** (separación).

**Rango:** Entre -1 y 1. Valores cercanos a 1 indican clusters bien definidos y separados.

```python
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X = sc.fit_transform(X)   # ¡Importante escalar antes de calcular Silhouette!

for n_cluster in n_cluster_list:
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found = kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))
```

**Nota crítica del código**: El escalado con `MinMaxScaler` es fundamental antes de calcular el coeficiente silhouette, ya que las distancias son sensibles a la escala de las variables.

**Resultado en el ejemplo**: El coeficiente silhouette es máximo en `K=2`, lo cual sugiere que el dataset Iris (con las características 2 y 3 utilizadas) se agrupa mejor en 2 grupos.

---

## Flujo General de la Unidad

```
Problema de datos sin etiquetas
         │
         ▼
┌─────────────────────────────────┐
│  ¿Qué distribución tienen       │
│  los datos dentro de cada       │
│  clase?                         │
└──────────┬──────────────────────┘
           │
     ┌─────▼──────────────┐
     │  Modelos Generativos│
     │  (GMM)              │
     └─────┬───────────────┘
           │  Una Gaussiana no es suficiente
           │  si los datos son multimodales
           ▼
     Modelo de Mezcla de Gaussianas
           │
           ▼
     Algoritmo EM para estimar:
     μ, Σ, ω por cada componente
           │
           ▼
     Casos especiales → K-means
     (igual peso, igual varianza)
           │
           ▼
┌──────────────────────────────────┐
│  ¿Cómo agrupar datos sin usar    │
│  distribuciones explícitas?      │
└──────────┬───────────────────────┘
           │
     Métodos de Clustering
           │
     ┌─────┴────────────────┐
     ▼                      ▼
Jerárquicos            K-means
(Dendrograma)          (Lloyd)
     │                      │
     └───────────┬──────────┘
                 │
     ¿Cuántos clusters usar?
                 │
     ┌───────────┴──────────┐
     ▼                      ▼
  Curva del Codo       Coeficiente
  (Elbow Method)        Silhouette
```

---

## Conceptos Clave

| Concepto | Definición |
|---|---|
| **Aprendizaje No Supervisado** | Paradigma donde los datos no tienen etiqueta de salida y se busca descubrir estructura inherente. |
| **Modelo Generativo** | Modelo que estima la función de densidad de probabilidad de los datos para cada clase. |
| **Modelo Discriminativo** | Modelo que aprende directamente la frontera de decisión entre clases. |
| **GMM (Gaussian Mixture Model)** | Suma ponderada de Gaussianas que modela distribuciones multimodales. |
| **Algoritmo EM** | Método iterativo (Expectation-Maximization) para estimar parámetros de modelos con variables latentes. |
| **Responsabilidad (γ_kl)** | Probabilidad de que una muestra haya sido generada por un conglomerado específico. |
| **Clustering Jerárquico** | Método que construye una jerarquía de clusters fusionando iterativamente los más cercanos. |
| **Dendrograma** | Representación visual en árbol del proceso de fusión en clustering jerárquico. |
| **K-means** | Algoritmo de clustering que minimiza la dispersión intra-cluster asignando puntos al centroide más cercano. |
| **Inercia** | Suma de distancias cuadráticas de cada punto a su centroide; métrica de calidad en K-means. |
| **Curva del Codo** | Método para seleccionar K basado en el porcentaje de varianza explicada. |
| **Coeficiente Silhouette** | Métrica entre -1 y 1 que mide cohesión intra-cluster y separación inter-cluster. |
| **Diagrama de Voronoi** | Partición del espacio en regiones según el centroide más cercano; ilustra K-means. |
| **k-means++** | Estrategia de inicialización de centroides en puntos distantes para evitar mínimos locales malos. |
| **Efecto cadena** | Problema del método `single linkage` donde puntos intermedios encadenan clusters que deberían estar separados. |

---

## Conceptos Aprendidos

1. **Diferencia entre modelos generativos y discriminativos** y cuándo conviene usar cada enfoque.
2. **Limitaciones de una Gaussiana única** para modelar distribuciones multimodales.
3. **Formulación matemática del GMM**: suma ponderada de Gaussianas con restricción sobre los pesos.
4. **Derivación e interpretación del concepto de responsabilidad** en el contexto del algoritmo EM.
5. **Pasos del algoritmo EM**: inicialización → Paso E → Paso M → convergencia.
6. **Garantías y limitaciones del algoritmo EM**: convergencia garantizada pero solo a máximos locales.
7. **Tipos de matrices de covarianza** en GMM (`full`, `diag`, `spherical`) y sus implicaciones.
8. **Implementación práctica del GMM con sklearn** (`GaussianMixture`).
9. **Fundamentos del clustering jerárquico aglomerativo** y construcción de dendrogramas.
10. **Estrategias de linkage** (`single`, `complete`, `ward`, etc.) y sus diferencias.
11. **Métricas de distancia** entre puntos (`euclidean`, `minkowski`).
12. **Formulación del problema K-means** como minimización de inercia.
13. **Relación entre K-means y el algoritmo EM** como caso particular simplificado.
14. **Implementación del algoritmo de Lloyd** para K-means.
15. **Importancia de la inicialización** en K-means y uso de `k-means++`.
16. **Método del codo** para selección de K basado en varianza explicada.
17. **Coeficiente Silhouette**: definición, fórmula, interpretación y uso práctico.
18. **Importancia del escalado** de datos antes de aplicar métricas basadas en distancias.

---

## Conclusiones

### Conclusiones Técnicas

- Los **Modelos de Mezcla de Gaussianas** son una generalización poderosa de los modelos gaussianos simples, capaces de representar distribuciones de cualquier forma mediante la superposición de Gaussianas.
- El **algoritmo EM** resuelve elegantemente el problema de estimación en GMM, pero su convergencia a óptimos locales exige una buena inicialización y, en la práctica, múltiples ejecuciones.
- **K-means** es un caso límite del algoritmo EM que, gracias a sus supuestos simplificadores, es computacionalmente eficiente y ampliamente aplicable, aunque sufre del mismo problema de óptimos locales y requiere especificar `K` a priori.
- Los **métodos jerárquicos** son más flexibles al no requerir `K` a priori, pero tienen un coste computacional que los hace inadecuados para grandes volúmenes de datos.
- Para **determinar el número óptimo de clusters**, ningún criterio es universalmente perfecto; la combinación de la curva del codo y el coeficiente Silhouette ofrece mayor robustez.

### Conclusiones Académicas

- La unidad establece un puente conceptual claro entre los **modelos probabilísticos generativos** y los **algoritmos de clustering**, mostrando que K-means no es más que un caso especial del algoritmo EM.
- El tratamiento matemático riguroso (derivación de las ecuaciones de actualización del EM mediante máxima verosimilitud con multiplicadores de Lagrange) proporciona una comprensión profunda de por qué los algoritmos funcionan, más allá de su uso práctico.
- La evaluación visual de resultados (dendrogramas, diagramas de dispersión con clusters coloreados, curvas de varianza y Silhouette) es esencial en el aprendizaje no supervisado, donde no hay métricas de error estándar como en clasificación.

---

## Recursos y Referencias

### Librerías utilizadas

| Librería | Uso en la unidad |
|---|---|
| `numpy` | Operaciones matriciales, generación de datos aleatorios |
| `matplotlib` | Visualización de distribuciones, clusters y dendrogramas |
| `scipy.cluster.hierarchy` | Clustering jerárquico (`linkage`, `dendrogram`) |
| `scipy.spatial.distance` | Cálculo de matrices de distancia (`pdist`, `cdist`) |
| `scipy.cluster.vq` | K-means de scipy para curva del codo |
| `sklearn.cluster.KMeans` | Implementación de K-means |
| `sklearn.mixture.GaussianMixture` | Implementación del algoritmo EM para GMM |
| `sklearn.metrics.silhouette_score` | Cálculo del coeficiente Silhouette |
| `sklearn.preprocessing.MinMaxScaler` | Normalización de datos |
| `sklearn.datasets` | Dataset Iris para ejemplos |

### Referencias académicas mencionadas

- [Cluster Analysis on Wikipedia](https://en.wikipedia.org/wiki/Cluster_analysis)
- Modern hierarchical, agglomerative clustering algorithms
- Basic Principles of Clustering Methods
- Documentación del curso: `https://jdariasl.github.io/Intro_ML_2025/`

### Conceptos relacionados para profundizar

- Criterio de Información de Akaike (AIC) y Bayesiano (BIC) para selección de `M` en GMM.
- DBSCAN como alternativa de clustering basada en densidad.
- Clustering espectral para datos no convexos.
- Variational Bayes para GMM con selección automática de componentes.
