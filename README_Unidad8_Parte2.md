# Unidad 8 Parte 2: Reducción de Dimensión — Métodos Avanzados y Selección de Características

---

## Introducción

Esta segunda parte de la Unidad 8 profundiza en tres grandes bloques temáticos del preprocesamiento y reducción dimensional en Machine Learning:

1. **Análisis Discriminante de Fisher (FDA/LDA):** Un método lineal supervisado de reducción de dimensión que, a diferencia de PCA, aprovecha la información de las etiquetas de clase para encontrar proyecciones que maximizan la separabilidad entre clases.

2. **Métodos No Lineales de Reducción de Dimensión:** Técnicas que van más allá de las proyecciones lineales, capturando estructuras complejas en los datos mediante Kernel PCA, t-SNE, UMAP y LLE (Local Linear Embedding).

3. **Selección de Características:** Estrategias para identificar el subconjunto óptimo de variables originales que mejor representa el problema, sin transformarlas, preservando su interpretabilidad física.

Los tres archivos de esta unidad corresponden a notebooks de clase exportados como PDF, pertenecientes al curso *Introducción al Machine Learning* (2025), y cubren desde la formulación matemática hasta implementaciones prácticas con `scikit-learn`.

---

## Contenido de la Unidad

| Archivo | Tema Principal |
|---|---|
| `Reducción_de_dimensión__Análisis_Discriminante_de_Fisher___Introducción_al_Machine_Learning.pdf` | FDA / LDA — proyección supervisada que maximiza separabilidad entre clases |
| `Reducción_de_dimensión__Métodos_no_lineales___Introducción_al_Machine_Learning.pdf` | Kernel PCA, t-SNE, UMAP, LLE — reducción no lineal y aprendizaje de variedades |
| `Selección_de_Características___Introducción_al_Machine_Learning.pdf` | Criterios y estrategias de búsqueda para selección de subconjuntos de variables |

---

## Estructura de Archivos

---

### Archivo 1: `Análisis_Discriminante_de_Fisher.pdf`

#### Propósito

Presentar el **Análisis Discriminante de Fisher (FDA)**, también conocido como **Linear Discriminant Analysis (LDA)** en su forma multivariable, como un método de reducción de dimensión *supervisado*. El objetivo es encontrar direcciones de proyección que maximicen la separación entre clases en el espacio reducido, superando la limitación de PCA que sólo considera varianza global.

---

#### Contenido y Funcionamiento

##### Motivación: Limitación de PCA para Clasificación

El notebook comienza mostrando un ejemplo con dos clases donde PCA falla como herramienta de separación:

```python
np.random.seed(1)
X = np.dot(np.random.random(size=(2, 2)), np.random.normal(size=(2, 200))*4).T
X2 = np.dot(np.random.random(size=(2, 200))).T + [offset]
X3 = np.r_[X, X2]
Y = np.r_[np.ones(200), np.zeros(200)]
X3 = X3 - np.mean(X3, axis=0)
```

- Se generan **dos clases** de 200 muestras cada una con distribuciones Gaussianas multivariables.
- Los datos se centran restando la media global.
- Al aplicar PCA con 1 componente, la proyección resultante **mezcla ambas clases**, evidenciando que la dirección de mayor varianza no es necesariamente la de mayor discriminación.

**Conclusión clave:** PCA busca direcciones de proyección que son eficientes para *representar* los datos. El análisis discriminante busca direcciones que son eficientes para *discriminar*, es decir, que permiten una mejor separación de las clases en el espacio de menor dimensión.

---

##### Formulación Matemática del FDA Binario

Se considera un problema de clasificación de **2 clases**, con muestras $\mathbf{x}_i$ de dimensión $d$.

**Proyección sobre un vector unitario $\mathbf{v}$:**

El escalar $\mathbf{v}^T \mathbf{x}_i$ es la distancia proyectada de $\mathbf{x}_i$ desde el origen sobre la dirección $\mathbf{v}$.

**Medias de las proyecciones:**

$$\widetilde{\mu}_1 = \frac{1}{n_1} \sum_{\mathbf{x}_i \in C_1} \mathbf{v}^T \mathbf{x}_i = \mathbf{v}^T \mu_1$$

$$\widetilde{\mu}_2 = \mathbf{v}^T \mu_2$$

donde $\mu_1$, $\mu_2$ son las medias de cada clase en el espacio original.

**¿Por qué no basta con maximizar $|\widetilde{\mu}_1 - \widetilde{\mu}_2|$?**

El problema con usar solo la distancia entre medias proyectadas es que **ignora la varianza intra-clase**. Dos distribuciones pueden tener medias muy separadas pero con tanta dispersión que se solapan. El notebook ilustra esto con un ejemplo donde el eje horizontal tiene mayor separación entre medias pero las distribuciones se solapan más que en el eje vertical.

**Solución — Criterio de Fisher:**

Se define la **dispersión** de un conjunto de muestras proyectadas de la clase $k$ como:

$$\widetilde{S}_k = \sum_{y_i \in C_k} (y_i - \widetilde{\mu}_k)^2$$

El **criterio de Fisher** normaliza la separación entre medias por la dispersión total:

$$J(\mathbf{v}) = \frac{(\widetilde{\mu}_1 - \widetilde{\mu}_2)^2}{\widetilde{S}_1^2 + \widetilde{S}_2^2}$$

Maximizar $J(\mathbf{v})$ garantiza clases bien separadas *y* compactas en el espacio proyectado.

---

##### Matrices de Dispersión en el Espacio Original

Para expresar $J(\mathbf{v})$ en términos de $\mathbf{v}$ y poder maximizarlo:

**Matrices de dispersión intra-clase por clase:**

$$S_1 = \sum_{\mathbf{x}_i \in C_1} (\mathbf{x}_i - \mu_1)(\mathbf{x}_i - \mu_1)^T$$

$$S_2 = \sum_{\mathbf{x}_i \in C_2} (\mathbf{x}_i - \mu_2)(\mathbf{x}_i - \mu_2)^T$$

**Matriz de dispersión intra-clase total (Within-class scatter):**

$$S_W = S_1 + S_2$$

Se puede demostrar que: $\widetilde{S}_1 + \widetilde{S}_2 = \mathbf{v}^T S_W \mathbf{v}$

**Matriz de dispersión entre clases (Between-class scatter):**

$$S_B = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T$$

Esta matriz mide la separación entre las medias de las dos clases. Se demuestra que:

$$(\widetilde{\mu}_1 - \widetilde{\mu}_2)^2 = \mathbf{v}^T S_B \mathbf{v}$$

**Criterio de Fisher reformulado:**

$$J(\mathbf{v}) = \frac{\mathbf{v}^T S_B \mathbf{v}}{\mathbf{v}^T S_W \mathbf{v}}$$

---

##### Solución: Problema de Valores Propios Generalizados

Al derivar $J(\mathbf{v})$ con respecto a $\mathbf{v}$ e igualar a cero, se obtiene:

$$S_W^{-1} S_B \mathbf{v} = \lambda \mathbf{v}$$

Este es un **problema de valores propios** de la matriz $S_W^{-1} S_B$.

**Solución cerrada para el caso binario:** El vector que maximiza el criterio de Fisher es:

$$\mathbf{v} = S_W^{-1} (\mu_1 - \mu_2)$$

No es necesario calcular todos los vectores propios — en el caso de 2 clases, basta con esta fórmula directa.

---

##### Generalización a Múltiples Clases

Para $K > 2$ clases, el criterio a optimizar es:

$$J(\mathbf{V}) = \frac{\det(\mathbf{V}^T S_B \mathbf{V})}{\det(\mathbf{V}^T S_W \mathbf{V})}$$

donde $\mathbf{V}$ es la **matriz de proyección** (no un solo vector). La solución se obtiene calculando los vectores propios de $S_W^{-1} S_B$. El número máximo de componentes discriminantes es $\min(K-1, d)$.

---

##### Implementación con scikit-learn

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy.linalg import inv

clf = LinearDiscriminantAnalysis(solver='eigen')
X_t = clf.fit_transform(X3, Y)

# Encontrar el vector de proyección desde los datos transformados
u0 = np.dot(np.dot(inv(np.dot(X3.T, X3)), X3.T), X_t)
```

- `solver='eigen'`: Usa descomposición en valores propios para encontrar los discriminantes.
- `fit_transform`: Ajusta el modelo y proyecta los datos en un único paso.
- El vector `u0` se recupera mediante pseudoinversa, permitiendo visualizar la dirección de proyección en el espacio original.

**Visualización del resultado:**

```python
plt.hist(c[Y==1], color='purple')
plt.hist(c[Y==0], color='yellow')
```

Los histogramas de las proyecciones muestran dos distribuciones bien separadas, validando que FDA encontró una dirección que discrimina efectivamente las clases.

---

##### Ejemplo con Dataset Iris (Multiclase)

```python
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target

# PCA: 1 componente
pca = PCA(1)
X_E = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_E)

# LDA
clf = LinearDiscriminantAnalysis(solver='eigen')
X_t = clf.fit_transform(X, y)
u = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), X_t)
```

Se comparan las proyecciones de PCA y LDA para todas las combinaciones de pares de variables de Iris (4 variables → 4×4 = 16 subplots):

- **PCA:** Las líneas rojas representan las direcciones de mayor varianza. Las clases no siempre quedan separadas.
- **LDA:** Las líneas rojas representan las direcciones de mayor discriminación. Las tres clases de Iris quedan claramente separadas y compactas.

---

#### Conceptos Clave del Archivo 1

| Concepto | Definición |
|---|---|
| **Análisis Discriminante de Fisher (FDA)** | Método supervisado de reducción dimensional que maximiza la ratio entre dispersión entre clases e intra-clases |
| **Matriz $S_W$ (Within-class scatter)** | Suma de matrices de dispersión intra-clase; mide compactez de las clases |
| **Matriz $S_B$ (Between-class scatter)** | Producto externo de la diferencia de medias; mide separación entre clases |
| **Criterio de Fisher $J(\mathbf{v})$** | Ratio $\mathbf{v}^T S_B \mathbf{v} / \mathbf{v}^T S_W \mathbf{v}$ a maximizar |
| **LDA (Linear Discriminant Analysis)** | Nombre alternativo, especialmente en contexto multiclase |
| **Proyección discriminante** | Dirección $\mathbf{v} = S_W^{-1}(\mu_1 - \mu_2)$ para el caso binario |
| **Problema de valores propios generalizados** | $S_W^{-1} S_B \mathbf{v} = \lambda \mathbf{v}$, solución general para múltiples clases |

---

---

### Archivo 2: `Métodos_no_lineales.pdf`

#### Propósito

Presentar métodos de reducción de dimensión **no lineales** que superan las limitaciones de PCA y LDA cuando los datos yacen sobre estructuras geométricas complejas (variedades no lineales). Se cubre teoría, implementación práctica e interpretación visual comparando cuatro métodos: Kernel PCA, t-SNE, UMAP y LLE.

---

#### Contenido y Funcionamiento

##### Motivación: Aprendizaje de Variedades (Manifold Learning)

En muchos problemas del mundo real, los datos de alta dimensionalidad **yacen sobre una variedad de menor dimensión** incrustada en el espacio original. Un ejemplo clásico es el *Swiss Roll* en 3D, que realmente tiene estructura 2D enrollada.

El aprendizaje de variedades (**manifold learning**) asume que existe una representación intrínseca de menor dimensión y busca descubrirla sin perder la estructura interna de los datos.

---

##### Sección 1 — Kernel PCA

**Concepto:**

Kernel PCA extiende PCA clásico aplicando el *truco del kernel* para capturar relaciones no lineales. En lugar de trabajar con la matriz de covarianza en el espacio original ($\frac{1}{N}X^TX$, de dimensión $d \times d$), trabaja con la **matriz kernel** $K$ (de dimensión $N \times N$):

$$K_{ij} = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$$

donde $\phi(\cdot)$ es una función de transformación (potencialmente a espacio infinito-dimensional), y el truco del kernel permite calcular $K_{ij}$ sin conocer explícitamente $\phi$.

**Problema de valores propios:**

$$N\lambda \mathbf{u} = K\mathbf{u}$$

Las componentes principales se obtienen como vectores propios de $K$.

**Nota importante:** La matriz $K$ es $N \times N$ (número de muestras), no $d \times d$. Esto es crítico en datasets grandes: el costo computacional escala con el número de muestras, no con la dimensionalidad.

**Proyección de nuevas muestras:** Kernel PCA no tiene una proyección directa de nuevas muestras. Se usa una aproximación que implica estimar el kernel entre la nueva muestra y todas las muestras de entrenamiento.

**Implementación:**

```python
from sklearn.decomposition import KernelPCA

# Kernel RBF (Radial Basis Function) = Gaussiano
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.01)
X_kpca = kpca.fit_transform(X)

# Proyección de nuevas muestras
X_new_kpca = kpca.transform(X_new)
```

- `kernel='rbf'`: $K_{ij} = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$. Captura similaridad basada en distancia Euclidiana.
- `gamma=0.01`: Controla el ancho de la función Gaussiana. Valores pequeños → vecindades más grandes.
- `fit_inverse_transform=True`: Permite reconstrucción aproximada al espacio original.

**Ejemplo Swiss Roll:**

El Swiss Roll es un dataset en 3D que tiene estructura 2D enrollada. PCA estándar no puede "desenrollar" esta estructura, mientras que Kernel PCA con RBF sí logra separar la variedad subyacente.

**Ejemplo Círculos Concéntricos:**

```python
X, y = make_circles(n_samples=1_000, factor=0.3, noise=0.05, random_state=0)
```

Dos círculos concéntricos son linealmente inseparables. PCA proyecta ambos círculos superpuestos; Kernel PCA los separa eficientemente en el espacio transformado, permitiendo luego aplicar LDA para clasificación perfecta.

---

##### Sección 2 — t-SNE (t-distributed Stochastic Neighbor Embedding)

**Concepto:**

t-SNE es una técnica **no lineal** diseñada principalmente para **visualización**. Funciona en dos pasos:

**Paso 1 — Distribución en el espacio original:**

Se construye una distribución de probabilidad condicional sobre pares de muestras:

$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

Luego se simetriza:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$

- Muestras **similares** (cercanas) → alta probabilidad de ser escogidas.
- Muestras **diferentes** (lejanas) → baja probabilidad.
- $\sigma_i$ se ajusta localmente según la densidad de cada punto (parámetro `perplexity`).

**Paso 2 — Distribución en el espacio reducido:**

Se define una nueva distribución usando una **cola pesada t de Student** (en lugar de Gaussiana):

$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_k \sum_{l \neq k} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

Las posiciones $\mathbf{y}_i$ en el espacio reducido se optimizan minimizando la **divergencia Kullback-Leibler**:

$$\text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

La cola pesada t de Student **evita el problema de aglomeración** (crowding problem) que sufre la versión original con Gaussianas: puntos moderadamente alejados en alta dimensión se colocan a distancias razonables en baja dimensión.

**Implementación:**

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='random', random_state=42)
X_tsne = tsne.fit_transform(X)
```

- `perplexity=30`: Número efectivo de vecinos considerados. Típicamente entre 5 y 50.
- `learning_rate=200`: Tasa de aprendizaje del gradiente descendente.
- `init='random'`: Inicialización aleatoria (alternativa: `'pca'` para mayor estabilidad).

**Limitaciones de t-SNE:**
- **No está diseñado para generalizar** — no puede proyectar nuevas muestras directamente.
- Para datasets de muy alta dimensión, se recomienda aplicar PCA primero (reducir a ~50 dimensiones) antes de t-SNE.
- Preserva bien las **vecindades locales** pero las distancias globales pueden no ser fiables.

---

##### Sección 3 — UMAP (Uniform Manifold Approximation and Projection)

**Concepto:**

UMAP es similar a t-SNE pero con ventajas adicionales:
- Más rápido computacionalmente.
- Preserva tanto estructura **local** como **global**.
- Puede proyectar nuevas muestras (a diferencia de t-SNE).

El algoritmo se basa en dos fases:

1. **Construcción de un grafo pesado de k-vecinos** en el espacio original, usando una métrica $D$ dada. Cada par de vecinos tiene un peso asociado basado en distancias.

2. **Optimización** de vectores de baja dimensión $\mathbf{y}$ para reproducir la topología del grafo original.

La base matemática involucra geometría diferencial (variedades de Riemann) y teoría de categorías, por lo que la formulación completa queda fuera del alcance del curso.

**Implementación:**

```python
import umap.umap_ as umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)

# Proyectar nuevas muestras
X_new_umap = reducer.transform(X_new)

# Reconstrucción al espacio original (aproximada)
X_reconstructed_umap = reducer.inverse_transform(X_test_umap)
```

- `n_neighbors=15`: Tamaño de la vecindad local. Valores pequeños → más detalle local; valores grandes → más estructura global.
- `min_dist=0.1`: Distancia mínima entre puntos en el espacio reducido. Controla cuán compactos quedan los clusters.

**Ventaja sobre t-SNE:** UMAP soporta `transform()` para nuevas muestras e `inverse_transform()` para reconstrucción aproximada.

---

##### Sección 4 — LLE (Local Linear Embedding)

**Concepto:**

LLE preserva **relaciones lineales locales** entre puntos vecinos. Funciona en dos pasos:

**Paso 1 — Estimación de pesos locales:**

Para cada punto $\mathbf{x}_i$, se encuentran sus $k$ vecinos más cercanos y se estima un conjunto de pesos $w_{ji}$ tales que:

$$\min \sum_i \left\| \mathbf{x}_i - \sum_j w_{ji} \mathbf{x}_j \right\|^2 \quad \text{sujeto a} \quad \sum_j w_{ji} = 1$$

Cada punto se reconstruye como combinación lineal de sus vecinos en el espacio original.

**Paso 2 — Embedding en baja dimensión:**

Se encuentran vectores $\mathbf{y}_i$ en baja dimensión minimizando:

$$\min_{\mathbf{y}} \sum_i \left\| \mathbf{y}_i - \sum_j w_{ji} \mathbf{y}_j \right\|^2$$

Los mismos pesos $w_{ji}$ se usan para ambas etapas: la idea es que si la estructura local es lineal, los mismos coeficientes que reconstruyen un punto en alta dimensión deben reconstruirlo también en baja dimensión.

**Implementación:**

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=12)
X_lle = lle.fit_transform(X)

# Para test
lle.fit(X_train)
X_test_lle = lle.transform(X_test)
```

- `n_neighbors=12`: Tamaño de la vecindad. Crítico: valores muy pequeños → ruido; valores muy grandes → pierde estructura local.

**Limitaciones:**
- Sensible al ruido y al tamaño de la vecindad.
- Puede fallar si la variedad tiene alta curvatura.

---

##### Sección 5 — Comparación en MNIST

Se aplican los cuatro métodos sobre el dataset `digits` de scikit-learn (subconjunto de MNIST, 1797 imágenes de 8×8 píxeles, 10 clases):

```python
digits = load_digits()
X_digits = StandardScaler().fit_transform(digits.data)  # Normalización previa
y_digits = digits.target

X_digits_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.01).fit_transform(X_digits)
X_digits_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(X_digits)
X_digits_umap = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X_digits)
X_digits_lle  = LocallyLinearEmbedding(n_components=2, n_neighbors=12).fit_transform(X_digits)
```

**Resultados observados en las visualizaciones:**

| Método | Separación de clases | Observación |
|---|---|---|
| **Kernel PCA** | Parcial | Los dígitos no quedan completamente separados; clusters solapados |
| **t-SNE** | Excelente | Clusters compactos y bien separados para casi todos los dígitos |
| **UMAP** | Muy buena | Clusters separados con mejor preservación de estructura global que t-SNE |
| **LLE** | Regular | Algunos clusters visibles pero con más ruido y solapamiento |

**Conclusión práctica:** Para visualización de datasets complejos como imágenes, t-SNE y UMAP son claramente superiores a Kernel PCA y LLE.

---

#### Conceptos Clave del Archivo 2

| Concepto | Definición |
|---|---|
| **Variedad (Manifold)** | Estructura geométrica de menor dimensión sobre la que yacen los datos de alta dimensión |
| **Aprendizaje de variedades** | Familia de técnicas que descubren la representación intrínseca de baja dimensión |
| **Truco del kernel** | Permite calcular productos internos en espacios de alta dimensión sin computar $\phi(\mathbf{x})$ explícitamente |
| **Kernel RBF** | $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma\|\mathbf{x}-\mathbf{z}\|^2)$; captura similaridad por proximidad |
| **Divergencia KL** | Medida de discrepancia entre dos distribuciones de probabilidad; función objetivo de t-SNE |
| **Perplexity (t-SNE)** | Número efectivo de vecinos; controla el balance entre estructura local y global |
| **n_neighbors (UMAP/LLE)** | Tamaño de la vecindad local; impacta directamente la calidad del embedding |
| **Swiss Roll** | Dataset 3D clásico con estructura 2D enrollada, usado para evaluar métodos no lineales |
| **Crowding problem** | Problema en embedding donde puntos moderadamente lejanos quedan aglomerados en baja dimensión |

---

---

### Archivo 3: `Selección_de_Características.pdf`

#### Propósito

Presentar la **selección de características** como estrategia complementaria a la extracción de características para reducir la dimensionalidad. A diferencia de la extracción (que transforma variables), la selección **elige un subconjunto** de las variables originales, preservando su interpretación física y semántica.

---

#### Contenido y Funcionamiento

##### Introducción y Motivación

El documento parte recordando el problema de la **maldición de la dimensionalidad**: a mayor número de variables, mayor dificultad para aprender modelos eficaces. Los beneficios de reducir dimensión incluyen:

- Reducir complejidad del modelo (menos parámetros a ajustar).
- Simplificar el análisis e interpretación de resultados.
- Mejorar el desempeño a través de representaciones más estables.
- Remover información redundante o irrelevante.
- Descubrir estructuras subyacentes en los datos.

**Dos estrategias principales:**

1. **Selección de características:** Encontrar el *mejor subconjunto de variables originales* de todos los subconjuntos posibles. Las variables conservan su significado original.
2. **Extracción de características:** Encontrar la *mejor transformación* (combinación de variables). Las nuevas variables pueden no tener interpretación directa (PCA, LDA, etc.).

---

##### Ventajas Específicas de la Selección

- **Reducción de costo de adquisición:** Algunas variables son costosas de medir (sensores, exámenes médicos, etc.). Si el modelo funciona bien sin ellas, se evita su costo.
- **Interpretabilidad:** Al conservar las variables originales, las reglas del modelo conservan sentido "físico".
- **Manejo de características no numéricas:** Más fácil trabajar con variables categóricas sin necesidad de transformarlas.

---

##### Por qué el Análisis Individual no es Suficiente

El documento ilustra este punto con un ejemplo visual de 5 clases y 4 variables:

- En el espacio $(X_1, X_2)$: las clases están bien separadas.
- En el espacio $(X_3, X_4)$: las clases se solapan más.

Si se analiza cada variable individualmente (por ejemplo, usando el índice discriminante de Fisher o la correlación), el resultado indica que **variable 1 es la mejor y variable 4 es la peor**. Sin embargo, al evaluar el *subconjunto* $\{X_1, X_4\}$ en conjunto, la combinación puede ser **mejor que cualquier variable individual**, debido a efectos de complementariedad.

**Conclusión:** No existe garantía de que las mejores variables individuales formen el mejor subconjunto conjunto.

---

##### El Problema Combinatorio

Dado un conjunto de $d$ variables, el número de posibles subconjuntos de tamaño $p$ es:

$$\binom{d}{p} = \frac{d!}{(d-p)! \cdot p!}$$

**Ejemplo:** Seleccionar 10 de 25 variables → $\binom{25}{10} = 3.268.760$ subconjuntos a evaluar. Y esto sólo para un valor fijo de $p$ — si $p$ también es desconocido, el espacio crece aún más.

Evaluar todos los subconjuntos (**Fuerza Bruta**) garantizaría encontrar el óptimo global, pero es computacionalmente inviable para valores moderados de $d$. Por tanto, se requieren **métodos de búsqueda subóptimos** que encuentren soluciones buenas sin evaluar todas las combinaciones.

Los métodos constan de dos componentes:
1. Un **criterio de selección** (función objetivo).
2. Una **estrategia de búsqueda** (cómo explorar el espacio de subconjuntos).

---

##### Criterios de Selección

###### A. Criterios Tipo Filtro

Los filtros evalúan subconjuntos basándose en propiedades intrínsecas de los datos, **independientemente del modelo de aprendizaje**.

**1. Distancia Probabilística (Divergencia de Bhattacharyya / Jeffreys):**

$$J_D(C_1, C_2) = \int [p(\mathbf{x}|C_1) - p(\mathbf{x}|C_2)] \log \frac{p(\mathbf{x}|C_1)}{p(\mathbf{x}|C_2)} d\mathbf{x}$$

Para distribuciones Gaussianas, la integral tiene solución cerrada:

$$J_D = \frac{1}{2}(\mu_1 - \mu_2)^T(\Sigma_1^{-1} + \Sigma_2^{-1})(\mu_1 - \mu_2) + \text{Tr}\{2\Sigma_1^{-1}\Sigma_2 - 2I\}$$

Para múltiples clases se extiende como:
- $J = \max_{i,j} J_D(C_i, C_j)$ — distancia entre las dos clases más difíciles.
- $J = \sum_{i<j} J_D(C_i, C_j) p(C_i) p(C_j)$ — distancia promedio ponderada entre todas las clases.

**2. Distancia entre Clases — Criterio de Fisher:**

Usando las matrices de dispersión ya conocidas:

$$J = \frac{\text{Tr}\{S_B\}}{\text{Tr}\{S_W\}}$$

Esta medida evalúa globalmente qué tan bien separadas están las clases en relación a su dispersión interna.

**3. Criterio Basado en Correlación:**

$$J = \frac{\sum_{i=1}^p \rho_{ic}}{\sum_{i=1}^p \sum_{j=i+1}^p \rho_{ij}}$$

- **Numerador:** Suma de correlaciones de cada variable con la salida $c$ → variables *relevantes*.
- **Denominador:** Suma de correlaciones entre variables de entrada → penaliza *redundancia*.

El objetivo: variables altamente correlacionadas con la salida y poco correlacionadas entre sí.

**4. Información Mutua:**

$$J = I(X_m; c) = H(c) - H(c | X_m)$$

- $H(c)$: Entropía de la variable de salida (incertidumbre sobre la clase).
- $H(c | X_m)$: Entropía condicional de la salida dado el subconjunto $X_m$.
- $I(X_m; c)$: Reducción en incertidumbre sobre $c$ al conocer $X_m$.

A diferencia de la correlación, la información mutua captura **relaciones no lineales**.

###### B. Criterios Tipo Wrapper

Los wrappers utilizan el **rendimiento de un modelo de aprendizaje** (como un clasificador) como función objetivo. Se entrena el modelo con el subconjunto de variables candidato, se evalúa mediante validación cruzada, y se usa $1 - \text{Error}$ como criterio.

**Ventajas y Desventajas comparadas:**

| Aspecto | Filtro | Wrapper |
|---|---|---|
| Velocidad | ✅ Rápido (cálculos directos) | ❌ Lento (entrena modelo por subconjunto) |
| Generalidad | ✅ Independiente del modelo | ❌ Específico al modelo usado |
| Exactitud | ❌ Puede no ser óptimo para el modelo final | ✅ Ajustado para reducir error de validación |
| Sobre-ajuste | ❌ Tendencia a seleccionar conjuntos grandes | ✅ Usa validación; mejor generalización |

---

##### Estrategias de Búsqueda

###### 1. Selección Secuencial Hacia Adelante (SFS — Sequential Forward Selection)

**Idea:** Comenzar con el conjunto vacío e ir añadiendo variables una a una.

**Algoritmo:**
1. Inicializar $X_0 = \{\emptyset\}$
2. $x^+ = \arg\max_{x \notin X_k} J(X_k + x)$ — seleccionar la variable que más mejora el criterio
3. $X_{k+1} = X_k + x^+$; $k = k+1$
4. Volver al paso 2.

**Ventaja:** Eficiente cuando el conjunto óptimo tiene pocas variables.

**Desventaja:** Una variable añadida no puede ser eliminada si posteriormente se vuelve redundante al añadir otras.

---

###### 2. Selección Secuencial Hacia Atrás (SBS — Sequential Backward Selection)

**Idea:** Comenzar con el conjunto completo e ir eliminando variables una a una.

**Algoritmo:**
1. Inicializar $X_0 = X$ (todas las variables)
2. $x^- = \arg\max_{x \in X_k} J(X_k - x)$ — eliminar la variable cuya ausencia menos deteriora (o más mejora) el criterio
3. $X_{k+1} = X_k - x^-$; $k = k+1$
4. Volver al paso 2.

**Ventaja:** Eficiente cuando el conjunto óptimo tiene muchas variables.

**Desventaja:** Una variable eliminada no puede ser reincorporada si posteriormente se vuelve útil.

---

###### 3. Selección Más-L Menos-R (LRS)

**Idea:** Permite retractación limitada combinando pasos hacia adelante y hacia atrás.

- Si $L > R$: Proceso predominantemente hacia adelante. Se añaden $L$ variables (SFS) y se eliminan $R$ (SBS) en cada ciclo.
- Si $L < R$: Proceso predominantemente hacia atrás.

**Algoritmo:**
1. Si $L > R$: comenzar con $X_0 = \emptyset$; si $L < R$: comenzar con $X_0 = X$.
2. Repetir $L$ veces: $X_{k+1} = X_k + \arg\max_{x \notin X_k} J(X_k + x)$.
3. Repetir $R$ veces: $X_{k+1} = X_k - \arg\max_{x \in X_k} J(X_k - x)$.
4. Volver al paso 2.

**Desventaja:** Introduce dos parámetros adicionales ($L$ y $R$) sin guía teórica para ajustarlos.

---

###### 4. Búsqueda Bidireccional (BDS — Bidirectional Search)

**Idea:** Ejecutar SFS y SBS simultáneamente con restricciones de consistencia:

- Variables añadidas por SFS **no pueden** ser eliminadas por SBS.
- Variables eliminadas por SBS **no pueden** ser añadidas por SFS.

**Algoritmo:**
1. SFS comienza con $X_F = \emptyset$; SBS comienza con $X_B = X$.
2. SFS añade: $x^+ = \arg\max_{x \notin X_{F_k}, x \in X_{B_k}} J(X_{F_k} + x)$; actualiza $X_{F_{k+1}}$.
3. SBS elimina: $x^- = \arg\max_{x \in X_{B_k}, x \notin X_{F_{k+1}}} J(X_B - x)$; actualiza $X_{B_{k+1}}$.
4. Volver al paso 2.

Las restricciones garantizan convergencia: en algún momento $X_F = X_B$ y el proceso termina.

---

###### 5. Selección Secuencial Flotante (SFFS y SFBS)

**Idea:** Extensión flexible de LRS donde los valores de $L$ y $R$ no son fijos sino que se determinan adaptativamente a partir de los datos.

La dimensionalidad del conjunto seleccionado "**flota**" hacia arriba y hacia abajo durante las iteraciones.

**SFFS (Sequential Floating Forward Selection):**

1. Comenzar con $X_0 = \emptyset$.
2. Añadir la mejor variable: $X_{k+1} = X_k + x^+$.
3. Intentar eliminar la peor variable:
   - $x^- = \arg\max_{x \in X_k} J(X_k - x)$
   - Si $J(X_k - x^-) > J(X_k)$: eliminar y volver al paso 3 (retractación).
   - Si no: volver al paso 2 (avanzar).

**SFBS (Sequential Floating Backward Selection):** Análogo, comenzando con el conjunto completo.

**Ventaja sobre LRS:** No requiere especificar $L$ y $R$ a priori; la flexibilidad es automática.

---

#### Conceptos Clave del Archivo 3

| Concepto | Definición |
|---|---|
| **Selección de características** | Elección de subconjunto de variables originales sin transformarlas |
| **Extracción de características** | Creación de nuevas variables como combinación/transformación de las originales |
| **Fuerza Bruta** | Evaluación de todos los $\binom{d}{p}$ subconjuntos posibles; garantiza óptimo global pero inviable |
| **Criterio Filtro** | Función objetivo independiente del modelo de aprendizaje; evalúa propiedades de los datos |
| **Criterio Wrapper** | Función objetivo basada en el rendimiento de un clasificador/regresor específico |
| **Información Mutua** | $I(X_m; c) = H(c) - H(c|X_m)$; mide dependencia estadística no lineal entre variables y salida |
| **SFS** | Selección secuencial hacia adelante; añade variables una a una |
| **SBS** | Selección secuencial hacia atrás; elimina variables una a una |
| **LRS** | Selección Más-L Menos-R; combina pasos de SFS y SBS con parámetros fijos |
| **BDS** | Búsqueda Bidireccional; ejecuta SFS y SBS simultáneamente con restricciones |
| **SFFS/SFBS** | Selección flotante; adapta dinámicamente los pasos de adición/eliminación |
| **Redundancia** | Información compartida entre variables de entrada; penalizada en criterios de correlación |

---

---

## Flujo General de la Unidad

```
Problema de alta dimensionalidad
            │
            ▼
¿Las clases son conocidas (supervisado)?
    │                    │
   SÍ                   NO
    │                    │
    ▼                    ▼
FDA / LDA           ¿Estructura lineal?
(Archivo 1)             │          │
                       SÍ         NO
                        │          │
                        ▼          ▼
                      PCA*    Kernel PCA,
                           t-SNE, UMAP, LLE
                           (Archivo 2)

¿Se requiere preservar interpretabilidad?
    │
   SÍ
    │
    ▼
Selección de Características
    │
    ├── Criterio: Filtro (rápido, general)
    │       ├── Distancia Probabilística
    │       ├── Índice de Fisher
    │       ├── Correlación
    │       └── Información Mutua
    │
    ├── Criterio: Wrapper (exacto, específico)
    │
    └── Estrategia de Búsqueda:
            ├── SFS (pocos óptimos)
            ├── SBS (muchos óptimos)
            ├── LRS (retractación limitada)
            ├── BDS (bidireccional)
            └── SFFS/SFBS (flotante, adaptativo)
```

*PCA fue tratado en la Parte 1 de la Unidad 8.

**Relaciones entre archivos:**

- El **Archivo 1 (FDA)** complementa directamente a PCA: cuando se tiene información de clase, FDA es superior a PCA para discriminación. Ambos son métodos *lineales*.
- El **Archivo 2 (Métodos No Lineales)** extiende el Archivo 1 al caso no lineal: cuando FDA y PCA no capturan la estructura, se recurre a Kernel PCA, t-SNE, UMAP o LLE. Además, se muestra cómo combinar Kernel PCA + LDA (Archivo 1) para clasificación no lineal.
- El **Archivo 3 (Selección)** es ortogonal a los anteriores: en lugar de transformar las variables, las selecciona. Los criterios de selección (como el índice de Fisher del Archivo 1) aparecen como funciones objetivo en los métodos de búsqueda.

---

## Conceptos Aprendidos

### Análisis Discriminante de Fisher
- Diferencia fundamental entre reducción no supervisada (PCA) y supervisada (FDA/LDA).
- Formulación de las matrices $S_B$ (between-class scatter) y $S_W$ (within-class scatter).
- El criterio de Fisher como ratio de varianza entre clases e intra-clases.
- Solución del problema como eigenvalues de $S_W^{-1} S_B$.
- Solución directa para el caso binario: $\mathbf{v} = S_W^{-1}(\mu_1 - \mu_2)$.
- Generalización a múltiples clases con matriz de proyección $\mathbf{V}$.
- Implementación con `LinearDiscriminantAnalysis(solver='eigen')` de scikit-learn.

### Métodos No Lineales
- Concepto de variedad (manifold) y por qué los métodos lineales fallan en datos con estructura no lineal.
- Kernel PCA: truco del kernel, matriz $K$ (N×N), relación con el problema de eigenvalues.
- t-SNE: distribución de probabilidad condicional en alta dimensión, distribución t en baja dimensión, minimización de divergencia KL, parámetro perplexity.
- UMAP: construcción de grafo pesado de k-vecinos, optimización de embedding, ventaja de poder proyectar nuevas muestras.
- LLE: estimación de pesos locales, preservación de relaciones lineales locales en embedding.
- Comparación práctica de los cuatro métodos en Swiss Roll, círculos concéntricos y MNIST.
- Limitaciones de cada método: t-SNE no generaliza, Kernel PCA es costoso para $N$ grande, LLE es sensible al ruido.

### Selección de Características
- Justificación de análisis conjunto frente a análisis individual de variables.
- Complejidad combinatoria del problema ($\binom{d}{p}$ subconjuntos).
- Criterios Filtro: distancia probabilística, índice de Fisher, correlación, información mutua.
- Criterios Wrapper: rendimiento del clasificador como función objetivo.
- Ventajas y desventajas de Filtros vs Wrappers (velocidad vs exactitud, generalidad vs especificidad).
- Estrategias de búsqueda: SFS, SBS, LRS, BDS, SFFS/SFBS — diferencias, algoritmos, ventajas y limitaciones.
- El problema de la irreversibilidad en SFS y SBS y cómo los métodos flotantes lo mitigan.

---

## Conclusiones

**Técnicas:**

1. **PCA vs FDA:** PCA es útil para compresión general, pero cuando el objetivo es clasificación, FDA debe preferirse si se dispone de etiquetas. FDA maximiza explícitamente la separación entre clases, lo que PCA no garantiza.

2. **Lineal vs No Lineal:** Ningún método único domina en todos los escenarios. Para datos con estructura no lineal (Swiss Roll, círculos), los métodos de manifold learning son indispensables. Para visualización, t-SNE y UMAP son los más efectivos en la práctica actual.

3. **Selección vs Extracción:** La extracción (PCA, LDA, UMAP, etc.) generalmente logra mejor compresión porque explora todas las combinaciones posibles de variables. La selección sacrifica poder expresivo pero gana en interpretabilidad y costo de adquisición de datos.

4. **Búsqueda Heurística:** Ningún método de búsqueda subóptimo garantiza el óptimo global. SFFS/SFBS es generalmente la estrategia más robusta entre las presentadas por su capacidad adaptativa de retractación.

**Académicas:**

5. La unidad establece una progresión clara: métodos lineales supervisados (FDA) → métodos no lineales (manifold learning) → estrategias de selección pura. Esta jerarquía refleja la evolución histórica del campo.

6. La elección entre métodos debe guiarse por: naturaleza de los datos (lineal/no lineal), disponibilidad de etiquetas (supervisado/no supervisado), requerimientos de interpretabilidad, costo computacional y necesidad de proyectar nuevas muestras.

---

## Recursos y Referencias

### Librerías Utilizadas

| Librería | Versión / Módulo | Uso |
|---|---|---|
| `numpy` | `numpy` | Operaciones matriciales, generación de datos |
| `matplotlib` | `matplotlib.pyplot` | Visualización de datos y proyecciones |
| `scikit-learn` | `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` | FDA/LDA |
| `scikit-learn` | `sklearn.decomposition.PCA`, `KernelPCA` | PCA lineal y con kernel |
| `scikit-learn` | `sklearn.manifold.TSNE`, `LocallyLinearEmbedding` | t-SNE y LLE |
| `scikit-learn` | `sklearn.datasets` | Swiss Roll, círculos, MNIST digits, Iris |
| `scikit-learn` | `sklearn.preprocessing.StandardScaler` | Normalización |
| `umap-learn` | `umap.umap_` | UMAP |
| `numpy.linalg` | `inv` | Inversión de matrices |

### Datasets Utilizados

- **Swiss Roll:** Dataset 3D sintético con estructura enrollada 2D (`make_swiss_roll`).
- **Círculos concéntricos:** Dataset 2D con dos clases linealmente inseparables (`make_circles`).
- **Iris:** Dataset clásico multiclase con 4 variables y 3 especies de flores (`datasets.load_iris`).
- **MNIST Digits:** Subconjunto de dígitos escritos a mano, 8×8 píxeles (`load_digits`).

### Referencias Conceptuales

- **Fisher, R.A. (1936):** "The Use of Multiple Measurements in Taxonomic Problems" — artículo original del Análisis Discriminante de Fisher.
- **van der Maaten, L. & Hinton, G. (2008):** "Visualizing Data using t-SNE" — paper original de t-SNE.
- **McInnes, L. et al. (2018):** "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" — paper original de UMAP.
- **Roweis, S.T. & Saul, L.K. (2000):** "Nonlinear Dimensionality Reduction by Locally Linear Embedding" — paper original de LLE.
- Documentación de scikit-learn: [https://scikit-learn.org/stable/modules/manifold.html](https://scikit-learn.org/stable/modules/manifold.html)
- Documentación de UMAP: [https://umap-learn.readthedocs.io](https://umap-learn.readthedocs.io)

---

*Documento generado para la Unidad 8 Parte 2 del curso Introducción al Machine Learning — 2025.*
