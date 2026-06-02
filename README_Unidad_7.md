# Unidad 7: Máquinas de Vectores de Soporte y Estrategias Multiclase

---

## Introducción

La Unidad 7 del curso de Introducción al Machine Learning aborda dos grandes temas complementarios:

1. **Máquinas de Vectores de Soporte (SVM)**: uno de los algoritmos de Machine Learning supervisado más sólidos matemáticamente, que busca la frontera de decisión de máximo margen entre clases. La unidad cubre su formulación matemática completa, el truco del kernel para datos no linealmente separables, la extensión al margen suave, la regresión por vectores de soporte (SVR) y la detección de anomalías con One-Class SVM.

2. **Estrategias Multiclase basadas en clasificadores binarios**: técnicas que permiten extender cualquier clasificador binario (como SVM) a problemas con más de dos clases. Se cubren tres estrategias: One vs All, One vs One y Clasificación Jerárquica, incluyendo también el caso de clasificación multi-etiqueta (multi-label).

El objetivo central es entender cómo las SVM maximizan el margen de separación y cómo se generaliza este principio para resolver problemas de clasificación más complejos.

---

## Contenido de la Unidad

| Archivo | Clase | Tema |
|---|---|---|
| `Máquinas_de_Vectores_de_Soporte___Introducción_al_Machine_Learning.pdf` | Clase 16 | SVM, kernels, SVR, One-Class SVM |
| `One_vs_all__one_vs_the_rest____Introducción_al_Machine_Learning.pdf` | Clase 17 | One vs All, One vs One, Clasificación Jerárquica |

---

## Estructura de Archivos

---

### Archivo 1: `Máquinas_de_Vectores_de_Soporte___Introducción_al_Machine_Learning.pdf`

#### Propósito

Presentar la formulación matemática completa de las Máquinas de Vectores de Soporte (SVM), desde la idea de máximo margen hasta la optimización con multiplicadores de Lagrange, el truco del kernel para espacios no lineales, la SVM de margen suave para datos traslapados, la regresión SVR y la detección de anomalías con One-Class SVM.

---

#### Contenido y Funcionamiento Detallado

##### 1. Motivación — El problema de la frontera óptima

Se parte del modelo lineal de clasificación biclase:

```
y(x) = wᵀφ(x) + b
```

donde:
- El conjunto de entrenamiento tiene `N` pares `(xᵢ, tᵢ)`, con `tᵢ ∈ {−1, 1}`.
- Una nueva muestra se clasifica según el signo de `y(x)`.

Para datos linealmente separables existen **infinitas fronteras de decisión** que clasifican perfectamente. La pregunta clave es: **¿cuál es la mejor?**

La respuesta de las SVM: la frontera que **maximiza el margen**, es decir, la distancia más corta entre la frontera y cualquiera de las muestras de entrenamiento.

---

##### 2. Distancia de un punto a un hiperplano

La distancia perpendicular de un punto `x` al hiperplano `y(x) = 0` es:

```
distancia = |y(x)| / ||w||
```

Si todas las muestras están correctamente clasificadas se cumple:
- Si `tₙ = −1` → `y(xₙ) < 0`
- Si `tₙ = 1` → `y(xₙ) > 0`

Por lo tanto: `tₙ · y(xₙ) > 0` para toda muestra `(xₙ, tₙ)`.

La distancia de cada muestra a la frontera es entonces:

```
tₙ · y(xₙ) / ||w|| = tₙ(wᵀφ(xₙ) + b) / ||w||
```

---

##### 3. Formulación del Máximo Margen

La solución de máximo margen se encuentra resolviendo:

```
arg max_{w,b} { (1/||w||) · min_n [tₙ(wᵀφ(xₙ) + b)] }
```

Para simplificar, se reescala el vector `w` tal que para el punto más cercano a la frontera:

```
tₙ(wᵀφ(xₙ) + b) = 1
```

Con esto, todos los puntos cumplen: `tₙ(wᵀφ(xₙ) + b) ≥ 1, n = 1, ..., N`

El problema de máximo margen se convierte en su formulación estándar:

```
min_{w,b}  (1/2)||w||²
sujeto a   tₙ(wᵀφ(xₙ) + b) ≥ 1
```

Este es un **problema de programación cuadrática**: función cuadrática con restricciones lineales.

---

##### 4. Formulación Dual con Multiplicadores de Lagrange

Se introducen multiplicadores de Lagrange `aₙ ≥ 0`, uno por restricción, para definir el Lagrangiano:

```
L(w, b, a) = (1/2)||w||² − Σₙ aₙ {tₙ(wᵀφ(xₙ) + b) − 1}
```

Derivando respecto a `w` y `b` e igualando a cero:

```
w = Σₙ aₙ tₙ φ(xₙ)
Σₙ aₙ tₙ = 0
```

Sustituyendo en el Lagrangiano se obtiene el **problema dual**:

```
max_a  L̃(a) = Σₙ aₙ − (1/2) ΣₙΣₘ aₙaₘtₙtₘ φ(xₙ)ᵀφ(xₘ)
sujeto a  aₙ ≥ 0,  Σₙ aₙtₙ = 0
```

La función de decisión resultante es:

```
y(x) = Σₙ aₙ tₙ k(x, xₙ) + b
```

donde `k(xₙ, xₘ) = φ(xₙ)ᵀφ(xₘ)` es la función kernel.

---

##### 5. Vectores de Soporte

De las condiciones de optimalidad (KKT):

```
aₙ ≥ 0
tₙ y(xₙ) − 1 ≥ 0
aₙ {tₙ y(xₙ) − 1} = 0
```

La tercera condición implica que para cada muestra:
- Si `aₙ = 0`: la muestra no está en el margen, **no influye** en la función de decisión.
- Si `aₙ ≠ 0`: la muestra está exactamente en el margen (`tₙy(xₙ) = 1`) → es un **vector de soporte**.

**Implicación práctica:** solo es necesario almacenar los vectores de soporte, no todo el conjunto de datos. Esto hace a la SVM eficiente en memoria.

El término independiente `b` se calcula promediando sobre todos los vectores de soporte:

```
b = (1/Nsv) Σ_{n∈sv} (tₙ − Σ_{m∈sv} aₘtₘk(xₙ, xₘ))
```

---

##### 6. El Truco del Kernel (Kernel Trick)

**Observación clave:** la función objetivo dual depende únicamente del producto `φ(xₙ)ᵀφ(xₘ)`, no del vector `w` explícitamente.

Esto permite **reemplazar** el producto punto por cualquier función `k(xₙ, xₘ)` que cumpla las **condiciones de Mercer** (garantizan que corresponde a un producto interno en algún espacio de características).

La ventaja es que se puede operar en espacios de alta dimensión (o incluso infinita dimensión) sin calcular explícitamente la transformación `φ(·)`.

**Kernels más comunes:**

| Kernel | Fórmula | Parámetros |
|---|---|---|
| **Lineal** | `k(xₙ, xₘ) = xₙᵀxₘ` | Ninguno |
| **Polinomial** | `k(xₙ, xₘ) = (xₙᵀxₘ + c)^d` | Grado `d`, constante `c` |
| **Gaussiano (RBF)** | `k(xₙ, xₘ) = exp(−||xₙ−xₘ||² / (2σ²))` | Ancho `σ` (o `gamma = 1/(2σ²)`) |
| **Sigmoide** | `k(xₙ, xₘ) = tanh(α xₙᵀxₘ + c)` | `α`, `c` |

**Ejemplo de mapeo kernel con datos circulares:**

El notebook muestra que datos en 2D organizados como círculos concéntricos (no separables linealmente) se vuelven separables linealmente al proyectarlos a un espacio 3D polinomial:

```python
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

# Mapeo polinomial manual a 3D
x1, x2 = X[:, 0], X[:, 1]
X_3d = np.vstack((x1**2, x2**2, x1*x2)).T
```

En el espacio 3D generado por `{x₁², x₂², x₁·x₂}`, las dos clases circulares se separan linealmente.

---

##### 7. SVM de Margen Suave (Soft-Margin SVM)

Cuando las clases están **traslapadas** (no hay separación perfecta), se introduce la SVM de margen suave, que permite que algunas muestras queden mal clasificadas, penalizando proporcionalmente a su distancia a la frontera.

**Variables de relajación** `ζₙ ≥ 0`:
- `ζₙ = 0`: muestra en el lado correcto del margen.
- `0 < ζₙ ≤ 1`: muestra dentro del margen, pero en el lado correcto.
- `ζₙ = 1`: muestra exactamente en la frontera.
- `ζₙ > 1`: muestra mal clasificada.

**Restricción modificada:**
```
tₙ y(xₙ) ≥ 1 − ζₙ
```

**Función objetivo:**
```
min_{w,b,ζ}  C·Σₙ ζₙ + (1/2)||w||²
sujeto a      tₙ y(xₙ) ≥ 1 − ζₙ,   ζₙ ≥ 0
```

El parámetro `C` controla el compromiso entre margen y errores:

| Valor de C | Efecto | Vectores de soporte |
|---|---|---|
| `C → ∞` | SVM dura (hard-margin), sin errores permitidos | Pocos (solo los del borde) |
| `C grande` | Pocos errores tolerados, margen pequeño | Pocos |
| `C pequeño` | Muchos errores tolerados, margen grande | Muchos |

**Formulación dual (igual estructura, restricción diferente):**

```
max_a  L̃(a) = Σₙ aₙ − (1/2) ΣₙΣₘ aₙaₘtₙtₘ φ(xₙ)ᵀφ(xₘ)
sujeto a  0 ≤ aₙ ≤ C,  Σₙ aₙtₙ = 0
```

Los vectores de soporte son ahora los puntos con `0 < aₙ < C`.

**Formulación ν-SVM:** variante donde el parámetro `C` se reemplaza por `ν ∈ [0,1]`, que tiene una interpretación más directa: es un límite inferior de la fracción de vectores de soporte y un límite superior de la fracción de muestras mal clasificadas. Disponible como `sklearn.svm.NuSVC`.

---

##### 8. Ejemplos Prácticos con scikit-learn

###### Efecto del parámetro C

```python
from sklearn import svm

# C grande: margen pequeño, pocos vectores de soporte
svc_high_C = svm.SVC(kernel='rbf', C=1e6)
svc_high_C.fit(X, y)

# C pequeño: margen grande, muchos vectores de soporte
svc_low_C = svm.SVC(kernel='rbf', C=1e-2)
svc_low_C.fit(X, y)
```

Los gráficos muestran claramente:
- Con `C=1e6`: frontera irregular, se ajusta muy bien al entrenamiento (riesgo de sobreajuste).
- Con `C=1e-2`: frontera suave, margen amplio, muchos puntos dentro del margen.

###### Efecto del kernel

```python
svc_lin  = svm.SVC(kernel='linear')           # Frontera recta
svc_poly3 = svm.SVC(kernel='poly', degree=3)   # Frontera cúbica
svc_poly7 = svm.SVC(kernel='poly', degree=7)   # Frontera de grado 7
svc_rbf   = svm.SVC(kernel='rbf', gamma=1e2)   # Kernel gaussiano
```

Observaciones de los gráficos:
- **Lineal**: frontera recta, adecuada para datos bien separados linealmente.
- **Polinomial grado 3**: frontera suavemente curva.
- **Polinomial grado 7**: frontera más compleja, mayor riesgo de sobreajuste.
- **RBF**: fronteras locales alrededor de cada grupo de puntos, muy flexible pero puede sobreajustar con `gamma` alto.

---

##### 9. Regresión por Vectores de Soporte (SVR)

La SVM se extiende a problemas de **regresión** mediante la función de error **ε-insensitiva**:

```
Eε(y(x) − t) = 0                    si |y(x) − t| < ε
              = |y(x) − t| − ε      en otro caso
```

Esta función ignora errores menores que `ε` (el tubo de tolerancia), penalizando solo lo que excede el margen.

**Función objetivo regularizada:**

```
min  C·Σₙ Eε(y(xₙ) − tₙ) + (1/2)||w||²
```

**Variables de relajación dobles** (por encima y por debajo del tubo):
- `ξₙ⁺ ≥ 0`: muestra por encima del tubo superior.
- `ξₙ⁻ ≥ 0`: muestra por debajo del tubo inferior.

**Restricciones:**
```
tₙ ≤ y(xₙ) + ε + ξₙ⁺
tₙ ≥ y(xₙ) − ε − ξₙ⁻
```

**Función objetivo completa:**
```
min  C·Σₙ (ξₙ⁺ + ξₙ⁻) + (1/2)||w||²
```

**Formulación dual** (con multiplicadores de Lagrange `aₙ⁺`, `aₙ⁻`):

```
L(a⁺, a⁻) = −(1/2)ΣₙΣₘ (aₙ⁺−aₙ⁻)(aₘ⁺−aₘ⁻)k(xₙ,xₘ) − ε·Σₙ(aₙ⁺+aₙ⁻) + Σₙ(aₙ⁺−aₙ⁻)tₙ
sujeto a  0 ≤ aₙ⁺ ≤ C,  0 ≤ aₙ⁻ ≤ C,  Σₙ(aₙ⁺−aₙ⁻) = 0
```

**Función de predicción:**
```
y(x) = Σₙ (aₙ⁺ − aₙ⁻) k(x, xₙ) + b
b = tₙ − ε − Σₘ (aₘ⁺ − aₘ⁻) k(xₙ, xₘ)
```

Los **vectores de soporte** en SVR son las muestras que caen **en el límite del tubo o fuera de él**.

**Ejemplo práctico:**

```python
from sklearn.svm import SVR

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8))  # Añadir ruido

svr_rbf  = SVR(kernel='rbf',    C=1e3, gamma=0.1)
svr_lin  = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly',   C=1e3, degree=2)
```

Los resultados muestran el efecto de los kernels y del parámetro `C` sobre la suavidad de la curva de regresión ajustada. Con `C` muy alto los modelos lineal y RBF se sobreajustan; con `C=1` el modelo RBF sigue bien la curva seno.

---

##### 10. One-Class SVM — Detección de Anomalías

Aunque la SVM es un método supervisado, puede adaptarse a **detección de anomalías** de forma no supervisada mediante **One-Class SVM**:

- Se entrena únicamente con datos normales.
- La única "muestra negativa" implícita es el **origen del espacio de características**.
- Se aprende una frontera que encierra la región de datos normales.
- Nuevas muestras fuera de la frontera se consideran **anomalías (outliers)**.

Utiliza la formulación **ν-SVM** porque `ν` permite controlar directamente la proporción esperada de anomalías.

**Implementación:**

```python
from sklearn.svm import OneClassSVM

ocsvm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.06  # proporción esperada de anomalías (~6%)
)
ocsvm.fit(X_normal)  # Solo datos normales en entrenamiento
y_pred = ocsvm.predict(X_test)  # +1 = normal, -1 = anomalía
```

**Generación de datos para el ejemplo:**
```python
from sklearn.datasets import make_blobs

X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.6)
X_anomalies = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack([X_normal, X_anomalies])
```

**Visualización de la frontera:**
- Se calcula `decision_function(grid)` sobre una malla densa.
- `f(x) = 0`: frontera de decisión (línea negra en el gráfico).
- `f(x) > 0`: región normal (puntos rojos).
- `f(x) < 0`: región de anomalías (puntos azules).

El gráfico resultante muestra una frontera elíptica que encierra el cluster normal, con las anomalías correctamente identificadas fuera de ella.

---

### Archivo 2: `One_vs_all__one_vs_the_rest____Introducción_al_Machine_Learning.pdf`

#### Propósito

Presentar las tres estrategias principales para extender clasificadores binarios (como SVM o LDA) a problemas multiclase: One vs All (OvA), One vs One (OvO) y Clasificación Jerárquica, incluyendo también el caso de clasificación multi-etiqueta (multi-label).

---

#### Contenido y Funcionamiento Detallado

##### 1. One vs All (OvA) — One vs the Rest

**Concepto:** se entrena un clasificador `Cᵢ` por cada clase `i` del problema. Cada clasificador separa la clase `i` del **resto de todas las clases** combinadas.

**En entrenamiento:**
- Para la clase `i`: las muestras de clase `i` son etiquetadas como `+1`, todas las demás como `0`.
- Se entrenan `Nc` clasificadores en total (uno por clase).

**En predicción:**
- **Multi-clase**: se evalúan todos los clasificadores y se elige la clase `i` cuyo clasificador produce la mayor confianza (consenso).
- **Multi-etiqueta**: cada clasificador decide de forma independiente si la muestra pertenece o no a su clase, permitiendo múltiples etiquetas simultáneas.

**Número de clasificadores:** `Nₑ = Nc` (igual al número de clases).

**Desventaja importante:** cada clasificador se entrena sobre un dataset **desbalanceado** (una clase vs. todas las demás), lo que puede requerir técnicas de corrección (oversampling, undersampling, class weights).

**Implementación manual (para entender el concepto):**

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

for i in range(3):
    Ytrain = Y.copy()
    Ytrain[Y == i] = 1
    Ytrain[Y != i] = 0
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, Ytrain.flatten())
    # Cada clf genera su propia frontera de decisión
```

El gráfico resultante muestra tres fronteras de decisión (una por clase), cada una separando su clase del resto.

**Con scikit-learn:**

```python
from sklearn.multiclass import OneVsRestClassifier

# Con LDA lineal
Z = OneVsRestClassifier(LinearDiscriminantAnalysis()).fit(X, Y).predict(...)

# Con QDA cuadrático (fronteras no lineales)
Z = OneVsRestClassifier(QuadraticDiscriminantAnalysis()).fit(X, Y).predict(...)
```

Los clasificadores base pueden ser de cualquier tipo, permitiendo fronteras lineales o no lineales según el clasificador elegido.

---

##### 2. Ejemplo Multi-etiqueta (Multi-label) — Clasificación de Texto

El notebook incluye un ejemplo completo de clasificación de textos en dos categorías simultáneas: **New York** y **London**.

**Dataset:**
```python
X_train = [
    "new york is a hell of a town",
    "london hosts the british museum",
    "new york is great and so is london",  # Multi-etiqueta: ambas
    ...
]
y_train_text = [["new york"], ["london"], ["new york", "london"], ...]
```

**Pipeline completo:**
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Binarizar las etiquetas múltiples
mlb = MultiLabelBinarizer()
Yt = mlb.fit_transform(y_train_text)
# Resultado: matriz binaria [london, new_york]
# Ej: ["new york", "london"] → [1, 1]

# Pipeline NLP + clasificación
classifier = Pipeline([
    ('vectorizer', CountVectorizer()),    # Bag of Words
    ('tfidf', TfidfTransformer()),        # TF-IDF weighting
    ('clf', OneVsRestClassifier(LinearSVC()))  # OvA con SVM lineal
])

classifier.fit(X_train, Yt)
predicted = classifier.predict(X_test)
```

**Resultados:**
```
nice day in nyc                                    => new york
welcome to london                                  => london
it is raining in britian and nyc                   => london, new york
hello welcome to new york. enjoy it here and london too => london, new york
```

El sistema detecta correctamente menciones a ambas ciudades simultáneamente.

**Componentes del Pipeline:**

| Componente | Función |
|---|---|
| `CountVectorizer` | Convierte texto a vectores de frecuencia de palabras (Bag of Words) |
| `TfidfTransformer` | Pondera las frecuencias por TF-IDF (penaliza palabras muy frecuentes) |
| `OneVsRestClassifier(LinearSVC)` | Un clasificador SVM lineal por etiqueta |
| `MultiLabelBinarizer` | Convierte listas de etiquetas en vectores binarios |

---

##### 3. One vs One (OvO) — All vs All

**Concepto:** se entrena un clasificador **por cada par de clases**. Para `Nc` clases se entrenan todos los pares posibles.

**En predicción:** cada clasificador emite un voto para una de las dos clases que considera. La clase con **más votos** gana (sistema de votación).

**Número de clasificadores:**
```
Nₑ = Nc · (Nc − 1) / 2
```

Por ejemplo, con 3 clases: `3·2/2 = 3` clasificadores (C0 vs C1, C0 vs C2, C1 vs C2).

**Comparación con OvA:**

| Estrategia | Nº Clasificadores | Datasets por clf | Desbalance |
|---|---|---|---|
| **One vs All** | `Nc` | Todo el dataset | Sí (1 clase vs. todas) |
| **One vs One** | `Nc(Nc-1)/2` | Solo muestras de 2 clases | Menor (datos de 2 clases) |

**Ventaja de OvO:** cada clasificador se entrena con un dataset más pequeño y más balanceado (solo las muestras de las dos clases relevantes).

**Desventaja de OvO:** requiere entrenar muchos más clasificadores cuando `Nc` es grande.

**Implementación:**

```python
from sklearn.multiclass import OneVsOneClassifier

# Con LDA lineal
Z = OneVsOneClassifier(LinearDiscriminantAnalysis()).fit(X, Y.flatten()).predict(...)

# Con QDA cuadrático
Z = OneVsOneClassifier(QuadraticDiscriminantAnalysis()).fit(X, Y.flatten()).predict(...)
```

Los gráficos muestran fronteras de decisión resultantes del sistema de votación entre los tres clasificadores binarios entrenados.

---

##### 4. Clasificación Jerárquica

**Concepto:** estrategia que organiza las clases en una **estructura de árbol**. En cada nodo del árbol se realiza una clasificación binaria entre dos grupos de clases, hasta llegar a diferenciar todas las clases de forma individual.

**Ejemplo de jerarquía:**
```
            Base Class
           /          \
        Food          Nonfood
       /    \              \
   Meats   Fruits      Paper Goods
           /    \
        Juices  Fresh Fruit
```

**Número de clasificadores:** `Nₑ = Nc − 1` (un clasificador por cada nodo interno del árbol).

**Ventaja:** si el conocimiento del dominio permite una jerarquía natural, puede ser más eficiente y precisa.

**Limitación crítica:** el orden de la jerarquía debe ser **definido manualmente** por el experto del dominio, lo que requiere conocimiento previo del problema.

**scikit-learn no incluye implementación nativa** para esta estrategia; se usa la librería externa `sklearn-hierarchical-classification`:

```bash
pip install sklearn-hierarchical-classification
```

**Implementación:**

```python
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn import svm

# Definición de la jerarquía como diccionario
class_hierarchy = {
    ROOT: ['0', 'A'],  # Raíz tiene dos ramas: clase 0 y grupo A
    'A': ['1', '2'],   # Grupo A se divide en clases 1 y 2
}

# Clasificador base: SVM con kernel RBF
clf_base = svm.SVC(gamma=0.001, kernel="rbf", probability=True)

clf = HierarchicalClassifier(
    base_estimator=clf_base,
    class_hierarchy=class_hierarchy
)

# Las etiquetas deben ser strings
Y_str = Y.astype(int).astype(str)
clf.fit(X, Y_str.flatten())
predictions = clf.predict(X_test)
```

El gráfico resultante muestra fronteras acordes a la estructura jerárquica definida.

---

##### 5. Generación de Datos de Prueba

El notebook usa datos gaussianos sintéticos con tres clases para ilustrar todas las estrategias:

```python
N = 100
Cov  = np.identity(2) * 1.1                     # Covarianza isótropa
Cov2 = np.array([[1.1, 0.5], [0.5, 1.1]])       # Correlación positiva
Cov3 = np.array([[1.1, -0.5], [-0.5, 1.1]])     # Correlación negativa

Mean  = [1.1, 2.1]   # Clase 0 (gris): parte inferior
Mean2 = [4.1, 4.1]   # Clase 1 (verde): parte central-derecha
Mean3 = [0, 8]       # Clase 2 (teal): parte superior-izquierda

# Generar muestras
x, y = np.random.multivariate_normal(Mean, Cov, N).T
...
X = np.r_[np.c_[x,y], np.c_[x2,y2], np.c_[x3,y3]]  # (300, 2)
Y = np.r_[2*np.ones((N,1)), np.ones((N,1)), np.zeros((N,1))]  # (300, 1)
```

Las tres clases tienen distribuciones parcialmente traslapadas, lo que permite comparar visualmente las diferentes estrategias multiclase.

---

## Flujo General de la Unidad

```
Clase 16 — SVM (Máquinas de Vectores de Soporte)
       ↓
   Motivación: ¿cuál frontera es la mejor?
   Respuesta: la de máximo margen
   Formulación primal → dual (Lagrange)
   Vectores de soporte: solo los puntos en el margen importan
   Kernel trick: clasificación no lineal sin calcular φ(x) explícitamente
   Soft-margin: tolerancia a errores con parámetro C
   Extensión a regresión (SVR): ε-tubo insensitivo
   Extensión a detección de anomalías (One-Class SVM)
       ↓
Clase 17 — Estrategias Multiclase
       ↓
   Problema: clasificadores binarios → múltiples clases
   Estrategia 1 — One vs All: Nc clasificadores, votación por confianza
   Estrategia 2 — One vs One: Nc(Nc-1)/2 clasificadores, votación por mayoría
   Estrategia 3 — Jerárquica: Nc-1 clasificadores, árbol de decisión
   Caso especial: Multi-label (varias etiquetas por muestra)
```

Las SVM entrenadas en Clase 16 son un ejemplo perfecto de clasificadores binarios que pueden usarse como base en cualquiera de las estrategias multiclase de Clase 17.

---

## Conceptos Aprendidos

### Máquinas de Vectores de Soporte

- **Máximo margen**: criterio óptimo para elegir entre múltiples fronteras de decisión; maximiza la distancia mínima a cualquier muestra.
- **Programación cuadrática**: tipo de problema de optimización que resuelve la SVM (función cuadrática con restricciones lineales).
- **Formulación dual**: reformulación equivalente que depende solo de productos internos entre muestras, no del vector `w`.
- **Multiplicadores de Lagrange**: herramienta matemática para incorporar restricciones en la optimización.
- **Condiciones KKT**: condiciones de optimalidad que dan lugar a los vectores de soporte.
- **Vectores de soporte**: las únicas muestras que definen la frontera; permiten representación compacta del modelo.
- **Kernel trick**: permite operar en espacios de alta dimensión sin calcular la transformación explícita.
- **Condiciones de Mercer**: garantizan que una función es un kernel válido (corresponde a un producto interno).
- **Kernel Gaussiano (RBF)**: el más utilizado en la práctica; `k(xₙ,xₘ) = exp(−||xₙ−xₘ||²/(2σ²))`.
- **Parámetro C**: controla el compromiso entre tamaño del margen y número de errores de clasificación.
- **Soft-margin SVM**: generalización que tolera errores mediante variables de relajación `ζₙ`.
- **ν-SVM**: formulación alternativa con interpretación probabilística del número de vectores de soporte.
- **SVR (Support Vector Regression)**: extensión de SVM a regresión usando ε-tubo insensitivo.
- **ε-tubo**: región alrededor de la función de regresión donde los errores no se penalizan.
- **One-Class SVM**: uso no supervisado de SVM para detección de anomalías.
- **`gamma` en RBF**: controla la influencia de cada muestra; `gamma` alto → influencia local, sobreajuste.

### Estrategias Multiclase

- **One vs All (OvA)**: `Nc` clasificadores, uno por clase; cada uno separa su clase del resto.
- **One vs One (OvO)**: `Nc(Nc-1)/2` clasificadores, uno por par de clases; votación por mayoría.
- **Clasificación Jerárquica**: árbol de clasificaciones binarias; `Nc-1` clasificadores; requiere jerarquía manual.
- **Multi-clase vs. Multi-etiqueta**: multi-clase → una sola clase ganadora; multi-etiqueta → múltiples clases simultáneas.
- **MultiLabelBinarizer**: convierte listas de etiquetas en matrices binarias para clasificación multi-etiqueta.
- **Pipeline de scikit-learn**: encadenamiento de transformaciones y clasificadores en un flujo unificado.
- **TF-IDF**: ponderación de términos en NLP que combina frecuencia local (TF) e inversa del documento (IDF).
- **Desbalance de clases**: problema inherente al OvA; requiere técnicas correctivas.
- **`OneVsRestClassifier`**: wrapper de scikit-learn que implementa OvA con cualquier clasificador base.
- **`OneVsOneClassifier`**: wrapper de scikit-learn que implementa OvO con cualquier clasificador base.
- **`sklearn-hierarchical-classification`**: librería externa para clasificación jerárquica compatible con scikit-learn.

---

## Conclusiones

1. **Las SVM son una de las formulaciones más elegantes del ML**: la idea de encontrar el hiperplano de máximo margen tiene sólida justificación teórica (teoría VC), lo que las hace robustas y con buenas propiedades de generalización, especialmente en espacios de alta dimensión.

2. **El kernel trick es el corazón de la potencia no lineal de las SVM**: permite separar clases que no son linealmente separables sin aumentar explícitamente la dimensionalidad, trasladando la no-linealidad del espacio de entrada a la elección de la función kernel.

3. **El parámetro C es el principal hiperparámetro a ajustar**: determina si el modelo prioriza un margen amplio (generalización) o minimiza errores en entrenamiento (ajuste exacto). Requiere ser calibrado mediante validación cruzada.

4. **La SVR extiende elegantemente la SVM a regresión**: el ε-tubo introduce un concepto de "tolerancia al error" que hace al modelo robusto frente a ruido, análogo al margen en clasificación.

5. **One-Class SVM amplía el alcance a detección de anomalías**: cuando solo se tienen datos normales disponibles (escenario común en práctica), la One-Class SVM proporciona una frontera de decisión basada en la densidad de los datos normales.

6. **No existe una estrategia multiclase universalmente superior**: OvA es simple y eficiente; OvO usa datasets más pequeños y balanceados pero requiere más clasificadores; la jerárquica puede ser muy eficiente si se conoce la estructura del problema. La elección depende del tamaño del dataset, el número de clases y el conocimiento del dominio.

7. **La clasificación multi-etiqueta es cualitativamente diferente de la multi-clase**: en multi-etiqueta, OvA es la estrategia natural porque cada clasificador decide de forma independiente, mientras que en multi-clase se necesita un mecanismo de consenso para elegir una única clase ganadora.

---

## Recursos y Referencias

### Librerías utilizadas

| Librería | Uso |
|---|---|
| `numpy` | Generación de datos y operaciones matriciales |
| `matplotlib` | Visualización de fronteras de decisión y datos |
| `scikit-learn` | SVM, SVR, OneClassSVM, OneVsRestClassifier, OneVsOneClassifier, Pipeline, TfidfTransformer, CountVectorizer, MultiLabelBinarizer |
| `sklearn-hierarchical-classification` | Clasificación jerárquica compatible con scikit-learn |
| `local.library.svmclass` | Funciones auxiliares del curso para visualización de SVM |

### Clases de scikit-learn relevantes

| Clase | Módulo | Descripción |
|---|---|---|
| `SVC` | `sklearn.svm` | SVM para clasificación (kernel flexible) |
| `NuSVC` | `sklearn.svm` | Formulación ν-SVM |
| `SVR` | `sklearn.svm` | SVR para regresión |
| `OneClassSVM` | `sklearn.svm` | One-Class SVM para detección de anomalías |
| `LinearSVC` | `sklearn.svm` | SVM lineal optimizado |
| `OneVsRestClassifier` | `sklearn.multiclass` | Estrategia One vs All |
| `OneVsOneClassifier` | `sklearn.multiclass` | Estrategia One vs One |
| `Pipeline` | `sklearn.pipeline` | Encadenamiento de pasos de ML |
| `MultiLabelBinarizer` | `sklearn.preprocessing` | Codificación multi-etiqueta |
| `CountVectorizer` | `sklearn.feature_extraction.text` | Bag of Words para texto |
| `TfidfTransformer` | `sklearn.feature_extraction.text` | TF-IDF para texto |

### Referencias bibliográficas

1. **Bishop, C.M.** — *Pattern Recognition and Machine Learning*, Springer, 2006. *(Referencia principal para la formulación matemática de SVM)*
2. **Schölkopf, B.; Smola, A.J.** — *Learning with Kernels*, MIT Press, 2002. *(Referencia para kernels y ν-SVM)*
3. **scikit-learn documentation** — [https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html)
4. **sklearn-hierarchical-classification** — [https://github.com/globality-corp/sklearn-hierarchical-classification](https://github.com/globality-corp/sklearn-hierarchical-classification)
