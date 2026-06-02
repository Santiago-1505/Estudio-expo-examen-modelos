# Unidad 5: Árboles de Decisión, Métodos de Ensamble y Detección de Anomalías

---

## Introducción

Esta unidad aborda una familia de modelos de Machine Learning fundamentalmente diferente a los vistos en unidades anteriores: los **Árboles de Decisión** y sus extensiones mediante **métodos de ensamble**. La motivación central es que los modelos lineales (regresión lineal, logística, clasificadores gaussianos) no son capaces de capturar estructuras complejas y no lineales en los datos, especialmente cuando los patrones presentan comportamientos radicalmente distintos en diferentes regiones del espacio de características.

La unidad progresa de lo simple a lo complejo en tres etapas:

1. **Clase 10** — Árboles de decisión individuales y sus primeras extensiones en ensamble (Voting, Bagging, Random Forest).
2. **Clase 11** — Métodos de ensamble secuenciales (Boosting: AdaBoost y Gradient Boosting) y combinación avanzada (Stacking).
3. **Clase 12** — Aplicación de los bosques a la detección de anomalías mediante Isolation Forest, un algoritmo no supervisado.

---

## Contenido de la Unidad

| Archivo | Clase | Tema |
|---|---|---|
| `Árboles_de_decisión.pdf` | Clase 10 | Árboles de decisión, Voting, Bagging, Random Forest |
| `Boosting.pdf` | Clase 11 | AdaBoost, Gradient Boosting, Stacking |
| `Isolation_Forest.pdf` | Clase 12 | Detección de anomalías con Isolation Forest |

---

## Estructura de Archivos

---

### Archivo 1: `Árboles_de_decisión.pdf`

#### Propósito

Este notebook introduce los árboles de decisión desde su motivación intuitiva hasta su implementación práctica, tanto para clasificación como para regresión. Además presenta los primeros métodos de ensamble: Voting, Bagging y Random Forest.

---

#### Contenido y Funcionamiento

##### 1. Motivación e Intuición

Se genera un dataset artificial compuesto por **dos segmentos con comportamientos distintos**:

```python
a1 = np.array([1, 2, 3])      # Coeficientes del polinomio (segmento parabólico)
a2 = np.array([-5, 263])      # Coeficientes del segmento lineal

x1 = np.linspace(-10, 10, 100)    # Rango izquierdo: comportamiento cuadrático
x2 = np.linspace(10.1, 30.1, 100) # Rango derecho: comportamiento lineal

for i in range(len(x1)):
    b1 = np.array([x1[i], x1[i]**2, 1])
    b2 = np.array([x2[i], 1])
    y1[i] = sum(a1*b1 + 30*(np.random.rand() - 0.5))  # Con ruido
    y2[i] = sum(a2*b2 + 30*(np.random.rand() - 0.5))  # Con ruido
```

**¿Por qué es difícil este problema?** Un único modelo lineal o polinomial no puede ajustar bien ambos segmentos simultáneamente porque tienen naturalezas matemáticas distintas (parabólica vs. lineal) y están ubicados en regiones distintas del espacio de características.

**Solución conceptual:** construir un modelo distinto para cada región y elegir cuál usar en función de la ubicación de la nueva muestra. Esto es exactamente el principio de los árboles de decisión.

---

##### 2. Definición de Árbol de Decisión

Un árbol de decisión es un método de aprendizaje inductivo que:

- Aprende una **función discreta representada como árbol**.
- Puede reformularse como **conjuntos de reglas if-then** encadenadas.
- Utiliza **una sola característica por nodo interno** (variante monoteísta).
- **Subdivide recursivamente** el espacio de características hasta obtener grupos homogéneos.

**Componentes del árbol (diagrama del notebook):**

```
         [x₆ < 2]          ← Nodo raíz (círculo): variable x₆, umbral 2
        /        \
   [x₄ < 1]   [x₅ < 5]    ← Nodos internos (círculos): evalúan condiciones
   /     \     /     \
[x₁<2]  [W₁] [W₃]  [W₂]   ← Hojas (cuadros): predicción final (clase o valor)
 /   \
[W₂] [W₁]
```

- **Nodos internos (círculos):** evalúan una condición `xᵢ < umbral`. Tienen dos ramas: `Sí` y `No`.
- **Hojas (cuadros):** contienen la predicción final (clase mayoritaria en clasificación, media en regresión).
- **Raíz:** el primer nodo que se evalúa.

**Ejemplo de inferencia:** Para la muestra `x = {5, 4, 6, 2, 2, 3}`:
1. Raíz: ¿x₆ < 2? → x₆=3, entonces **No** → ir a rama derecha.
2. ¿x₅ < 5? → x₅=2, entonces **Sí** → llegar a hoja **W₃**.

**Historia del algoritmo:**
- **ID3** (Ross Quinlan, 1986): algoritmo básico, construye el árbol de arriba hacia abajo.
- **C4.5** y **C5.0**: versiones mejoradas con manejo de variables continuas y podado más sofisticado.

---

##### 3. Medida de Impureza — Entropía

Para decidir qué variable y qué umbral usar en cada nodo, se necesita una medida que cuantifique la **calidad de la partición**. El objetivo es lograr nodos lo más puros posibles (muestras de una sola clase).

**Entropía de Shannon:**

$$I(U) = -\sum_j P(w_j) \log_2 P(w_j)$$

Donde:
- **U** es la partición evaluada.
- **P(w_j)** es la proporción de muestras de la clase *j* en la partición.

**Propiedades:**
- `I = 0`: nodo perfectamente puro (todas las muestras son de una sola clase). ✓ Situación ideal.
- `I = máxima`: cuando todas las clases están igualmente representadas (máxima incertidumbre). ✗ Peor caso.

**Impureza de Gini** (alternativa):

$$I(U) = \sum_{i \neq j} P(w_i) P(w_j)$$

Tiene la misma interpretación conceptual pero con diferente formulación matemática. Ambas son ampliamente usadas.

---

##### 4. Ganancia de Información

La **Ganancia de Información** mide cuánto reduce la impureza una partición basada en el atributo `a`:

$$\text{Gain}(U, a) = I(U) - \left( I(U_L) P_L + I(U_R) P_R \right)$$

Donde:
- **I(U)**: impureza del nodo padre antes de la partición.
- **U_L, U_R**: subconjuntos izquierdo y derecho tras la partición.
- **P_L, P_R**: proporciones de muestras que van a cada hijo.

Se elige la variable `a` **y su umbral** que maximicen la ganancia de información. Esto se evalúa de manera exhaustiva sobre todos los posibles umbrales de todas las variables.

---

##### 5. Algoritmo de Entrenamiento

El proceso es **greedy y recursivo** (divide y vencerás):

1. Calcular la ganancia de información para todos los atributos y todos los posibles umbrales.
2. Seleccionar el atributo y umbral con mayor ganancia.
3. Crear dos nodos hijos y dividir las muestras según la condición.
4. **Repetir recursivamente** en cada nodo hijo hasta alcanzar nodos puros (impureza = 0) o algún criterio de parada.

**Problema: Sobreajuste.** Si se deja crecer el árbol hasta impureza cero, cada hoja puede contener una sola muestra, lo que genera un modelo perfecto en entrenamiento pero con pobre generalización.

---

##### 6. Podado (Pruning)

El podado reduce el árbol para evitar sobreajuste. El método **Podado de Error Reducido** funciona así:

1. Todos los nodos de decisión son candidatos a ser convertidos en hojas.
2. Para cada candidato, la clase asignada sería la mayoritaria de todas las muestras bajo ese nodo.
3. Un nodo se elimina **solo si el árbol resultante mantiene o mejora** el desempeño en un **conjunto de validación**.
4. Se repite hasta que no haya más nodos eliminables sin pérdida de desempeño.

---

##### 7. Implementación de Árbol de Clasificación con scikit-learn

```python
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
clf = clf.fit(iris.data, iris.target)
```

**Parámetros clave:**
- `criterion`: métrica de impureza. `'entropy'` usa entropía de Shannon; `'gini'` usa la impureza de Gini.
- `max_depth`: profundidad máxima del árbol. `None` significa que crece hasta impureza cero. Valores pequeños (2, 3, 4) funcionan como regularización implícita.

**Visualización del árbol entrenado:**
```python
import pydotplus
dot_data = tree.export_graphviz(clf, feature_names=iris.feature_names,
                                 filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
```

El árbol visualizado muestra en cada nodo:
- La condición evaluada (e.g., `petal width (cm) <= 0.8`).
- La entropía actual del nodo.
- El número de muestras en ese nodo (`samples`).
- La distribución de clases (`value`).
- El color indica la clase dominante (saturación indica pureza).

La primera partición en el dataset Iris es `petal width <= 0.8`, que sepera perfectamente la clase *setosa* (50 muestras, entropía = 0) del resto.

**Visualización de fronteras de decisión:**

```python
Z[i,j] = clf.predict(np.array([xx[1,i], yy[j,1]]).reshape(1,2))[0]
plt.pcolormesh(xx, yy, Z.T, cmap=cmap_light)
plt.scatter(X[:,1], X[:,2], c=y)
```

Las fronteras son **paralelas a los ejes** (cortes ortogonales), lo cual es una característica estructural inherente de los árboles de decisión: cada partición evalúa solo una variable a la vez con un umbral.

---

##### 8. Árbol de Regresión

Para regresión, el valor predicho en cada hoja es el **promedio de los valores reales** de las muestras asignadas a ese nodo:

$$\hat{y}(\tau_l) = \frac{1}{N(\tau_l)} \sum_{x_i \in \tau_l} y_i$$

El error total del árbol de regresión se mide como:

$$R = \frac{1}{N} \sum_{l=1}^{L} \sum_{x_i \in \tau_l} (y_i - \hat{y}(\tau_l))^2$$

Y la ganancia de una partición en el nodo τ es:

$$\Delta R(\tau) = R(\tau) - R(\tau_L) - R(\tau_R)$$

Se elige la partición que maximiza `ΔR`.

**Ejemplo con profundidades diferentes:**

```python
clf_1 = DecisionTreeRegressor(max_depth=2)  # Modelo simple, pocas particiones
clf_2 = DecisionTreeRegressor(max_depth=5)  # Modelo más complejo
```

La gráfica muestra la clásica aproximación por **función escalera**: el árbol divide el espacio en segmentos y predice la media en cada uno. Con `max_depth=5` las predicciones se aproximan mejor a la función subyacente pero presentan más varianza (picos y caídas bruscas).

---

##### 9. Comité de Máquinas — Voting

El método de **Voting** combina las predicciones de múltiples modelos **distintos** entrenados sobre los mismos datos:

```python
from sklearn.ensemble import VotingRegressor

clf_a = DecisionTreeRegressor(max_depth=3)
clf_b = DecisionTreeRegressor(max_depth=5)
clf_c = LinearRegression()

clf = VotingRegressor([('DT1', clf_a), ('DT2', clf_b), ('lr', clf_c)]).fit(X, y)
```

- En **regresión**: la predicción final es el **promedio** de las predicciones individuales.
- En **clasificación**: la predicción final es la **clase con más votos** (mayoría).

El Voting combina las fortalezas de cada modelo: el árbol profundo captura no linealidades locales, el árbol poco profundo aporta suavidad, y la regresión lineal contribuye tendencias globales. La predicción combinada es más robusta que cualquier modelo individual.

---

##### 10. Bagging — Bootstrap Aggregating

**Bagging** usa el mismo tipo de modelo base pero con diferentes **subconjuntos de datos** generados por muestreo con reemplazo (bootstrap):

```
Dataset original (N muestras)
        ↓
Bootstrap 1 → Modelo 1 ─┐
Bootstrap 2 → Modelo 2 ─┼─→ Voto → Predicción final
    ...                  │
Bootstrap B → Modelo B ─┘
```

**Procedimiento:**
1. Crear *B* subconjuntos del conjunto de entrenamiento mediante **muestreo con reemplazo** (cada subconjunto tiene el mismo tamaño que el original, pero con muestras repetidas y otras faltantes).
2. Entrenar un modelo idéntico en cada subconjunto.
3. Combinar las predicciones: **moda** para clasificación, **promedio** para regresión.

**Implementación:**

```python
from sklearn.ensemble import BaggingRegressor

clf_b = DecisionTreeRegressor(max_depth=5)
clf_1 = BaggingRegressor(base_estimator=clf_b, n_estimators=10, random_state=0)
clf_2 = BaggingRegressor(base_estimator=clf_b, n_estimators=20, random_state=0)
```

**Parámetros:**
- `base_estimator`: el modelo base a usar (puede ser cualquier estimador de sklearn).
- `n_estimators`: número de modelos base (*B*). Más modelos = menor varianza pero mayor costo computacional.
- `random_state`: semilla para reproducibilidad.

**Efecto del Bagging:** reduce la **varianza** del error, produciendo predicciones más suaves y estables que un árbol individual profundo. La gráfica con 10 y 20 estimadores muestra predicciones muy similares (la varianza ya se redujo con 10 árboles).

---

##### 11. Random Forest

Random Forest es una extensión del Bagging específicamente diseñada para árboles que añade un componente adicional de aleatoriedad:

> En cada nodo, **en lugar de evaluar todas las variables disponibles**, se selecciona aleatoriamente un subconjunto de *m* variables y la partición se busca únicamente dentro de ese subconjunto.

Esto desacopla las correlaciones entre árboles: si una variable es muy dominante, en Bagging todos los árboles la usarían en la raíz y serían muy similares entre sí. Al limitar las variables candidatas en cada nodo, los árboles se diversifican más, mejorando el ensamble.

**Aplicación al dataset de dígitos escritos a mano (Digits):**

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()  # 1797 imágenes 8x8 de dígitos 0-9

for i in range(1, 50, 2):
    clf = RandomForestClassifier(n_estimators=i, max_depth=6, random_state=0)
    clf.fit(digits.data[train_idx], digits.target[train_idx])
    Performance.append(clf.score(digits.data[test_idx], digits.target[test_idx]))
```

**Resultado:** La precisión sube rápidamente de ~0.70 (1 árbol) hasta ~0.94 (50 árboles), mostrando la curva típica de convergencia del Random Forest: mejora rápida al inicio y estabilización posterior.

**Importancia de características (`feature_importances_`):**

```python
plt.bar(np.arange(digits.data.shape[1]), clf.feature_importances_)
```

El atributo `feature_importances_` devuelve la contribución promedio de cada característica a la reducción de impureza en todos los árboles. En el dataset de dígitos (imágenes 8x8 = 64 píxeles), los píxeles centrales tienen mayor importancia porque contienen más información discriminativa que los bordes.

**Nota sobre Árboles Extremadamente Aleatorios (Extra Trees):** El notebook menciona esta variante donde el umbral de cada nodo también se elige aleatoriamente (en lugar de buscar el óptimo), lo que aumenta aún más la diversidad entre árboles a cambio de potencialmente mayor sesgo.

---

##### 12. Múltiples Salidas (Multi-output)

Los árboles pueden extenderse para predecir **K variables simultáneamente**. Para regresión multi-salida, la función criterio usa la **distancia de Mahalanobis**:

$$R = \frac{1}{N} \sum_{l=1}^{L} \sum_{x_i \in \tau_l} (y_i - \hat{y}(\tau_l))^T \Sigma(\tau_l)^{-1} (y_i - \hat{y}(\tau_l))$$

**Ejemplo: Completado de rostros (Face Completion):**

```python
from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesRegressor

# Se usa la mitad superior de cada imagen para predecir la mitad inferior
X_train = train[:, :int(np.ceil(0.5 * n_pixels))]   # Mitad superior (entradas)
y_train = train[:, int(np.floor(0.5 * n_pixels)):]   # Mitad inferior (salidas)
```

Se comparan tres modelos multi-salida:
- **Extra Trees**: mejor reconstrucción visual, captura detalles.
- **K-NN**: reconstrucción suave pero menos definida.
- **Regresión Lineal**: peor resultado, no captura la no linealidad.

---

### Archivo 2: `Boosting.pdf`

#### Propósito

Este notebook presenta los métodos de Boosting, que a diferencia del Bagging (paralelo), entrenan los modelos de forma **secuencial**, haciendo que cada modelo corrija los errores del anterior. También introduce el método de Stacking.

---

#### Contenido y Funcionamiento

##### 1. Diferencia entre Bagging y Boosting

| Característica | Bagging | Boosting |
|---|---|---|
| Entrenamiento | Paralelo (independiente) | Secuencial (dependiente) |
| Datos | Bootstrap por cada modelo | Pesos adaptativos por muestra |
| Combinación | Promedio simple / Voto | Promedio ponderado |
| Reduce principalmente | **Varianza** | **Bias** |
| Riesgo | Bajo sobreajuste | Puede sobreajustar |

La imagen del notebook ilustra visualmente la diferencia: en Bagging todos los modelos se entrenan en subconjuntos paralelos; en Boosting se entrenan secuencialmente y el error de un modelo alimenta el siguiente.

---

##### 2. AdaBoost — Adaptive Boosting

**Definición del modelo ensamble:**

$$F(\mathbf{x}) = \sum_{m=1}^{M} \alpha_m f_m(\mathbf{x})$$

Donde:
- **α_m**: peso del clasificador *m* en la decisión final (mayor peso = mayor confiabilidad).
- **f_m(·)**: clasificador débil *m* (generalmente un árbol de profundidad 1, llamado "decision stump").

**Algoritmo AdaBoost (paso a paso):**

1. **Inicialización:** todos los pesos de las muestras son iguales: `w₁(i) = 1/N`.

2. **Para m = 1 hasta M:**

   a. Entrenar un clasificador débil `fₘ: ℝᵈ → {-1, +1}` usando los pesos `wₘ`.
   
   b. Calcular el **error ponderado** (tasa de error pesando más las muestras difíciles):
   $$\epsilon_m = \sum_{i: f_m(x_i) \neq y_i} w_m(i)$$
   
   c. Calcular el **peso del clasificador** (clasificadores con menor error reciben mayor peso):
   $$\alpha_m = \log \frac{1 - \epsilon_m}{\epsilon_m}$$
   
   d. **Actualizar los pesos de las muestras** (las muestras mal clasificadas reciben más peso):
   $$w_{m+1}(i) = w_m(i) \exp(-\alpha_m y_i f_m(x_i))$$

3. **Modelo final:**
$$F(\mathbf{x}) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m f_m(\mathbf{x})\right)$$

**Intuición gráfica (diagrama D1→D4):**
- **D1**: distribución uniforme inicial. El primer clasificador (Box 1) comete algunos errores.
- **D2**: las muestras mal clasificadas reciben más peso. El segundo clasificador (Box 2) se enfoca en ellas.
- **D3**: ídem. El tercer clasificador se enfoca en los errores del segundo.
- **D4**: el ensamble final (Box 4) combina los tres y clasifica casi perfectamente.

**Dos formas de usar los pesos en la implementación:**

1. **Muestreo:** usar los pesos para generar una nueva distribución de muestreo `D_{m+1}`, de modo que las muestras difíciles aparezcan más frecuentemente. Se implementa con `numpy.random.multinomial`.

2. **Peso explícito:** pasar los pesos directamente al método `fit` del clasificador base mediante el parámetro `sample_weight`. Los árboles de decisión de sklearn lo soportan.

**Implementación:**

```python
from sklearn.ensemble import AdaBoostRegressor

clf_b = DecisionTreeRegressor(max_depth=5)
clf_1 = AdaBoostRegressor(base_estimator=clf_b, n_estimators=10, random_state=0)
clf_2 = AdaBoostRegressor(base_estimator=clf_b, n_estimators=20, random_state=0)
```

**Atributos del modelo entrenado:**

```python
clf_1.estimator_weights_   # Pesos α_m de cada árbol
# array([1.896, 1.761, 1.682, 1.652, 1.772, 2.337, 1.132, 1.526, 0.900, 1.316])

clf_1.estimator_errors_    # Errores ε_m de cada árbol
# array([0.131, 0.147, 0.157, 0.161, 0.145, 0.088, 0.244, 0.179, 0.289, 0.211])

clf_1.estimators_          # Lista de los árboles individuales entrenados
```

Nótese que el árbol 6 (índice 5) tiene el menor error (0.088) y por tanto el mayor peso (2.337): AdaBoost le da más autoridad en la votación final.

---

##### 3. Gradient Boosting

**Gradient Boosting** aborda el problema desde una perspectiva diferente a AdaBoost: en lugar de ajustar los pesos de las muestras, **cada nuevo modelo aprende a predecir los residuales del modelo anterior**.

**Modelo acumulativo:**

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m f_m(\mathbf{x})$$

Donde:
- **F₀ = 0** (inicialización).
- **γ_m**: longitud del paso (similar a la tasa de aprendizaje).
- **f_m**: el nuevo modelo base ajustado a los residuales.

**Algoritmo:**

En cada iteración *m*, el nuevo modelo `f_m` se entrena usando como objetivos los **residuales** del paso anterior:

$$r_m(i) = -\frac{\partial L(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}$$

Para pérdida cuadrática: `r_m(i) = y_i - F_{m-1}(x_i)` (diferencia entre el valor real y la predicción actual).

La longitud del paso se optimiza como:

$$\gamma_m = \arg\min_\gamma \sum_{i=1}^{N} L\left(y_i, F_{m-1}(\mathbf{x}_i) + \gamma f_m(\mathbf{x}_i)\right)$$

**Diagrama conceptual:**
```
(X, y)      → Tree 1 → predice ŷ₁
r₁ = y - ŷ₁
(X, r₁)     → Tree 2 → predice r̂₁
r₂ = r₁ - r̂₁
(X, r₂)     → Tree 3 → predice r̂₂
...
F(x) = ŷ₁ + r̂₁ + r̂₂ + ...
```

Cada árbol aprende a corregir el error residual del ensamble hasta ese momento.

**Implementación:**

```python
from sklearn.ensemble import GradientBoostingRegressor

clf_1 = GradientBoostingRegressor(loss='absolute_error', n_estimators=20, random_state=0)
clf_2 = GradientBoostingRegressor(loss='absolute_error', n_estimators=30, random_state=0)
```

**Parámetros clave:**
- `loss`: función de pérdida. `'absolute_error'` usa el error absoluto medio; `'squared_error'` usa el ECM.
- `n_estimators`: número de etapas de boosting (número de árboles).
- `max_depth`: profundidad de cada árbol base. Árboles pequeños (3-5 niveles) son preferidos.

**Evaluación en digits:**

```python
for i in range(1, 200, 2):
    clf = GradientBoostingClassifier(n_estimators=i, max_depth=4, random_state=0)
    ...
```

La curva de precisión vs. número de árboles muestra que el Gradient Boosting alcanza ~0.96 de accuracy con suficientes árboles (~100-150), superando al Random Forest (que llegaba a ~0.94).

**Variantes avanzadas mencionadas en el notebook:**
- **Stochastic Gradient Boosting:** agrega muestreo aleatorio de filas y columnas para mayor robustez.
- **XGBoost:** implementación muy eficiente con regularización L1 y L2, ampliamente usado en competencias.
- **LightGBM:** variante de Microsoft, más rápida en datasets grandes usando Gradient-based One-Side Sampling (GOSS) y Exclusive Feature Bundling (EFB).
- **CatBoost:** variante de Yandex, maneja naturalmente variables categóricas.

---

##### 4. Stacking

**Stacking** es un método de ensamble de **dos capas**:

```
Capa 1 (Modelos base):
    Modelo A (DT profundidad 3)  ─┐
    Modelo B (DT profundidad 5)  ─┼─→ predicciones intermedias
    Modelo C (Regresión Lineal)  ─┘
                                   ↓
Capa 2 (Meta-modelo):
    RandomForest (meta-aprendiz) → predicción final
```

A diferencia del Voting (que promedia las predicciones), el Stacking **aprende cómo combinar** los modelos base usando un meta-modelo entrenado sobre las predicciones de la primera capa.

**Implementación:**

```python
from sklearn.ensemble import StackingRegressor

clf_a = DecisionTreeRegressor(max_depth=3)
clf_b = DecisionTreeRegressor(max_depth=5)
clf_c = LinearRegression()

estimators = [('DT1', clf_a), ('DT2', clf_b), ('lr', clf_c)]

clf = StackingRegressor(
    estimators=estimators,
    final_estimator=RandomForestRegressor(n_estimators=10, ...)
)
```

**Estructura del modelo (mostrada en el notebook):**
```
StackingRegressor
├── DT1: DecisionTreeRegressor(max_depth=3)
├── DT2: DecisionTreeRegressor(max_depth=5)
├── lr:  LinearRegression()
└── final_estimator: RandomForestRegressor(n_estimators=10)
```

El meta-modelo (Random Forest en este caso) aprende qué modelos base son más confiables para qué regiones del espacio de entrada.

---

### Archivo 3: `Isolation_Forest.pdf`

#### Propósito

Este notebook aplica el mecanismo de partición de los árboles de decisión a un problema completamente diferente: la **detección de anomalías** (también llamado detección de outliers). Es un algoritmo **no supervisado** que no requiere ejemplos etiquetados de anomalías para entrenarse.

---

#### Contenido y Funcionamiento

##### 1. Motivación — Clasificación de Una Clase

El problema de detección de anomalías se enmarca como **clasificación de una sola clase**: solo se tiene acceso a ejemplos "normales" durante el entrenamiento, y el objetivo es identificar nuevas muestras que no pertenecen a esa clase.

**Aplicaciones reales:**
- **Fraude financiero**: transacciones que no siguen el patrón habitual de un usuario.
- **Tráfico de red malicioso**: paquetes con características inusuales.
- **Detección de anomalías industriales**: sensores que reportan valores fuera de lo normal.

**Enfoque clásico:** modelar la densidad de probabilidad de los datos normales y definir un umbral — muestras con densidad baja son anomalías. Problema: requiere asumir una distribución (e.g., gaussiana) y puede fallar con distribuciones complejas.

---

##### 2. Intuición de Isolation Forest

**Premisa fundamental:** las anomalías son más fáciles de aislar que los puntos normales porque son pocas y están lejos del grueso de los datos.

El algoritmo genera particiones **aleatorias y recursivas**:
1. Seleccionar aleatoriamente una variable.
2. Seleccionar aleatoriamente un umbral entre el mínimo y máximo de esa variable.
3. Dividir los datos según esa condición.
4. Repetir recursivamente.

**Métrica clave:** cuántas particiones se necesitan para aislar un punto.

| Tipo de punto | Particiones necesarias |
|---|---|
| **Normal** | Muchas (está rodeado de vecinos) |
| **Anomalía** | Pocas (está lejos, se aísla rápido) |

---

##### 3. Generación del Dataset de Ejemplo

```python
from sklearn.datasets import make_blobs

# Datos normales: cluster compacto
X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.6, random_state=42)

# Anomalías: puntos uniformemente distribuidos en todo el espacio
X_anomalies = np.random.uniform(low=-6, high=6, size=(20, 2))

X = np.vstack([X_normal, X_anomalies])  # 300 normales + 20 anomalías
```

La gráfica resultante muestra claramente los dos grupos:
- **Azul (normales):** cluster denso centrado alrededor de (-2, 9).
- **Rojo (anomalías):** puntos dispersos por todo el espacio [-6, 6] × [-6, 6].

---

##### 4. Entrenamiento del Modelo

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=100,      # Número de árboles de aislamiento
    contamination=0.06,    # Proporción esperada de anomalías en el dataset
    random_state=42
)
iso_forest.fit(X)
```

**Parámetros importantes:**
- `n_estimators`: número de árboles de aislamiento. Más árboles = estimación más estable del score. Generalmente 100 es suficiente.
- `contamination`: fracción esperada de anomalías. Define el umbral de decisión. En el ejemplo: 20/320 ≈ 0.0625, por eso se usa 0.06.
- `random_state`: semilla para reproducibilidad.

**¿Qué aprende el modelo?** No aprende una frontera de decisión tradicional sino la **longitud de ruta promedio** para aislar cada punto a través de todos los árboles.

---

##### 5. Detección y Puntuación de Anomalías

```python
# Predicción: 1 = normal, -1 = anomalía
y_pred = iso_forest.predict(X)

# Score de anomalía (valores negativos = más anómalo)
scores = iso_forest.decision_function(X)
anomaly_strength = -scores  # Invertimos: valores altos = más anómalo
```

**`decision_function`**: devuelve el score de anomalía promediado sobre todos los árboles. Valores negativos indican anomalías.

**Visualización con tamaño proporcional al score:**

```python
# Normalización del score para tamaño del marcador
sizes = 50 + 400 * (anomaly_strength - anomaly_strength.min()) / \
        (anomaly_strength.max() - anomaly_strength.min())
```

En la gráfica resultante:
- Puntos **azules (normales)** son pequeños: scores bajos (normales).
- Puntos **naranjas (anomalías detectadas)** son grandes: scores altos (anómalos).
- El tamaño es proporcional a qué tan anómalo es cada punto.

---

##### 6. Análisis de las Particiones de un Árbol Individual

```python
def extract_splits(tree, feature_names):
    splits = []
    for i in range(tree.node_count):
        if tree.children_left[i] != tree.children_right[i]:  # Es nodo interno
            splits.append((
                feature_names[tree.feature[i]],  # Variable usada
                tree.threshold[i]                 # Umbral de corte
            ))
    return splits

tree = iso_forest.estimators_[0].tree_
splits = extract_splits(tree, ["x1", "x2"])
```

**Explicación línea a línea:**
- `tree.node_count`: número total de nodos en el árbol.
- `tree.children_left[i] != tree.children_right[i]`: condición que indica si el nodo *i* es un nodo interno (no es hoja). Las hojas tienen `children_left == children_right == -1`.
- `tree.feature[i]`: índice de la característica usada en el nodo *i*.
- `tree.threshold[i]`: umbral de corte en el nodo *i*.

Los primeros 10 splits del árbol 0 son todos en la variable **x2** (que tiene mayor rango de variación en las anomalías), mostrando que el algoritmo automáticamente prioriza las dimensiones con más dispersión.

**Visualización de splits:**

```python
for feature, threshold in splits:
    if feature == "x1":
        plt.axvline(threshold, color="gray", alpha=0.2)   # Corte vertical
    else:
        plt.axhline(threshold, color="gray", alpha=0.2)   # Corte horizontal
```

La gráfica muestra la densa malla de cortes horizontales y verticales que genera un árbol individual. Las regiones con más cortes son las más densamente habitadas por puntos normales.

---

##### 7. Mapa de Calor del Anomaly Score

```python
# Crear grilla para evaluar el score en todo el espacio
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scores = iso_forest.decision_function(grid).reshape(xx.shape)

# Visualizar como mapa de calor
plt.contourf(xx, yy, grid_scores, levels=30, cmap="coolwarm", alpha=0.35)
```

El mapa de calor muestra:
- **Zona roja/cálida** (scores positivos): región del cluster normal → scores normales (normales en efecto, bajo score de anomalía).
- **Zona azul/fría** (scores negativos): regiones alejadas del cluster → alto score de anomalía.

La frontera entre zonas define implícitamente el umbral de decisión del modelo, que separa puntos normales de anomalías en el espacio 2D.

---

## Flujo General de la Unidad

```
Clase 10: Árboles de Decisión
    ├── Motivación: datos con comportamientos múltiples
    ├── Estructura: nodos internos, hojas, raíz
    ├── Entrenamiento: ID3 con Ganancia de Información
    │       ├── Entropía / Impureza de Gini
    │       └── Algoritmo greedy recursivo
    ├── Podado: reducción de sobreajuste
    ├── Árbol de Regresión: predicción por media
    ├── Métodos de Ensamble (Paralelos)
    │       ├── Voting: combinar modelos distintos
    │       ├── Bagging: bootstrap + mismo modelo
    │       └── Random Forest: Bagging + aleatoriedad en variables
    └── Múltiples salidas
            ↓
Clase 11: Boosting y Stacking
    ├── Boosting (Secuencial)
    │       ├── AdaBoost: pesos adaptativos por muestra
    │       │       └── α_m según error ε_m
    │       └── Gradient Boosting: ajuste de residuales
    │               └── Variantes: XGBoost, LightGBM, CatBoost
    └── Stacking: meta-modelo aprende a combinar
            ↓
Clase 12: Isolation Forest
    ├── Problema: detección de anomalías (no supervisado)
    ├── Premisa: anomalías son fáciles de aislar
    ├── Algoritmo: particiones aleatorias recursivas
    ├── Score: longitud de ruta promedio para aislar
    └── Visualización: mapa de calor del anomaly score
```

---

## Conceptos Aprendidos

### Árboles de Decisión
- Estructura de árbol con nodos internos (condiciones) y hojas (predicciones).
- Algoritmo ID3: construcción greedy de arriba a abajo.
- Entropía de Shannon como medida de impureza.
- Impureza de Gini como medida alternativa.
- Ganancia de Información: criterio para seleccionar la mejor variable y umbral.
- Las fronteras de decisión de los árboles son siempre paralelas a los ejes.
- Sobreajuste y estrategias de podado (Podado de Error Reducido).
- Árboles de regresión: predicción por media de las hojas.
- La profundidad máxima (`max_depth`) controla el trade-off bias-varianza.

### Métodos de Ensamble Paralelos
- Voting: combinación de modelos heterogéneos por promedio o mayoría.
- Bagging (Bootstrap Aggregating): muestreo con reemplazo + mismo modelo base.
- Random Forest: Bagging con selección aleatoria de variables por nodo.
- El ensamble reduce la varianza respecto a un modelo individual.
- Importancia de características: las RF pueden cuantificar la relevancia de cada variable.
- Extensión a múltiples salidas usando distancia de Mahalanobis como criterio.

### Métodos de Ensamble Secuenciales (Boosting)
- Los modelos base se entrenan secuencialmente, cada uno corrigiendo al anterior.
- Boosting reduce principalmente el **sesgo (bias)**.
- AdaBoost: pesos adaptativos por muestra; mayor peso a las más difíciles.
  - Peso del clasificador α_m inversamente proporcional a su error.
  - Actualización exponencial de pesos de muestras.
- Gradient Boosting: entrena cada árbol sobre los residuales del modelo acumulado.
  - Equivalente a hacer gradiente descendente en el espacio funcional.
  - La longitud del paso γ_m se optimiza en cada iteración.
- Stacking: dos capas donde el meta-modelo aprende a combinar los modelos base.
- Variantes industriales: XGBoost, LightGBM y CatBoost.

### Isolation Forest
- Detección de anomalías como problema no supervisado de clasificación de una clase.
- Premisa: anomalías son más fáciles de aislar (requieren menos particiones).
- Particiones aleatorias recursivas (variable y umbral elegidos aleatoriamente).
- Score = longitud de ruta promedio para aislar cada punto.
- `contamination`: hiperparámetro que define el umbral de decisión.
- `decision_function`: devuelve el score continuo; valores negativos = anomalía.
- El mapa de calor del score revela la estructura del espacio aprendida por el bosque.

---

## Conclusiones

1. **Los árboles de decisión son intrínsecamente no lineales**: producen fronteras de decisión en escalera (paralelas a los ejes) que pueden aproximar cualquier función, pero son propensos al sobreajuste con profundidades grandes.

2. **Los métodos de ensamble superan consistentemente a los modelos individuales**: combinar múltiples modelos débiles casi siempre produce un modelo fuerte. La diversidad entre los modelos base es clave para el éxito del ensamble.

3. **Bagging vs. Boosting son complementarios**: Bagging reduce varianza (útil cuando el modelo base se sobreajusta), Boosting reduce bias (útil cuando el modelo base es demasiado simple). Random Forest es el método de ensamble más robusto para uso general.

4. **Gradient Boosting alcanza los mejores resultados en datos tabulares**: en el dataset Digits, llegó al 96% de accuracy, superando al Random Forest (94%). No es casualidad que XGBoost y LightGBM dominen las competencias de ML en Kaggle.

5. **Isolation Forest extiende la idea de los árboles al aprendizaje no supervisado**: aprovecha la misma lógica de partición pero con una métrica completamente diferente (longitud de camino vs. impureza), demostrando la versatilidad del paradigma de árboles.

6. **La interpretabilidad disminuye al aumentar la complejidad**: un árbol individual es completamente interpretable (reglas if-then); un Random Forest o un Gradient Boosting ya no lo son directamente, aunque métricas como `feature_importances_` ayudan a entender el modelo.

---

## Recursos y Referencias

### Librerías Utilizadas

| Librería | Uso |
|---|---|
| `numpy` | Manipulación de arrays y operaciones matemáticas |
| `matplotlib.pyplot` | Visualización de datos, fronteras y curvas de aprendizaje |
| `pandas` | Manipulación de datos tabulares (Isolation Forest) |
| `sklearn.tree.DecisionTreeClassifier` | Árbol de clasificación |
| `sklearn.tree.DecisionTreeRegressor` | Árbol de regresión |
| `sklearn.tree.export_graphviz` | Exportar árbol a formato Graphviz |
| `sklearn.ensemble.VotingClassifier/Regressor` | Ensamble por votación |
| `sklearn.ensemble.BaggingClassifier/Regressor` | Bagging |
| `sklearn.ensemble.RandomForestClassifier` | Random Forest para clasificación |
| `sklearn.ensemble.ExtraTreesRegressor` | Árboles extremadamente aleatorios |
| `sklearn.ensemble.AdaBoostClassifier/Regressor` | AdaBoost |
| `sklearn.ensemble.GradientBoostingClassifier/Regressor` | Gradient Boosting |
| `sklearn.ensemble.StackingClassifier/Regressor` | Stacking |
| `sklearn.ensemble.IsolationForest` | Detección de anomalías |
| `sklearn.datasets.load_iris` | Dataset Iris (clasificación 3 clases) |
| `sklearn.datasets.load_digits` | Dataset de dígitos manuscritos (64 características) |
| `sklearn.datasets.fetch_olivetti_faces` | Rostros de Olivetti (multi-output) |
| `sklearn.datasets.make_blobs` | Generación de datos sintéticos con clusters |
| `graphviz` / `pydotplus` | Visualización de árboles de decisión |

### Datasets

- **Iris Dataset**: 150 muestras, 4 características, 3 clases de flores.
- **Digits Dataset**: 1797 imágenes 8×8 de dígitos 0-9 (64 características).
- **Olivetti Faces**: 400 imágenes 64×64 de 40 personas (multi-output regression).
- **Dataset HPI**: problema de predicción del Índice de Prosperidad (mencionado como referencia).

### Referencias Externas Mencionadas

- Quinlan, J.R. (1986). Induction of Decision Trees. *Machine Learning*, 1(1), 81-106. — Algoritmo ID3.
- https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/ — Comparación visual Bagging vs Boosting.
- https://towardsdatascience.com/understanding-adaboost-2f94f22d5bfe — Ilustración del algoritmo AdaBoost.
- Liu et al. (2008). Isolation Forest. *ICDM*. — Paper original del algoritmo.
- Sitio web del curso: https://jdariasl.github.io/Intro_ML_2025/

---

*README generado para la Unidad 5 del curso Introducción al Machine Learning — 2025.*
