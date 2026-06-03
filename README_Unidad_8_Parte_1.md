# Unidad 8 - Parte 1: Técnicas Avanzadas en Machine Learning

## Introducción

Esta unidad abarca tres grandes temas que complementan y profundizan el ciclo completo de desarrollo de modelos de Machine Learning: la **regularización y selección de variables con LASSO**, la **reducción de dimensionalidad con PCA (Análisis de Componentes Principales)** y la **explicabilidad de modelos de ML (XAI - Explainable Artificial Intelligence)**. Estos temas son fundamentales para construir modelos que no solo sean precisos, sino también interpretables, eficientes y aplicables en contextos reales como la medicina, las finanzas y la seguridad.

Los tres notebooks corresponden a las **Clases 19, 20 y 23** del curso *Introducción al Machine Learning* (2025), disponible en `https://jdariasl.github.io/Intro_ML_2025/`.

---

## Contenido de la Unidad

| Archivo | Clase | Tema |
|--------|-------|------|
| `LASSO__Least_Absolute_Shrinkage_and_Selection_Operator____Introducción_al_Machine_Learning.pdf` | Clase 19 | Regularización L1, selección de variables y Elastic Net |
| `Reducción_de_dimensión__Análisis_de_Componentes_Principales___Introducción_al_Machine_Learning.pdf` | Clase 20 | PCA: teoría, implementación y aplicación a reconocimiento de rostros |
| `Explicabilidad_de_modelos_de_Machine_Learning___Introducción_al_Machine_Learning.pdf` | Clase 23 | SHAP, LIME y explicabilidad contrafactual |

---

## Estructura de Archivos

---

### Archivo 1: `LASSO__Least_Absolute_Shrinkage_and_Selection_Operator____Introducción_al_Machine_Learning.pdf`

#### Propósito

Este notebook profundiza en el método LASSO, estudiado inicialmente en semanas anteriores del curso. El objetivo es entender sus efectos de regularización, cómo realiza selección automática de variables al llevar coeficientes a cero, y cómo se determina el parámetro de regularización óptimo.

---

#### Contenido y Funcionamiento Detallado

##### 1. Formulación del Modelo de Regresión con LASSO

Se parte de un modelo lineal donde se busca explicar una variable respuesta `y` a través de `d` variables predictoras `x₁, x₂, ..., xd`:

```
y = wᵀx + ε
donde x = [x₁, x₂, ..., xd, 1]
```

La variable `ε` representa el ruido de observación. En la práctica, se trabaja con la forma simplificada `y ≈ wᵀx`.

La hipótesis de partida de LASSO es que **no todas las variables son pertinentes**. El objetivo es identificar y eliminar las variables "inútiles" llevando sus coeficientes `wᵢ` exactamente a cero.

---

##### 2. Función Objetivo de LASSO

LASSO minimiza la suma de los errores cuadráticos más un término de penalización en norma L1:

```
argmin_w Σᵢ (yᵢ - Σⱼ wⱼ·xᵢⱼ)² + λ · Σⱼ |wⱼ|   con λ ≥ 0
```

Forma equivalente con restricción explícita:

```
argmin_w Σᵢ (yᵢ - Σⱼ wⱼ·xᵢⱼ)²
Sujeto a: Σⱼ |wⱼ| ≤ s
```

**Nota importante:** el término de regularización **no incluye** al coeficiente `w₀` (intercepto), ya que este representa la línea base del modelo y no debería ser penalizado.

---

##### 3. Características Fundamentales de LASSO

- **Norma L1 como penalización:** a diferencia de Ridge que usa norma L2, LASSO utiliza el valor absoluto de los coeficientes. Geométricamente, la región factible de L1 tiene vértices afilados sobre los ejes, lo que provoca que la solución óptima sea exactamente cero en muchas variables.

- **Anulación de coeficientes:** la penalización L1 convierte directamente en 0 los coeficientes de variables no relevantes, haciendo innecesarios los test de hipótesis clásicos.

- **Control mediante λ y s:**
  - Si `λ = 0` → LASSO equivale a regresión lineal estándar (sin penalización).
  - Si `λ → ∞` → todos los coeficientes tienden a 0.
  - El parámetro `s` (forma alternativa) actúa inversamente a `λ`.

- **Estimación del λ óptimo:** se obtiene mediante **validación cruzada**.

- **Algoritmos de solución:** LARS (Least Angle Regression) y descenso por coordenadas reducen la complejidad computacional de la optimización.

---

##### 4. Ejemplos Implementados

**Ejemplo 1 - Entrenamiento básico sobre datos Diabetes (sklearn):**

```python
from sklearn import linear_model, datasets
diabetes = datasets.load_diabetes()
clf = linear_model.Lasso(alpha=0.5)
clf.fit(diabetes.data, diabetes.target)
print(clf.coef_)
# Resultado: [ 0.  -0.  471.04  136.51  -0.  -0.  -58.32  0.  408.02  0. ]
```

Se observa que varios coeficientes son exactamente 0, lo que indica que LASSO **descartó automáticamente** esas variables del modelo. Los coeficientes no nulos (471, 136, 408) corresponden a las variables con mayor poder predictivo.

**Ejemplo 2 - Entrenamiento y predicción con datos pequeños:**

```python
clf = linear_model.Lasso(alpha=1)
y_pred = clf.fit([[0,0],[1,1],[2,5]], [0,1,4]).predict([3,7])
# coef_: [0. 0.571]
# intercept_: 0.524
# r² en test: 0.983
```

Con `alpha=1`, el primer coeficiente se anula completamente. El R² de 0.983 indica muy buen ajuste.

**Ejemplo 3 - Datos sintéticos de alta dimensión (200 variables, solo 10 relevantes):**

```python
np.random.seed(42)
n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
coef[inds[10:]] = 0  # Solo 10 variables son verdaderamente relevantes
y = np.dot(X, coef)
y += 0.01 * np.random.normal((n_samples,))

lasso = Lasso(alpha=0.1)
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
# r² = 0.385
```

La gráfica generada compara los coeficientes LASSO estimados (en azul discontinuo) con los coeficientes originales (en verde). LASSO logra identificar correctamente los picos de coeficientes no nulos, aunque con r² = 0.385 no se recupera perfectamente la señal (efecto del ruido y el bajo n/p ratio).

**Ejemplo 4 - LASSO Path (influencia de λ sobre los coeficientes):**

```python
alphas, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)
xx = np.sum(np.abs(coefs.T), axis=1) / np.sum(np.abs(coefs.T), axis=1)[-1]
plt.plot(xx, coefs.T)
plt.xlabel('|coef| / max|coef| (Alpha)')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
```

El **LASSO Path** es un gráfico diagnóstico fundamental que muestra cómo los coeficientes del modelo se van incorporando (saliendo de cero) a medida que `λ` disminuye (o equivalentemente, a medida que `|coef|/max|coef|` aumenta). Permite visualizar el orden de entrada de variables al modelo y cuáles son las más robustas (permanecen no-nulas en el mayor rango de λ).

---

##### 5. Ventajas y Desventajas de LASSO

**Ventajas:**
- Selección automática de variables sin necesidad de test de hipótesis.
- Ampliamente implementado en Matlab, Python (sklearn), R y Octave.
- Aplicable a problemas de alta dimensionalidad.

**Desventajas:**
- La penalización L1 se aplica uniformemente a todos los parámetros.
- **No garantiza** recuperar siempre el conjunto correcto de variables relevantes.
- Matemáticamente complejo para calcular el error estándar (la penalización L1 no es diferenciable en 0); se requiere un cálculo probabilístico de la varianza del error.
- Con variables altamente correlacionadas, LASSO tiende a elegir una arbitrariamente y descartar las demás.

---

##### 6. Elastic Net: Variante de LASSO

Elastic Net combina las penalizaciones L1 (LASSO) y L2 (Ridge) en un único término de regularización:

```
argmin_w Σᵢ (yᵢ - Σⱼ wⱼ·xᵢⱼ)² + λ₁·Σⱼ|wⱼ| + λ₂·Σⱼ wⱼ²
```

**Características:**
- Especialmente útil cuando `d >> N` (más variables que observaciones).
- El término L1 promueve **dispersión** (coeficientes en cero).
- El término L2 promueve que variables correlacionadas tengan **coeficientes similares** (soluciona el problema de arbitrariedad de LASSO).

**Comparación experimental:**

```python
lasso = Lasso(alpha=0.1)
enet = ElasticNet(alpha=0.1, l1_ratio=0.7)
# r² LASSO:       0.384710
# r² ElasticNet:  0.240176
```

En este ejemplo LASSO supera a Elastic Net en r², pero en datasets con alta correlación entre variables, Elastic Net suele ser superior.

Las gráficas incluidas muestran los paths de regularización para: LASSO, LASSO positivo, Elastic Net y Elastic Net positivo. Se aprecia que Elastic Net produce una transición más suave en los coeficientes.

**Otras variantes mencionadas:**
- Regresión Lasso Adaptativo
- Regresión Lasso Relajado

---

### Archivo 2: `Reducción_de_dimensión__Análisis_de_Componentes_Principales___Introducción_al_Machine_Learning.pdf`

#### Propósito

Este notebook presenta el **Análisis de Componentes Principales (PCA)**, la técnica de extracción de características más utilizada en Machine Learning. Explica tanto la base matemática (eigenvectores, valores propios, covarianza) como su aplicación práctica, incluyendo un ejemplo completo de reconocimiento de rostros (EigenFaces) con SVM.

---

#### Contenido y Funcionamiento Detallado

##### 1. Motivación: Selección vs. Extracción de Características

La reducción de dimensionalidad tiene dos grandes enfoques:

- **Selección de características:** elige un subconjunto de las variables originales.
  - Ventaja: mantiene la interpretabilidad.
  - Desventaja: computacionalmente costosa (especialmente con criterios wrapper); puede perder información al eliminar variables.

- **Extracción de características (PCA):** crea nuevas variables como combinaciones lineales de las originales.
  - Ventaja: maximiza la varianza retenida con pocas componentes.
  - Desventaja: las nuevas variables (componentes) son más difíciles de interpretar.

**Propiedad clave de PCA:** en conjuntos de datos con alta dependencia entre variables, menos del 20% de las nuevas variables (componentes principales) suelen explicar más del 80% de la variabilidad original.

---

##### 2. Fundamento Matemático de PCA

**Configuración inicial:**
- Matriz de datos `X` con `N` muestras y `d` variables.
- Los datos deben estar **centrados en cero** (media extraída por variable).
- Matriz de covarianza: `S = (1/N) · Xᵀ·X`

**Objetivo de la proyección:**
Encontrar la dirección `u₁ = (u₁₁, ..., u₁d)ᵀ` (vector unitario) que maximice la varianza de las proyecciones. La proyección de cada punto `xᵢ` sobre esta dirección es:

```
zᵢ = u₁ᵀ · xᵢ
```

**Problema de optimización:**
Se busca minimizar la distancia entre cada punto y su proyección, lo que equivale a maximizar la varianza proyectada:

```
max_{u₁}  Σᵢ zᵢ²  =  max_{u₁}  Σᵢ u₁ᵀ·xᵢᵀ·xᵢ·u₁
```

Considerando toda la matriz: `z₁ = X·u₁`, el problema se reescribe como:

```
(1/N) · z₁ᵀ·z₁  =  (1/N) · u₁ᵀ·Xᵀ·X·u₁  =  u₁ᵀ·S·u₁
```

**Restricción (multiplicadores de Lagrange):**
Para que `u₁` no crezca indefinidamente, se agrega la restricción `|u₁| = 1`:

```
M = u₁ᵀ·S·u₁ - λ·(u₁ᵀ·u₁ - 1)
```

Derivando e igualando a cero:

```
∂M/∂u₁ = 2·S·u₁ - 2·λ·u₁ = 0
S·u₁ = λ·u₁
```

**Conclusión fundamental:** `u₁` es un **vector propio (eigenvector)** de la matriz de covarianza `S`, y `λ` es su **valor propio (eigenvalue)** asociado.

Como `u₁ᵀ·S·u₁ = λ`, la varianza proyectada es igual al valor propio. Para maximizarla, se selecciona el **eigenvector asociado al mayor eigenvalue** → este es el **Primer Componente Principal**.

**Generalización:** para proyectar en un espacio de dimensión `p < d`, se seleccionan los `p` eigenvectores asociados a los `p` mayores eigenvalues de `S`. Las proyecciones sobre estas direcciones son los **Componentes Principales**.

---

##### 3. Implementación con Scikit-learn

**PCA básico (1 componente):**

```python
from sklearn.decomposition import PCA

# Generación de datos correlacionados
np.random.seed(1)
X = np.dot(np.random.random(size=(2,2)), np.random.normal(size=(2,200))).T + 10
X = X - np.mean(X, axis=0)  # Centrar en 0

pca = PCA(n_components=1)
pca.fit(X)
Xt = pca.transform(X)[:, 0]  # Coordenadas en el espacio reducido
```

El vector del primer componente resultante es aproximadamente `[0.944, 0.329]`, indicando que el primer componente captura principalmente la variación en la dirección de `x₁`.

**Reconstrucción en 2D:**

```python
u0 = pca.components_[0]
c = X.dot(u0)
Xr = np.r_[[i * u0 for i in c]]
```

La reconstrucción proyecta todos los puntos sobre la línea del primer componente principal, mostrando visualmente cuánta información se pierde al reducir la dimensión.

**Proyecciones aleatorias vs. PCA:**

Se generan 3 proyecciones aleatorias con distintos ángulos y se calcula la desviación estándar de cada proyección. PCA selecciona automáticamente la dirección con mayor desviación estándar (mayor variabilidad).

**Extensión a datos de alta dimensión (4D → 1D):**

```python
Cov = np.array([[2.9, -2.2], [-2.2, 6.5]])
X = np.random.multivariate_normal([1,2], Cov, size=200)
X_HD = np.dot(X, np.random.uniform(0.2, 3, (2,4)) * ...)  # Shape (200, 4)

pca = PCA(1)
X_E = pca.fit_transform(X_HD)  # Shape (200, 1)
X_reconstructed = pca.inverse_transform(X_E)  # Reconstrucción aproximada
```

La gráfica de matriz de dispersión 4×4 muestra que la reconstrucción (azul) sigue muy de cerca la estructura de los datos originales (rojo), confirmando que 1 componente es suficiente para capturar la variabilidad principal.

---

##### 4. Aplicación: EigenFaces (Reconocimiento de Rostros)

Este es el ejemplo más avanzado del notebook, demostrando PCA en un problema real de visión computacional.

**Carga del dataset:**

```python
from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data         # Shape: (n_samples, 1850) — imágenes aplanadas
y = lfw_people.target       # Etiqueta de identidad
names = lfw_people.target_names
n_samples, n_features = X.shape  # ~1140 muestras, 1850 píxeles por imagen
_, h, w = lfw_people.images.shape  # Dimensiones de la imagen (50×37)
```

El dataset LFW (Labeled Faces in the Wild) contiene imágenes de rostros de figuras públicas. Con `min_faces_per_person=70` se filtran solo las identidades con suficientes ejemplos (Bush, Blair, Powell, Sharon, Schroeder, Rumsfeld).

**División train/test:**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# X_train.shape: (966, 1850)
```

**Extracción de EigenFaces:**

```python
n_components = 150
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca.fit(X_train)
eigenfaces = pca.components_.reshape((n_components, h, w))
```

- `n_components=150`: se extraen las 150 direcciones de mayor varianza.
- `svd_solver='randomized'`: algoritmo eficiente para datos de alta dimensión.
- `whiten=True`: normaliza los componentes para que tengan varianza unitaria, mejorando el rendimiento del clasificador posterior.
- Los **EigenFaces** son los vectores propios visualizados como imágenes: capturan patrones de iluminación, orientación y rasgos faciales que maximizan la varianza entre imágenes.

**Clasificación con SVM:**

```python
X_train_pca = pca.transform(X_train)  # (966, 150) — espacio reducido

from sklearn.svm import SVC
svm = SVC(kernel='rbf', class_weight='balanced')

# Validación cruzada
cv = StratifiedShuffleSplit(n_splits=3, test_size=0.20)
svm_cv_scores = cross_val_score(svm, X_train_pca, y_train, scoring='accuracy', cv=cv)
# Resultado: [0.851, 0.756, 0.782, 0.793, 0.746]
```

**Búsqueda de hiperparámetros:**

```python
param_grid = {
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}
clf = GridSearchCV(svm, param_grid, scoring='accuracy', cv=cv, n_jobs=2)
clf.fit(X_train_pca, y_train)
# Mejores parámetros: {'C': 1000.0, 'gamma': 0.005}
# Mejor score de validación: 0.842
```

**Predicción y visualización:**

```python
X_test_pca = pca.transform(X_test)
y_pred = clf.predict(X_test_pca)
```

La galería de imágenes de test muestra sobre cada foto el nombre predicho y el nombre verdadero, permitiendo identificar visualmente los errores del modelo.

---

##### 5. Selección del Número de Componentes

Estrategias para elegir el número óptimo de componentes `p`:

1. **Gráfico de eigenvalues vs. índice (Scree Plot):** buscar el "codo" de la curva donde la varianza explicada por componente cae bruscamente.

2. **Varianza acumulada:** seleccionar los primeros `p` componentes que cubran el 80%, 90% o 95% de la varianza total.

3. **Umbral de eigenvalue:** descartar componentes con eigenvalue inferior a la varianza media.

4. **Criterio wrapper:** evaluar el rendimiento del modelo completo con diferentes valores de `p` (más costoso pero más preciso).

**Ejemplo con EigenFaces:**

```python
plt.plot(pca.explained_variance_)                # Varianza por componente
plt.plot(np.cumsum(pca.explained_variance_ /      # Varianza acumulada
                   np.sum(pca.explained_variance_)))
```

El gráfico de varianza acumulada muestra que con los primeros ~20 componentes se captura aproximadamente el 80% de la varianza, y con 150 componentes se supera el 97%.

---

### Archivo 3: `Explicabilidad_de_modelos_de_Machine_Learning___Introducción_al_Machine_Learning.pdf`

#### Propósito

Este notebook aborda la **explicabilidad de modelos de Machine Learning (XAI)**, un área de creciente importancia tanto técnica como regulatoria. Se implementan tres métodos: SHAP Values, LIME y Explicabilidad Contrafactual, aplicados a datasets reales de diagnóstico médico y riesgo crediticio.

---

#### Contenido y Funcionamiento Detallado

##### 1. Motivación de la Explicabilidad

Muchos modelos modernos (Random Forest, Gradient Boosting, Redes Neuronales, SVM con kernel) alcanzan alta precisión pero son **cajas negras** — difíciles de interpretar. Esto genera problemas en:

- **Medicina:** ¿por qué el modelo diagnostica cáncer?
- **Finanzas:** ¿por qué se rechaza un crédito?
- **Seguridad:** ¿cómo se puede atacar o engañar el sistema?
- **Regulación:** normativas internacionales (como el GDPR en Europa) exigen que los sistemas de IA puedan explicar sus decisiones.

Preguntas clave que la explicabilidad debe responder:
- ¿Por qué el modelo tomó esta decisión?
- ¿Qué variables fueron más importantes?
- ¿Qué pasaría si cambiamos una característica?

---

##### 2. Tipos de Explicabilidad

**Según el alcance:**

| Tipo | Descripción |
|------|-------------|
| **Global** | Describe el comportamiento general del modelo (importancia de variables promedio) |
| **Local** | Explica una predicción concreta para una observación específica |

**Según el modelo:**

| Tipo | Descripción | Ejemplos |
|------|-------------|---------|
| **Model-specific** | Depende del tipo de modelo | Importancia de impureza en árboles, pesos en redes neuronales |
| **Model-agnostic** | Aplicable a cualquier modelo | SHAP, LIME |
| **Contrafactual** | ¿Qué cambiaría la decisión? | Búsqueda de contrafactuales |

---

##### 3. Configuración del Experimento Base

**Dataset Breast Cancer (para SHAP y LIME):**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
```

El dataset Breast Cancer Wisconsin tiene 569 muestras con 30 características numéricas (radio, textura, perímetro, área, etc. de núcleos celulares) para clasificar tumores como malignos (0) o benignos (1).

---

##### 4. SHAP Values (SHapley Additive exPlanations)

**Fundamento teórico:**

SHAP se basa en la **teoría de juegos cooperativos de Shapley**. Cada variable es tratada como un "jugador" que contribuye a la "ganancia" (predicción final). La predicción se descompone como:

```
f(x) = φ₀ + Σᵢ φᵢ(x)
```

Donde:
- `φ₀ = E[f(X)]` es el valor base (predicción media del modelo sobre todos los datos)
- `φᵢ(x)` es la contribución local de la característica `i` para la observación `x`

**Fórmula de los valores de Shapley:**

```
φᵢ(x) = Σ_{S ⊆ N\{i}} [|S|! · (d-|S|-1)! / d!] · [f_{S∪{i}}(x) - f_S(x)]
```

Donde:
- `N = {1,...,d}` es el conjunto de todas las características
- `S ⊆ N\{i}` es un subconjunto que no incluye la característica `i`
- `f_S(x)` es el modelo evaluado usando solo las características en `S`
- El término `|S|! · (d-|S|-1)! / d!` es el peso que da importancia igual a todas las combinaciones posibles

En esencia, el valor SHAP de una variable mide su **contribución marginal promedio** a la predicción, considerando todas las posibles coaliciones de variables.

**Implementación:**

```python
import shap

y_est = model.predict(X_test)
explainer = shap.TreeExplainer(model)    # Explainer específico para Random Forest
shap_values = explainer(X_test)          # Calcula los SHAP values
```

`shap.TreeExplainer` es eficiente para modelos basados en árboles (aprovecha la estructura del árbol para calcular los valores de Shapley exactos en tiempo polinomial, en lugar de la complejidad exponencial general).

**Explicación Global (todos los datos de test):**

```python
shap.summary_plot(shap_values[:, :, 1], X_test)
shap.plots.beeswarm(shap_values[:, :, 1])
```

El gráfico **beeswarm** muestra:
- Cada punto es una observación del conjunto de test.
- El eje X es el valor SHAP (impacto en la salida del modelo).
- El color indica el valor de la característica (rojo = alto, azul = bajo).
- Las características están ordenadas de mayor a menor importancia.

**Interpretación del beeswarm para Breast Cancer:**
- `worst concave points`: la característica más importante. Valores altos (rojo) tienden a producir SHAP positivos (empujan hacia predicción benigna), mientras que valores bajos (azul) tienden hacia SHAP negativos (empujan hacia maligno).
- `worst perimeter`, `worst area`, `mean concave points`, `worst radius`: también muy importantes.
- Las características al fondo (como `concavity error`) tienen poca influencia.

**Explicación Local (observación individual):**

```python
i = 10
shap.plots.waterfall(shap_values[i, :, y_est[i]])
```

El gráfico **waterfall** para la observación i=10 muestra:
- Valor base `E[f(X)] = 0.628`
- Cada barra representa la contribución de una variable (rojo = empuja hacia clase positiva, azul = empuja hacia clase negativa)
- El valor final `f(x) = 0.955` se construye sumando las contribuciones al valor base
- `worst concave points = 0.06` aporta +0.06, `worst perimeter = 91.29` aporta +0.05, etc.

**Force Plot (visualización alternativa local):**

```python
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values.values[i,:,1], X.iloc[i])
```

El force plot compacta toda la información en una sola línea horizontal donde las fuerzas rojas empujan el valor de la predicción hacia arriba y las azules hacia abajo.

---

##### 5. LIME (Local Interpretable Model-agnostic Explanations)

**Principio:**

LIME explica una predicción **local** aproximando el modelo complejo con un modelo simple (como regresión lineal) en la vecindad de la observación de interés.

**Procedimiento paso a paso:**

1. **Perturbación:** dada la observación `x`, genera un dataset "falso" aplicando ligeras perturbaciones aleatorias a sus características.
2. **Ponderación:** asigna un peso a cada muestra falsa según su distancia a la observación original.
3. **Predicción:** obtiene las predicciones del modelo original para cada muestra perturbada.
4. **Modelo sustituto:** entrena una regresión lineal ponderada. Los coeficientes de esta regresión son las medidas de importancia local de las variables.

**Implementación:**

```python
from lime.lime_tabular import LimeTabularExplainer

explainer_lime = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=["maligno", "benigno"],
    mode="classification"
)

exp = explainer_lime.explain_instance(
    X_test.iloc[i].values,
    model.predict_proba,
    num_features=8        # Mostrar las 8 características más importantes
)
exp.show_in_notebook()
```

**Resultado para la observación i=10:**

```
Probabilidades:  maligno=0.04,  benigno=0.95

Características más influyentes (para clase "benigno"):
520.70 < worst area ≤ ...     +0.06
worst concave points ...       +0.06
worst texture > 29.49          +0.05
84.47 < worst perimeter ≤ ... +0.05
0.02 < mean concave points ≤  +0.04
mean texture > 21.59           +0.04
0.11 < worst concavity ≤ ...  +0.04
13.06 < worst radius ≤ ...    +0.03
```

LIME presenta las reglas en forma de intervalos (condiciones sobre las variables), lo que facilita su interpretación clínica.

**Diferencia entre SHAP y LIME:**
- SHAP tiene fundamento teórico sólido (valores de Shapley) y produce explicaciones consistentes globalmente.
- LIME es más flexible (funciona con texto e imágenes también) pero las explicaciones pueden variar entre ejecuciones por la aleatoriedad de las perturbaciones.

---

##### 6. Explicabilidad Contrafactual

**Concepto:**

Mientras SHAP y LIME responden "¿por qué el modelo tomó esta decisión?", la explicabilidad contrafactual responde: **"¿Qué debería cambiar para obtener un resultado diferente?"**

Es especialmente valiosa en contextos como:
- Un cliente cuyo crédito fue rechazado: ¿qué debería mejorar para ser aprobado?
- Un paciente de alto riesgo: ¿qué factores modificar para reducir el riesgo?

**Consideraciones al generar contrafactuales:**
- Rangos dinámicos factibles (no se puede tener un ingreso negativo).
- Variables inmutables (la edad no se puede reducir).
- Objetivos variados: solución más cercana, solución que cambie menos variables, etc.

---

##### 7. Implementación del Sistema Contrafactual

**Dataset: Give Me Some Credit (Kaggle)**

```python
import kagglehub
path = kagglehub.competition_download('GiveMeSomeCredit')
```

Dataset de riesgo crediticio con variables como:
- `RevolvingUtilizationOfUnsecuredLines`: utilización de crédito rotativo
- `age`: edad
- `NumberOfTime30-59DaysPastDueNotWorse`: pagos atrasados 30-59 días
- `DebtRatio`: ratio de deuda
- `MonthlyIncome`: ingreso mensual
- `NumberOfTimes90DaysLate`: pagos atrasados 90+ días
- `NumberOfDependents`: número de dependientes

**Entrenamiento del modelo:**

```python
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",   # Importante: dataset desbalanceado
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
# AUC-ROC: 0.849
# Recall Alto Riesgo: 0.73 (detecta 73% de los realmente en mora)
# Precision Alto Riesgo: 0.22 (muchos falsos positivos, esperado con desbalance)
```

**Métricas del modelo:**

```
              precision  recall  f1-score
Low Risk        0.98     0.80     0.88
High Risk       0.22     0.73     0.34
AUC-ROC: 0.849
```

El modelo detecta bien los casos de alto riesgo (recall 0.73) pero con muchos falsos positivos debido al desbalance de clases.

**Función generadora de contrafactuales (búsqueda greedy):**

```python
def generate_counterfactual(
    model,
    instance: np.ndarray,
    feature_names: list,
    desired_class: int,
    n_iter: int = 5_000,
    step_scale: float = 0.05,
    feature_ranges: dict = None,
    inmutable_features = {},
):
```

**Lógica paso a paso:**

1. **Inicialización:** copia la instancia original como punto de partida del contrafactual.
2. **Identificación de características mutables:** excluye las características inmutables (`inmutable_features`).
3. **Iteraciones (5000 por defecto):**
   - Selecciona aleatoriamente una característica mutable.
   - Genera una perturbación aleatoria en la **dirección correcta** (según `direction_map`).
   - Aplica la perturbación a una copia candidata.
   - Recorta el candidato a los rangos permitidos (`feature_ranges`).
   - Calcula la probabilidad del candidato para la clase deseada.
   - **Acepta el cambio solo si mejora** la probabilidad hacia la clase deseada (búsqueda greedy).
   - Para cuando el modelo clasifica el contrafactual en la clase deseada.
4. **Retorno:** contrafactual encontrado y diccionario de cambios por variable.

**Direction Map (dirección favorable para reducir riesgo):**

```python
direction_map = {
    0: -1,   # RevolvingUtilization → reducir (menos uso de crédito)
    1: +1,   # age → aumentar (no se puede controlar)
    2: -1,   # late30_59 → reducir (menos atrasos)
    3: -1,   # DebtRatio → reducir (menos deuda)
    4: +1,   # income → aumentar (más ingresos)
    5: +1,   # open credit lines → ligeramente más
    6: -1,   # late90 → reducir
    7:  0,   # real estate loans → sin dirección clara
    8: -1,   # late60_89 → reducir
    9: -1,   # dependents → reducir carga financiera
}
```

**Rangos permitidos:**

```python
RANGES = {
    "RevolvingUtilizationOfUnsecuredLines": (0, 1),
    "age": (18, 90),
    "NumberOfTime30-59DaysPastDueNotWorse": (0, 20),
    "DebtRatio": (0, 1),
    "MonthlyIncome": (1_000, 1_000_000),
    ...
}
```

**Experimento 1 — Sin restricciones de inmutabilidad:**

```python
high_risk_idx = np.where((y_test == 1) & (y_pred == 1))[0][:5]
instances = X_test[high_risk_idx]

for inst in instances:
    cf, changes = generate_counterfactual(model, inst, FEATURE_NAMES, desired_class=0, feature_ranges=RANGES)
```

Resultados: 4 de 5 instancias flippan exitosamente de Alto Riesgo a Bajo Riesgo.

**Gráfico: Variación normalizada por característica (Plot 4)**

```python
norm_delta = (cf_0 - instance_0) / feature_ranges_span
```

El gráfico muestra el cambio necesario en cada variable expresado como porcentaje del rango total de esa variable. En el primer experimento, la variable `Age` domina con un cambio de +34.1% (el algoritmo "usa" el envejecimiento para reducir el riesgo), lo cual no es realista.

**Experimento 2 — Con edad como variable inmutable:**

```python
cf, changes = generate_counterfactual(
    model, inst, FEATURE_NAMES,
    desired_class=0,
    feature_ranges=RANGES,
    inmutable_features={"age"}
)
```

Con la restricción de que la edad no puede cambiar, el algoritmo encuentra cambios en variables accionables como:
- Reducción leve de atrasos 30-59 días (-0.1%)
- Ligero aumento de ingresos (+0.1%)
- La clase no siempre se logra flipar (solo 2 de 5 logran cambiar a Bajo Riesgo).

**Gráfico: Contribución de cada cambio a la probabilidad (Plot 5 - Waterfall)**

```python
contributions = []
for i in range(len(FEATURE_NAMES)):
    modified = instance_0.copy()
    modified[i] = cf_0[i]
    p = model.predict_proba(modified.reshape(1,-1))[0, 1]
    contributions.append(prob_orig - p)  # Positivo = reduce riesgo
```

Mediante ablación (aplicar un cambio a la vez), se cuantifica cuánto contribuye cada variable al cambio total en probabilidad:
- `Revolving Util.`: +0.267 (la mayor reducción de riesgo)
- `Monthly Income`: +0.007
- `Debt Ratio`: +0.004

La reducción de la utilización de crédito rotativo es por lejos el cambio más impactante (probabilidad original 77.47% → contrafactual 48.73%).

**Gráfico: Distribución de probabilidades (Plot 7)**

```python
probs_low = y_prob[y_test == 0]
probs_high = y_prob[y_test == 1]
ax.axvline(prob_orig, color=C_BAD, ...)    # Instancia original: 77.47%
ax.axvline(prob_cf,   color=C_GOOD, ...)   # Contrafactual: 48.73%
```

Muestra histogramas de probabilidades predichas para bajo y alto riesgo, con líneas verticales indicando la posición de la instancia original y su contrafactual.

**Gráfico: Mapa de calor multi-instancia (Plot 6)**

```python
heatmap_data = np.zeros((5, len(FEATURE_NAMES)))
for j, (inst, cf) in enumerate(zip(instances, cfs_list)):
    for i in range(len(FEATURE_NAMES)):
        span = X[:, i].max() - X[:, i].min() + 1e-8
        heatmap_data[j, i] = abs(cf[i] - inst[i]) / span
```

Para las 5 instancias de alto riesgo, el heatmap muestra qué tan grande fue el cambio necesario en cada variable (en escala normalizada). Se observa que principalmente `Open Credit Lines` fue la variable que más varió en la mayoría de instancias.

**Función auxiliar: Inferencia automática de dirección:**

```python
def infer_feature_directions(model, X, feature_names, step_fraction=0.05):
```

Calcula automáticamente si aumentar cada variable reduce o aumenta el riesgo, perturbando levemente el dataset completo:

```
RevolvingUtilizationOfUnsecuredLines  Δrisk = +0.1886  (aumentar = más riesgo)
age                                   Δrisk = -0.0094  (aumentar = menos riesgo)
NumberOfTime30-59DaysPastDueNotWorse  Δrisk = +0.2804  (aumentar = más riesgo)
MonthlyIncome                         Δrisk = -0.0262  (aumentar = menos riesgo)
NumberOfTimes90DaysLate               Δrisk = +0.4060  (el más dañino)
```

**Variables que mejoran al aumentar:** `MonthlyIncome`, `age`  
**Variables que empeoran al aumentar:** todas las demás (principalmente variables de mora)

---

## Flujo General de la Unidad

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIDAD 8 - PARTE 1                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
   │   CLASE 19   │    │   CLASE 20   │    │    CLASE 23      │
   │    LASSO     │    │    PCA       │    │ EXPLICABILIDAD   │
   └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘
          │                   │                     │
          ▼                   ▼                     ▼
   Selección de         Reducción de          SHAP / LIME /
   Variables            Dimensión             Contrafactual
   (antes del           (antes del            (después del
   modelado)            modelado)             modelado)
```

Los tres temas abordan distintas fases del ciclo de ML:
- **LASSO** actúa en el preprocesamiento/modelado: selecciona qué variables incluir.
- **PCA** actúa en el preprocesamiento: transforma las variables para reducir dimensión.
- **Explicabilidad (SHAP/LIME/Contrafactual)** actúa en el post-procesamiento: interpreta las decisiones del modelo ya entrenado.

Todos comparten la biblioteca **scikit-learn** como base y se complementan en un pipeline completo de ML responsable.

---

## Conceptos Aprendidos

### LASSO y Regularización
- Formulación matemática de LASSO y diferencia con Ridge (L1 vs L2).
- Por qué la norma L1 produce soluciones dispersas (coeficientes exactamente cero).
- Interpretación del parámetro de regularización λ y cómo calibrarlo por validación cruzada.
- LASSO Path como herramienta diagnóstica del orden de selección de variables.
- Elastic Net como solución al problema de variables correlacionadas.

### PCA (Análisis de Componentes Principales)
- Objetivo de PCA: encontrar proyecciones que maximicen la varianza.
- Relación entre PCA y la descomposición en valores propios de la matriz de covarianza.
- El primer componente principal es el eigenvector del mayor eigenvalue.
- Implementación en sklearn y cómo interpretar `explained_variance_`.
- Aplicación a datos de alta dimensión (imágenes) con EigenFaces.
- Reconstrucción aproximada de datos desde el espacio reducido.
- Estrategias para elegir el número de componentes (scree plot, varianza acumulada).

### Explicabilidad de ML (XAI)
- Motivación regulatoria y ética de la explicabilidad.
- Diferencia entre explicabilidad global y local.
- Fundamento teórico de SHAP en teoría de juegos de Shapley.
- Cómo interpretar gráficos beeswarm, waterfall y force plot de SHAP.
- Principio de LIME: modelo sustituto simple en vecindad local.
- Concepto de contrafactual: ¿qué cambiar para obtener otra decisión?
- Implementación de búsqueda greedy de contrafactuales con restricciones.
- Importancia de definir variables inmutables en contrafactuales.

---

## Conclusiones

1. **LASSO es una herramienta poderosa de selección automática de variables** cuando se trabaja con datos de alta dimensión, pero sus limitaciones (correlación entre variables, garantías estadísticas) sugieren complementarlo con Elastic Net en escenarios complejos.

2. **PCA es la técnica de reducción de dimensionalidad más robusta y versátil**, con fundamento matemático sólido. La clave está en centrar los datos correctamente y seleccionar el número de componentes con cuidado. El ejemplo de EigenFaces demuestra que incluso reduciendo de 1850 a 150 dimensiones se mantiene suficiente información para clasificación.

3. **La explicabilidad no es opcional** en aplicaciones críticas. SHAP ofrece la explicación más rigurosa (global y local), LIME es más flexible aunque menos estable, y los contrafactuales son los más actionable (indican qué hacer para cambiar la decisión).

4. **Los tres temas son complementarios:** LASSO/PCA mejoran los modelos antes de entrenarlos, mientras que SHAP/LIME/Contrafactual los hacen comprensibles después. Un pipeline responsable de ML debería incorporar los tres.

5. **Las variables inmutables son críticas en contrafactuales:** ignorarlas produce recomendaciones inútiles (como "envejece 15 años para reducir tu riesgo crediticio"). La generación de contrafactuales realistas requiere un profundo conocimiento del dominio.

---

## Recursos y Referencias

### Bibliotecas Python Utilizadas

| Biblioteca | Versión | Uso |
|------------|---------|-----|
| `scikit-learn` | — | LASSO, ElasticNet, PCA, RandomForest, SVM, GridSearchCV |
| `shap` | — | SHAP values, TreeExplainer, beeswarm, waterfall, force_plot |
| `lime` | — | LimeTabularExplainer |
| `numpy` | — | Operaciones matriciales y vectoriales |
| `matplotlib` | — | Visualizaciones |
| `seaborn` | — | Heatmaps |
| `pandas` | — | Manipulación de datos |
| `kagglehub` | — | Descarga del dataset Give Me Some Credit |

### Datasets

- **Breast Cancer Wisconsin (Diagnostic):** `sklearn.datasets.load_breast_cancer()` — 569 muestras, 30 características, clasificación binaria.
- **Diabetes:** `sklearn.datasets.load_diabetes()` — datos de progresión de diabetes, regresión.
- **LFW (Labeled Faces in the Wild):** `sklearn.datasets.fetch_lfw_people()` — reconocimiento de identidad en imágenes de rostros.
- **Give Me Some Credit (Kaggle):** dataset de riesgo crediticio con ~150,000 observaciones y 10 variables financieras.

### Referencias Bibliográficas

- Tibshirani, R. (1996). *Regression shrinkage and selection via the lasso*. Journal of the Royal Statistical Society, Series B.
- Lundberg, S.M. & Lee, S.I. (2017). *A unified approach to interpreting model predictions* (SHAP). NeurIPS 2017.
- Ribeiro, M.T., Singh, S. & Guestrin, C. (2016). *"Why should I trust you?": Explaining the predictions of any classifier* (LIME). KDD 2016.
- Wold, S., Esbensen, K. & Geladi, P. (1987). *Principal component analysis*. Chemometrics and Intelligent Laboratory Systems.

### Fuente del Curso

**Introducción al Machine Learning 2025** — Julián D. Arias Londoño  
Disponible en: `https://jdariasl.github.io/Intro_ML_2025/`
