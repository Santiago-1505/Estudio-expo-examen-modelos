# Unidad 1: Introducción al Machine Learning

---

## Introducción

Esta unidad constituye el punto de partida del curso de **Introducción al Machine Learning**. Su propósito es establecer los fundamentos conceptuales, terminológicos y metodológicos que guiarán todo el aprendizaje posterior. A lo largo de tres clases progresivas, se construye una comprensión sólida que va desde la definición misma del Machine Learning, pasando por los modelos básicos de aprendizaje supervisado (regresión lineal y regresión logística), hasta los clasificadores basados en funciones de densidad gaussianas.

La unidad responde tres preguntas fundamentales:
1. ¿Qué es el Machine Learning y por qué importa?
2. ¿Cómo se construyen y optimizan los modelos más básicos?
3. ¿Cómo se puede abordar la clasificación desde una perspectiva probabilística?

---

## Contenido de la Unidad

La unidad está compuesta por **tres notebooks** exportados como PDF, cada uno correspondiente a una clase del curso:

| Archivo | Clase | Tema |
|---|---|---|
| `Introducción_al_Machine_Learning.pdf` | Clase 01 | Fundamentos, tipos de aprendizaje y diseño de modelos |
| `Modelos_básicos_de_aprendizaje.pdf` | Clase 02 | Regresión lineal múltiple y regresión logística |
| `Modelos_de_clasificación_empleando_funciones_de_densidad_Gausianas.pdf` | Clase 03 | Clasificadores gaussianos y modelos generativos |

---

## Estructura de Archivos

---

### Archivo 1: `Introducción_al_Machine_Learning.pdf`

#### Propósito

Este notebook introductorio establece el marco conceptual del Machine Learning: qué es, de dónde vienen los modelos, qué tipos de aprendizaje existen, qué retos se enfrentan y cómo se diseña formalmente un sistema de ML.

---

#### Contenido y Funcionamiento

##### 1. Definición de Machine Learning

El Machine Learning (ML) se define como una rama de la inteligencia artificial enfocada en el estudio de sistemas que pueden **"aprender" a partir de datos**. Formalmente, es un conjunto de métodos capaces de:

- Detectar automáticamente patrones en un conjunto de datos.
- Aprender dichos patrones.
- Usarlos para **predecir datos futuros** o tomar decisiones bajo incertidumbre.

Se presentan aplicaciones reales como motivación:
- **Reconocimiento de rostros**: sistemas de verificación de asistencia.
- **Reconocimiento de objetos**: detección de peatones, vehículos y señales en tiempo real (visión computacional).
- **Reconocimiento de voz**: asistentes virtuales.
- **Traducción automática**: sistemas multilingües.
- **IA Generativa**: modelos como Stable Diffusion que generan imágenes a partir de texto, lo que también constituye ML.

---

##### 2. Modelos a partir de datos

Se contrastan dos grandes paradigmas para construir modelos:

**Aproximación Mecanicista:**
Se conocen los principios físicos o las leyes que gobiernan la interacción entre las variables del problema. El modelo se deduce analíticamente.

**Aproximación en Machine Learning:**
Se desconocen las leyes que rigen las relaciones entre variables. En su lugar, se **descubren correlaciones a partir de los datos mismos**, usando un enfoque correlacional.

El flujo general de ML se puede representar como:

```
[Datos de ejemplo: X → y]
        ↓
[Creación del Modelo / Calibración]
        ↓
[Modelo entrenado]
        ↓
[Nuevos datos X_new → Y_predicted]
```

**Conceptos clave del diagrama:**
- **Template Model (Plantilla)**: Es la estructura matemática del modelo elegida a priori (por ejemplo, un árbol de decisión, una red neuronal, una regresión polinomial). Corresponde a los algoritmos disponibles en librerías como scikit-learn.
- **Model Calibration (Calibración)**: Es el proceso de ajustar los parámetros internos de la plantilla usando los datos de entrenamiento.
- **Model (Modelo calibrado)**: El resultado final, listo para hacer predicciones.

> **Nota importante:** La calibración depende completamente de la muestra de datos disponible. Con más datos, menor variación en los parámetros obtenidos.

Se ilustra esto con un ejemplo en Python usando `scipy.stats`:

```python
from scipy import stats
d1 = stats.norm(loc=10, scale=2)   # Distribución normal: clase 1, media=10, std=2
d2 = stats.norm(loc=17, scale=3)   # Distribución normal: clase 2, media=17, std=3
```

Se busca la **frontera óptima de separación** entre dos poblaciones gaussianas: el punto donde las dos distribuciones se intersectan. El resultado fue `frontera óptima en 13.15`.

Luego se demuestra que cuando se trabaja con **muestras reales** (no las distribuciones teóricas completas), la frontera obtenida varía dependiendo de:
1. La **plantilla de modelo** utilizada (árbol de decisión vs. Naive Bayes gaussiano).
2. El **tamaño de la muestra** (con más datos hay menos variación).

---

##### 3. ¿Qué implica resolver un problema de ML?

Se establece una distinción fundamental entre dos roles:

| Rol | Enfoque |
|---|---|
| **Diseñador de algoritmos de ML** | Cómo se genera/construye un nuevo modelo |
| **Usuario de algoritmos de ML** | Cómo calibrar modelos existentes sobre datos concretos |

El flujo de trabajo típico del **usuario** de ML incluye cinco etapas:
1. **Get Data** – Obtener los datos del problema.
2. **Clean, Prepare & Manipulate Data** – Limpiar y transformar los datos.
3. **Train Model** – Entrenar el modelo seleccionado.
4. **Test Data** – Evaluar el modelo en datos no vistos.
5. **Improve** – Iterar y mejorar el modelo.

Cuando las soluciones convencionales no son suficientes, el usuario debe asumir el rol de diseñador y proponer modificaciones o modelos nuevos.

---

##### 4. Tipos de Aprendizaje de Máquina

Se presentan cinco paradigmas de aprendizaje, organizados en una tabla comparativa:

| Tipo | Definición | Objetivo | Nota |
|---|---|---|---|
| **Supervised Learning** | La tarea contiene respuestas correctas para todos los ejemplos | Replicar las respuestas correctas | El más común |
| **Unsupervised Learning** | Los ejemplos no tienen salidas correctas; se busca estructura | Encontrar la estructura de los datos | — |
| **Semi-supervised Learning** | Es supervisado pero para algunos ejemplos no hay respuesta | Replicar las respuestas correctas | — |
| **Self-supervised Learning** | Se usa parte de la información del ejemplo como respuesta | Encontrar estructura, pre-entrenamiento | El más reciente |
| **Reinforcement Learning** | Un agente toma decisiones según recompensas obtenidas | Encontrar la ruta de mayor recompensa | — |

**Aprendizaje Supervisado — Formalización matemática:**

El objetivo es aprender un mapeo desde entradas **x** = [x₁, x₂, ..., x_d] a salidas *y*, dado un conjunto de entrenamiento:

$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$$

Donde:
- **x_i** es un vector *d*-dimensional de **características** (*features*) del ejemplo *i*.
- **y_i** es la etiqueta o valor de salida del ejemplo *i*.
- *N* es el número total de muestras de entrenamiento.

**Aprendizaje No Supervisado — Formalización:**

Solo se dispone de: $\mathcal{D} = \{(\mathbf{x}_i)\}_{i=1}^{N}$

El objetivo es descubrir "patrones interesantes" (también llamado **Descubrimiento de Conocimiento**). Las técnicas incluyen clustering, reducción de dimensionalidad, etc.

---

##### 5. Tipos de problemas supervisados

Dependiendo del tipo de variable de salida *y_i*, los problemas se clasifican en:

| Tipo de salida | Nombre del problema |
|---|---|
| Valores **discretos** (categorías) | **Clasificación** |
| Valores **reales** (continuos) | **Regresión** |

Se ilustra un problema de **clasificación multiclase** usando el dataset Iris de scikit-learn (3 clases), visualizando dos características: `sepal width` vs. `petal length`.

Se ilustra también un **problema de regresión**, ajustando una función cuadrática con ruido aleatorio, donde se puede ver la diferencia entre los datos reales (con ruido) y la función subyacente.

---

##### 6. Retos del aprendizaje de máquina

Los principales desafíos mencionados en el contexto de clasificación son:

- **Variabilidad intra-clase**: Ejemplos de la misma clase pueden verse muy diferentes entre sí (e.g., una misma persona fotografiada desde distintos ángulos o con distintas expresiones).
- **Similitud inter-clase**: Ejemplos de clases distintas pueden parecerse mucho entre sí (e.g., dos personas diferentes que se ven similares).

Estos retos hacen que el problema de clasificación no sea trivial.

---

##### 7. Diseño de un modelo de ML

Se establecen las **tres preguntas fundamentales** al enfrentar un problema de ML:

1. **¿Cuál es el modelo?**
   Definir la estructura matemática que representará la solución. Todo modelo tiene parámetros ajustables. Ejemplo: `f(x) = wᵀx` (modelo lineal).

2. **¿Cuál es el criterio de ajuste (función de costo)?**
   Expresar matemáticamente qué queremos que el modelo logre con los datos de entrenamiento. La función de costo más común para regresión es el **Error Cuadrático Medio (ECM)**:
   $$\arg\min_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2$$

3. **¿Cuál es el algoritmo de optimización?**
   Encontrar los parámetros del modelo que minimizan la función de costo. El ejemplo clásico es el **Gradiente Descendente**.

Se definen además **tres niveles de abstracción** para trabajar con ML:

| Nivel | Descripción |
|---|---|
| **Alto nivel** | Seleccionar una plantilla de modelo de una librería, ajustar sobre los datos y validar. No se modifica la función de costo ni el optimizador. |
| **Nivel intermedio** | Seleccionar un modelo pero modificar la función de costo. Se usa cálculo simbólico para la optimización. |
| **Bajo nivel** | Modificar el modelo, la función de costo y/o el algoritmo de optimización. Necesario en problemas avanzados. |

---

### Archivo 2: `Modelos_básicos_de_aprendizaje.pdf`

#### Propósito

Este notebook profundiza en los dos modelos supervisados más fundamentales del ML: la **regresión lineal/polinomial múltiple** para problemas de regresión y la **regresión logística** para problemas de clasificación. Se demuestra cómo implementar estos modelos en los tres niveles de abstracción mencionados en la clase anterior.

---

#### Contenido y Funcionamiento

##### 1. Regresión Múltiple — El Modelo

Se parte de una función polinomial general:

$$f(\mathbf{x}, \mathbf{w}) = w_0 + w_1 x + w_2 x^2 + \cdots + w_M x^M = \sum_{j=0}^{M} w_j x^j$$

Donde:
- **M** es el orden del polinomio (hiperparámetro que define la complejidad del modelo).
- **w** = [w₀, w₁, ..., w_M] son los parámetros a ajustar (pesos).

**Función de costo — Error Cuadrático Medio (ECM):**

$$E(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^{N} \{f(x_i, \mathbf{w}) - y_i\}^2$$

Esta función mide la distancia perpendicular promedio al cuadrado entre las predicciones del modelo y los valores reales. Minimizarla equivale a encontrar la recta (o curva) que mejor se ajusta a los datos.

---

##### 2. Implementación en Alto Nivel (Caja Negra)

Se genera un dataset artificial 3D basado en el polinomio real: `y = 3x₁ + 7x₂ - 2`, con ruido gaussiano añadido.

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(np.c_[x1, x2], y2)
reg.predict(np.array([8, 5]).reshape(1, -1))
# Resultado: array([[55.56288392]])

print(reg.coef_)      # [[2.82755481  7.00129031]]
print(reg.intercept_) # [-2.06400611]
```

**Explicación:**
- `LinearRegression()` de scikit-learn encapsula tanto el modelo lineal como la función ECM y el algoritmo de optimización (solución analítica por mínimos cuadrados).
- `.fit(X, y)` realiza el entrenamiento: estima los pesos óptimos.
- `.predict(X_new)` aplica el modelo calibrado a nuevos datos.
- Los coeficientes obtenidos (~2.83, ~7.00, ~-2.06) se aproximan mucho a los valores reales (3, 7, -2), demostrando que el modelo aprendió correctamente.

---

##### 3. Función ECM Personalizada (Nivel Intermedio)

Se define manualmente la función de costo:

```python
def ECM(w, x, y):
    w = w.reshape(1, -1)
    x = np.c_[x, np.ones((x1.shape[0], 1))]  # Añade columna de unos para el término independiente
    return np.mean((np.dot(w, x.T).T - y)**2)  # Error cuadrático medio
```

**Parámetros:**
- `w`: Vector de pesos (incluye el término independiente al final).
- `x`: Matriz de características de entrenamiento.
- `y`: Vector de valores objetivo.

**Retorno:** Valor escalar del ECM, que cuantifica el error del modelo con los pesos actuales.

Luego se usa `scipy.optimize.minimize` para minimizar la función ECM sin necesidad de derivar manualmente:

```python
from scipy.optimize import minimize
r1 = minimize(lambda w: ECM(w, np.c_[x1, x2], y2), np.random.random(size=3))
# r1.x ≈ [2.82755477, 7.0012903, -2.06400599]
```

Los resultados son idénticos a los de scikit-learn, validando la implementación.

---

##### 4. Gradiente Descendente — Nivel Bajo

El algoritmo de Gradiente Descendente actualiza iterativamente los parámetros en dirección contraria al gradiente de la función de costo:

$$w_j[\tau] = w_j[\tau-1] - \eta \frac{\partial E(\mathbf{w})}{\partial w_j}$$

Donde:
- **τ** es la iteración actual.
- **η** (eta) es la **tasa de aprendizaje** (*learning rate*): controla el tamaño del paso en cada iteración.

La derivada de la función ECM respecto a w_j es:

$$\frac{\partial E(\mathbf{w})}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (f(x_i, \mathbf{w}) - y_i) x_{ij}$$

La regla de actualización completa queda:

$$w_j[\tau] = w_j[\tau-1] - \frac{\eta}{N} \sum_{i=1}^{N} \left(\sum_{k=0}^{d} w_k[\tau-1] x_{ik} - y_i \right) x_{ij}$$

**Implementación en Python:**

```python
MaxIter = 1000000
w = np.ones(3).reshape(3, 1)   # Inicialización de pesos
eta = 0.0001                    # Tasa de aprendizaje
N = len(x1)
X = np.array([x1, x2, np.ones((100,1))]).reshape(3, 100)  # Matriz extendida (con unos)

for i in range(MaxIter):
    tem = np.dot(X.T, w)             # Predicciones: X^T · w
    tem2 = tem - np.array(y2)        # Error: predicciones - valores reales
    Error[i] = np.sum(tem2**2) / (2*N)  # ECM
    tem = np.dot(X, tem2)            # Gradiente: X · error
    w = w - eta * tem / N            # Actualización de pesos
```

**Resultado:** Tras 1,000,000 iteraciones, los pesos convergen a los mismos valores (~2.83, ~7.00, ~-2.06), con un ECM de ~41.48.

La gráfica de convergencia muestra cómo el error cae rápidamente en las primeras iteraciones y luego se estabiliza, lo cual es el comportamiento esperado de un problema convexo.

**Nota sobre convexidad:** El ECM de la regresión lineal es una función convexa, por lo que cualquier algoritmo basado en gradiente converge al mínimo global. Para funciones no convexas (como en redes neuronales profundas), se requieren estrategias más sofisticadas (momento, tasas adaptativas).

---

##### 5. Regresión Logística — El Modelo

Para problemas de clasificación binaria (y ∈ {0, 1}), la idea es encontrar una **frontera de separación lineal** en el espacio de características:

$$f(\mathbf{x}) = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_d x_d$$

La clasificación se realiza como:

$$\text{Clase} = \begin{cases} 1 & \text{si } w_0 + w_1x_1 + \cdots + w_dx_d \geq 0 \\ 0 & \text{en otro caso} \end{cases}$$

**Problema con la función signo (sgn):** Es discontinua y no diferenciable, por lo que no puede ser usada directamente en algoritmos basados en gradiente.

**Solución — Función Sigmoide:**

$$g(u) = \frac{e^u}{1 + e^u}$$

Propiedades:
- Es continua y diferenciable en todo su dominio.
- Toma valores en el intervalo (0, 1).
- Se comporta asióticamente como la función signo.
- Puede interpretarse como una **probabilidad**: g(f(x)) ≈ P(y=1 | x).

**Función de costo — Entropía Cruzada (Cross-Entropy):**

$$J(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^{N} \left[ -y_i \log(g(f(\mathbf{x}))) - (1-y_i) \log(1 - g(f(\mathbf{x}))) \right]$$

Esta función penaliza fuertemente las clasificaciones incorrectas y tiene propiedades convenientes para la optimización mediante gradiente.

**Derivada de J respecto a w_j:**

$$\frac{\partial J(\mathbf{w})}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (g(f(x_i, \mathbf{w})) - y_i) x_{ij}$$

La única diferencia respecto a la regresión lineal es la presencia de la función sigmoide g(·). Por eso se puede reutilizar casi el mismo algoritmo de gradiente descendente.

---

##### 6. Implementación de Regresión Logística

```python
def sigmoide(u):
    g = np.exp(u) / (1 + np.exp(u))
    return g

def Gradiente(X2, y2, MaxIter=10000, eta=0.001):
    w = np.ones(3).reshape(3, 1)
    N = len(y2)
    Xent = np.concatenate((X2, np.ones((100, 1))), axis=1)  # Añade columna de unos
    
    for i in range(MaxIter):
        tem = np.dot(Xent, w)
        tem2 = sigmoide(tem.T) - np.array(y2)   # Error con función sigmoide
        Error[i] = np.sum(abs(tem2)) / N
        tem = np.dot(Xent.T, tem2.T)
        w = w - eta * tem / N                   # Actualización de pesos
    
    return w, Error
```

Se aplica al dataset Iris (primeras 2 clases, primeras 2 características) y se visualiza cómo la frontera de clasificación evoluciona con el número de iteraciones.

**Visualización del espacio de búsqueda:** Se grafica la función de costo (entropía cruzada) como un mapa de calor en el espacio de parámetros (w₁ vs. w₂), mostrando que la función es convexa con un mínimo global bien definido.

---

### Archivo 3: `Modelos_de_clasificación_empleando_funciones_de_densidad_Gausianas.pdf`

#### Propósito

Este notebook presenta un enfoque alternativo al de la regresión logística para resolver problemas de clasificación: en lugar de encontrar una frontera de separación directamente, se modelan las **funciones de densidad de probabilidad (fdp)** de cada clase por separado y se clasifica según qué clase tiene mayor probabilidad de haber generado el nuevo dato.

---

#### Contenido y Funcionamiento

##### 1. Motivación y Enfoque Probabilístico

La idea central es: en lugar de buscar una función discriminante, modelar cada clase como una distribución de probabilidad. Para clasificar una nueva muestra, se evalúa en cada distribución y se asigna a la clase con mayor probabilidad.

Se usa el dataset Iris (clases 0 y 1, características 3 y 4: longitud y ancho del pétalo), donde las dos clases son claramente separables.

---

##### 2. Distribución Gaussiana Multivariada

Se asume que cada clase sigue una **distribución normal multivariada**:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left[-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right]$$

**Parámetros:**
- **μ** = {μ₁, μ₂, ..., μ_d}: Vector de medias (media de cada característica).
- **Σ**: Matriz de covarianza, que captura varianzas individuales y correlaciones entre características:

$$\Sigma = \begin{bmatrix} \sigma_1^2 & \rho_{1,2}\sigma_1\sigma_2 & \cdots & \rho_{1,d}\sigma_1\sigma_d \\ \rho_{2,1}\sigma_2\sigma_1 & \sigma_2^2 & \cdots & \rho_{2,d}\sigma_2\sigma_d \\ \vdots & \vdots & \ddots & \vdots \\ \rho_{d,1}\sigma_d\sigma_1 & \cdots & \cdots & \sigma_d^2 \end{bmatrix}$$

---

##### 3. Estimación de Parámetros por Máxima Verosimilitud

Dado un conjunto de *N* muestras i.i.d. **x_i** ~ N(μ, Σ), los estimadores de máxima verosimilitud son:

**Media muestral:**
$$\hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i$$

**Matriz de covarianza muestral:**
$$\hat{\Sigma} = \frac{1}{N} \sum_{i=1}^{N} (\mathbf{x}_i - \hat{\mu})(\mathbf{x}_i - \hat{\mu})^T$$

Estos son los parámetros que se estiman durante la fase de **entrenamiento**.

---

##### 4. Procedimiento de Entrenamiento

1. Tomar el conjunto de entrenamiento y **separarlo en C subconjuntos**, uno por clase.
2. Para cada subconjunto, **estimar μ̂_c y Σ̂_c** (media y covarianza de la clase c).
3. Con esos parámetros, **definir la fdp Gaussiana p_c(x)** para cada clase c.

---

##### 5. Procedimiento de Clasificación

Para clasificar una nueva muestra **x***:

1. **Evaluar x* en cada fdp p_c(x*)**, para c = 1, ..., C.
2. **Asignar x* a la clase con mayor probabilidad:**

$$C^* = \arg\max_k \; p_k(\mathbf{x}^*)$$

**Implementación de la distribución gaussiana:**

```python
def DistribucionGaussiana(X, Mu, Sigma):
    d = X.shape[1]
    SigmaInversa = np.linalg.inv(np.array(Sigma))  # Inversa de la matriz de covarianza
    PrimerTermino = (1 / (((2 * math.pi)**(d/2)) * math.sqrt(np.linalg.det(Sigma))))
    
    primerDot = np.dot((X - Mu), SigmaInversa)
    segundoDot = np.dot(primerDot, (X - Mu).T)
    Exponencial = math.exp(-0.5 * segundoDot)
    
    Probabilidad = PrimerTermino * Exponencial
    return Probabilidad
```

**Explicación paso a paso:**
- `d`: número de características (dimensionalidad).
- `SigmaInversa`: inversa de Σ, necesaria para calcular la distancia de Mahalanobis.
- `PrimerTermino`: constante de normalización de la Gaussiana.
- `primerDot` y `segundoDot`: calculan la forma cuadrática `(x-μ)ᵀ Σ⁻¹ (x-μ)`, que mide la distancia de Mahalanobis entre x y μ.
- `Exponencial`: aplica la función exponencial negativa.
- `Probabilidad`: densidad de probabilidad de que x provenga de esta distribución.

---

##### 6. Los Tres Casos del Modelo Gaussiano

La forma de la **frontera de decisión** depende de cómo se trate la matriz de covarianza:

**Caso 1 — Σ = σ²I (Matriz identidad escalada):**
- Todos las características son **estadísticamente independientes** e **igual varianza**.
- Cada clase se representa como una **hiperesfera** alrededor de su media.
- La frontera de decisión es **lineal**.
- El clasificador resultante es el **Análisis Discriminante Lineal (LDA)** bajo esta suposición.

**Caso 2 — Σ diagonal:**
- Las características son **estadísticamente independientes** pero con **varianzas distintas**.
- Cada clase se representa como un **elipsoide alineado con los ejes**.
- Este caso corresponde al **Naïve Bayes Classifier** (Clasificador Bayesiano Ingenuo).
- La frontera puede ser cuadrática.

**Caso 3 — Σ completa (caso general):**
- Las características pueden estar **correlacionadas** y con varianzas distintas.
- Cada clase se representa como un **elipsoide en cualquier orientación**.
- Es el caso más **flexible** pero requiere más parámetros y más datos.
- Corresponde al **Análisis Discriminante Cuadrático (QDA)**.

---

##### 7. Visualización de las Fronteras con scikit-learn

Se comparan los tres clasificadores sobre el mismo dataset:

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    # Caso 1
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis # Caso 3
from sklearn.naive_bayes import GaussianNB                              # Caso 2

clf1 = LinearDiscriminantAnalysis()
clf2 = GaussianNB()
clf3 = QuadraticDiscriminantAnalysis()
```

La gráfica resultante muestra claramente las diferencias entre las fronteras:
- **Verde (LDA)**: Frontera lineal — separa de forma simple.
- **Azul (Naive Bayes)**: Frontera cuadrática suave.
- **Naranja (QDA)**: Frontera cuadrática más flexible, puede ser cóncava.

---

##### 8. Nota Final: Modelos Generativos vs. Discriminativos

Se hace una distinción conceptual fundamental:

| Tipo | Modelo | Enfoque |
|---|---|---|
| **Generativo** | Clasificador Gaussiano (LDA, QDA, Naive Bayes) | Modela cada clase por separado con una fdp; puede generar nuevas muestras de cada clase. |
| **Discriminativo** | Regresión Logística | Entrena con muestras de todas las clases simultáneamente; aprende directamente la frontera de separación. |

> Los modelos generativos aprenden P(x|clase), los discriminativos aprenden P(clase|x) directamente.

---

## Flujo General de la Unidad

```
Clase 01: Fundamentos
    ├── ¿Qué es ML? → Definición y aplicaciones
    ├── Paradigmas: Mecanicista vs. ML
    ├── Tipos de aprendizaje (Supervisado, No supervisado, etc.)
    ├── Tipos de problemas: Clasificación vs. Regresión
    ├── Retos: Variabilidad intra-clase, similitud inter-clase
    └── Diseño de un modelo: Modelo + Criterio + Algoritmo
            ↓
Clase 02: Modelos Básicos
    ├── Regresión Lineal Múltiple
    │       ├── Alto nivel: sklearn.LinearRegression
    │       ├── Nivel intermedio: scipy.minimize + ECM personalizado
    │       └── Bajo nivel: Gradiente Descendente manual
    └── Regresión Logística
            ├── Función sigmoide como alternativa a signo
            ├── Función de costo: Entropía Cruzada
            └── Gradiente Descendente con sigmoide
            ↓
Clase 03: Clasificadores Gaussianos
    ├── Enfoque probabilístico: modelar P(x|clase)
    ├── Estimación de parámetros: μ̂ y Σ̂ por MLE
    ├── Tres casos: LDA, Naive Bayes, QDA
    └── Comparación: Modelos Generativos vs. Discriminativos
```

---

## Conceptos Aprendidos

### Fundamentos de ML
- Definición formal de Machine Learning como detección automática de patrones.
- Diferencia entre aproximación mecanicista y ML.
- Conceptos de plantilla de modelo, calibración y modelo entrenado.
- Importancia del tamaño de la muestra en la variabilidad del modelo.

### Tipos de Aprendizaje
- Supervised, Unsupervised, Semi-supervised, Self-supervised y Reinforcement Learning.
- Diferencia entre problemas de Clasificación y Regresión.

### Diseño de Modelos
- Las tres preguntas del diseño: modelo, criterio, algoritmo.
- Tres niveles de abstracción: alto, intermedio, bajo.
- Concepto de función de costo o función de pérdida.
- Error Cuadrático Medio (ECM) como criterio de regresión.
- Entropía Cruzada (Cross-Entropy) como criterio de clasificación.

### Optimización
- Gradiente Descendente: formulación, regla de actualización, tasa de aprendizaje.
- Importancia de la convexidad en la garantía de convergencia al mínimo global.
- Diferencia entre problemas convexos y no convexos.

### Regresión Lineal y Logística
- Función polinomial paramétrica y sus pesos.
- Función sigmoide y su ventaja sobre la función signo.
- Similitud matemática entre la actualización de pesos en regresión lineal y logística.

### Clasificación Probabilística Gaussiana
- Distribución normal multivariada: parámetros μ y Σ.
- Estimadores de máxima verosimilitud para μ̂ y Σ̂.
- Distancia de Mahalanobis implícita en la función gaussiana.
- Los tres casos: LDA (Σ=σ²I), Naive Bayes (Σ diagonal), QDA (Σ completa).
- Diferencia entre modelos generativos y discriminativos.

---

## Conclusiones

1. **El ML es un enfoque correlacional**, no causal: aprende relaciones a partir de los datos sin requerir conocimiento explícito de las leyes subyacentes. Esto lo hace poderoso pero también dependiente de la calidad y cantidad de datos.

2. **La elección del modelo importa**: diferentes plantillas (árbol de decisión vs. Naive Bayes) producen fronteras distintas. No hay un modelo universalmente superior; la elección depende del problema y los datos.

3. **La calibración es sensible al tamaño muestral**: con pocos datos, los modelos son variables e inestables. Con más datos, convergen hacia la frontera óptima teórica.

4. **El Gradiente Descendente es el motor del ML moderno**: una regla de actualización simple y elegante que, bien configurada (tasa de aprendizaje apropiada, suficientes iteraciones), permite optimizar una enorme variedad de funciones de costo.

5. **Regresión logística y modelos gaussianos son complementarios**: uno es discriminativo (aprende la frontera directamente), el otro es generativo (modela cada clase). Ambos tienen fortalezas y debilidades según el problema.

6. **Los tres casos del clasificador gaussiano ofrecen un espectro de flexibilidad**: desde LDA (asunciones fuertes, pocos parámetros) hasta QDA (asunciones débiles, más parámetros y más datos necesarios).

---

## Recursos y Referencias

### Librerías Utilizadas

| Librería | Uso |
|---|---|
| `numpy` | Álgebra lineal, operaciones matriciales, generación de datos |
| `scipy.stats` | Distribuciones de probabilidad (norm) |
| `scipy.optimize` | Optimización numérica (minimize) |
| `matplotlib.pyplot` | Visualización de datos y resultados |
| `sklearn.linear_model.LinearRegression` | Regresión lineal (alto nivel) |
| `sklearn.tree.DecisionTreeClassifier` | Árbol de decisión (comparación de plantillas) |
| `sklearn.naive_bayes.GaussianNB` | Naive Bayes Gaussiano |
| `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` | LDA |
| `sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` | QDA |
| `sklearn.datasets` | Dataset Iris |
| `math` | Funciones matemáticas escalares (exp, sqrt, pi) |
| `seaborn` | Visualización estadística |
| `mpl_interactions` | Widgets interactivos para Jupyter |
| `ipywidgets` | Controles interactivos en notebooks |

### Datasets

- **Iris Dataset** (sklearn.datasets.load_iris): Dataset clásico de clasificación con 150 muestras, 4 características y 3 clases de flores.

### Referencias Bibliográficas del Curso

- Murphy, K. P. — *Machine Learning: A Probabilistic Perspective* (referencia [1] citada en la definición de ML).
- Sitio web del curso: [https://jdariasl.github.io/Intro_ML_2025/](https://jdariasl.github.io/Intro_ML_2025/)

---

*README generado para la Unidad 1 del curso Introducción al Machine Learning — 2025.*
