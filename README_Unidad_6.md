# Unidad 6: Redes Neuronales — Artificiales, Auto-Organizables y Recurrentes

---

## Introducción

La Unidad 6 del curso de Introducción al Machine Learning cubre tres grandes familias de redes neuronales que forman la base del aprendizaje profundo moderno:

1. **Redes Neuronales Artificiales (ANN)**: Modelos supervisados inspirados en el cerebro humano, capaces de aprender fronteras de decisión no lineales mediante el algoritmo de retropropagación (Backpropagation).
2. **Mapas Auto-Organizables (SOM)**: Redes neuronales no supervisadas basadas en aprendizaje competitivo que permiten visualizar y agrupar datos de alta dimensionalidad en un espacio bidimensional.
3. **Redes Neuronales Recurrentes (RNN)**: Arquitecturas diseñadas para procesar datos secuenciales mediante el mantenimiento de un estado interno, incluyendo variantes avanzadas como las redes LSTM y bidireccionales.

El objetivo central de esta unidad es comprender cómo distintas arquitecturas neuronales abordan diferentes tipos de problemas: clasificación, clustering y modelamiento temporal/secuencial.

---

## Contenido de la Unidad

La unidad está compuesta por tres archivos PDF exportados desde notebooks de Jupyter, correspondientes a tres clases del curso:

| Archivo | Clase | Tema |
|---|---|---|
| `Redes_Neuronales_Artificiales___Introducción_al_Machine_Learning.pdf` | Clase 13 | Perceptrón, Backpropagation, MLP |
| `Mapas_Auto-Organizables___Introducción_al_Machine_Learning.pdf` | Clase 14 | SOM, aprendizaje competitivo |
| `Redes_Neuronales_Recurrentes___Introducción_al_Machine_Learning.pdf` | Clase 15 | RNN, BPTT, LSTM |

---

## Estructura de Archivos

---

### Archivo 1: `Redes_Neuronales_Artificiales___Introducción_al_Machine_Learning.pdf`

#### Propósito

Introducir las redes neuronales artificiales desde sus fundamentos biológicos hasta la implementación práctica de un perceptrón multicapa (MLP) con el algoritmo de Backpropagation, incluyendo las diferentes estrategias de entrenamiento (Batch, Mini-batch, On-line).

---

#### Contenido y Funcionamiento Detallado

##### 1. Motivación biológica

El documento abre contextualizando el poder de cómputo del cerebro humano:

- El cerebro humano contiene aproximadamente **10¹¹ neuronas**.
- Cada neurona se conecta en promedio con otras **10⁴ neuronas**.
- La velocidad de cambio de estado es del orden de **10⁻³ segundos** (muy lento comparado con los computadores), pero el paralelismo masivo permite tareas cognitivas complejas en tiempo real (por ejemplo, reconocer visualmente a una persona en ~0.1 segundos).

Esta motivación justifica la construcción de modelos computacionales inspirados en dicha arquitectura.

---

##### 2. El Perceptrón

El **perceptrón** es la unidad básica de una ANN. Recibe un vector de valores reales, calcula una combinación lineal ponderada y produce una salida binaria.

**Función de salida:**

```
O(x₁, x₂, ..., x_d) = 1   si  w₀ + w₁x₁ + w₂x₂ + ... + w_d·x_d > 0
                      = -1  en otro caso
```

En forma matricial: `O(x) = sgn(wᵀx)`

donde `sgn(u) = 1 si u > 0`, y `-1` en otro caso.

**Relación con regresión logística:** El perceptrón es equivalente a la regresión logística vista en clases anteriores. El umbral `-w₀` equivale al término independiente negado.

**Interpretación geométrica:** El perceptrón define un hiperplano de decisión en el espacio d-dimensional. Las muestras a un lado del hiperplano reciben salida 1, y las del otro lado reciben -1.

---

##### 3. Entrenamiento del Perceptrón

**Regla de actualización de pesos** (cuando una muestra es mal clasificada):

```
wᵢ ← wᵢ + Δwᵢ
Δwᵢ = η(yⱼ − O(xⱼ)) · xⱼᵢ
```

Donde:
- `η` es la tasa de aprendizaje.
- `yⱼ` es la salida deseada para la muestra `xⱼ`.
- `O(xⱼ)` es la salida calculada por el perceptrón.

**Limitación:** Este procedimiento solo converge cuando los datos son **linealmente separables**. Para datos no separables se requiere minimizar un error mediante gradiente descendente.

---

##### 4. Limitaciones del Perceptrón Simple — El problema XOR

El documento ilustra tres problemas de clasificación (OR, AND, XOR):

- **OR y AND**: separables linealmente → resolubles con un solo perceptrón.
- **XOR**: **no separable linealmente** → requiere combinar múltiples perceptrones.

La solución al XOR requiere dos fronteras de decisión lineales combinadas, lo que lleva naturalmente a la idea de **red neuronal multicapa**.

---

##### 5. Redes Neuronales Artificiales (ANN)

Una ANN combina múltiples perceptrones en capas:

| Capa | Rol |
|---|---|
| **Capa de entrada** | Recibe las características/variables de cada muestra |
| **Capas ocultas** | Definen fronteras de decisión complejas |
| **Capa de salida** | Proporciona la predicción final (una o múltiples salidas) |

**Funciones de activación no lineales** permiten que las fronteras de decisión no sean rectas. Las principales son:

- **Función sigmoide**: `f(u) = exp(u) / (1 + exp(u))`, rango `[0, 1]`
  - Derivada: `∂f(u)/∂wᵢ = f(u)(1 − f(u)) · ∂u/∂wᵢ`
- **Tangente hiperbólica**: `f(u) = (exp(u) − exp(−u)) / (exp(u) + exp(−u))`, rango `[−1, 1]`

La necesidad de funciones derivables nace porque el entrenamiento multicapa requiere calcular gradientes. La función `sgn` no es derivable, por lo que se reemplaza por la sigmoide.

**Función de salida según tipo de problema:**
- **Regresión**: `yₖ = aₖ` (salida lineal)
- **Clasificación binaria**: `yₖ = σ(aₖ)` (sigmoide)
- **Clasificación multiclase**: función **softmax**

---

##### 6. Algoritmo Backpropagation

El **Backpropagation** ajusta los pesos de una red multicapa mediante gradiente descendente, minimizando el error cuadrático medio entre salidas y valores deseados.

**Notación utilizada:**
- `d`: número de variables de entrada (características)
- `xⱼᵢ`: entrada del nodo `i` a la unidad `j`
- `wⱼᵢ⁽¹⁾`: peso correspondiente en la capa 1
- `Mₖ`: número de neuronas en la capa `k`

**Construcción de la red (hacia adelante):**

Capa oculta 1:
```
aⱼ = Σᵢ wⱼᵢ⁽¹⁾ · xᵢ     (j = 1, ..., M₁)
zⱼ = h(aⱼ)               (función de activación)
```

Capa de salida:
```
aₖ = Σⱼ wₖⱼ⁽²⁾ · zⱼ     (k = 1, ..., M₂)
yₖ(x, w) = σ(Σⱼ wₖⱼ⁽²⁾ · h(Σᵢ wⱼᵢ⁽¹⁾ · xᵢ + wⱼ₀⁽¹⁾) + wₖ₀⁽²⁾)
```

**Regla de actualización:**
```
w(τ+1) = w(τ) − η · ∇E(w(τ))
```

**Función de error (Batch):**
```
E(w) = (1/N) · Σₙ Eₙ(w)
```

**Etapas del Backpropagation:**

1. **Propagación hacia adelante**: presentar la muestra `xₙ` y calcular todas las activaciones de las capas ocultas y de salida.
2. **Cálculo de deltas en la capa de salida**: `δₖ = yₖ − tₖ`
3. **Retropropagación de deltas a capas ocultas**:
   ```
   δⱼ = ḣ(aⱼ) · Σₖ wₖⱼ · δₖ
   ```
4. **Cálculo de gradientes**:
   ```
   ∂Eₙ/∂wⱼᵢ = δⱼ · zᵢ
   ```

**Ventaja del Backpropagation**: es una forma computacionalmente eficiente de evaluar el gradiente de la función de error para todos los pesos de la red.

**Limitación importante**: en redes multicapa la función de error puede tener **múltiples mínimos locales**, por lo que no se garantiza convergencia al mínimo global.

---

##### 7. Batch, Mini-batch y On-line Learning

| Estrategia | Descripción | Ventaja | Desventaja |
|---|---|---|---|
| **Batch** | Acumula todos los errores del dataset antes de actualizar | Estable, convergencia suave | Lento, problemas de memoria, se atasca en puntos silla (saddle points) |
| **On-line (SGD)** | Actualiza con cada muestra individual | Escapa de mínimos locales, requiere poca memoria | Convergencia ruidosa, muchas iteraciones |
| **Mini-batch** | Actualiza con subconjuntos del dataset | Balance entre velocidad y estabilidad | Requiere ajustar el tamaño del batch |

El documento ilustra estas tres estrategias con el `MLPClassifier` de scikit-learn, mostrando las curvas de pérdida para cada una.

**Código relevante:**
```python
from sklearn.neural_network import MLPClassifier

# Batch: batch_size=len(X)
clf1 = MLPClassifier(activation='logistic', solver='sgd', learning_rate_init=...)

# Mini-batch: batch_size intermedio
clf2 = MLPClassifier(activation='logistic', solver='sgd', ...)

# On-line: batch_size=1
clf3 = MLPClassifier(activation='logistic', solver='sgd', ...)
```

Las gráficas muestran que On-line y Mini-batch convergen más rápido que Batch, mientras que Batch tiende a estancarse.

---

##### 8. Desventajas de las Aproximaciones Clásicas

- Requieren reformular completamente las ecuaciones si cambia la arquitectura o la función de costo.
- No aprovechan herramientas de diferenciación automática (cálculo simbólico).
- Soporte limitado de regularización avanzada.
- No usan paralelismo eficientemente (GPUs).
- No soportan estrategias modernas como **Transfer Learning**.

Esta sección motiva el uso de frameworks modernos como **PyTorch** o **TensorFlow**.

---

##### 9. Ejemplo práctico — Clasificación con neurolab

El notebook incluye un ejemplo completo de clasificación con datos de dos medias lunas (`make_moons`):

```python
from sklearn import datasets
X, y = datasets.make_moons(n_samples=1500, noise=.05)

import neurolab as nl
net = nl.net.newff([[X.min(), X.max()], [y.min(), y.max()]], [5, 1], ...)
err = net.train(input, target, show=15)
```

- **Red**: 2 entradas → 5 neuronas ocultas → 1 salida
- **Resultado**: la red aprende la frontera no lineal de separación entre las dos clases con error cercano a 0.48 en 15 épocas.

---

### Archivo 2: `Mapas_Auto-Organizables___Introducción_al_Machine_Learning.pdf`

#### Propósito

Presentar los **Mapas Auto-Organizables (Self-Organizing Maps — SOM)** como una alternativa no supervisada para agrupamiento y visualización de datos de alta dimensionalidad, destacando sus ventajas frente a algoritmos de clustering convencionales como K-means.

---

#### Contenido y Funcionamiento Detallado

##### 1. Introducción a los SOM

Los SOM son redes neuronales **no supervisadas** basadas en **aprendizaje competitivo**:

- Solo **una neurona de salida** puede estar activada en un momento dado (la "neurona ganadora").
- Las neuronas están ubicadas en los nodos de una **cuadrícula** (generalmente 1D o 2D).
- Durante el entrenamiento, las neuronas se ajustan selectivamente a diferentes patrones de entrada.
- La ubicación espacial de las neuronas ganadoras forma un **mapa topográfico** que refleja las características estadísticas intrínsecas de los datos.

**Modelos principales:**
- **Willshaw-von der Malsburg**: requiere que el número de entradas igual al de salidas; uso limitado.
- **Kohonen**: el más utilizado; captura los elementos esenciales de los mapas corticales del cerebro humano.

---

##### 2. Fases del Entrenamiento SOM

El entrenamiento tiene **tres fases principales** (más la inicialización aleatoria de pesos):

###### Fase 1 — Competición

Dada una muestra de entrada `x`, se determina la **neurona ganadora** (`winning neuron`) como aquella cuyo vector de pesos es más cercano al vector de entrada:

```
i(x) = argmín_j ||x − wⱼ||,   j ∈ A
```

Donde `A` es el conjunto de todas las neuronas. Esto es equivalente a encontrar la neurona con mayor producto interno `wⱼᵀx`.

- `x = [x₁, x₂, ..., xₘ]ᵀ`: vector de entrada (m características)
- `wⱼ = [wⱼ₁, wⱼ₂, ..., wⱼₘ]ᵀ`: vector de pesos de la neurona j

###### Fase 2 — Cooperación

La neurona ganadora `i` se convierte en el centro de una **vecindad topológica** de neuronas cooperantes, emulando el comportamiento biológico de la corteza cerebral.

La vecindad se define mediante una función gaussiana:

```
hⱼ,ᵢ(x) = exp(−dⱼᵢ² / (2σ²)),   j ∈ A
```

Donde:
- `dⱼᵢ²`: distancia en la cuadrícula entre las neuronas `i` y `j`
  - En 1D: `dⱼᵢ = |j − i|`
  - En 2D: `dⱼᵢ² = ||rⱼ − rᵢ||²`
- `σ`: ancho efectivo de la vecindad topológica

El ancho de vecindad **decrece con el tiempo** para permitir la convergencia:

```
σ(t) = σ₀ · exp(−t/τ₁),   t = 0, 1, 2, ...
```

Donde `σ₀` es el valor inicial y `τ₁` es una constante de tiempo definida por el diseñador.

###### Fase 3 — Adaptación Sináptica

Los pesos de las neuronas en la vecindad de la neurona ganadora se actualizan para acercarse al vector de entrada:

```
wⱼ(t+1) = wⱼ(t) + η(t) · hⱼ,ᵢ(x)(t) · (x(t) − wⱼ(t))
```

La tasa de aprendizaje también decrece con el tiempo:

```
η(t) = η₀ · exp(−t/τ₂),   t = 0, 1, 2, ...
```

**Valores típicos:** `η₀ = 0.1`, `τ₂ = 1000`

---

##### 3. Ejemplo 1 — Mapa de Colores

Se entrena un SOM con 15 colores representados como vectores RGB:

```python
colors = np.array([
    [0., 0., 0.],   # black
    [0., 0., 1.],   # blue
    ...
    [.66, .66, .66] # lightgrey
])

sm = sompy.SOMFactory().build(colors, [20, 30], normalization='var', ...)
sm.train(n_job=4, verbose=False, train_len_factor=0.5, ...)
```

- **Cuadrícula**: 20 × 30 = 600 neuronas de salida
- **Resultado**: el mapa organiza los colores de forma coherente, colocando colores similares (azul, azul oscuro, azul cielo) en regiones adyacentes del mapa.

---

##### 4. Ejemplo 2 — Clustering con estructuras anulares (ventaja sobre K-means)

Este ejemplo demuestra una ventaja clave del SOM: la capacidad de detectar **estructuras no convexas** que K-means no puede manejar.

Se generan 4 anillos concéntricos (2800 puntos en total):

```python
dlen = 700
tetha = np.random.uniform(low=0, high=2*np.pi, size=dlen)[:, np.newaxis]

X1 = 3*np.cos(tetha) + ruido    # Radio 3
X2 = 1*np.cos(tetha) + ruido    # Radio 1
X3 = 5*np.cos(tetha) + ruido    # Radio 5
X4 = 8*np.cos(tetha) + ruido    # Radio 8
Data = np.concatenate((X1, X2, X3, X4), axis=0)  # (2800, 2)
```

K-means fallaría porque los centroides no pueden representar estructuras circulares. El SOM con una cuadrícula de 50×50 y la **U-Matrix** detecta correctamente los 4 clusters anulares.

```python
mapsize = [50, 50]
som = sompy.SOMFactory.build(Data2, mapsize, ...)
som.train(n_job=1, verbose=False, train_finetune_len=200)

# U-Matrix para visualización de clusters
u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', ...)
UMAT = u.build_u_matrix(som, distance=1, row_normalized=False)
UMAT = u.show(som, distance2=1, show_data=True, ...)
```

La **U-Matrix** (Unified Distance Matrix) muestra las distancias promedio entre neuronas vecinas en el espacio de pesos: valores altos (zonas claras/rojas) indican fronteras entre clusters.

---

##### 5. Ejemplo 3 — Dataset Iris

Se aplica el SOM al clásico dataset Iris (150 muestras, 4 características):

```python
from sklearn.datasets import load_iris
data = load_iris()
X = data.data   # Shape: (150, 4)

mapsize = [20, 20]
som = sompy.SOMFactory.build(X, mapsize, ...)
som.train(n_job=1, verbose=False, train_finetune_len=200)
```

La U-Matrix generada muestra la separación entre las 3 especies de iris, evidenciando que el SOM puede capturar la estructura inherente del dataset sin usar las etiquetas de clase.

---

### Archivo 3: `Redes_Neuronales_Recurrentes___Introducción_al_Machine_Learning.pdf`

#### Propósito

Introducir las **Redes Neuronales Recurrentes (RNN)** como herramienta para el procesamiento de datos secuenciales, cubriendo el algoritmo de entrenamiento BPTT, las redes bidireccionales y la arquitectura LSTM para modelar dependencias temporales de largo plazo.

---

#### Contenido y Funcionamiento Detallado

##### 1. Motivación y Concepto de RNN

Las RNN están diseñadas para procesar **datos secuenciales** (series de tiempo, texto, audio, etc.) mediante el principio de **compartir parámetros** a lo largo del tiempo:

- Si se tuviera un parámetro por cada índice de tiempo, no se podría generalizar a secuencias de longitudes no vistas durante el entrenamiento.
- Las RNN mantienen un **estado interno** (memoria) a través de nodos de contexto que influencian la capa oculta en entradas subsecuentes.

**Arquitecturas conocidas:**
- **Red de Elman**: la salida de la capa oculta se realimenta como entrada en el siguiente paso de tiempo.
- **Red de Jordan**: la salida de la capa de salida se realimenta.

---

##### 2. Paradigmas de Aprendizaje de las RNN

Las RNN pueden manejar diferentes configuraciones de entrada-salida (diagrama de Karpathy):

| Configuración | Descripción | Ejemplo de aplicación |
|---|---|---|
| **One to one** | Una entrada → una salida (MLP convencional) | Clasificación de imágenes estáticas |
| **One to many** | Una entrada → secuencia de salidas | Caption generation (imagen → descripción) |
| **Many to one** | Secuencia de entradas → una salida | Sentiment Analysis (texto → positivo/negativo) |
| **Many to many** (con delay) | Secuencia de entrada → secuencia de salida distinta | Traducción automática |
| **Many to many** (simultáneo) | Entrada y salida sincronizadas | Predicción de series de tiempo, etiquetado morfosintáctico |

---

##### 3. Arquitectura Elman (la más usada)

La descripción matemática de una RNN Elman con una capa oculta:

```
a⁽ᵗ⁾ = b + V·h⁽ᵗ⁻¹⁾ + U·x⁽ᵗ⁾
h⁽ᵗ⁾ = tanh(a⁽ᵗ⁾)
o⁽ᵗ⁾ = c + W·h⁽ᵗ⁾
```

Donde:
- `x⁽ᵗ⁾`: entrada en el tiempo t
- `h⁽ᵗ⁾`: estado oculto (memoria) en el tiempo t
- `o⁽ᵗ⁾`: salida en el tiempo t
- `U`: matriz de pesos para las entradas
- `V`: matriz de pesos de realimentación (estado oculto anterior)
- `W`: matriz de pesos hacia la capa de salida
- `b`, `c`: vectores de términos independientes

---

##### 4. Backpropagation Through Time (BPTT)

El entrenamiento de una RNN usa Backpropagation adaptado al tiempo:

**Función de costo total** (suma sobre todos los tiempos):
```
L = Σₜ L⁽ᵗ⁾
```

En clasificación:
```
L⁽ᵗ⁾ = − log Pmodel(ŷ⁽ᵗ⁾ | {x⁽¹⁾, ..., x⁽ᵗ⁾})
```

**Proceso:**

1. **Propagación hacia adelante**: calcular `h⁽ᵗ⁾` y `o⁽ᵗ⁾` para cada t, almacenando todos los estados intermedios.
2. **Gradiente en la capa de salida** (tiempo τ):
   ```
   ∇ₒ₍ₜ₎L = y⁽ᵗ⁾ ⊙ (ŷ⁽ᵗ⁾ − 1)
   ```
   donde `⊙` es el producto Hadamard (elemento a elemento).
3. **Gradiente en el estado oculto final (τ)**:
   ```
   ∇_h(τ)L = Wᵀ · ∇ₒ₍τ₎L
   ```
4. **Propagación hacia atrás (t < τ)**:
   ```
   ∇_h(t)L = Vᵀ · (∇_h(t+1)L) · diag(1 − (h⁽ᵗ⁺¹⁾)²) + Wᵀ · ∇ₒ₍ₜ₎L
   ```
5. **Gradientes de los parámetros:**
   ```
   ∇_cL = Σₜ (∂oᵗ/∂c)ᵀ · ∇ₒ₍ₜ₎L
   ∇_WL = Σₜ (∇ₒ₍ₜ₎L) · h⁽ᵗ⁾ᵀ
   ∇_VL = Σₜ diag(1 − (h⁽ᵗ⁾)²) · (∇_h(t)L) · h⁽ᵗ⁻¹⁾ᵀ
   ∇_UL = Σₜ diag(1 − (h⁽ᵗ⁾)²) · (∇_h(t)L) · x⁽ᵗ⁾ᵀ
   ```

**Limitación del BPTT:** las operaciones de forward y backward **no son paralelizables** (completamente secuenciales).

**Truncated BPTT:** para secuencias largas, se propaga hacia adelante hasta un tiempo intermedio `t`, se realiza el backward y se actualiza, luego se continúa desde `t+1`. Esto reduce el uso de memoria y acelera la convergencia.

---

##### 5. Ejemplos de Entrenamiento RNN

###### Con neurolab (red Elman)

```python
import neurolab as nl

# Señales de entrada: seno de amplitud 1 y amplitud 2
i1 = np.sin(np.arange(0, 20))
i2 = np.sin(np.arange(0, 20)) * 2
input = np.array([i1, i2, i1, i2]).reshape(80, 1)
target = np.array([t1, t2, t1, t2]).reshape(80, 1)

# Red Elman: 1 entrada → 10 neuronas ocultas → 1 salida
net = nl.net.newelm([[-2, 2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
error = net.train(input, target, epochs=500, show=100, goal=0.01)
```

**Tarea:** detectar la amplitud de la señal seno (1 o 2), que es una tarea de memoria: la red debe recordar si está procesando la señal de amplitud 1 o 2.

**Resultado:** error final ~0.064 a las 500 épocas.

###### Con PyTorch (SimpleRNN)

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

model = SimpleRNN(input_size=1, hidden_size=10, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

- **Datos:** secuencia de 80 puntos (4 repeticiones de señales seno)
- **Resultado:** error final ~0.0136 en 500 épocas, con predicciones que siguen bien el patrón de la señal objetivo.

---

##### 6. Redes Neuronales Recurrentes Bidireccionales (BRNN)

En algunos problemas (reconocimiento de voz, traducción) es necesario conocer **contexto futuro y pasado** para hacer una predicción en un punto de la secuencia.

Las **BRNN** tienen **dos capas ocultas**:
- Una que transmite información de `t` hacia `t+1` (dirección normal).
- Otra que transmite información de `t` hacia `t−1` (dirección inversa).

Ambas capas ocultas no se interconectan entre sí, pero sí se conectan a la capa de salida.

**Aplicaciones:** predicción de una única salida para toda la secuencia, o salida en cada paso de tiempo con contexto completo.

---

##### 7. Long Short-Term Memory (LSTM)

###### Problema que resuelve

Las RNN convencionales sufren del problema de **desvanecimiento del gradiente**: la matriz `V` de realimentación se multiplica por sí misma `(τ−1)` veces durante el BPTT, lo que hace que:

- Si sus valores son < 1: los gradientes se **desvanecen** → la red no aprende dependencias largas.
- Si sus valores son > 1: los gradientes **divergen** → inestabilidad en el entrenamiento.

Además, la aplicación consecutiva de múltiples funciones de activación agrava el desvanecimiento.

###### Solución: Celdas LSTM

Las LSTM reemplazan los perceptrones por **celdas con memoria explícita y compuertas entrenables** (propuestas en 1997). Tienen la capacidad de aprender dependencias de tiempo corto, largo, y decidir cuándo olvidar.

**Principio base — EWMA (Exponential Weighted Moving Average):**

```
μ⁽ᵗ⁾ ← β·μ⁽ᵗ⁻¹⁾ + (1−β)·υ⁽ᵗ⁾
```

El parámetro `β` controla cuánto peso se da al pasado vs. el dato actual. Las LSTM generalizan esto con compuertas entrenables.

###### Ecuaciones de la celda LSTM

Para el tiempo `t` y la celda `l`:

| Elemento | Ecuación | Rol |
|---|---|---|
| **Compuerta de olvido** | `f⁽ᵗ⁾ = σ(bᶠ + Uᶠ·x⁽ᵗ⁾ + Vᶠ·h⁽ᵗ⁻¹⁾)` | Controla cuánto del estado anterior conservar |
| **Compuerta de entrada** | `i⁽ᵗ⁾ = σ(bⁱ + Uⁱ·x⁽ᵗ⁾ + Vⁱ·h⁽ᵗ⁻¹⁾)` | Controla cuánta información nueva escribir |
| **Estado de la celda** | `c⁽ᵗ⁾ = f⁽ᵗ⁾·c⁽ᵗ⁻¹⁾ + i⁽ᵗ⁾·tanh(bᶜ + Uᶜ·x⁽ᵗ⁾ + Vᶜ·h⁽ᵗ⁻¹⁾)` | Memoria interna de la celda |
| **Compuerta de salida** | `o⁽ᵗ⁾ = σ(bᵒ + Uᵒ·x⁽ᵗ⁾ + Vᵒ·h⁽ᵗ⁻¹⁾)` | Controla el nivel de activación de la salida |
| **Estado oculto** | `h⁽ᵗ⁾ = tanh(c⁽ᵗ⁾) · o⁽ᵗ⁾` | Salida de la celda al siguiente tiempo |

Las tres compuertas (`f`, `i`, `o`) usan funciones **sigmoide** → valores en `(0, 1)`, lo que permite control gradual (no binario) de la memoria.

###### Parámetros entrenables completos

```
Vᶠ, Uᶠ, bᶠ, Vᶜ, Uᶜ, bᶜ, Vⁱ, Uⁱ, bⁱ, Vᵒ, Uᵒ, bᵒ
```

###### Ejemplo con PyTorch — Predicción de serie de tiempo

```python
class LSTM_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        super(LSTM_net, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (out, _) = self.rnn(x)
        out = out.view(x.shape[0], self.hidden_size)
        out = self.fc(out)
        return out
```

**Dataset:** serie de tiempo de pasajeros de aerolínea internacional (144 meses)
**Preprocesamiento:**
- Normalización MinMax a `[0, 1]`
- División 67% entrenamiento / 33% prueba
- Creación de ventanas deslizantes con `look_back=1`
- Reshape a `(samples, time_steps, features)` para PyTorch

**Resultados:**
- Train RMSE: ~24.08
- Test RMSE: ~47.46
- Las predicciones siguen bien la tendencia y la estacionalidad de la serie.

---

## Flujo General de la Unidad

```
Clase 13 — ANN (Perceptrón y Backpropagation)
       ↓
   Fundamentos: cómo una neurona aprende
   Arquitectura MLP: capas entrada-oculta-salida
   Algoritmo: Backpropagation + gradiente descendente
   Estrategias: Batch, Mini-batch, On-line
       ↓
Clase 14 — SOM (Aprendizaje No Supervisado)
       ↓
   Extensión hacia problemas no supervisados
   Competición: encontrar la neurona ganadora
   Cooperación: vecindad topológica
   Adaptación: actualización de pesos
       ↓
Clase 15 — RNN y LSTM (Datos Secuenciales)
       ↓
   Extensión hacia datos con dependencia temporal
   BPTT: Backpropagation aplicado en el tiempo
   BRNN: contexto pasado y futuro
   LSTM: memoria selectiva a largo plazo
```

Los tres documentos construyen sobre el mismo principio base (redes de neuronas interconectadas), evolucionando desde aprendizaje supervisado estático (ANN) hacia aprendizaje no supervisado (SOM) y finalmente hacia el modelamiento de secuencias temporales (RNN/LSTM).

---

## Conceptos Aprendidos

### Fundamentos de ANN
- **Perceptrón**: unidad básica con función signo, equivalente a la regresión logística.
- **Hiperplano de decisión**: separación lineal del espacio d-dimensional.
- **Funciones de activación**: sigmoide, tanh, ReLU; necesidad de diferenciabilidad.
- **Multicapa**: combinación de perceptrones para resolver problemas no lineales (XOR).
- **Backpropagation**: propagación eficiente del gradiente hacia atrás usando la regla de la cadena.
- **Deltas**: errores propagados hacia atrás que permiten calcular gradientes en capas ocultas.
- **Batch vs. SGD vs. Mini-batch**: compromisos entre velocidad, memoria y estabilidad.
- **Mínimos locales**: la función de error en redes multicapa no es convexa.

### Mapas Auto-Organizables
- **Aprendizaje competitivo**: solo la neurona más cercana al input se activa.
- **Neurona ganadora (BMU)**: la de menor distancia euclidiana al vector de entrada.
- **Vecindad topológica gaussiana**: las neuronas cercanas a la ganadora también se actualizan.
- **Decaimiento temporal**: tanto el ancho de vecindad como la tasa de aprendizaje disminuyen con las iteraciones.
- **Mapa topográfico**: preservación de la estructura de los datos en el espacio de salida.
- **U-Matrix**: herramienta de visualización para detectar fronteras entre clusters en el espacio del SOM.
- **Ventaja sobre K-means**: capacidad de detectar clusters de forma no convexa.

### Redes Neuronales Recurrentes
- **Estado interno**: mecanismo de memoria que persiste entre pasos de tiempo.
- **Compartición de parámetros**: los mismos pesos se usan en todos los tiempos.
- **Paradigmas secuenciales**: one-to-one, one-to-many, many-to-one, many-to-many.
- **BPTT**: suma de gradientes sobre todos los tiempos de la secuencia.
- **Desvanecimiento del gradiente**: el gradiente tiende a cero en secuencias largas.
- **Truncated BPTT**: alternativa para secuencias largas con actualización parcial.
- **BRNN**: procesamiento bidireccional para tareas que requieren contexto futuro.
- **LSTM**: celdas con compuertas de olvido, entrada y salida para memoria selectiva.
- **EWMA**: principio matemático base de las LSTM.
- **Compuertas entrenables**: parámetros aprendidos que controlan el flujo de información.

---

## Conclusiones

1. **Escalabilidad de la complejidad**: el perceptrón simple resuelve solo problemas linealmente separables. La combinación en capas (MLP) permite resolver cualquier problema con suficiente capacidad, gracias al teorema de aproximación universal.

2. **El Backpropagation es el corazón del aprendizaje supervisado**: permite ajustar eficientemente millones de parámetros propagando el error desde la salida hacia las capas internas. Su limitación de múltiples mínimos locales motivó el desarrollo de frameworks modernos con optimizadores avanzados (Adam, RMSProp).

3. **Los SOM llenan el vacío no supervisado**: cuando no hay etiquetas disponibles, los SOM permiten organizar y visualizar los datos preservando la topología, algo que K-means convencional no puede hacer para estructuras no convexas (anillos, espirales).

4. **Las RNN habilitan el modelamiento temporal**: al compartir parámetros en el tiempo, las RNN generalizan a secuencias de cualquier longitud. Sin embargo, el desvanecimiento del gradiente limita su capacidad de aprender dependencias largas.

5. **Las LSTM resuelven el problema de memoria a largo plazo**: mediante compuertas entrenables que aprenden cuándo recordar y cuándo olvidar, las LSTM pueden modelar dependencias de orden superior, siendo la arquitectura más exitosa para datos secuenciales hasta la llegada de los Transformers.

6. **PyTorch como framework moderno**: los ejemplos con PyTorch demuestran cómo los frameworks de diferenciación automática simplifican enormemente la implementación de arquitecturas complejas, eliminando la necesidad de derivar manualmente las ecuaciones de actualización.

---

## Recursos y Referencias

### Librerías utilizadas

| Librería | Uso |
|---|---|
| `numpy` | Operaciones matriciales y generación de datos |
| `matplotlib` | Visualización de datos y resultados |
| `scikit-learn` | `MLPClassifier`, `make_moons`, `load_iris`, `MinMaxScaler` |
| `neurolab` | Implementación de redes Elman y perceptrones multicapa |
| `sompy` | Implementación de mapas auto-organizables (SOMPY) |
| `torch` (`PyTorch`) | Implementación moderna de RNN y LSTM |
| `pandas` | Carga y manejo del dataset de pasajeros de aerolínea |

### Referencias bibliográficas

1. **Simon Haykin** — *Neural Networks and Learning Machines*, 3ra Edición, Prentice Hall, USA, 2009. *(Referencia principal para ANN y SOM)*
2. **Goodfellow, I.; Bengio, Y.; Courville, A.** — *Deep Learning*, The MIT Press, Cambridge, MA, USA, 2016. *(Referencia principal para RNN y LSTM)*
3. **Andrej Karpathy** — Diagrama de paradigmas de RNN: [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
4. **Machine Learning Mastery** — Tutorial de LSTM para series de tiempo: [Time Series Prediction with LSTM](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
5. **SOMPY library** — [https://github.com/sevamoo/SOMPY](https://github.com/sevamoo/SOMPY)

### Datasets utilizados

- **Colors dataset**: 15 colores en espacio RGB, construido manualmente.
- **Anillos concéntricos**: generados sintéticamente con funciones trigonométricas.
- **Iris dataset**: `sklearn.datasets.load_iris()` — 150 muestras, 4 características, 3 clases.
- **International Airline Passengers**: serie de tiempo de 144 meses de tráfico aéreo.
- **make_moons**: datos de dos medias lunas generados con `sklearn.datasets.make_moons`.
