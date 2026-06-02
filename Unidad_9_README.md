# Unidad 9: Aprendizaje por Refuerzo y Aprendizaje Semi-Supervisado

---

## Introducción

Esta unidad aborda dos paradigmas avanzados del aprendizaje automático que van más allá del aprendizaje supervisado clásico:

- **Aprendizaje por Refuerzo (Reinforcement Learning — RL):** Un agente aprende a tomar decisiones óptimas interactuando con un entorno, recibiendo señales de recompensa o penalidad según sus acciones. El objetivo es maximizar la recompensa acumulada a lo largo del tiempo.

- **Aprendizaje Semi-Supervisado:** Combina una pequeña cantidad de datos etiquetados con una gran cantidad de datos sin etiquetar para construir modelos de clasificación más robustos, reduciendo la necesidad de anotación manual costosa.

Ambos paradigmas son fundamentales en aplicaciones modernas de inteligencia artificial, desde agentes que juegan videojuegos hasta sistemas de clasificación con pocos datos anotados.

---

## Contenido de la Unidad

| Archivo | Descripción |
|---|---|
| `Aprendizaje_por_refuerzo___Introducción_al_Machine_Learning.pdf` | Teoría e implementación de RL: MDP, Q-Learning y Deep Q-Learning |
| `Aprendizaje_semi-supervisado___Introducción_al_Machine_Learning.pdf` | Teoría e implementación de métodos semi-supervisados: Self-Training, Co-Training y Label Propagation |

---

## Estructura de Archivos

---

### Archivo: `Aprendizaje_por_refuerzo___Introducción_al_Machine_Learning.pdf`

#### Propósito

Introduce los fundamentos teóricos y prácticos del Aprendizaje por Refuerzo (RL), cubriendo desde la formalización matemática con Procesos de Decisión de Markov hasta implementaciones concretas en Python con Q-Learning y Deep Q-Learning.

---

#### Contenido Detallado

##### 1. Introducción al Aprendizaje por Refuerzo

El aprendizaje por refuerzo es un tipo de aprendizaje automático donde un **agente** aprende a tomar decisiones a través de la interacción con un **ambiente**. A diferencia del aprendizaje supervisado, no existe un conjunto de etiquetas predefinidas; en cambio, el agente recibe señales de recompensa o penalidad que guían su aprendizaje.

**Componentes clave del RL:**

| Componente | Descripción |
|---|---|
| **Agente** | El aprendiz o tomador de decisiones |
| **Ambiente** | Todo con lo que el agente puede interactuar |
| **Acción (A)** | Las elecciones que el agente puede hacer en cada momento |
| **Estado (S)** | La situación actual del agente dentro del ambiente |
| **Recompensa (R)** | La señal de realimentación que el ambiente envía al agente tras cada acción |

El objetivo central del agente es aprender una **política** (estrategia de comportamiento) que maximice la recompensa acumulada a lo largo del tiempo.

---

##### 2. Proceso de Decisión de Markov (MDP)

El MDP es el marco matemático formal que sustenta la mayoría de los algoritmos de RL. Se define mediante una tupla de cinco elementos:

| Elemento | Notación | Descripción |
|---|---|---|
| Conjunto de estados | `S` | Todos los posibles estados del entorno |
| Conjunto de acciones | `A` | Todas las posibles acciones del agente |
| Probabilidad de transición | `P(s' \| s, a)` | Probabilidad de llegar al estado `s'` desde el estado `s` tomando la acción `a` |
| Función de recompensa | `R(s, a)` | Recompensa recibida al tomar la acción `a` en el estado `s` |
| Factor de descuento | `γ ∈ [0, 1]` | Pondera la importancia de recompensas futuras vs. inmediatas |

**Política y valor esperado:**

La política `π: S → A` es una función que mapea cada estado a una acción. El valor esperado de seguir una política desde un estado inicial `s₀` se define como:

```
V^π(s) = E[ Σ(t=0 → ∞) γᵗ · R(sₜ, aₜ) | s₀ = s, aₜ = π(sₜ) ]
```

- Cuando `γ` es cercano a 0, el agente es **miope** (solo le importan recompensas inmediatas).
- Cuando `γ` es cercano a 1, el agente considera **recompensas a largo plazo**.

---

##### 3. Q-Learning

###### Concepto

La **Tabla Q** (Q-table) es una estructura de datos (matriz) que almacena el valor esperado de tomar cada acción `a` en cada estado `s`. Es decir, `Q(s, a)` representa la máxima recompensa futura esperada al estar en el estado `s` y ejecutar la acción `a`.

La tabla Q guía al agente hacia la mejor acción en cada estado durante la ejecución.

###### Regla de actualización

La tabla Q se actualiza iterativamente con la siguiente regla de Bellman:

```
Q(s, a) ← Q(s, a) + α · [R(s, a) + γ · max_a' Q(s', a') - Q(s, a)]
```

Donde:
- `α` es la **tasa de aprendizaje** (learning rate): controla cuánto se actualiza el valor Q en cada paso.
- `γ` es el **factor de descuento**: importancia de recompensas futuras.
- `R(s, a)` es la recompensa inmediata obtenida.
- `max_a' Q(s', a')` es la mejor recompensa esperada desde el siguiente estado `s'`.
- El término `R(s, a) + γ · max_a' Q(s', a') - Q(s, a)` se llama **error TD (Temporal Difference)**.

###### Implementación en Python

```python
import numpy as np
import random

# Definición del entorno
states = [0, 1, 2, 3]   # 4 estados posibles
actions = [0, 1]         # 0 = izquierda, 1 = derecha

# Matriz de recompensas (determinística)
rewards = np.array([
    [-1, 0],    # Estado 0: izquierda → -1, derecha → 0
    [-1, 0],    # Estado 1
    [-1, 10],   # Estado 2: derecha → estado terminal con recompensa 10
    [0, 0]      # Estado terminal
])

# Inicializar tabla Q con ceros
Q = np.zeros((len(states), len(actions)))
```

**Explicación del entorno:**
- Hay 4 estados linealmente ordenados (0 → 1 → 2 → 3).
- El agente puede moverse a la izquierda (acción 0) o a la derecha (acción 1).
- Al llegar al estado 3 (terminal) desde el estado 2 yendo a la derecha, recibe una recompensa de +10.
- Moverse a la izquierda siempre genera penalidad de -1 (excepto en el estado terminal).

**Función de selección de acción (ε-greedy):**

```python
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Exploración: acción aleatoria
    return np.argmax(Q[state])         # Explotación: mejor acción conocida
```

- `epsilon`: probabilidad de explorar aleatoriamente vs. explotar el conocimiento actual.
- Esta estrategia balancea **exploración** (descubrir nuevas acciones) vs. **explotación** (usar lo aprendido).

**Función de actualización de la tabla Q:**

```python
def update_q_table(state, action, reward, next_state, alpha, gamma):
    best_next_action = np.max(Q[next_state])
    Q[state, action] += alpha * (reward + gamma * best_next_action - Q[state, action])
```

**Bucle de entrenamiento:**

```python
num_episodes = 100
alpha = 0.1    # Tasa de aprendizaje
gamma = 0.9    # Factor de descuento
epsilon = 0.2  # Tasa de exploración

for episode in range(num_episodes):
    state = 0
    while state != 3:   # Hasta llegar al estado terminal
        action = choose_action(state, epsilon)
        reward = rewards[state, action]
        next_state = state + 1 if action == 1 else max(0, state - 1)
        update_q_table(state, action, reward, next_state, alpha, gamma)
        state = next_state
```

**Resultado obtenido tras el entrenamiento:**

```
Trained Q-table:
[[1.87  8.09]
 [2.76  8.99]
 [5.56  9.99]
 [0.    0.  ]]
```

**Interpretación:** Para cada estado, la acción "derecha" (columna 1) tiene siempre un valor Q mayor que "izquierda" (columna 0), confirmando que la política óptima aprendida es siempre moverse a la derecha hacia la recompensa de +10.

**Extracción de la política óptima:**

```python
def extract_policy(Q):
    return [np.argmax(Q[state]) for state in range(len(Q))]

policy = extract_policy(Q)
# Resultado: [1, 1, 1, 0] → siempre ir a la derecha (excepto estado terminal)
```

---

##### 4. Deep Q-Learning (DQN)

###### Motivación

Q-Learning con tabla no es viable cuando el espacio de estados es **grande o continuo** (por ejemplo, imágenes de píxeles). Deep Q-Learning (DQN) reemplaza la tabla Q por una **red neuronal** que aproxima la función Q.

###### Componentes principales de una DQN

| Componente | Descripción |
|---|---|
| **Red neuronal de política** | Toma un estado como entrada y devuelve valores Q para cada acción posible |
| **Memoria de experiencias (Experience Replay)** | Buffer que almacena tuplas `(s, a, r, s')` para entrenamiento por lotes |
| **Red objetivo (Target Network)** | Red separada que provee valores Q estables para calcular el objetivo de entrenamiento |

###### Función de costo

La DQN se entrena minimizando el error cuadrático medio entre el valor Q predicho y el valor Q objetivo:

```
L(θ) = E[(r + γ · max_a' Q_target(s', a'; θ⁻) - Q(s, a; θ))²]
```

Donde:
- `θ` son los parámetros de la red de política.
- `θ⁻` son los parámetros de la **red objetivo** (se actualizan con menor frecuencia).
- `r = R(s, a)` es la recompensa inmediata del entorno.
- `D` es la memoria que almacena experiencias previas.

**¿Por qué usar una red objetivo separada?**

Sin la red objetivo, el modelo perseguiría sus propias predicciones cambiantes, generando inestabilidad en el entrenamiento. La red objetivo actúa como un **punto de referencia estable** que se actualiza periódicamente (cada `n` episodios), permitiendo un entrenamiento más consistente.

###### Implementación en Python

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

state_size = 4
action_size = 2

# Definición de la red neuronal
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 24),  # Capa de entrada
            nn.ReLU(),
            nn.Linear(24, 24),           # Capa oculta
            nn.ReLU(),
            nn.Linear(24, action_size)   # Capa de salida: un valor Q por acción
        )

    def forward(self, x):
        return self.fc(x)
```

**Arquitectura de la red:**
- **Entrada:** vector de estado de dimensión 4.
- **Capas ocultas:** dos capas densas de 24 neuronas con activación ReLU.
- **Salida:** vector de dimensión 2 (un valor Q por cada acción posible).

**Buffer de experiencias:**

```python
memory = deque(maxlen=1000)

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))
```

- El buffer almacena hasta 1000 experiencias pasadas.
- `done` indica si el episodio terminó en ese paso.
- El muestreo aleatorio del buffer rompe la correlación temporal entre experiencias consecutivas, estabilizando el entrenamiento.

**Función de acción (ε-greedy):**

```python
def act(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(action_size)   # Exploración
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        return policy_net(state).argmax().item()   # Explotación
```

**Función de entrenamiento (replay):**

```python
def replay(batch_size, gamma):
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.BoolTensor(dones)

    q_values = policy_net(states).gather(1, actions).squeeze()
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + (gamma * next_q_values * (~dones))

    loss = criterion(q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Pasos del replay:**
1. Se muestrea un mini-lote aleatorio del buffer de memoria.
2. Se calculan los valores Q actuales con la red de política.
3. Se calculan los valores Q objetivo con la red objetivo.
4. Se calcula la pérdida MSE entre ambos.
5. Se realiza retropropagación y se actualizan los pesos de la red de política.

**Actualización periódica de la red objetivo:**

```python
def update_target_network():
    target_net.load_state_dict(policy_net.state_dict())

# Se actualiza cada 5 episodios
if episode % 5 == 0:
    update_target_network()
```

---

### Archivo: `Aprendizaje_semi-supervisado___Introducción_al_Machine_Learning.pdf`

#### Propósito

Introduce el aprendizaje semi-supervisado: cómo aprovechar datos no etiquetados junto con pocos datos etiquetados para mejorar el rendimiento de clasificadores. Se implementan y comparan cuatro enfoques: baseline supervisado, Self-Training, Co-Training y Label Propagation.

---

#### Contenido Detallado

##### 1. Preparación del Conjunto de Datos

Se utiliza el dataset sintético **Moons** de scikit-learn, ideal para visualización 2D de fronteras de decisión no lineales.

```python
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_labeled = 40
X_labeled   = X_train[:n_labeled]      # Solo 40 muestras etiquetadas
y_labeled   = y_train_full[:n_labeled]
X_unlabeled = X_train[n_labeled:]      # El resto sin etiquetar
y_unlabeled = y_train_full[n_labeled:] # Verdad oculta (solo para evaluación)
```

**Descripción:**
- 500 muestras totales, distribuidas en dos clases (0 y 1) con forma de medias lunas.
- Solo **40 muestras** (≈11.4%) están etiquetadas para el entrenamiento semi-supervisado.
- El resto de muestras de entrenamiento se tratan como **no etiquetadas**.
- El conjunto de prueba (`X_test`, `y_test`) se usa para evaluar todos los modelos.

---

##### 2. Aprendizaje Supervisado (Baseline)

Modelo de referencia entrenado **únicamente con los 40 datos etiquetados**.

```python
model_supervised = SVC(probability=True)
model_supervised.fit(X_labeled, y_labeled)
y_pred_sup = model_supervised.predict(X_test)
# Accuracy: 0.8933
```

- **Modelo:** Support Vector Classifier (SVC) con estimación de probabilidades habilitada.
- **Accuracy obtenida:** ~89.3%
- La frontera de decisión muestra regiones amplias pero no perfectamente ajustadas a la geometría de los datos, debido a la escasez de ejemplos etiquetados.

---

##### 3. Autoaprendizaje (Self-Training)

###### Concepto

El Self-Training es un método *wrapper* que usa un clasificador base para etiquetar iterativamente las muestras no etiquetadas sobre las que tiene **mayor confianza**. También conocido como **pseudo-labels**.

###### Implementación

```python
class SelfTrainingClassifier:
    def __init__(self, base_estimator, confidence_threshold=0.8, max_iter=10):
        self.base_estimator = base_estimator
        self.confidence_threshold = confidence_threshold
        self.max_iter = max_iter
```

**Parámetros:**
- `base_estimator`: clasificador base (en este caso SVC con probabilidades).
- `confidence_threshold`: umbral de confianza mínimo (0.8 = 80%) para aceptar una etiqueta predicha.
- `max_iter`: número máximo de iteraciones del proceso de etiquetado.

**Método `fit`:**

```python
def fit(self, X_labeled, y_labeled, X_unlabeled):
    X_l = X_labeled.copy()
    y_l = y_labeled.copy()
    X_u = X_unlabeled.copy()

    for iteration in range(self.max_iter):
        model = clone(self.base_estimator)
        model.fit(X_l, y_l)                         # Entrenar con datos etiquetados actuales
        probs = model.predict_proba(X_u)             # Predecir probabilidades para no etiquetados
        confident = np.max(probs, axis=1) >= self.confidence_threshold  # Filtrar por confianza

        if not any(confident):
            break  # Detener si no hay predicciones confiables

        confident_indices = np.where(confident)[0]
        X_new = X_u[confident_indices]
        y_new = np.argmax(probs[confident_indices], axis=1)  # Pseudo-etiquetas

        # Agregar muestras confiables al conjunto etiquetado
        X_l = np.vstack([X_l, X_new])
        y_l = np.concatenate([y_l, y_new])
        X_u = np.delete(X_u, confident_indices, axis=0)  # Remover del conjunto no etiquetado
    
    self.model = model
```

**Flujo del algoritmo:**
1. Entrenar el clasificador base con los datos etiquetados disponibles.
2. Predecir probabilidades sobre los datos no etiquetados.
3. Seleccionar las muestras cuya probabilidad máxima supere el umbral de confianza.
4. Asignarles pseudo-etiquetas y agregarlas al conjunto etiquetado.
5. Repetir hasta agotar las iteraciones o no encontrar más muestras confiables.

**Resultado:**
```
Self-Training Accuracy: 0.8733
```

- La accuracy (~87.3%) es ligeramente inferior al baseline supervisado en este caso, posible indicador de que las pseudo-etiquetas introducidas pudieron añadir algo de ruido.

---

##### 4. Co-Aprendizaje (Co-Training)

###### Concepto

El Co-Training utiliza **dos clasificadores diferentes** que se etiquetan mutuamente los datos. Idealmente, cada clasificador usa diferentes conjuntos de características o fuentes de información, lo que permite perspectivas complementarias.

###### Implementación

```python
def co_training(X_labeled, y_labeled, X_unlabeled,
                base_estimator_1, base_estimator_2,
                max_iter=10, k=5):
    X_l = X_labeled.copy()
    y_l = y_labeled.copy()
    X_u = X_unlabeled.copy()

    for _ in range(max_iter):
        model_1 = clone(base_estimator_1)
        model_2 = clone(base_estimator_2)
        model_1.fit(X_l, y_l)
        model_2.fit(X_l, y_l)

        probs_1 = model_1.predict_proba(X_u)
        probs_2 = model_2.predict_proba(X_u)

        # Seleccionar las k muestras más confiables de cada modelo
        confident_1 = np.argsort(-np.max(probs_1, axis=1))[:k]
        confident_2 = np.argsort(-np.max(probs_2, axis=1))[:k]

        new_X = np.vstack((X_u[confident_1], X_u[confident_2]))
        new_y = np.hstack((
            np.argmax(probs_1[confident_1], axis=1),
            np.argmax(probs_2[confident_2], axis=1)
        ))

        X_l = np.vstack([X_l, new_X])
        y_l = np.concatenate([y_l, new_y])
        X_u = np.delete(X_u, np.union1d(confident_1, confident_2), axis=0)

        if len(X_u) == 0:
            break

    final_model = clone(base_estimator_1)
    final_model.fit(X_l, y_l)
    return final_model
```

**Parámetros:**
- `base_estimator_1` y `base_estimator_2`: dos clasificadores distintos (SVC y KNN en este ejemplo).
- `max_iter`: número máximo de iteraciones.
- `k`: número de muestras más confiables que cada modelo aporta al conjunto etiquetado por iteración.

**Flujo del algoritmo:**
1. Entrenar ambos clasificadores con los datos etiquetados actuales.
2. Cada clasificador predice probabilidades sobre los datos no etiquetados.
3. Cada uno selecciona sus `k` predicciones más confiables.
4. Ambos conjuntos de pseudo-etiquetas se agregan al conjunto etiquetado.
5. Repetir hasta agotar iteraciones o datos no etiquetados.
6. Entrenar el modelo final con todo el conjunto etiquetado ampliado.

**Resultado:**
```
Co-Training Accuracy: 0.9667
```

- Mejora significativa (~96.7%) sobre el baseline supervisado (~89.3%), demostrando la efectividad de usar dos perspectivas complementarias.

---

##### 5. Propagación de Etiquetas (Label Propagation)

###### Concepto

La Propagación de Etiquetas es un método basado en grafos que propaga la información de etiquetas a través de la **variedad de los datos** (*manifold*). La intuición es que puntos cercanos en el espacio de características deben pertenecer a la misma clase.

###### Implementación

```python
# Combinar datos etiquetados y no etiquetados
X_lp = np.vstack((X_labeled, X_unlabeled))
y_lp = np.concatenate((y_labeled, [-1] * len(X_unlabeled)))
# Las muestras sin etiquetar se marcan con -1

label_prop_model = LabelPropagation(kernel='knn')
label_prop_model.fit(X_lp, y_lp)
y_pred_lp = label_prop_model.predict(X_test)
```

**Parámetros:**
- `kernel='knn'`: usa k vecinos más cercanos para construir el grafo de similitud entre puntos.
- Las muestras etiquetadas tienen su etiqueta real; las no etiquetadas se marcan con `-1`.
- El algoritmo propaga las etiquetas desde los nodos conocidos hacia los desconocidos siguiendo la estructura del grafo.

**Resultado:**
```
Label Propagation Accuracy: 0.9867
```

- La mayor accuracy obtenida (~98.7%), superando a todos los métodos anteriores.
- La frontera de decisión se adapta muy bien a la geometría de medias lunas de los datos.

**Variante mencionada:** `LabelSpreading`, una versión más robusta al ruido que permite a los nodos etiquetados también modificar levemente sus etiquetas durante la propagación.

---

## Flujo General de la Unidad

```
Unidad 9
│
├── Aprendizaje por Refuerzo
│   ├── Fundamentos teóricos (MDP)
│   ├── Q-Learning (tabla discreta)
│   │   ├── Entorno simple de 4 estados
│   │   ├── Estrategia ε-greedy
│   │   └── Regla de actualización de Bellman
│   └── Deep Q-Learning (DQN)
│       ├── Red neuronal (PyTorch)
│       ├── Experience Replay
│       └── Red objetivo (Target Network)
│
└── Aprendizaje Semi-Supervisado
    ├── Dataset: make_moons (500 muestras, 40 etiquetadas)
    ├── Baseline supervisado (SVC) → 89.3%
    ├── Self-Training              → 87.3%
    ├── Co-Training                → 96.7%
    └── Label Propagation          → 98.7%
```

Los dos grandes temas de la unidad son independientes entre sí, pero comparten la filosofía de aprender con **información limitada o indirecta**: en RL la señal de supervisión es la recompensa diferida, y en semi-supervisado la supervisión es escasa (pocos datos etiquetados).

---

## Conceptos Clave

### Aprendizaje por Refuerzo

| Concepto | Definición |
|---|---|
| **Agente** | Entidad que toma decisiones en el entorno |
| **Ambiente** | Sistema con el que el agente interactúa |
| **Estado** | Representación de la situación actual del agente |
| **Acción** | Decisión tomada por el agente en un estado dado |
| **Recompensa** | Señal numérica de retroalimentación del ambiente |
| **Política (π)** | Función que mapea estados a acciones |
| **MDP** | Marco matemático formal del RL con estados, acciones, transiciones y recompensas |
| **Factor de descuento (γ)** | Parámetro que pondera recompensas futuras vs. presentes |
| **Q-function** | Valor esperado de tomar una acción en un estado bajo una política óptima |
| **Tabla Q** | Representación tabular de la Q-function para espacios discretos |
| **Regla de Bellman** | Ecuación de actualización iterativa de los valores Q |
| **ε-greedy** | Estrategia de exploración/explotación con probabilidad ε de explorar |
| **DQN** | Q-Learning con red neuronal para espacios de estado grandes |
| **Experience Replay** | Buffer de experiencias pasadas para entrenamiento por lotes |
| **Target Network** | Red auxiliar estable para calcular los objetivos de entrenamiento |
| **Error TD** | Diferencia temporal entre el valor Q estimado y el valor Q objetivo |

### Aprendizaje Semi-Supervisado

| Concepto | Definición |
|---|---|
| **Datos etiquetados** | Muestras con clase conocida usadas para supervisión directa |
| **Datos no etiquetados** | Muestras sin clase conocida que aportan estructura al modelo |
| **Pseudo-etiquetas** | Etiquetas predichas por el modelo usadas como si fueran reales |
| **Self-Training** | Método iterativo donde el modelo se auto-etiqueta con alta confianza |
| **Co-Training** | Dos clasificadores se etiquetan mutuamente usando perspectivas complementarias |
| **Label Propagation** | Método basado en grafos que difunde etiquetas entre puntos similares |
| **LabelSpreading** | Variante de Label Propagation más robusta al ruido |
| **Umbral de confianza** | Probabilidad mínima requerida para aceptar una pseudo-etiqueta |
| **Manifold** | Estructura geométrica subyacente en los datos de alta dimensión |
| **Kernel KNN** | Función de similitud basada en k vecinos más cercanos para construir el grafo |

---

## Comparativa de Resultados (Aprendizaje Semi-Supervisado)

| Método | Accuracy | Observaciones |
|---|---|---|
| Supervisado (solo 40 etiquetados) | 89.3% | Baseline de referencia |
| Self-Training | 87.3% | Levemente inferior; riesgo de propagación de errores |
| Co-Training | 96.7% | Gran mejora; beneficio de dos perspectivas distintas |
| Label Propagation | 98.7% | Mejor resultado; aprovecha la geometría de los datos |

---

## Conclusiones

### Aprendizaje por Refuerzo

1. **El MDP provee un marco riguroso** para formalizar problemas de decisión secuencial donde el agente aprende por interacción con el entorno.

2. **Q-Learning es efectivo para espacios discretos pequeños**, pero escala mal a entornos complejos con espacios de estado continuos o de alta dimensión.

3. **Deep Q-Learning (DQN) supera las limitaciones del Q-Learning tabular** usando redes neuronales para aproximar la función Q, habilitando aplicaciones a problemas complejos como juegos de Atari o control robótico.

4. **El Experience Replay y la Red Objetivo** son las dos innovaciones clave que estabilizan el entrenamiento de DQN, rompiendo correlaciones temporales y evitando la persecución de objetivos inestables.

5. **El balance exploración-explotación** (estrategia ε-greedy) es crítico: demasiada exploración desperdicia tiempo; demasiada explotación impide descubrir mejores políticas.

### Aprendizaje Semi-Supervisado

1. **El aprendizaje semi-supervisado es poderoso cuando etiquetar datos es costoso**, permitiendo aprovechar grandes volúmenes de datos no etiquetados para mejorar significativamente el rendimiento.

2. **Self-Training es simple pero frágil**: los errores de etiquetado temprano pueden propagarse y acumularse, degradando el rendimiento.

3. **Co-Training es más robusto** al introducir diversidad mediante dos clasificadores complementarios, reduciendo el riesgo de sesgo en el etiquetado.

4. **Label Propagation es altamente efectivo** cuando la estructura geométrica de los datos es coherente con las clases, logrando la mayor accuracy en este experimento (~98.7%).

5. **La elección del método semi-supervisado depende del contexto**: si los datos tienen estructura de manifold clara, Label Propagation es ideal; si se necesita flexibilidad, Co-Training es una excelente opción.

---

## Recursos y Referencias

### Librerías Utilizadas

| Librería | Uso en la unidad |
|---|---|
| `numpy` | Operaciones matriciales, manejo de arrays |
| `scikit-learn` | Modelos ML, datasets, métricas, LabelPropagation |
| `torch` (PyTorch) | Implementación de redes neuronales para DQN |
| `collections.deque` | Buffer circular para Experience Replay |
| `random` | Estrategia ε-greedy y muestreo del buffer |

### Modelos de scikit-learn Utilizados

| Modelo | Módulo |
|---|---|
| `SVC` | `sklearn.svm` |
| `KNeighborsClassifier` | `sklearn.neighbors` |
| `LabelPropagation` | `sklearn.semi_supervised` |
| `make_moons` | `sklearn.datasets` |
| `accuracy_score` | `sklearn.metrics` |

### Referencias Conceptuales

- **Gymnasium:** Biblioteca recomendada para entornos estándar de RL (sucesor de OpenAI Gym).
- **Bellman, R.** (1957): Ecuaciones de optimización dinámica que fundamentan Q-Learning.
- **Mnih et al.** (2015): *Human-level control through deep reinforcement learning* — paper original de DQN (DeepMind/Nature).
- **Blum & Mitchell** (1998): Co-Training original paper `[2]`.
- **Zhu & Ghahramani** (2002): Label Propagation original paper `[3]`.
