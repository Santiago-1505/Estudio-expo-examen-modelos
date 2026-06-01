# Plan de Estudio — Examen Final de Machine Learning (7 días)

> **Alcance:** Clases 1 a 23 (semestre completo).
> **Modalidad:** plan intensivo de 1 semana.
> **Fuente:** los 23 notebooks del curso (`Clase 01 …` a `Clase_23 …`), convertidos a `.md` en esta carpeta.

---

## 0. Cómo usar este plan

1. **Cada día tiene 3 fases:** (A) leer/entender, (B) formular a mano, (C) autoevaluarte sin mirar.
2. **La regla de oro:** no avances al siguiente tema hasta poder responder las preguntas de autoevaluación **sin mirar los apuntes**. Recordar ≠ reconocer.
3. **Escribí las fórmulas a mano.** El formulario del final (sección 9) es tu hoja maestra — reescribilo de memoria cada día, no lo copies.
4. **Cuando algo no cierre**, abrí el `.md` de esa clase (tiene la derivación completa con gráficos) o preguntale a tu NotebookLM (skill ya instalada): *"explicame X de la Clase N con un ejemplo"*.
5. **Bloque diario sugerido:** 3–4 h. Mañana = teoría nueva (fase A+B). Tarde/noche = autoevaluación + repaso del día anterior (fase C). El repaso espaciado del día previo es **no negociable**.

---

## 1. El mapa del curso (entendé la estructura antes de estudiar)

Todo modelo de ML se define por **3 componentes** (Clase 1, y es la columna vertebral de TODO el examen):

> **(1) Modelo** (estructura + parámetros) · **(2) Criterio** (función de costo) · **(3) Algoritmo** (optimización)

El semestre se organiza en 5 grandes bloques:

| Bloque | Clases | Pregunta que responde |
|---|---|---|
| **I. Fundamentos supervisados** | 1–5 | ¿Cómo aprende un modelo y cómo mido si está bien? |
| **II. Generalización y ensembles** | 6–12 | ¿Cómo evito sobreajustar y cómo combino modelos? |
| **III. Modelos no lineales potentes** | 13–17 | Redes neuronales y SVM |
| **IV. Reducción de dimensionalidad** | 18–22 | ¿Cómo bajo la dimensión sin perder lo importante? |
| **V. Explicabilidad** | 23 | ¿Por qué el modelo decidió esto? |

**Dependencias clave** (qué necesitás saber antes de qué):
- Clase 1 (los 3 componentes) → habilita TODO.
- Clase 2 (gradiente descendente + sigmoide) → backprop (13), regresión logística como base del perceptrón.
- Clase 5 (métricas) → validación (6) → selección de hiperparámetros en TODOS los modelos.
- Clase 7 (regularización L2) → LASSO/Elastic Net (19) y el parámetro C de SVM (16).
- Clase 3 (Gaussianas, $\Sigma$) → LDA/Fisher (21) y PCA (elipses de covarianza, 20).

---

## 2. Calendario de 7 días (vista general)

| Día | Bloque | Clases | Foco |
|---|---|---|---|
| **1** | Fundamentos | 1, 2, 3 | Los 3 componentes · regresión lineal/logística · gradiente · Gaussianas (generativo vs discriminativo) |
| **2** | Fundamentos | 4, 5 | No paramétricos (kNN/Parzen) · métricas y desbalance |
| **3** | Generalización | 6, 7, 8, 9 | Bias-varianza · validación · regularización · GMM/EM · clustering |
| **4** | Ensembles | 10, 11, 12 | Árboles · Bagging/RF · Boosting/Stacking · Isolation Forest |
| **5** | No lineales | 13, 14, 15 | Redes neuronales + backprop · SOM · RNN/LSTM |
| **6** | No lineales + Reducción | 16, 17, 18, 19 | SVM (margen/kernel/dual) · multiclase · selección de features · LASSO |
| **7** | Reducción + Explicabilidad + REPASO | 20, 21, 22, 23 | PCA · LDA-Fisher · manifold (t-SNE/UMAP) · SHAP/LIME + **simulacro general** |

> Si vas adelantado, adelantá el repaso. Si vas atrasado, **nunca saltees el Día 1 ni el Día 3**: son las bases de las que cuelga todo lo demás.

---

## 3. DÍA 1 — Fundamentos (Clases 1, 2, 3)

### Conceptos clave
- **Los 3 componentes** (modelo/criterio/algoritmo). Sabelos aplicar a cualquier modelo del curso.
- **Regresión lineal:** modelo $f(\mathbf{x},\mathbf{w})=\sum_j w_j x^j$, costo ECM, optimización por gradiente descendente.
- **Regresión logística:** misma estructura + **sigmoide** + **entropía cruzada**. Frontera **lineal**, función de salida no lineal. Es **discriminativa**.
- **Gaussianas discriminantes:** modelo **generativo** $p(\mathbf{x}|C_k)$, parámetros por **Máxima Verosimilitud**. La forma de $\Sigma$ define la frontera (¡tabla abajo!).

### Formulario imprescindible
- Costo ECM: $E(\mathbf{w})=\frac{1}{2N}\sum_i (f(\mathbf{x}_i,\mathbf{w})-y_i)^2$
- Actualización: $w_j \leftarrow w_j - \frac{\eta}{N}\sum_i (\text{pred}_i - y_i)\,x_{ij}$
- Sigmoide: $g(u)=\frac{e^u}{1+e^u}$
- Entropía cruzada: $J=\frac{1}{N}\sum_i[-y_i\log g(f) - (1-y_i)\log(1-g(f))]$
- Gaussiana: $p(\mathbf{x})=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp[-\frac12(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)]$
- Estimadores ML: $\hat\mu=\frac1N\sum_i \mathbf{x}_i$, $\hat\Sigma=\frac1N\sum_i(\mathbf{x}_i-\hat\mu)(\mathbf{x}_i-\hat\mu)^T$

### Tabla CRÍTICA — forma de $\Sigma$ ↔ frontera
| $\Sigma$ | Frontera | Equivale a |
|---|---|---|
| $\sigma^2 I$ | circular | — |
| diagonal | parabólica | **Naïve Bayes** Gaussiano |
| completa, **igual** entre clases | **lineal** | **LDA** |
| completa, **distinta** por clase | **cuadrática** | **QDA** |

### Errores que cuestan puntos
- Decir que regresión logística da frontera no lineal (la frontera **es lineal**).
- Confundir generativo ($p(\mathbf{x}|C_k)$) con discriminativo ($p(C_k|\mathbf{x})$).
- Creer que LDA siempre da frontera lineal sin la condición de **$\Sigma$ compartida**.
- Usar la función signo como criterio (no es derivable → por eso la sigmoide).

### Autoevaluación (respondé sin mirar)
1. ¿Cuáles son los 3 componentes de un problema de ML?
2. ¿En qué difiere la regla de actualización de regresión logística vs lineal? (respuesta: solo la sigmoide; la estructura del gradiente es idéntica)
3. Dame la frontera para $\Sigma$ diagonal, $\Sigma$ compartida y $\Sigma$ distinta por clase.
4. ¿Qué significa que un modelo sea generativo? Dame un ejemplo del curso.

📖 Profundizar: `Clase 01…md`, `Clase 02…md`, `Clase 03…md`

---

## 4. DÍA 2 — No paramétricos y métricas (Clases 4, 5)

### Conceptos clave
- **Paramétrico vs no paramétrico:** el no paramétrico no asume forma de la fdp y su complejidad crece con $N$ (guarda las muestras).
- **kNN:** clasificación = moda de los $k$ vecinos; regresión = promedio. **Normalización OBLIGATORIA** (es por distancia).
- **Ventana de Parzen / Nadaraya-Watson:** pondera TODAS las muestras con un kernel; $h$ controla el suavizado.
- **Función de costo ≠ métrica de evaluación.** El costo se optimiza (continuo/derivable); la métrica interpreta (puede ser discontinua).
- **Desbalance:** Accuracy MIENTE. Usar **MCC, BACC, F1, G-mean**.

### Formulario imprescindible
- Recall (sensibilidad) $=\frac{TP}{TP+FN}$ · Precision $=\frac{TP}{TP+FP}$ · Especificidad $=\frac{TN}{TN+FP}$
- $F_\beta=(1+\beta^2)\frac{P\cdot R}{\beta^2 P + R}$ ; $F_1=2\frac{PR}{P+R}$
- $\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}$
- $MCC=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$
- $R^2=1-\frac{\sum(y_i-f_i)^2}{\sum(y_i-\bar y)^2}$  (puede ser **negativo**)
- Parzen: $f(\mathbf{x}^*)=\frac{1}{Nh^d}\sum_i K(u_i)$, $u_i=\frac{\text{dist}(\mathbf{x}^*,\mathbf{x}_i)}{h}$

### Errores que cuestan puntos
- Reportar solo Accuracy en problema desbalanceado.
- No normalizar antes de kNN.
- Confundir Recall (sobre positivos) con Especificidad (sobre negativos).
- Olvidar que el AUC-ROC es **independiente del umbral**.

### Autoevaluación
1. ¿Por qué hay que normalizar en kNN?
2. Un detector de fraude predice siempre "legítimo" con 99% accuracy. ¿Por qué es inútil? ¿Qué métrica usar?
3. ¿Qué mide el AUC y cuánto vale para un clasificador aleatorio? (0.5)
4. ¿Cuándo $R^2 < 0$?

📖 Profundizar: `Clase 04…md`, `Clase 05…md`

---

## 5. DÍA 3 — Generalización, regularización y no supervisado (Clases 6, 7, 8, 9)

### Conceptos clave
- **Bias-varianza:** alto bias = subajuste (simplifica de más); alta varianza = sobreajuste (aprende el ruido). + error irreducible.
- **Sobreajuste:** error de train ≈ 0, error de test alto.
- **Validación:** k-fold, LOO ($k=N$), estratificada (obligatoria con desbalance), bootstrapping. **Nunca tocar el test para elegir hiperparámetros.**
- **Regularización:** penalizar la complejidad. **L2 (Ridge):** encoge coeficientes. **L1 (LASSO):** los lleva a **cero exacto** (selecciona).
- **GMM + EM:** mezcla de Gaussianas; EM alterna paso E (responsabilidades) y paso M (re-estimar $\pi,\mu,\Sigma$).
- **Clustering:** K-means (clusters convexos), jerárquico, etc.

### Formulario imprescindible
- Ridge: $\min \sum_i(y_i-\mathbf{w}^T\mathbf{x}_i)^2 + \lambda\sum_j w_j^2$
- LASSO: $\min \sum_i(y_i-\mathbf{w}^T\mathbf{x}_i)^2 + \lambda\sum_j |w_j|$  (no penaliza $w_0$)
- $\lambda=0$ → mínimos cuadrados; $\lambda\to\infty$ → todos los $w_j\to 0$.

### Errores que cuestan puntos
- Usar el test set para seleccionar hiperparámetros (contaminación).
- Decir que Ridge anula coeficientes (eso es LASSO; Ridge solo los encoge).
- Confundir alto bias con alta varianza.

### Autoevaluación
1. Dibujá (mentalmente) las curvas de error train/test vs complejidad. ¿Dónde está el sobreajuste?
2. ¿Por qué L1 produce sparsidad y L2 no? (geometría: diamante vs esfera)
3. ¿Qué hace el paso E y el paso M en EM?
4. ¿Cuándo usás validación estratificada?

📖 Profundizar: `Clase 06…md` (clave), `Clase 07…md`, `Clase 08…md`, `Clase 09…md`

---

## 6. DÍA 4 — Ensembles (Clases 10, 11, 12)

### Conceptos clave
- **Árbol de decisión:** divide por impureza (**Gini / entropía**). Interpretable pero sobreajusta.
- **Bagging:** entrena muchos modelos en **bootstraps** y promedia → reduce **varianza**. **Random Forest** = bagging de árboles + subconjunto aleatorio de features por split.
- **Boosting:** modelos secuenciales, cada uno corrige los errores del anterior → reduce **bias** (AdaBoost, Gradient Boosting).
- **Stacking:** un meta-modelo aprende a combinar las salidas de varios modelos base.
- **Isolation Forest:** detección de anomalías; las anomalías se **aíslan** con pocos cortes (camino corto en el árbol).

### Cuadro mental clave
| | Bagging/RF | Boosting |
|---|---|---|
| Entrenamiento | paralelo, independiente | secuencial |
| Ataca | varianza | bias |
| Riesgo | menos sobreajuste | más sensible a ruido/outliers |

### Errores que cuestan puntos
- Decir que bagging reduce bias (reduce **varianza**) o que boosting reduce varianza (reduce **bias**).
- Confundir RF (features aleatorias por split) con bagging genérico.

### Autoevaluación
1. ¿Qué reduce bagging y qué reduce boosting?
2. ¿Qué agrega Random Forest sobre bagging de árboles?
3. ¿Cómo detecta anomalías Isolation Forest?

📖 Profundizar: `Clase 10…md`, `Clase 11…md`, `Clase 12…md`

---

## 7. DÍA 5 — Redes neuronales y secuencias (Clases 13, 14, 15)

### Conceptos clave
- **Perceptrón:** $O(\mathbf{x})=\text{sgn}(\mathbf{w}^T\mathbf{x})$. Equivale a regresión logística con salida discreta. Solo separa linealmente (no resuelve XOR solo).
- **MLP + backpropagation:** forward (activaciones) → backward (propagar $\delta$ con regla de la cadena). **No garantiza mínimo global.**
- **SOM (Kohonen):** no supervisado, competitivo. 3 fases: **competición** (neurona ganadora), **cooperación** (vecindad gaussiana), **adaptación**. $\eta$ y $\sigma$ **decaen** con el tiempo.
- **RNN:** estado interno, comparte parámetros en el tiempo. **BPTT**. Sufre **vanishing/exploding gradient**.
- **LSTM:** 3 compuertas (forget, input, output) + estado de célula → resuelve dependencias largas.

### Formulario imprescindible
- Regla delta: $\Delta w_i=\eta(y_j-O(\mathbf{x}_j))x_{ji}$
- Gradiente backprop: $\frac{\partial E_n}{\partial w_{ji}}=\delta_j z_i$
- $\delta$ oculto: $\delta_j=\dot h(a_j)\sum_k w_{kj}\delta_k$ ; $\delta$ salida: $\delta_k=y_k-t_k$
- Neurona ganadora SOM: $i(\mathbf{x})=\arg\min_j\|\mathbf{x}-\mathbf{w}_j\|$
- Update SOM: $\mathbf{w}_j(t{+}1)=\mathbf{w}_j(t)+\eta(t)h_{j,i}(t)(\mathbf{x}(t)-\mathbf{w}_j(t))$

### Errores que cuestan puntos
- Confundir $\delta$ de salida ($y_k-t_k$) con $\delta$ oculto (lleva $\dot h$ y suma ponderada).
- Decir que backprop encuentra el mínimo global (NO).
- En SOM: creer que solo se actualiza la ganadora (se actualizan **todas**, ponderadas por $h$); olvidar que $\sigma$ **y** $\eta$ decaen.
- Confundir estado de célula $c^{(t)}$ (largo plazo) con salida $h^{(t)}$ (corto plazo) en LSTM.

### Autoevaluación
1. ¿Por qué el perceptrón no resuelve XOR y el MLP sí?
2. ¿Por qué se reemplaza sgn por sigmoide en el MLP?
3. Nombrá y explicá las 3 fases del SOM.
4. ¿Qué problema de las RNN resuelve la LSTM y cómo?

📖 Profundizar: `Clase_13…md`, `Clase_14…md`, `Clase_15…md`

---

## 8. DÍA 6 — SVM, multiclase, selección y LASSO (Clases 16, 17, 18, 19)

### Conceptos clave
- **SVM:** frontera de **máximo margen**. **Vectores de soporte** = muestras con $a_n>0$ (las únicas que importan). Se resuelve el **dual** → habilita el **kernel trick**.
- **C (soft-margin):** C alto → pocos SV, margen chico, sobreajuste; C bajo → muchos SV, margen amplio, más regularización.
- **Multiclase:** OvR ($N_c$ clasif.), OvO ($\frac{N_c(N_c-1)}{2}$), jerárquica ($N_c-1$).
- **Selección de features:** filtro (intrínseco, rápido) vs wrapper (usa modelo, exacto, lento). SFS/SBS/SFFS (subóptimos).
- **LASSO** = selección **embebida** (L1 lleva coeficientes a cero durante el entrenamiento). **Elastic Net** = L1+L2 (mejor con features correlacionadas).

### Formulario imprescindible
- Primal: $\min \frac12\|\mathbf{w}\|^2$ s.a. $t_n(\mathbf{w}^T\phi(\mathbf{x}_n)+b)\ge 1$
- Dual: $\max \sum_n a_n - \frac12\sum_n\sum_m a_n a_m t_n t_m k(\mathbf{x}_n,\mathbf{x}_m)$ s.a. $0\le a_n\le C,\ \sum_n a_n t_n=0$
- KKT (la clave): $a_n\{t_n y(\mathbf{x}_n)-1\}=0$
- Kernel RBF: $k(\mathbf{x}_n,\mathbf{x}_m)=\exp(-\|\mathbf{x}_n-\mathbf{x}_m\|^2/2\sigma^2)$
- Nº clasificadores: OvR $=N_c$ · OvO $=\frac{N_c(N_c-1)}{2}$ · jerárquica $=N_c-1$

### Errores que cuestan puntos
- Decir que todos los puntos son vectores de soporte (solo los de $a_n>0$).
- Invertir el efecto de C (C **alto** = margen **chico**, pocos SV).
- Confundir selección (subconjunto de originales) con extracción (variables nuevas).
- Creer que Ridge selecciona; el que selecciona es **LASSO**.

### Autoevaluación
1. ¿Qué son los vectores de soporte y por qué solo ellos definen la frontera?
2. ¿Por qué se pasa al dual? (kernel trick)
3. Para 5 clases: ¿cuántos clasificadores OvR / OvO / jerárquica?
4. ¿Por qué Elastic Net es mejor que LASSO con features correlacionadas?

📖 Profundizar: `Clase_16…md`, `Clase_17…md`, `Clase_18…md`, `Clase_19…md`

---

## 9. DÍA 7 — Reducción de dimensión, explicabilidad y SIMULACRO (Clases 20, 21, 22, 23)

### Conceptos clave
- **PCA:** extracción **no supervisada**, maximiza **varianza**. Componentes = autovectores de $\Sigma$; la varianza del $k$-ésimo componente es $\lambda_k$. **Centrar los datos** antes. Limitación: la dirección de máxima varianza **no** es necesariamente la mejor para clasificar.
- **LDA (Fisher):** extracción **supervisada**, maximiza separabilidad ($S_B$) sobre dispersión interna ($S_W$). Máx. $C-1$ componentes.
- **Manifold (no lineal):** Kernel PCA, **t-SNE** (solo visualización, no generaliza), **UMAP** (rápido, generaliza, local+global), **LLE** (lineal local).
- **Explicabilidad:** global vs local; agnóstico vs específico. **SHAP** (teoría de juegos, Shapley), **LIME** (modelo lineal local), **contrafactuales** (¿qué cambiar para otro resultado?).

### Formulario imprescindible
- PCA: $\mathbf{S}=\frac1N\mathbf{X}^T\mathbf{X}$, problema $\mathbf{S}\mathbf{u}=\lambda\mathbf{u}$; varianza explicada acumulada $\frac{\sum_{k\le p}\lambda_k}{\sum_k \lambda_k}$
- Fisher: $J(\mathbf{v})=\frac{\mathbf{v}^T S_B \mathbf{v}}{\mathbf{v}^T S_W \mathbf{v}}$ → $S_W^{-1}S_B\mathbf{v}=\lambda\mathbf{v}$; bi-clase: $\mathbf{v}=S_W^{-1}(\mu_1-\mu_2)$
- SHAP: $f(\mathbf{x})=\phi_0+\sum_i \phi_i$, con $\phi_0=\mathbb{E}[f(X)]$

### Tabla CRÍTICA — PCA vs LDA
| | PCA | LDA |
|---|---|---|
| Supervisado | No | Sí |
| Maximiza | varianza total | separabilidad entre clases |
| Nº máx. componentes | $\min(d,N)$ | $\min(d, C-1)$ |

### Errores que cuestan puntos
- No centrar los datos antes de PCA.
- Decir que PCA es óptimo para clasificación (es **no supervisado**; para eso está LDA).
- Usar t-SNE en producción (no generaliza a nuevas muestras).
- Confundir explicabilidad global con local.

### Autoevaluación
1. ¿Qué relación hay entre $\lambda_k$ y la varianza en PCA?
2. ¿Por qué LDA puede tener máximo $C-1$ componentes?
3. ¿Por qué t-SNE no sirve para inferir sobre datos nuevos?
4. ¿En qué se basa SHAP y qué es $\phi_0$?

### 🎯 Simulacro general (tarde del Día 7)
Hacé esto **sin apuntes**, cronometrado:
1. Escribí el **formulario maestro completo** de memoria (sección 10).
2. Respondé 1 pregunta de autoevaluación de **cada día** (1→7).
3. Para 3 modelos al azar, decí sus **3 componentes** (modelo/criterio/algoritmo).
4. Armá de memoria 2 tablas: $\Sigma$↔frontera (Día 1) y PCA vs LDA (Día 7).

📖 Profundizar: `Clase_20…md`, `Clase_21…md`, `Clase_22…md`, `Clase_23…md`

---

## 10. Formulario maestro (tu hoja de una sola página)

**Optimización base**
- ECM: $E=\frac{1}{2N}\sum_i(f_i-y_i)^2$ · Update: $w_j\leftarrow w_j-\frac{\eta}{N}\sum_i(\text{pred}_i-y_i)x_{ij}$
- Sigmoide $g(u)=\frac{e^u}{1+e^u}$ · Entropía cruzada $J=\frac1N\sum_i[-y_i\log g(f)-(1-y_i)\log(1-g(f))]$

**Generativo (Gaussiano)**
- $p(\mathbf{x})=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}e^{-\frac12(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)}$ · $\hat\mu,\hat\Sigma$ por ML

**Métricas**
- $R=\frac{TP}{TP+FN}$ · $P=\frac{TP}{TP+FP}$ · $F_1=2\frac{PR}{P+R}$ · $R^2=1-\frac{\sum(y_i-f_i)^2}{\sum(y_i-\bar y)^2}$

**Regularización**
- Ridge $+\lambda\sum w_j^2$ · LASSO $+\lambda\sum|w_j|$ · Elastic Net $+\lambda_1\sum|w_j|+\lambda_2\sum w_j^2$

**Redes**
- $\delta_j=\dot h(a_j)\sum_k w_{kj}\delta_k$ · $\frac{\partial E_n}{\partial w_{ji}}=\delta_j z_i$
- SOM: $\mathbf{w}_j(t{+}1)=\mathbf{w}_j(t)+\eta(t)h_{j,i}(t)(\mathbf{x}(t)-\mathbf{w}_j(t))$

**SVM**
- Dual: $\max\sum_n a_n-\frac12\sum_{n,m}a_n a_m t_n t_m k(\mathbf{x}_n,\mathbf{x}_m)$, $0\le a_n\le C$, $\sum a_n t_n=0$
- KKT: $a_n\{t_n y(\mathbf{x}_n)-1\}=0$ · RBF: $k=e^{-\|\mathbf{x}_n-\mathbf{x}_m\|^2/2\sigma^2}$

**Reducción**
- PCA: $\mathbf{S}\mathbf{u}=\lambda\mathbf{u}$ (varianza = $\lambda_k$) · Fisher: $\mathbf{v}=S_W^{-1}(\mu_1-\mu_2)$, $J=\frac{\mathbf{v}^T S_B\mathbf{v}}{\mathbf{v}^T S_W\mathbf{v}}$

**Conteo multiclase:** OvR $=N_c$ · OvO $=\frac{N_c(N_c-1)}2$ · jerárquica $=N_c-1$

---

## 11. Conexiones transversales (las preguntas "trampa" del final)

Estas relaciones son lo que distingue una nota buena de una excelente:
- **Perceptrón ↔ Regresión logística:** mismo modelo, distinta activación (sgn vs sigmoide).
- **Regularización L2 ↔ C de SVM:** C es el inverso de $\lambda$ (C grande = menos regularización).
- **LASSO ↔ Selección de features:** L1 es selección embebida.
- **PCA ↔ LDA:** varianza (no superv.) vs separabilidad (superv.).
- **Gaussiana $\Sigma$ ↔ LDA/QDA/Naïve Bayes:** la forma de $\Sigma$ define cuál es.
- **Bagging ↔ varianza / Boosting ↔ bias.**
- **Costo ↔ métrica:** se optimiza uno, se evalúa con otra.

---

## 12. Estrategia para el día del examen
1. **Apenas te den la hoja**, escribí el formulario maestro en el margen (memoria de descarga).
2. Leé TODO el examen primero; arrancá por lo que sabés seguro (asegurás puntos y ganás confianza).
3. En preguntas conceptuales, estructurá: **definición → fórmula → ejemplo/cuándo se usa → limitación**.
4. Si te piden comparar (PCA vs LDA, bagging vs boosting), **hacé una tabla**: es más rápido y se lee mejor.
5. Si te trabás, escribí los 3 componentes del modelo en cuestión: casi siempre destraba la respuesta.

---

## 13. Recursos
- **Notebooks completos** (`.md` en esta carpeta): la fuente con derivaciones y gráficos.
- **NotebookLM** (skill instalada, notebook *Intro ML — Curso completo* activo): para dudas puntuales mientras estudiás. Ej: *"explicame el paso M del algoritmo EM con un ejemplo"*. Ojo: límite ~50 consultas/día.

> Éxitos. Estudiá los CONCEPTOS, no memorices código. Si entendés los 3 componentes y por qué cada modelo existe, el examen se vuelve una conversación, no un interrogatorio. 💪
