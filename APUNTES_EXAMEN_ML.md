# Apuntes de Examen Final — Machine Learning (Clases 1 a 23)

> Documento de repaso profundo, organizado por clase. Para cada tema: idea central, conceptos clave, fórmulas, errores comunes y preguntas tipo examen.
> Acompaña al `PLAN_ESTUDIO_ML_7dias.md` (el cronograma). La fuente completa con derivaciones y gráficos son los notebooks `.md` de esta carpeta.

## Índice
- **Bloque I — Fundamentos:** Clases 1-5 (intro, regresión lineal/logística, Gaussianas, no paramétricos, métricas)
- **Bloque II — Generalización y Ensembles:** Clases 6-12 (validación, regularización, GMM, clustering, árboles, bagging/RF, boosting, isolation forest)
- **Bloque III — Modelos no lineales:** Clases 13-17 (redes neuronales, SOM, RNN/LSTM, SVM, multiclase)
- **Bloques IV y V — Reducción y Explicabilidad:** Clases 18-23 (selección, LASSO, PCA, LDA, manifold, SHAP/LIME)

---

# Bloque I — Fundamentos (Clases 1 a 5)

## CLASE 1 — Introducción al Machine Learning

### Idea central
ML detecta patrones en datos y los usa para predecir/decidir bajo incertidumbre (enfoque **correlacional**, no mecanicista). Todo modelo tiene 3 componentes: **modelo**, **criterio (costo)** y **algoritmo de optimización**.

### Conceptos clave
- **Supervisado:** aprender $\mathbf{x}\to y$. Discreto → clasificación; real → regresión.
- **No supervisado:** solo $\mathbf{x}$; descubrir patrones.
- **Modelo:** estructura con parámetros ajustables, ej. $f(\mathbf{x})=\mathbf{w}^T\mathbf{x}$.
- **Función de costo:** qué se minimiza; **algoritmo:** cómo se encuentran los parámetros.
- **Diseñador** (crea/modifica modelos) vs **usuario** (calibra modelos existentes).
- **Niveles de abstracción:** alto (usar librería), intermedio (modificar costo), bajo (cambiar modelo/criterio/algoritmo).

### Fórmulas
- Modelo lineal: $f(\mathbf{x})=\mathbf{w}^T\mathbf{x}$
- Costo cuadrático: $\arg\min_\mathbf{w}\frac1N\sum_i(y_i-f(\mathbf{x}_i))^2$

### Errores comunes
- Confundir función de costo (optimización, continua/derivable) con métrica de evaluación.
- Creer que cambiar el orden del polinomio $M$ es solo ajustar parámetros: cambia el **modelo**.
- Pensar que más parámetros siempre es mejor.

### Preguntas tipo examen
1. **¿Diferencia entre modelo mecanicista y de ML?** El mecanicista parte de leyes conocidas; ML descubre relaciones desde los datos (correlacional).
2. **¿Los 3 componentes de un problema de ML?** Modelo, criterio (costo), algoritmo de optimización.
3. **¿Por qué la frontera varía entre muestras?** Porque se calibra sobre muestra finita; con pocos datos la varianza es alta.
4. **¿Clasificación vs regresión?** $y$ discreto vs $y$ real.

---

## CLASE 2 — Regresión lineal y regresión logística

### Idea central
Modelos lineales base: regresión múltiple (regresión) y regresión logística (clasificación). Comparten gradiente descendente; difieren en activación (identidad vs sigmoide) y costo (ECM vs entropía cruzada). La logística es **discriminativa**.

### Conceptos clave
- **ECM**, **gradiente descendente**, **tasa de aprendizaje $\eta$**.
- **Sigmoide** $g(u)=\frac{e^u}{1+e^u}$: aproximación continua/derivable de la función signo.
- **Entropía cruzada** para clasificación binaria.
- La frontera de la logística es **lineal**; la no linealidad está en la salida (sigmoide).

### Fórmulas
- $f(\mathbf{x},\mathbf{w})=\sum_{j=0}^M w_j x^j$
- $E(\mathbf{w})=\frac{1}{2N}\sum_i(f_i-y_i)^2$
- Update: $w_j[\tau]=w_j[\tau-1]-\frac{\eta}{N}\sum_i(\text{pred}_i-y_i)x_{ij}$
- $J(\mathbf{w})=\frac1N\sum_i[-y_i\log g(f)-(1-y_i)\log(1-g(f))]$, $y_i\in\{0,1\}$
- Gradiente entropía cruzada: $\frac{\partial J}{\partial w_j}=\frac1N\sum_i(g(f_i)-y_i)x_{ij}$

### Algoritmo (gradiente descendente)
1. Inicializar $\mathbf{w}$. 2. Matriz extendida $\mathbf{X}$ (columna de unos para $w_0$). 3. Iterar: predicción → residuo → gradiente → update. 4. Parar por criterio. (Logística: aplicar sigmoide a la predicción.)

### Errores comunes
- Decir que la logística da frontera no lineal (la frontera **es lineal**).
- Olvidar la columna de unos para $w_0$.
- Usar la función signo como criterio (no derivable).

### Preguntas tipo examen
1. **¿Por qué no usar signo como criterio?** Discontinua, derivada 0/indefinida; la sigmoide la reemplaza.
2. **Update de la logística vs lineal:** igual estructura, solo se pasa la predicción por la sigmoide.
3. **¿Qué pasa con $\eta$ muy grande?** Diverge / oscila.
4. **¿Por qué es discriminativa?** Modela $p(y|\mathbf{x})$ directamente.

---

## CLASE 3 — Funciones discriminantes Gaussianas

### Idea central
Clasificación modelando la fdp de **cada clase** con Gaussiana multivariada y asignando por $\arg\max_k p_k(\mathbf{x})$. Modelo **generativo**; parámetros por **Máxima Verosimilitud**.

### Conceptos clave
- **Generativo** ($p(\mathbf{x}|C_k)$) vs **discriminativo** ($p(y|\mathbf{x})$).
- La forma de $\Sigma$ define la frontera (tabla abajo).

### Fórmulas
- $p(\mathbf{x})=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp[-\frac12(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)]$
- $\hat\mu=\frac1N\sum_i\mathbf{x}_i$, $\hat\Sigma=\frac1N\sum_i(\mathbf{x}_i-\hat\mu)(\mathbf{x}_i-\hat\mu)^T$
- Decisión: $C^*=\arg\max_k p_k(\mathbf{x}^*)$

### Tabla $\Sigma$ ↔ frontera
| $\Sigma$ | Frontera | Equivale a |
|---|---|---|
| $\sigma^2 I$ | circular | — |
| diagonal | parabólica | Naïve Bayes Gaussiano |
| completa, igual entre clases | **lineal** | LDA |
| completa, distinta por clase | **cuadrática** | QDA |

### Errores comunes
- Confundir generativo con discriminativo.
- Creer que LDA siempre da frontera lineal sin la condición de $\Sigma$ compartida.
- Estimar $\Sigma$ sobre todo el conjunto (se estima **por clase**).
- No verificar invertibilidad de $\Sigma$ cuando $N<d$.

### Preguntas tipo examen
1. **Generativo vs discriminativo + ejemplo.** Gaussianas (gen.) vs regresión logística (disc.).
2. **Estimadores ML de $\mu,\Sigma$.** (fórmulas arriba)
3. **Frontera con $\Sigma$ compartida vs distinta.** Lineal (LDA) vs cuadrática (QDA).
4. **¿Cuándo equivale a Naïve Bayes?** $\Sigma$ diagonal.
5. **¿Problema si $N<d$?** $\hat\Sigma$ singular → no invertible.

---

## CLASE 4 — Modelos no Paramétricos

### Idea central
Paramétrico (forma fija, nº params constante) vs no paramétrico (sin forma, complejidad crece con $N$). Métodos: histograma, **kNN**, **ventana de Parzen**. Trade-off: cómputo vs flexibilidad.

### Conceptos clave
- **kNN:** clasificación = moda de $k$ vecinos; regresión = promedio.
- **Parzen/Nadaraya-Watson:** pondera TODAS las muestras con kernel; $h$ controla suavizado.
- **Normalización obligatoria** (métodos por distancia).
- **Maldición de la dimensionalidad:** celdas crecen exponencialmente con $d$.

### Fórmulas
- Distancias: Euclidiana, Manhattan, Minkowski.
- Parzen: $f(\mathbf{x}^*)=\frac{1}{Nh^d}\sum_i K(u_i)$, $u_i=\frac{\text{dist}}{h}$, kernel Gaussiano $K(u)=e^{-u^2/2}$.
- Nadaraya-Watson: $y^*=\frac{\sum_i K(u_i)y_i}{\sum_i K(u_i)}$
- Normalización: max-min $\to[0,1]$; z-score $\to$ media 0, var 1 (variable a variable, con stats de **train**).

### Algoritmo kNN
Fijar $k$ y distancia → guardar train → para $\mathbf{x}^*$: calcular distancias, tomar $k$ menores, moda (clasif.) / promedio (regr.).

### Errores comunes
- Tratar kNN como paramétrico (no tiene params aprendidos).
- No normalizar.
- Creer que $k$ grande siempre es mejor (suaviza de más).
- $h$ grande = suave (subajuste); $h$ chico = ruidoso (sobreajuste).

### Preguntas tipo examen
1. **Paramétrico vs no paramétrico.** (forma fija/nº params vs sin forma/crece con N)
2. **¿Por qué normalizar en kNN?** Para que una variable de rango grande no domine la distancia.
3. **¿Qué predice Nadaraya-Watson?** Regresión no paramétrica con ventana de Parzen.
4. **kNN regresión vs clasificación.** Promedio vs moda.
5. **Efecto de $h$ grande.** Función más suave (subajuste si excesivo).

---

## CLASE 5 — Métricas de error

### Idea central
**Función de costo** (entrenamiento, continua/derivable) ≠ **métrica de evaluación** (interpreta desempeño, puede ser discontinua). Métricas de clasificación y regresión + manejo de **desbalance**.

### Conceptos clave
- Matriz de confusión: TP, TN, FP (Tipo I), FN (Tipo II).
- **Recall/sensibilidad**, **Precision**, **Especificidad**, **Accuracy**, $F_\beta$, **MCC**, **BACC**, **ROC/AUC**.
- **Desbalance:** Accuracy engaña; usar MCC, BACC, F1, G-mean.

### Fórmulas
- $R=\frac{TP}{TP+FN}$ · $P=\frac{TP}{TP+FP}$ · Espec. $=\frac{TN}{TN+FP}$ · $FPR=\frac{FP}{FP+TN}$
- $\text{Acc}=\frac{TP+TN}{\text{total}}$ · $F_\beta=(1+\beta^2)\frac{PR}{\beta^2P+R}$ · $F_1=2\frac{PR}{P+R}$
- $MCC=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$
- $BACC=$ promedio de Recall por clase · $G_{mean}=\sqrt{\text{Sens}\cdot\text{Espec}}$
- Regresión: MSE, MAE, MedAE (robusta), MAPE (falla si $y=0$), $R^2=1-\frac{\sum(y_i-f_i)^2}{\sum(y_i-\bar y)^2}$

### ROC/AUC
TPR vs FPR variando umbral. Perfecto: AUC=1; aleatorio: AUC=0.5. **Independiente del umbral.**

### Desbalance — estrategias
Submuestreo/sobremuestreo inteligente (SMOTE), muestreo apropiado en validación, pesos distintos al error.

### Errores comunes
- Reportar solo Accuracy con desbalance.
- Confundir Precision (penaliza FP) con Recall (penaliza FN); y Recall con Especificidad.
- Olvidar que $R^2$ puede ser negativo.

### Preguntas tipo examen
1. **Costo vs métrica + ejemplos.** (entropía cruzada vs error de clasificación)
2. **Fraude 99% legítimo, modelo predice siempre legítimo: ¿accuracy? ¿por qué engaña?** 99%, Recall=0; usar MCC/BACC/F1.
3. **¿Qué mide ROC/AUC?** Capacidad discriminativa independiente del umbral.
4. **¿MAE vs MSE vs MedAE?** Robustez creciente a outliers.
5. **¿Cuándo $R^2<0$?** Modelo peor que predecir el promedio.

---

# Bloque II — Generalización y Ensembles (Clases 6 a 12)

---

# Apuntes de Estudio — Examen Final de Machine Learning
## Clases 06 a 12

---

## CLASE 06 — Complejidad de Modelos, Sobreajuste y Metodologías de Validación

### 1. Objetivo / Idea Central

Entender el trade-off entre la complejidad del modelo y su capacidad de generalización (bias-varianza). Aprender a usar metodologías de validación para estimar correctamente el desempeño y seleccionar hiperparámetros sin contaminar el conjunto de test.

### 2. Conceptos Clave

**Bias (sesgo):** diferencia entre la predicción promedio del modelo y el valor real. Alto bias = subajuste = el modelo no captura la estructura subyacente de los datos ("pone poca atención a los datos de entrenamiento y sobre-simplifica el modelo").

**Variance (varianza):** error debido a alta sensibilidad a pequeñas fluctuaciones del conjunto de entrenamiento. Alta varianza = sobreajuste = el modelo se enfoca en el ruido. "Buenos en entrenamiento, altos errores en test."

**Error irreducible ($\sigma_e^2$):** ruido inherente al sistema; no se puede eliminar con ningún modelo.

**Sobreajuste:** el modelo aprende muy bien las muestras de entrenamiento pero no generaliza. Señal: error de entrenamiento ~0, error de test alto.

**Subajuste:** el modelo no puede ajustarse ni a los datos de entrenamiento. Error alto en ambos conjuntos.

**k-fold Cross-Validation:** metodología para seleccionar hiperparámetros usando únicamente el conjunto de entrenamiento, sin tocar el test.

**Validación Bootstrapping (Shuffle-Split):** muestreo con reemplazo; una misma muestra puede aparecer en múltiples conjuntos de validación.

**Leave-One-Out (LOO):** caso especial de k-fold con $k = N$ (máximo uso de datos, máximo costo computacional).

**Validación estratificada:** mantiene la proporción de clases en cada fold; obligatoria para conjuntos desbalanceados.

**Validación por grupos (Group CV):** evita que muestras del mismo sujeto/fuente aparezcan simultáneamente en entrenamiento y validación. Crítica en datos con dependencia entre muestras (ej: múltiples grabaciones del mismo paciente).

### 3. Fórmulas y Ecuaciones Importantes

Sistema a modelar:
$$y = f(\mathbf{x}) + e, \quad e \sim \mathcal{N}(0, \sigma_e^2)$$

Descomposición del error esperado:
$$\text{Err}(\mathbf{x}) = E\left[(y - \hat{f}(\mathbf{x}))^2\right]$$

$$\boxed{\text{Err}(\mathbf{x}) = \underbrace{\left(E[\hat{f}(\mathbf{x})] - f(\mathbf{x})\right)^2}_{\text{Bias}^2} + \underbrace{E\left[(\hat{f}(\mathbf{x}) - E[\hat{f}(\mathbf{x})])^2\right]}_{\text{Variance}} + \underbrace{\sigma_e^2}_{\text{Error irreducible}}}$$

### 4. Algoritmo Paso a Paso — k-Fold Cross-Validation

1. Dividir el dataset $\mathcal{D}$ en Training (80%) y **Test** (20%) — el test NUNCA se toca hasta la evaluación final.
2. Dividir el Training en $k$ subconjuntos disyuntos de igual tamaño.
3. Para cada iteración $i = 1, \ldots, k$:
   - Usar los $k-1$ folds restantes como entrenamiento.
   - Usar el fold $i$ como validación.
   - Calcular la métrica de desempeño.
4. Promediar las métricas: $\bar{\text{acc}} \pm 2\sigma$.
5. Seleccionar el hiperparámetro con mejor $\bar{\text{acc}}$ en validación.
6. Reentrenar con TODO el training usando ese hiperparámetro.
7. Evaluar **una sola vez** en el conjunto de test.

### 5. Hiperparámetros Clave

| Hiperparámetro | Efecto |
|---|---|
| Grado del polinomio / profundidad del modelo | Mayor → mayor varianza, menor bias |
| $k$ en KNN | Menor $k$ → mayor varianza; mayor $k$ → mayor bias |
| Número de folds $k$ en CV | Mayor → estimado más preciso, mayor costo computacional |
| `n_splits` en ShuffleSplit | Más repeticiones → IC más estrecho |

### 6. Ventajas / Limitaciones

**k-Fold CV:**
- (+) Todos los datos se usan para validación exactamente una vez.
- (+) Estimado más estable que una sola partición.
- (-) Porcentajes de train/val implícitamente fijos por $k$.

**Bootstrapping (ShuffleSplit):**
- (+) Porcentajes configurables libremente.
- (+) Puede tener menos muestras en cada fold.
- (-) Una muestra puede repetirse entre conjuntos de validación distintos.

**LOO:**
- (+) Máximo uso de datos de entrenamiento.
- (-) Costo computacional $O(N)$ entrenamientos; inviable para datasets grandes.

### 7. Conexiones entre Métodos

- **Validación cruzada → Bootstrapping:** ambos sirven para estimar hiperparámetros; Bootstrapping es más flexible en proporciones.
- **Validación estratificada + por grupos:** en la práctica, muchos problemas reales requieren AMBAS simultáneamente (`StratifiedGroupKFold`).
- **Curva de aprendizaje:** herramienta complementaria que muestra cómo evoluciona el error de train y validación al aumentar el tamaño del conjunto de entrenamiento; diagnostica si el problema es bias o varianza.

### 8. Errores Conceptuales Comunes en Examen

- **Usar el test para seleccionar hiperparámetros.** El test se usa UNA SOLA VEZ al final. Si lo usás para ajustar hiperparámetros, ya es parte del entrenamiento.
- **Confundir el error de validación con el error de test real.** El error de validación tiende a ser optimista; el test siempre da error un poco mayor.
- **No estratificar en datasets desbalanceados.** Un fold puede no contener representantes de la clase minoritaria.
- **No usar Group CV cuando las muestras tienen dependencia.** En el ejemplo de Parkinson: accuracy en validación = 0.947 vs. test = 0.814 (sin Group CV); con Group CV: 0.870 vs. 0.860. La diferencia es evidencia de leakage.
- **Confundir bias alto con varianza alta.** Bias alto → error alto en AMBOS conjuntos. Varianza alta → error bajo en train, alto en test.

### 9. Preguntas Tipo Examen

**P1:** ¿Por qué el error de validación puede ser "optimista" respecto del error real de test?
> Porque los hiperparámetros se eligieron para minimizar ese error de validación, introduciendo un sesgo de selección. El test, al no haber sido visto nunca, es el estimador imparcial.

**P2:** ¿Cuál es la diferencia fundamental entre k-fold CV y Bootstrapping?
> En k-fold, los folds son disyuntos y los porcentajes están implícitamente fijados por $k$. En Bootstrapping (ShuffleSplit), los porcentajes son configurables y una misma muestra puede aparecer en múltiples conjuntos de validación.

**P3:** Dado el trade-off bias-varianza, ¿qué le pasa al error total cuando aumentás la complejidad del modelo más allá del punto óptimo?
> El bias disminuye pero la varianza aumenta más rápido, por lo tanto el error total $\text{Bias}^2 + \text{Variance}$ crece (sobreajuste).

**P4:** ¿Cuándo es OBLIGATORIO usar Group CV?
> Cuando muestras del dataset provienen de la misma fuente en instancias múltiples (mismo paciente con varias grabaciones, mismo usuario con múltiples sesiones), ya que las muestras no son independientes entre sí.

**P5:** ¿Cuántas veces debe evaluarse el modelo en el conjunto de test?
> Una sola vez, al final, luego de elegir los hiperparámetros con el conjunto de validación.

---

## CLASE 07 — Regularización

### 1. Objetivo / Idea Central

La regularización es una técnica matemática para combatir el sobreajuste imponiendo restricciones sobre los parámetros del modelo, penalizando los valores de alta magnitud. Se aborda también la Maldición de la Dimensionalidad como causa estructural del sobreajuste.

### 2. Conceptos Clave

**Sobreajuste:** cuando el modelo aprende el ruido de los datos de entrenamiento en lugar de la estructura subyacente. Señal diagnóstica: error de entrenamiento muy bajo (~0) pero error de test alto.

**Subajuste:** el modelo es insuficientemente flexible. Error alto y similar en train y test.

**Regularización L2 (Ridge):** penaliza la norma cuadrática de los pesos $\|\mathbf{w}\|^2$. Reduce la magnitud de todos los pesos sin llevarlos exactamente a cero.

**Regularización L1 (LASSO):** penaliza la norma L1 $\|\mathbf{w}\|_1$. Causa que algunos pesos tomen exactamente valor 0 → selección de características implícita.

**Parámetro $\lambda$:** controla la fuerza de la regularización. Mayor $\lambda$ → mayor penalización → modelo más simple → mayor bias.

**Parada anticipada (Early Stopping):** detiene el entrenamiento cuando el error de validación deja de decrecer aunque el de entrenamiento siga bajando. Implementado en sklearn con `early_stopping=True`.

**Maldición de la Dimensionalidad:** cuando el número de características es muy grande pero el número de muestras es pequeño, el espacio entre muestras es enorme y los modelos tienden a memorizar. La solución es reducir dimensionalidad (selección/extracción de características).

### 3. Fórmulas y Ecuaciones Importantes

**Función de costo regularizada (L2):**
$$\boxed{E(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^{N}\left(y_i - f(\mathbf{x}_i, \mathbf{w})\right)^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2}$$

**Regla de actualización de pesos (gradiente regularizado L2):**
$$w_j = w_j - \eta\left(\lambda w_j + \sum_{i=1}^{N}\left(f(\mathbf{x}_i, \mathbf{w}) - y_i\right)x_{ij}\right)$$

**Solución analítica regularizada (Ridge/regresión lineal):**
$$\boxed{\mathbf{w} = \left(\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I}\right)^{-1}\mathbf{X}^T\mathbf{Y}}$$

> La adición de $\lambda \mathbf{I}$ garantiza que la matriz sea invertible incluso cuando $\mathbf{X}^T\mathbf{X}$ es singular (problema mal condicionado).

**Regla de actualización (regresión logística regularizada L2):**
$$w_j = w_j - \eta\left(\lambda w_j + \sum_{i=1}^{N}\left(g(f(\mathbf{x}_i, \mathbf{w})) - y_i\right)x_{ij}\right)$$

donde $g(\cdot)$ es la función sigmoidal.

### 4. Algoritmo Paso a Paso — Regresión Regularizada

1. Definir la función de costo con término de penalización.
2. Calcular el gradiente de la función de costo regularizada con respecto a cada $w_j$.
3. Actualizar los pesos con la regla de gradiente que incluye el término $\lambda w_j$.
4. Usar validación cruzada para seleccionar el valor óptimo de $\lambda$.
5. Evaluar en test con el $\lambda$ elegido.

### 5. Hiperparámetros Clave

| Hiperparámetro | Efecto |
|---|---|
| $\lambda$ (alpha en sklearn) | Mayor → más regularización → modelo más simple → mayor bias, menor varianza |
| Tipo de norma (L1 vs L2) | L1 → sparse (algunos $w_j = 0$); L2 → pesos pequeños pero no nulos |
| Número de iteraciones (early stopping) | Más iteraciones → potencial sobreajuste en modelos flexibles |

### 6. Ventajas / Limitaciones

**L2 (Ridge):**
- (+) Solución analítica cerrada.
- (+) Siempre invertible ($\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I}$).
- (-) No realiza selección de características (ningún peso llega a 0).

**L1 (LASSO):**
- (+) Selección implícita de características (pesos = 0 para variables irrelevantes).
- (-) No tiene solución analítica cerrada; requiere optimización iterativa.
- (-) Inestable cuando hay alta correlación entre variables.

**Early Stopping:**
- (+) Eficiente; no requiere modificar la arquitectura.
- (-) Requiere un conjunto de validación separado durante el entrenamiento.

### 7. Conexiones entre Métodos

- **L2 → GMM/clasificadores generativos:** la solución regularizada evita matrices de covarianza singulares.
- **L1 → Selección de Características (Clase 18):** LASSO es una de las técnicas de selección de características más usadas.
- **Early Stopping → Redes Neuronales (Clase 13):** la parada anticipada es el mecanismo de regularización más común en redes profundas.
- **Maldición de la Dimensionalidad → PCA/FDA (Clases 20-21):** la solución directa a este problema.

### 8. Errores Conceptuales Comunes en Examen

- **Creer que L1 y L2 producen modelos equivalentes.** L1 produce soluciones sparse (algunos pesos exactamente 0), L2 no.
- **Aumentar $\lambda$ siempre mejora el modelo.** Un $\lambda$ muy grande lleva a underfitting (bias alto).
- **La solución analítica sin regularización siempre funciona.** Sin regularización, $\mathbf{X}^T\mathbf{X}$ puede ser singular si hay más características que muestras o si hay características correlacionadas.
- **Confundir regularización con normalización.** Son conceptos distintos: regularización penaliza los pesos; normalización escala los datos de entrada.

### 9. Preguntas Tipo Examen

**P1:** ¿Por qué cuando hay sobreajuste los coeficientes del polinomio "tienden a aumentar en magnitud"?
> Porque el modelo usa valores extremos en los coeficientes para interpolar exactamente los puntos ruidosos de entrenamiento, generando oscilaciones que no reflejan la función subyacente.

**P2:** ¿Cuál es la diferencia entre regularización L1 y L2 en cuanto al efecto sobre los pesos?
> L2 reduce la magnitud de todos los pesos proporcionalmente sin llevarlos a cero. L1 causa que algunos pesos tomen exactamente el valor 0, realizando selección implícita de características.

**P3:** ¿Qué forma tiene la solución analítica de la regresión lineal con regularización L2?
> $\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{Y}$. La adición de $\lambda\mathbf{I}$ garantiza invertibilidad.

**P4:** ¿Qué es la maldición de la dimensionalidad y cómo se relaciona con el sobreajuste?
> Cuando el número de características es muy alto y el número de muestras es pequeño, el espacio entre muestras es muy grande y los modelos tienden a memorizar las respuestas de las pocas muestras de entrenamiento disponibles.

**P5:** ¿Cuándo es preferible usar early stopping sobre regularización L2?
> En modelos con muchos parámetros entrenados iterativamente (como redes neuronales), donde el número de parámetros hace que la regularización explícita sea computacionalmente costosa o difícil de calibrar.

---

## CLASE 08 — Modelos de Mezclas de Gaussianas (GMM)

### 1. Objetivo / Idea Central

Modelar distribuciones de probabilidad complejas (multimodales) como combinación lineal de gaussianas. El algoritmo EM estima los parámetros de forma iterativa. Se aplica tanto a clasificación generativa (cada clase modelada con un GMM) como a clustering no supervisado.

### 2. Conceptos Clave

**Modelos Discriminativos:** aprenden la frontera de separación directamente. Minimizan el error de clasificación usando muestras de ambas clases.

**Modelos Generativos:** estiman la función de densidad de probabilidad (fdp) de cada clase por separado. Una vez estimada la fdp, se puede usarla para generar nuevas muestras de esa distribución.

**GMM (Gaussian Mixture Model):** modelo de densidad que representa la distribución de una clase como suma ponderada de $M$ gaussianas.

**Parámetros del GMM para clase $i$:** $\Theta_i = \{w_{ij}, \mu_{ij}, \Sigma_{ij}\}_{j=1}^M$ — pesos, medias y matrices de covarianza.

**Restricción estocástica:** $\sum_{j=1}^M w_{ij} = 1$ para cada clase $i$.

**Responsabilidad $\gamma_{kl}$:** probabilidad de que la muestra $\mathbf{x}_k$ haya sido generada por el conglomerado $l$. Es el núcleo del paso E del algoritmo EM.

**Algoritmo EM:** esquema iterativo para maximizar la verosimilitud cuando existen variables latentes (asignación de muestras a conglomerados). Garantiza que la verosimilitud no decrece en cada iteración, pero puede converger a un máximo local.

**$n_l$:** número efectivo de puntos asignados al conglomerado $l$; $n_l = \sum_{k=1}^{N_i} \gamma_{kl}$.

### 3. Fórmulas y Ecuaciones Importantes

**Modelo GMM para clase $i$:**
$$\boxed{p(\mathbf{x}|\Theta_i) = \sum_{j=1}^{M} w_{ij}\,\mathcal{N}(\mathbf{x}|\mu_{ij}, \Sigma_{ij})}$$

**Función de verosimilitud (log-likelihood):**
$$\underset{\Theta}{\arg\max}\;\sum_{k=1}^{N_i}\log\sum_{j=1}^M w_{ij}\,\mathcal{N}(\mathbf{x}_k|\mu_{ij},\Sigma_{ij})$$

**Responsabilidad (Paso E):**
$$\boxed{\gamma_{kl} = \frac{w_{il}\,\mathcal{N}(\mathbf{x}_k|\mu_{il},\Sigma_{il})}{\sum_j w_{ij}\,\mathcal{N}(\mathbf{x}_k|\mu_{ij},\Sigma_{ij})}}$$

**Actualización de la media (Paso M):**
$$\boxed{\hat\mu_{il} = \frac{1}{n_l}\sum_{k=1}^{N_i}\gamma_{kl}\,\mathbf{x}_k} \quad \text{donde } n_l = \sum_{k=1}^{N_i}\gamma_{kl}$$

**Actualización de la covarianza (Paso M):**
$$\boxed{\hat\Sigma_{il} = \frac{1}{n_l}\sum_{k=1}^{N_i}\gamma_{kl}(\mathbf{x}_k - \mu_{il})(\mathbf{x}_k - \mu_{il})^T}$$

**Actualización de los pesos (Paso M):**
$$\boxed{\hat w_{il} = \frac{n_l}{N_i}}$$

### 4. Algoritmo Paso a Paso — EM para GMM

1. **Inicializar** los parámetros $\{w_{ij}^{(0)}, \mu_{ij}^{(0)}, \Sigma_{ij}^{(0)}\}$ (aleatorio o usando k-means).
2. **Paso E:** calcular la matriz de responsabilidades $\gamma$ de dimensión $N_i \times M$ usando la fórmula de $\gamma_{kl}$.
3. **Paso M:** actualizar todos los parámetros usando $\gamma$:
   - Calcular $n_l = \sum_k \gamma_{kl}$ para cada componente $l$.
   - Actualizar $\hat\mu_{il}$, $\hat\Sigma_{il}$, $\hat w_{il}$ con las fórmulas anteriores.
4. **Repetir** los pasos E y M hasta convergencia (la log-verosimilitud deja de crecer significativamente).

### 5. Hiperparámetros Clave

| Hiperparámetro | Efecto |
|---|---|
| $M$ (n_components) | Número de gaussianas por clase. Se selecciona con CV. Más componentes → más flexible, más riesgo de sobreajuste |
| `covariance_type` | `'full'`: matriz completa; `'diag'`: diagonal; `'spherical'`: escalar × identidad. Menor libertad → menos parámetros, menos riesgo de sobreajuste |
| `max_iter` | Número máximo de iteraciones EM. Pocas iteraciones → no converge |
| Inicialización (`means_init`) | Afecta fuertemente al máximo local encontrado |

### 6. Ventajas / Limitaciones

**Ventajas:**
- Modelo generativo: puede generar nuevas muestras.
- Modela distribuciones multimodales (múltiples conglomerados por clase).
- Proporciona probabilidades de pertenencia suaves (no asignaciones duras).
- Se puede usar para clasificación y para clustering.

**Limitaciones:**
- No garantiza convergencia al máximo global (sensible a la inicialización).
- El número de componentes $M$ es un hiperparámetro que hay que ajustar.
- Con `covariance_type='full'` el número de parámetros crece cuadráticamente con la dimensión: $(d(d+1)/2)$ por componente.
- Con pocas muestras, la estimación de $\Sigma$ puede ser inestable (necesita regularización).

### 7. Conexiones entre Métodos

- **GMM → K-means (Clase 09):** K-means es un caso especial del EM donde $w_j = 1/M$ para todos los componentes y la covarianza es esférica. Asignación dura en lugar de blanda.
- **GMM → Clasificación Bayesiana:** una vez estimadas las fdp por clase, la clasificación se hace con la regla de Bayes: asignar a la clase con mayor $P(w_i)P(\mathbf{x}|w_i)$.
- **GMM → Clustering (Clase 09):** EM sin etiquetas de clase = clustering blando.

### 8. Errores Conceptuales Comunes en Examen

- **Confundir GMM con una sola gaussiana.** Una sola gaussiana no puede modelar distribuciones multimodales. El GMM usa $M$ componentes.
- **Creer que el EM encuentra siempre el máximo global.** Solo garantiza que la log-verosimilitud NO DECRECE en cada iteración; puede quedar atrapado en un máximo local.
- **Las ecuaciones del paso M no son formas cerradas independientes.** Dependen de $\gamma_{kl}$, que a su vez depende de los parámetros. Por eso el esquema iterativo es necesario.
- **$\gamma_{kl}$ no es una asignación dura.** Es la probabilidad (número entre 0 y 1) de que la muestra $k$ pertenezca al conglomerado $l$. Todos los $\gamma_{kl}$ para un $k$ fijo suman 1.

### 9. Preguntas Tipo Examen

**P1:** ¿Por qué no se puede usar una sola gaussiana para modelar la distribución de una clase que tiene dos conglomerados separados?
> Porque una gaussiana solo puede representar una distribución unimodal (un solo máximo). Una clase con dos conglomerados tiene dos máximos, lo que requiere una mezcla de al menos dos gaussianas.

**P2:** ¿Qué garantiza el algoritmo EM en cada iteración y qué NO garantiza?
> Garantiza que la log-verosimilitud no decrece en cada iteración (E-step + M-step). No garantiza convergencia al máximo global; puede quedar en un máximo local dependiendo de la inicialización.

**P3:** ¿Qué significa $\gamma_{kl}$ y por qué tiene ese nombre de "responsabilidad"?
> $\gamma_{kl}$ es la probabilidad posterior de que la muestra $\mathbf{x}_k$ haya sido generada por el conglomerado $l$. Se llama "responsabilidad" porque mide cuánto "se responsabiliza" el componente $l$ de explicar la muestra $k$.

**P4:** ¿Cuál es la diferencia entre `covariance_type='full'` y `'spherical'`?
> `full`: cada componente tiene una matriz de covarianza completa (captura correlaciones entre variables). `spherical`: la covarianza es un escalar multiplicado por la identidad (asume varianza igual en todas las direcciones, sin correlaciones). `spherical` tiene menos parámetros y es más estable con pocas muestras.

**P5:** ¿Cómo se selecciona el hiperparámetro $M$ (número de componentes) en un GMM?
> Mediante validación cruzada, de manera análoga a como se selecciona $K$ en KNN: se entrena el modelo con diferentes valores de $M$ y se elige el que maximiza la métrica de desempeño en el conjunto de validación.

---

## CLASE 09 — Aprendizaje No Supervisado (Clustering)

### 1. Objetivo / Idea Central

El clustering agrupa muestras sin etiquetas en subconjuntos donde los elementos son más similares entre sí que con los de otros subconjuntos. Se estudian métodos jerárquicos/aglomerativos, K-means y métricas para elegir el número de clusters.

### 2. Conceptos Clave

**Aprendizaje No Supervisado:** las muestras solo tienen vector de características $\mathbf{x}_i$, sin variable de salida objetivo $y_i$.

**Clustering duro:** cada muestra se asigna exactamente a un cluster (K-means).

**Clustering blando:** cada muestra se asigna a diferentes clusters con grados de pertenencia (GMM/EM).

**Métodos aglomerativos/jerárquicos:** comienzan con cada muestra como su propio nodo y los van uniendo de a pares según similitud, hasta obtener un solo nodo. Representación visual: **dendrograma**.

**K-means:** minimiza la inercia (dispersión intra-cluster). Variante del EM con asignación dura y covarianzas esféricas iguales. Algoritmo de Lloyd para resolución iterativa.

**Inercia:** suma de distancias cuadráticas al centroide más cercano. Criterio que K-means minimiza.

**Diagrama de Voronoi:** representación visual de las regiones asignadas a cada centroide en K-means.

**Método del codo (Elbow):** graficar el porcentaje de varianza explicada vs. número de clusters $K$; el "codo" de la curva sugiere el número óptimo.

**Coeficiente Silhouette $S(i)$:** medida de coherencia del clustering. Rango $[-1, 1]$; valor 1 = perfectamente agrupado.

### 3. Fórmulas y Ecuaciones Importantes

**Función objetivo de K-means (inercia):**
$$\boxed{\underset{\mu_k}{\min}\;\sum_{k=1}^K\sum_{\mathbf{x}_k \in C_k}\|\mathbf{x}_k - \mu_k\|^2}$$

**Coeficiente Silhouette:**
$$\boxed{S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}}, \quad \text{si } |C_i| > 1$$

donde:
- $a(i) = \frac{1}{|C_i|-1}\sum_{j \in C_i, i \neq j} d(i,j)$ — distancia media intra-cluster
- $b(i) = \min_{k \neq i}\frac{1}{|C_k|}\sum_{j \in C_k} d(i,j)$ — distancia media mínima a otro cluster

**Porcentaje de varianza explicada (Elbow):**
$$\frac{\text{varianza entre grupos}}{\text{varianza total}} \times 100$$

**Estrategias de linkage jerárquico:**

| Nombre | Distancia entre clusters $A$ y $B$ |
|---|---|
| single | $\min_{a \in A, b \in B} d(a,b)$ |
| complete | $\max_{a \in A, b \in B} d(a,b)$ |
| average | $\frac{1}{|A||B|}\sum_{a,b} d(a,b)$ |
| Ward | $\sqrt{\frac{2|A||B|}{|A|+|B|}}\|\vec{c}_A - \vec{c}_B\|_2$ |
| centroid | $\|\vec{c}_A - \vec{c}_B\|_2$ |

### 4. Algoritmo Paso a Paso — K-means (Lloyd)

1. Elegir $K$ y **inicializar** los centroides $\mu_1,\ldots,\mu_K$ (recomendado: `init='k-means++'`).
2. **Asignación:** asignar cada muestra $\mathbf{x}_i$ al centroide más cercano: $c_i = \arg\min_k\|\mathbf{x}_i - \mu_k\|^2$.
3. **Actualización:** recalcular cada centroide como la media de las muestras asignadas: $\mu_k = \frac{1}{|C_k|}\sum_{\mathbf{x}_i \in C_k}\mathbf{x}_i$.
4. **Repetir** pasos 2-3 hasta convergencia (las asignaciones no cambian o la reducción de inercia es menor que un umbral).

**Recomendaciones sklearn:** usar `init='k-means++'` y `n_init=10` para múltiples inicializaciones.

### 5. Hiperparámetros Clave

| Hiperparámetro | Efecto |
|---|---|
| $K$ (n_clusters) | Número de clusters. Elegir con Elbow o Silhouette |
| `init` | `'random'`: sensible a inicialización; `'k-means++'`: más estable |
| `n_init` | Número de inicializaciones distintas; se devuelve la de menor inercia |
| Métrica de distancia | Euclidea (por defecto) o Minkowski; afecta la forma de los clusters |
| Método de linkage (jerárquico) | `single`, `complete`, `ward`, etc. |
| `color_threshold` en dendrogram | Umbral para cortar el dendrograma y obtener los clusters |

### 6. Ventajas / Limitaciones

**Métodos Aglomerativos:**
- (+) No requieren definir $K$ a priori; se decide cortando el dendrograma.
- (+) El dendrograma es informativo e interpretable.
- (-) Costo computacional alto; no escalan bien a datasets grandes.
- (-) No permiten backtracking.
- (-) Efecto de cadena (chain effect) en linkage `single`.

**K-means:**
- (+) Simple, eficiente, escala bien.
- (+) Interpretable: centroides son representativos.
- (-) Muy dependiente de la inicialización.
- (-) Asume clusters esféricos de igual varianza.
- (-) Hay que definir $K$ a priori.
- (-) Sensible a outliers (los centroides se mueven hacia ellos).

### 7. Conexiones entre Métodos

- **K-means → EM/GMM:** K-means es un caso especial de EM con asignación dura (argmax en lugar de probabilidades), varianzas esféricas iguales y pesos uniformes.
- **GMM → Clustering blando:** EM sin etiquetas de clase es un algoritmo de clustering que proporciona probabilidades de pertenencia en lugar de asignaciones duras.
- **Silhouette → Selección de $M$ en GMM:** se puede usar la misma lógica para elegir el número de componentes.

### 8. Errores Conceptuales Comunes en Examen

- **K-means minimiza la varianza inter-cluster.** FALSO: minimiza la inercia (varianza intra-cluster, dispersión dentro de cada cluster).
- **K-means siempre converge al óptimo global.** FALSO: es un método greedy que puede quedar en óptimos locales según la inicialización.
- **El coeficiente Silhouette puede ser negativo.** VERDADERO: un valor negativo indica que la muestra está más cerca del centroide de otro cluster que del suyo propio (asignación incorrecta).
- **Los métodos jerárquicos siempre dan el mismo resultado.** FALSO: el resultado depende fuertemente del método de linkage elegido.

### 9. Preguntas Tipo Examen

**P1:** ¿En qué se diferencia K-means de EM/GMM?
> K-means es una variante simplificada del EM con asignación dura (cada muestra va a exactamente un cluster), varianzas esféricas iguales y pesos uniformes. El EM/GMM usa asignaciones blandas (responsabilidades) y puede modelar covarianzas arbitrarias y pesos distintos.

**P2:** ¿Qué problema tiene el linkage `single` en métodos jerárquicos?
> Sufre el efecto de cadena (chain effect): si dos clusters están cerca en un solo punto, se unen aunque la mayoría de sus muestras estén lejos. Genera clusters alargados poco compactos.

**P3:** ¿Qué indica un Silhouette Score promedio cercano a 0?
> Que las muestras están en la frontera entre dos clusters, es decir, el clustering no tiene una estructura clara o el número de clusters elegido es incorrecto.

**P4:** ¿Cuál es la ventaja de K-means++ sobre la inicialización aleatoria?
> K-means++ inicializa los centroides en puntos distantes entre sí, reduciendo la probabilidad de converger a un mínimo local de alta inercia. Produce resultados más estables y de mejor calidad.

**P5:** ¿Por qué los métodos aglomerativos no escalan bien?
> Porque requieren calcular y actualizar la matriz de distancias entre todos los pares de clusters en cada paso de la unión. El costo computacional es $O(N^2)$ en memoria y $O(N^3)$ en tiempo en el peor caso.

---

## CLASE 10 — Árboles de Decisión, Voting, Bagging, Random Forest

### 1. Objetivo / Idea Central

Los árboles de decisión dividen el espacio de características con particiones binarias para predecir clases o valores continuos. Los métodos de ensamble (Voting, Bagging, Random Forest) reducen el error de varianza combinando múltiples modelos.

### 2. Conceptos Clave

**Árbol de decisión:** modelo que subdivide el espacio de características de forma recursiva usando reglas if-then sobre variables individuales (monoteísta). Nodos internos = variables + umbrales; hojas = predicción.

**ID3:** algoritmo básico de aprendizaje de árboles (Ross Quinlan, 1986). Construye de arriba hacia abajo usando una sola variable por nodo.

**Medida de impureza:** cuantifica la mezcla de clases en un nodo. Impureza 0 = nodo puro (todas las muestras de la misma clase).

**Ganancia de información:** reducción esperada de impureza debida a una partición. Guía la elección de la variable y el umbral en cada nodo.

**Podado (Pruning):** técnica para reducir el sobreajuste eliminando nodos que no mejoran el desempeño en el conjunto de validación.

**Voting:** ensambla modelos DISTINTOS; la predicción final es la moda (clasificación) o promedio (regresión) de las predicciones individuales.

**Bagging (Bootstrap Aggregating):** entrena $B$ modelos del MISMO tipo sobre $B$ muestras bootstrap (con reemplazo). Reduce la varianza.

**Random Forest:** Bagging de árboles con **selección aleatoria de $m$ variables en cada nodo**. Introduce mayor decorrelación entre árboles.

**Feature Importance:** promedio de la reducción de impureza producida por cada variable en todos los árboles del bosque.

### 3. Fórmulas y Ecuaciones Importantes

**Entropía (impureza de información):**
$$\boxed{I(U) = -\sum_j P(w_j)\log_2 P(w_j)}$$
- Mínima (= 0): nodo puro. Máxima: distribución uniforme entre clases.

**Impureza de Gini:**
$$\boxed{I(U) = \sum_{i \neq j} P(w_i)P(w_j)}$$

**Ganancia de información:**
$$\boxed{Gain(U, a) = I(U) - \left(I(U_L)\cdot P_L + I(U_R)\cdot P_R\right)}$$

**Error de regresión en nodo terminal $\tau_l$:**
$$\hat{y}(\tau_l) = \frac{1}{N(\tau_l)}\sum_{\mathbf{x}_i \in \tau_l} y_i$$

**Reducción del error en regresión por partición:**
$$\Delta R(\tau) = R(\tau) - R(\tau_L) - R(\tau_R)$$

**Criterio de regresión multi-salida (Mahalanobis):**
$$R = \frac{1}{N}\sum_{l=1}^L\sum_{\mathbf{x}_i \in \tau_l}(y_i - \hat{y}(\tau_l))^T\Sigma(\tau_l)^{-1}(y_i - \hat{y}(\tau_l))$$

### 4. Algoritmo Paso a Paso — Construcción de Árbol de Decisión

1. **Para el nodo actual** con conjunto de muestras $U$:
   - Para cada variable $a$ y cada posible umbral: calcular $Gain(U, a)$.
   - Seleccionar la variable y umbral con **mayor ganancia de información**.
2. **Partir** $U$ en $U_L$ y $U_R$ según la condición elegida.
3. **Crear** los nodos hijos izquierdo ($U_L$) y derecho ($U_R$).
4. **Repetir recursivamente** desde el paso 1 para cada nodo hijo.
5. **Condición de parada:** impureza = 0, o número mínimo de muestras por nodo, o profundidad máxima alcanzada.
6. **Post-procesamiento:** aplicar podado de error reducido si se requiere.

**Podado de Error Reducido:**
- Cada nodo interno es candidato a ser reemplazado por una hoja (con la clase mayoritaria).
- Se suprime el nodo si el árbol podado no empeora en el conjunto de validación.
- Se repite hasta que no se pueda eliminar ningún nodo sin perjudicar el desempeño.

### 5. Hiperparámetros Clave

| Hiperparámetro | Efecto |
|---|---|
| `max_depth` | Profundidad máxima. Mayor → más complejo → riesgo de sobreajuste |
| `criterion` | `'entropy'` (ganancia de información) vs `'gini'` |
| `min_samples_split` | Mínimo de muestras para dividir un nodo |
| `n_estimators` (Bagging/RF) | Número de árboles. Más → menor varianza (con rendimientos decrecientes) |
| `max_features` (RF) | Número $m$ de variables evaluadas en cada nodo. RF: típicamente $\sqrt{d}$ para clasificación, $d/3$ para regresión |

### 6. Ventajas / Limitaciones

**Árboles de Decisión:**
- (+) Interpretables; representables como reglas if-then.
- (+) No requieren normalización de datos.
- (+) Manejan variables categóricas y continuas.
- (-) Alta varianza: pequeños cambios en datos producen árboles muy distintos.
- (-) Fronteras solo paralelas a los ejes de características.
- (-) Fácilmente sobreajustables sin podado o limitación de profundidad.

**Bagging:**
- (+) Reduce la varianza del estimador base.
- (+) Funciona con cualquier estimador base.
- (-) No reduce el bias.
- (-) Los modelos pueden estar correlacionados si la variable dominante siempre se elige en la raíz.

**Random Forest:**
- (+) Reduce la correlación entre árboles → mayor reducción de varianza que Bagging puro.
- (+) Proporciona feature importance.
- (+) Robusto a sobreajuste con suficientes árboles.
- (-) Menos interpretable que un árbol individual.
- (-) Mayor costo computacional que un árbol solo.

### 7. Conexiones entre Métodos

- **Árbol de Decisión → Bagging:** Bagging es la generalización; usa bootstrap y promediado.
- **Bagging → Random Forest:** RF agrega selección aleatoria de $m$ variables por nodo sobre el Bagging.
- **Random Forest → Gradient Boosting (Clase 11):** ambos usan árboles; RF es paralelo y promedia; GBM es secuencial y ajusta residuales.
- **Árboles → Isolation Forest (Clase 12):** los árboles de aislamiento aplican la construcción de árboles para detectar anomalías.

### 8. Errores Conceptuales Comunes en Examen

- **Bagging reduce el bias.** FALSO: reduce la varianza promediando. El bias de cada árbol individual se mantiene.
- **Random Forest es simplemente Bagging con más árboles.** FALSO: RF agrega la selección aleatoria de $m < d$ variables por nodo, lo que decorrelaciona los árboles.
- **Un árbol con `max_depth=None` está bien regularizado.** FALSO: sin límite de profundidad, el árbol crece hasta impureza 0 → sobreajuste garantizado con datos ruidosos.
- **La ganancia de información y la impureza de Gini siempre eligen la misma partición.** No necesariamente; Gini es computacionalmente más eficiente (no requiere logaritmos).

### 9. Preguntas Tipo Examen

**P1:** ¿Por qué el Random Forest tiene menor varianza que un árbol de decisión individual?
> Porque promedia las predicciones de $B$ árboles entrenados sobre muestras bootstrap distintas con subconjuntos aleatorios de variables. El promedio de $B$ variables no correlacionadas reduce la varianza en factor $B$; la selección aleatoria de variables reduce la correlación entre árboles.

**P2:** ¿Cuál es la diferencia entre Voting y Bagging?
> Voting combina modelos base DISTINTOS (diferentes algoritmos); la combinación es por moda o promedio sin ningún mecanismo especial de entrenamiento. Bagging usa el MISMO modelo base entrenado sobre $B$ muestras bootstrap distintas.

**P3:** ¿Cuál es el criterio de partición en un árbol de regresión?
> La mayor reducción $\Delta R(\tau) = R(\tau) - R(\tau_L) - R(\tau_R)$, donde $R(\tau)$ es la suma de errores cuadráticos (MSE) en el nodo, y la predicción en cada hoja es la media de los valores $y_i$ de las muestras de ese nodo.

**P4:** ¿Cuántas variables evalúa Random Forest en cada nodo y por qué?
> Evalúa un subconjunto aleatorio de $m$ variables ($m < d$). Esto reduce la correlación entre árboles del bosque: si una variable es dominante, en Bagging puro todos los árboles la elegirían en la raíz → árboles correlacionados. Con $m < d$, algunos árboles no pueden elegirla y exploran otras variables.

**P5:** ¿Qué efecto tiene aumentar `n_estimators` en Random Forest?
> Reduce la varianza del modelo (mejor generalización) con rendimientos decrecientes. A partir de un cierto número de árboles, el desempeño se estabiliza. No aumenta el sobreajuste (a diferencia de profundidad).

---

## CLASE 11 — Boosting y Stacking

### 1. Objetivo / Idea Central

Boosting construye modelos de ensamble secuencialmente, haciendo que cada modelo siguiente se enfoque en los errores del anterior. Stacking usa un meta-modelo para aprender a combinar las predicciones de modelos base. A diferencia de Bagging (paralelo), Boosting es secuencial.

### 2. Conceptos Clave

**Boosting:** ensamble secuencial donde cada modelo base recibe mayor atención (mayor peso) sobre las muestras que el ensamble actual no predice bien.

**Clasificador débil:** un modelo que hace predicciones apenas mejor que azar. En la práctica se usan árboles con `max_depth` pequeño (stumps o árboles poco profundos).

**AdaBoost:** ajusta los pesos de las muestras en cada iteración. Las muestras mal clasificadas reciben mayor peso en la siguiente iteración.

**Gradient Boosting (GBM):** en lugar de pesos de muestras, ajusta residuales. Cada modelo base se entrena para predecir el gradiente negativo de la función de pérdida (los residuales del modelo actual).

**Residuales $r_m(i)$:** en GBM, el objetivo que debe aprender el modelo $m$ es $r_m(i) = -\frac{\partial L(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}$.

**Stacking:** meta-aprendizaje en dos capas. Capa 1: modelos base distintos. Capa 2: meta-modelo que aprende a combinar las salidas de capa 1.

**Diferencia Bagging vs Boosting:** Bagging → paralelo, reduce varianza. Boosting → secuencial, reduce bias (pero puede sobreajustar).

### 3. Fórmulas y Ecuaciones Importantes

**Modelo de ensamble (AdaBoost y GBM):**
$$F(\mathbf{x}) = \sum_{m=1}^M \alpha_m f_m(\mathbf{x})$$

**Error ponderado (AdaBoost):**
$$\epsilon_m = \sum_{i:\, f_m(\mathbf{x}_i) \neq y_i} w_m(i)$$

**Peso del clasificador (AdaBoost):**
$$\boxed{\alpha_m = \log\frac{1-\epsilon_m}{\epsilon_m}}$$
- Si $\epsilon_m < 0.5$ (mejor que azar): $\alpha_m > 0$.
- Si $\epsilon_m = 0.5$ (azar puro): $\alpha_m = 0$.
- Si $\epsilon_m > 0.5$ (peor que azar): $\alpha_m < 0$.

**Actualización de pesos de muestras (AdaBoost):**
$$\boxed{w_{m+1}(i) = w_m(i)\exp\left(-\alpha_m y_i f_m(\mathbf{x}_i)\right)}$$
- Muestra correctamente clasificada: $y_i f_m(\mathbf{x}_i) = +1$ → peso decrece.
- Muestra incorrectamente clasificada: $y_i f_m(\mathbf{x}_i) = -1$ → peso aumenta.

**Predicción final AdaBoost:**
$$F(\mathbf{x}) = \text{sign}\left(\sum_{m=1}^M \alpha_m f_m(\mathbf{x})\right)$$

**Actualización de función en GBM:**
$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m f_m(\mathbf{x})$$

**Residuales en GBM:**
$$\boxed{r_m(i) = -\frac{\partial L(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}}$$

**Paso de gradiente en GBM:**
$$\gamma_m = \arg\min_\gamma \sum_{i=1}^N L\left(y_i, F_{m-1}(\mathbf{x}_i) + \gamma f_m(\mathbf{x}_i)\right)$$

### 4. Algoritmo Paso a Paso

**AdaBoost:**
1. Inicializar pesos $w_1(i) = 1/N$ para todo $i$.
2. Para $m = 1$ a $M$:
   a. Entrenar clasificador $f_m$ con pesos $w_m$.
   b. Calcular error ponderado $\epsilon_m = \sum_{i: f_m(\mathbf{x}_i)\neq y_i} w_m(i)$.
   c. Calcular peso del clasificador $\alpha_m = \log\frac{1-\epsilon_m}{\epsilon_m}$.
   d. Actualizar pesos $w_{m+1}(i) = w_m(i)\exp(-\alpha_m y_i f_m(\mathbf{x}_i))$.
   e. Normalizar $w_{m+1}$ para que sume 1.
3. Predicción: $F(\mathbf{x}) = \text{sign}\left(\sum_m \alpha_m f_m(\mathbf{x})\right)$.

**Gradient Boosting:**
1. Inicializar $F_0(\mathbf{x}) = 0$ (o constante óptima).
2. Para $m = 1$ a $M$:
   a. Calcular pseudo-residuales $r_m(i) = -\partial L / \partial F_{m-1}(\mathbf{x}_i)$.
   b. Ajustar un árbol de decisión $f_m$ sobre los residuales $\{(\mathbf{x}_i, r_m(i))\}$.
   c. Encontrar el paso óptimo $\gamma_m$ por line search.
   d. Actualizar $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m f_m(\mathbf{x})$.
3. Devolver $F_M(\mathbf{x})$.

### 5. Hiperparámetros Clave

| Hiperparámetro | Efecto |
|---|---|
| `n_estimators` ($M$) | Número de modelos base. Más → mejor hasta cierto punto; luego sobreajuste (especialmente en Boosting) |
| `max_depth` del árbol base | Menor profundidad → clasificador más débil → menor sobreajuste |
| `learning_rate` ($\gamma$ en GBM) | Controla la magnitud del paso. Menor → más conservador, necesita más estimadores |
| `loss` (GBM) | Función de pérdida: `'squared_error'`, `'absolute_error'`, `'huber'` |
| `subsample` (Stochastic GBM) | Fracción de muestras usadas por árbol. < 1 → regularización |

### 6. Ventajas / Limitaciones

**AdaBoost:**
- (+) Teóricamente bien fundamentado; reduce bias.
- (+) No requiere ajuste de learning rate.
- (-) Muy sensible a outliers (les asigna pesos altos y los memoriza).
- (-) Puede sobreajustar si $M$ es demasiado grande.

**Gradient Boosting:**
- (+) Flexible: funciona con cualquier función de pérdida diferenciable.
- (+) Uno de los métodos más poderosos para datos tabulares.
- (-) Más lento de entrenar que Random Forest (secuencial).
- (-) Requiere ajuste cuidadoso de hiperparámetros (`n_estimators`, `learning_rate`, `max_depth`).
- (-) Puede sobreajustar; necesita regularización.

**Stacking:**
- (+) Puede capturar las fortalezas de modelos completamente distintos.
- (+) Muy flexible en la elección del meta-modelo.
- (-) Más costoso de entrenar y validar correctamente.
- (-) Riesgo de leakage si el meta-modelo se entrena con las mismas muestras que los modelos base (usar cross-val stacking).

### 7. Conexiones entre Métodos

- **Bagging → Boosting:** ambos son métodos de ensamble basados en el mismo modelo base; Bagging es paralelo y reduce varianza, Boosting es secuencial y reduce bias.
- **GBM → XGBoost/LightGBM/CatBoost:** variantes modernas con regularización adicional, manejo eficiente de memoria y velocidad de entrenamiento (mencionados en el curso para consulta).
- **AdaBoost → GBM:** GBM generaliza AdaBoost; con pérdida exponencial y árboles de profundidad 1, GBM es equivalente a AdaBoost.
- **Stacking → Voting:** Voting combina con moda/promedio; Stacking aprende la combinación óptima con un meta-modelo.

### 8. Errores Conceptuales Comunes en Examen

- **Boosting y Bagging reducen los mismos tipos de error.** FALSO: Bagging reduce principalmente varianza; Boosting reduce principalmente bias.
- **Agregar más estimadores en Boosting nunca sobreajusta.** FALSO: a diferencia de Bagging/RF (donde más árboles ayudan o no empeoran), en Boosting demasiados estimadores pueden sobreajustar.
- **En AdaBoost, todos los clasificadores tienen el mismo peso $\alpha_m$.** FALSO: $\alpha_m = \log\frac{1-\epsilon_m}{\epsilon_m}$ varía según el error del clasificador.
- **GBM ajusta los residuales directamente.** PARCIALMENTE CORRECTO: para pérdida cuadrática los residuales son $y_i - F_{m-1}(\mathbf{x}_i)$ exactamente; para otras funciones de pérdida son los pseudo-residuales (gradiente negativo de $L$).
- **Stacking es solo promedio de modelos.** FALSO: Stacking usa un meta-modelo entrenado (ej. Random Forest) que aprende a combinar óptimamente las salidas de capa 1. Voting es el que promedia.

### 9. Preguntas Tipo Examen

**P1:** ¿Cuál es la diferencia fundamental entre AdaBoost y Gradient Boosting?
> AdaBoost ajusta los pesos de las muestras en cada iteración, dando más importancia a las mal clasificadas. GBM ajusta los pseudo-residuales (gradiente negativo de la función de pérdida), entrenando cada árbol para corregir el error del modelo acumulado hasta ese punto.

**P2:** ¿Qué indica $\alpha_m = 0$ en AdaBoost?
> Que el clasificador $m$ tiene $\epsilon_m = 0.5$, es decir, clasifica al azar (no mejor que chance). Un clasificador con ese peso no aporta información al ensamble final.

**P3:** ¿Por qué Boosting puede sobreajustar y Bagging generalmente no?
> En Bagging, promediar $B$ estimadores independientes reduce la varianza monotónicamente; no sobreajusta al aumentar $B$. En Boosting, cada modelo está correlacionado con los anteriores (se enfoca en los mismos errores acumulados), y con muchas iteraciones el ensamble puede memorizar el ruido.

**P4:** ¿Qué diferencia Stacking de Voting?
> En Voting, la combinación es mecánica (moda o promedio simple). En Stacking, la combinación es aprendida: un meta-modelo de segunda capa recibe como entradas las predicciones de los modelos base y aprende la combinación óptima para el problema específico.

**P5:** ¿Qué son los pseudo-residuales en Gradient Boosting?
> Son el gradiente negativo de la función de pérdida evaluado en la predicción actual del modelo: $r_m(i) = -\frac{\partial L(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}$. Para pérdida cuadrática coinciden con los residuales ordinarios $y_i - F_{m-1}(\mathbf{x}_i)$.

---

## CLASE 12 — Isolation Forest

### 1. Objetivo / Idea Central

Isolation Forest detecta anomalías aprovechando que las muestras atípicas son más fáciles de aislar que las normales. En lugar de modelar lo "normal" y buscar lo que no encaja, construye árboles de partición aleatoria y mide cuántas divisiones se necesitan para aislar cada punto. Aplicaciones: detección de fraude, tráfico malicioso, anomalías en general.

### 2. Conceptos Clave

**Detección de anomalías (One-Class Classification):** identificar muestras que se desvían significativamente del comportamiento esperado, sin necesidad de etiquetas de clase "anómala" en el entrenamiento.

**Premisa del Isolation Forest:** las anomalías son pocas y distintas de la mayoría. Por lo tanto, son más fáciles de separar del resto mediante particiones aleatorias.

**Árbol de aislamiento (Isolation Tree):** árbol de decisión construido con variables y umbrales seleccionados aleatoriamente (no por ganancia de información). Las particiones se aplican recursivamente hasta aislar cada punto o alcanzar la profundidad máxima.

**Longitud de ruta (path length):** número de particiones necesarias para aislar una muestra en un árbol de aislamiento.

**Anomaly Score:** promedio normalizado de las longitudes de ruta en todos los árboles del bosque. Valores bajos de score (cercanos a -1 en sklearn o valores pequeños de path length) indican anomalías.

**`decision_function`:** en sklearn, devuelve scores donde valores negativos = más anómalo, valores positivos = más normal.

**`contamination`:** hiperparámetro que indica la proporción esperada de anomalías en el dataset; define el umbral de decisión.

**Partición aleatoria:** en cada nodo se selecciona una variable al azar y un umbral uniforme entre el mínimo y máximo de esa variable.

### 3. Fórmulas y Ecuaciones Importantes

**Intuición del score:**
- Anomalías: longitud de ruta corta → pocas particiones para aislarlas.
- Puntos normales: longitud de ruta larga → muchas particiones necesarias.

**Score de anomalía (normalizado):**
$$\text{score}(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

donde $E[h(x)]$ es la longitud de ruta promedio en el bosque y $c(n)$ es el factor de normalización basado en la longitud esperada de un BST (árbol de búsqueda binaria) con $n$ muestras:
$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$
con $H(i) = \ln(i) + 0.5772$ (número armónico).

- $\text{score} \approx 1$: anomalía clara.
- $\text{score} \approx 0.5$: punto indistinguible.
- $\text{score} \ll 0.5$: punto normal.

### 4. Algoritmo Paso a Paso — Isolation Forest

**Entrenamiento:**
1. Para cada uno de los $T$ árboles (n_estimators):
   a. Seleccionar aleatoriamente una submuestra de tamaño $\psi$ del dataset.
   b. Construir un árbol de aislamiento:
      - En cada nodo: seleccionar una variable aleatoria $q$ y un umbral $p$ aleatorio entre $[\min(q), \max(q)]$.
      - Partir las muestras según $q \leq p$ (izquierda) y $q > p$ (derecha).
      - Repetir recursivamente hasta que cada nodo tenga una sola muestra o se alcance la profundidad máxima $\lfloor\log_2\psi\rfloor$.
2. El bosque queda formado por los $T$ árboles de aislamiento.

**Predicción (score):**
1. Para cada muestra $\mathbf{x}$: recorrer cada árbol y medir la profundidad $h_t(\mathbf{x})$ hasta llegar a la hoja.
2. Calcular $E[h(\mathbf{x})]$ = promedio de las longitudes de ruta en los $T$ árboles.
3. Normalizar usando $c(n)$ para obtener el anomaly score.
4. Clasificar como anomalía si el score supera el umbral definido por `contamination`.

### 5. Hiperparámetros Clave

| Hiperparámetro | Efecto |
|---|---|
| `n_estimators` | Número de árboles. Más árboles → estimado de path length más estable. 100 es suficiente en la mayoría de los casos |
| `contamination` | Proporción esperada de anomalías. Define el umbral de decisión. Si no se conoce, usar `'auto'` |
| `max_samples` | Tamaño $\psi$ de la submuestra por árbol. Valor por defecto: `min(256, n_samples)`. Submuestras pequeñas aceleran el entrenamiento y mejoran la detección |
| `max_features` | Número de características disponibles en cada árbol |
| `random_state` | Semilla para reproducibilidad |

### 6. Ventajas / Limitaciones

**Ventajas:**
- No requiere etiquetas de anomalías en el entrenamiento (no supervisado / one-class).
- Eficiente computacionalmente: complejidad lineal en número de muestras.
- No requiere modelar la distribución de los datos normales.
- Escala bien a datasets grandes y de alta dimensión.
- Robusto a la maldición de la dimensionalidad (comparado con métodos basados en distancias).

**Limitaciones:**
- Dificultad con anomalías en regiones densas de datos normales (anomalías locales).
- Sensible al hiperparámetro `contamination`; si se desconoce la proporción real, el umbral puede ser incorrecto.
- Las particiones son completamente aleatorias: no aprende estructura de los datos normales, solo explota la facilidad de aislamiento.
- No proporciona explicaciones directas de por qué un punto es anómalo.

### 7. Conexiones entre Métodos

- **Isolation Forest → Árboles de Decisión (Clase 10):** usa la misma estructura de árbol pero con particiones completamente aleatorias (sin criterio de impureza).
- **Isolation Forest → Random Forest:** ambos son bosques de árboles; RF usa criterio de ganancia de información y clasifica con etiquetas; IF usa particiones aleatorias y mide longitud de ruta.
- **Isolation Forest → GMM (Clase 08):** el enfoque alternativo para one-class classification es modelar la fdp de los datos normales (GMM) y marcar como anómalos los puntos de baja verosimilitud. IF es más eficiente computacionalmente.
- **Detección de anomalías → Aplicaciones prácticas:** fraude financiero, tráfico de red malicioso, control de calidad industrial.

### 8. Errores Conceptuales Comunes en Examen

- **Isolation Forest es un clasificador supervisado.** FALSO: es no supervisado / one-class. Se entrena SOLO con datos normales (o con la mezcla, usando `contamination` para estimar la proporción de anómalos).
- **Path length largo = anomalía.** FALSO: es exactamente al revés. Path length CORTO = anomalía (fue fácil de aislar). Path length LARGO = punto normal (difícil de separar de la masa de datos).
- **Los árboles de Isolation Forest usan ganancia de información.** FALSO: las variables y umbrales se eligen de forma completamente aleatoria, sin ningún criterio de calidad de partición.
- **`decision_function` positivo = anómalo en sklearn.** FALSO: en sklearn, `decision_function` devuelve scores donde valores MÁS NEGATIVOS corresponden a anomalías. El umbral es 0 (por defecto ajustado por `contamination`).

### 9. Preguntas Tipo Examen

**P1:** ¿Cuál es la premisa fundamental del Isolation Forest y por qué funciona para detectar anomalías?
> La premisa es que las anomalías son pocas y distintas del resto de los datos, por lo que son más fáciles de aislar mediante particiones aleatorias. Un punto anómalo queda aislado con pocas particiones (path length corto), mientras que un punto normal en una región densa necesita muchas particiones para ser aislado.

**P2:** ¿Cómo difiere la construcción de un árbol de Isolation Forest de un árbol de decisión estándar?
> En un árbol de decisión estándar, la variable y el umbral de partición se eligen maximizando la ganancia de información (o reducción de impureza). En Isolation Forest, tanto la variable como el umbral se eligen completamente al azar (uniforme entre el mínimo y máximo de la variable), sin ningún criterio de calidad.

**P3:** ¿Qué representa el `anomaly_strength = -scores` calculado en el notebook?
> Es la inversión del `decision_function` de sklearn para que valores ALTOS indiquen mayor anomalía. En sklearn, `decision_function` devuelve valores más negativos para anomalías, por lo que negar el score lo convierte en una medida directamente proporcional al grado de anomalía.

**P4:** ¿Por qué Isolation Forest es eficiente computacionalmente comparado con métodos basados en densidad?
> Porque la construcción de cada árbol de aislamiento tiene complejidad $O(n\log n)$ y el path length se calcula en $O(\log n)$ por muestra. Los métodos basados en densidad (como GMM o KDE) requieren estimar y evaluar distribuciones de probabilidad complejas, que escalan peor con la dimensionalidad y el tamaño del dataset.

**P5:** ¿Qué efecto tiene el hiperparámetro `contamination` sobre el modelo?
> Define la proporción esperada de anomalías en el dataset. Se usa para calcular el umbral del score a partir del cual una muestra se clasifica como anómala. Si `contamination` es mayor que la proporción real de anomalías, el modelo marcará algunos puntos normales como anómalos (falsos positivos). Si es menor, dejará pasar anomalías reales (falsos negativos).

---

## TABLA RESUMEN COMPARATIVO

| Método | Tipo | Reduce | Hiperparámetro clave | Uso típico |
|---|---|---|---|---|
| K-fold CV | Validación | — | $k$ (num folds) | Selección de hiperparámetros |
| Regularización L2 | Penalización | Varianza | $\lambda$ | Regresión/logística con sobreajuste |
| Regularización L1 | Penalización + selección | Varianza + dims | $\lambda$ | Selección de features implícita |
| GMM + EM | Generativo | — | $M$ (componentes) | Clasificación multimodal, clustering |
| K-means | Clustering duro | — | $K$ | Agrupamiento no supervisado |
| Árbol de Decisión | Supervisado | Bias | `max_depth` | Clasificación/regresión interpretable |
| Bagging | Ensamble paralelo | Varianza | `n_estimators` | Reducir varianza de modelos inestables |
| Random Forest | Ensamble paralelo | Varianza | `n_estimators`, `max_features` | Clasificación/regresión general |
| AdaBoost | Ensamble secuencial | Bias | `n_estimators` | Clasificación con clases difíciles |
| Gradient Boosting | Ensamble secuencial | Bias + Varianza | `n_estimators`, `learning_rate`, `max_depth` | Tabular data, competencias |
| Stacking | Meta-aprendizaje | Ambos | Meta-modelo | Combinación óptima de modelos distintos |
| Isolation Forest | Detección anomalías | — | `contamination`, `n_estimators` | Fraude, intrusiones, outliers |

---

*Apuntes construidos exclusivamente a partir del contenido de los notebooks de las Clases 06 a 12. No se incluyó contenido externo.*agentId: abcf175aaaa344b5f (use SendMessage with to: 'abcf175aaaa344b5f' to continue this agent)
<usage>subagent_tokens: 86988
tool_uses: 7
duration_ms: 328145</usage>

---

# Bloque III — Modelos no lineales potentes (Clases 13 a 17)

## CLASE 13 — Redes Neuronales Artificiales (MLP)

### Idea central
Unidades (perceptrones) interconectadas que aprenden representaciones no lineales. Resuelven lo que un clasificador lineal no puede (ej. XOR).

### Conceptos clave
- **Perceptrón:** $O(\mathbf{x})=\text{sgn}(\mathbf{w}^T\mathbf{x})$. Equivale a regresión logística con salida discreta.
- **Capas ocultas:** permiten fronteras complejas; su salida deseada es desconocida (de ahí "ocultas").
- **Backpropagation:** forward (activaciones) + backward (propagar error con regla de la cadena).
- **$\delta_j$:** sensibilidad del error a la activación pre-sináptica. Salida: $\delta_k=y_k-t_k$. Oculto: $\delta_j=\dot h(a_j)\sum_k w_{kj}\delta_k$.
- **Batch / mini-batch / online (SGD):** todas las muestras / subconjunto / una a una. Online escapa de saddle points.

### Fórmulas
- Regla delta: $\Delta w_i=\eta(y_j-O(\mathbf{x}_j))x_{ji}$
- $a_j=\sum_i w_{ji}^{(1)}x_i$, $z_j=h(a_j)$
- $E_n=\frac12\sum_k(y_{nk}-t_{nk})^2$
- $\frac{\partial E_n}{\partial w_{ji}}=\delta_j z_i$ · $\delta_j=\dot h(a_j)\sum_k w_{kj}\delta_k$
- Derivada sigmoide: $f'(u)=f(u)(1-f(u))$

### Algoritmo backprop
Forward (calcular $a_j,z_j$) → $\delta_k=y_k-t_k$ en salida → propagar $\delta$ hacia atrás → gradientes $\delta_j z_i$ → update $w_{ji}\leftarrow w_{ji}-\eta\delta_j z_i$ → repetir por épocas.

### Errores comunes
- Confundir $\delta$ de salida ($y_k-t_k$) con $\delta$ oculto (lleva $\dot h$ y suma ponderada).
- Creer que backprop garantiza mínimo **global** (NO: hay mínimos locales).
- Usar sgn en capas ocultas (no derivable → no se puede backprop).

### Preguntas tipo examen
1. **¿Por qué el perceptrón no resuelve XOR y el MLP sí?** XOR no es linealmente separable; la capa oculta crea fronteras compuestas.
2. **¿Por qué sgn → sigmoide?** sgn no es derivable; la sigmoide sí ($f(1-f)$).
3. **Batch vs mini-batch vs online.** Estable/lento vs intermedio vs ruidoso/escapa óptimos.
4. **¿Qué es $\delta_j$ y su fórmula oculta?** Sensibilidad del error; $\delta_j=\dot h(a_j)\sum_k w_{kj}\delta_k$.
5. **Activación de salida: regresión vs clasificación.** Lineal vs sigmoide/softmax.

---

## CLASE 14 — Mapas Auto-Organizables (SOM / Kohonen)

### Idea central
Red **no supervisada** de **aprendizaje competitivo**: proyecta datos de alta dimensión a una cuadrícula 1D/2D preservando la topología (mapa topográfico).

### Conceptos clave
- **Neurona ganadora:** $i(\mathbf{x})=\arg\min_j\|\mathbf{x}-\mathbf{w}_j\|$.
- **Vecindad topológica $h_{j,i}$:** gaussiana centrada en la ganadora; se reduce con el tiempo.
- **U-Matrix:** visualiza distancias entre neuronas vecinas → identifica clusters.
- Tanto $\eta$ como $\sigma$ **decaen** con el tiempo.

### Fórmulas
- $h_{j,i}=\exp(-\frac{d_{j,i}^2}{2\sigma^2})$ · $\sigma(t)=\sigma_0 e^{-t/\tau_1}$ · $\eta(t)=\eta_0 e^{-t/\tau_2}$ (típicos $\eta_0=0.1,\tau_2=1000$)
- Update: $\mathbf{w}_j(t{+}1)=\mathbf{w}_j(t)+\eta(t)h_{j,i}(t)(\mathbf{x}(t)-\mathbf{w}_j(t))$

### Algoritmo (3 fases)
**Competición** (ganadora) → **Cooperación** (vecindad $h$) → **Adaptación** (update de todas, ponderado por $h$). Decrementar $\sigma,\eta$. Repetir.

### Errores comunes
- Confundir $d_{j,i}$ (en la cuadrícula) con $\|\mathbf{x}-\mathbf{w}_j\|$ (en espacio de entrada).
- Olvidar que $\eta$ **y** $\sigma$ decaen.
- Decir que requiere etiquetas (es no supervisado).
- Creer que solo se actualiza la ganadora (se actualizan **todas**).

### Preguntas tipo examen
1. **¿Las 3 fases del SOM?** Competición, cooperación, adaptación.
2. **¿Por qué la vecindad se reduce con el tiempo?** Organización global primero, ajuste fino después.
3. **¿Cuándo mín. distancia = máx. producto interno?** Con vectores normalizados.
4. **¿Qué es la U-Matrix?** Distancia promedio entre neuronas vecinas; revela fronteras de clusters.
5. **¿SOM vs K-means?** SOM captura estructuras no convexas (anillos) preservando topología.

---

## CLASE 15 — Redes Neuronales Recurrentes (RNN / LSTM)

### Idea central
RNN procesan **secuencias** con estado interno y parámetros compartidos en el tiempo. LSTM resuelve el vanishing/exploding gradient con compuertas.

### Conceptos clave
- **RNN (Elman):** matrices $\mathbf{U}$ (entrada→oculta), $\mathbf{V}$ (oculta→oculta), $\mathbf{W}$ (oculta→salida).
- **BPTT:** backprop sobre el grafo desplegado en el tiempo (no paralelizable). **Truncated BPTT:** corta la propagación temporal.
- **Vanishing gradient:** $\mathbf{V}$ se multiplica consigo misma $(\tau-1)$ veces; si <1 → 0; si >1 → explota.
- **LSTM:** compuertas **forget / input / output** (sigmoide ∈ (0,1)) + estado de célula $c^{(t)}$.
- **EWMA:** $\mu^{(t)}=\beta\mu^{(t-1)}+(1-\beta)\upsilon^{(t)}$ (inspira la compuerta forget).
- **BRNN:** bidireccional (pasado + futuro); no sirve para tiempo real.

### Fórmulas
- Elman: $\mathbf{a}^{(t)}=\mathbf{b}+\mathbf{V}\mathbf{h}^{(t-1)}+\mathbf{U}\mathbf{x}^{(t)}$, $\mathbf{h}^{(t)}=\tanh(\mathbf{a}^{(t)})$, $\mathbf{o}^{(t)}=\mathbf{c}+\mathbf{W}\mathbf{h}^{(t)}$
- Estado célula LSTM: $c_l^{(t)}=f_l^{(t)}c_l^{(t-1)}+i_l^{(t)}\tanh(\cdot)$ · salida: $h_l^{(t)}=\tanh(c_l^{(t)})o_l^{(t)}$

### Errores comunes
- Confundir estado de célula $c^{(t)}$ (largo plazo) con salida $h^{(t)}$ (corto plazo).
- Olvidar que en BPTT los gradientes **se acumulan** ($\sum_t$) por parámetros compartidos.
- Decir que BRNN sirve para tiempo real (usa el futuro).

### Preguntas tipo examen
1. **¿Por qué las RNN no aprenden dependencias largas?** Vanishing/exploding gradient + saturación de tanh.
2. **Rol de cada compuerta LSTM.** Forget (cuánto olvidar), input (cuánto incorporar), output (cuánto exponer).
3. **BPTT vs Truncated BPTT.** Toda la secuencia vs ventana cortada (menos memoria).
4. **¿Para qué las BRNN y por qué no en tiempo real?** Necesitan contexto futuro (voz, traducción).

---

## CLASE 16 — Máquinas de Vectores de Soporte (SVM)

### Idea central
Frontera de **máximo margen**. Solo importan los **vectores de soporte**. Se resuelve el **dual** → **kernel trick** para fronteras no lineales.

### Conceptos clave
- **Vectores de soporte:** muestras con $a_n>0$ (las únicas en la función de decisión).
- **C (soft-margin):** alto → pocos SV, margen chico, sobreajuste; bajo → muchos SV, margen amplio.
- **SVR:** regresión con tubo $\epsilon$-insensitivo. **One-Class SVM:** anomalías (formulación $\nu$).

### Fórmulas
- Primal: $\min\frac12\|\mathbf{w}\|^2$ s.a. $t_n(\mathbf{w}^T\phi(\mathbf{x}_n)+b)\ge1$
- Dual: $\max\sum_n a_n-\frac12\sum_{n,m}a_n a_m t_n t_m k(\mathbf{x}_n,\mathbf{x}_m)$, $0\le a_n\le C$, $\sum_n a_n t_n=0$
- Decisión: $y(\mathbf{x})=\sum_n a_n t_n k(\mathbf{x},\mathbf{x}_n)+b$
- KKT: $a_n\{t_n y(\mathbf{x}_n)-1\}=0$
- RBF: $k(\mathbf{x}_n,\mathbf{x}_m)=\exp(-\|\mathbf{x}_n-\mathbf{x}_m\|^2/2\sigma^2)$

### Errores comunes
- Decir que todos los puntos son SV (solo $a_n>0$).
- Invertir el efecto de C (C alto = margen chico, pocos SV).
- Creer que se resuelve el primal (se resuelve el **dual**, habilita kernels).

### Preguntas tipo examen
1. **¿Qué son los SV y por qué solo ellos importan?** $a_n>0$; por KKT, solo los del margen tienen $a_n\neq0$.
2. **¿Por qué el dual?** La función depende de $\phi^T\phi$ → reemplazable por $k$ (kernel trick).
3. **Efecto de C.** Compromiso margen vs errores; alto = poco margen, sobreajuste.
4. **Condiciones KKT.** $a_n\ge0$; $t_n y-1\ge0$; $a_n\{t_n y-1\}=0$.
5. **¿Cómo extiende SVM a regresión (SVR)?** Función $\epsilon$-insensitiva + tubo.

---

## CLASE 17 — Estrategias Multiclase

### Idea central
Extender clasificadores binarios a $N_c>2$ clases: **OvR**, **OvO**, **jerárquica**.

### Conceptos clave / conteo
- **OvR (One vs Rest):** $N_c$ clasificadores; sirve para **multi-label**; sufre **desbalance**.
- **OvO (One vs One):** $\frac{N_c(N_c-1)}{2}$ clasificadores; datasets balanceados; escala mal.
- **Jerárquica:** $N_c-1$ clasificadores; árbol manual; error cascading.
- **Multi-class** (una clase por muestra) vs **multi-label** (varias clases por muestra).

### Errores comunes
- Confundir multi-class con multi-label.
- Calcular mal el nº de clasificadores.
- Olvidar el desbalance de OvR.
- Creer que la jerarquía es automática (es manual).

### Preguntas tipo examen
1. **5 clases: ¿clasificadores por estrategia?** OvR 5, OvO 10, jerárquica 4.
2. **Multi-class vs multi-label + estrategia.** OvO/OvR/jerárquica vs OvR (natural).
3. **¿Por qué OvR genera desbalance?** Positiva $n_i$ vs negativa $N-n_i$ (domina).
4. **Mecanismo de predicción OvO.** Votación; gana la clase con más votos.
5. **Limitación de la jerárquica.** Árbol manual + propagación de errores.

---

# Bloques IV y V — Reducción de dimensión y Explicabilidad (Clases 18 a 23)

## CLASE 18 — Selección de Características

### Idea central
Encontrar el mejor **subconjunto** de variables originales (conserva interpretación). Distinto de extracción (crea variables nuevas).

### Conceptos clave
- **Filtro:** evalúa info intrínseca (distancias, correlación, info mutua), sin modelo. Rápido, general.
- **Wrapper:** usa un modelo como criterio (1−error con validación). Exacto, lento.
- **Embedded:** la selección ocurre durante el entrenamiento (ej. LASSO).
- **SFS** (forward), **SBS** (backward), **LRS** (más-L menos-R), **SFFS** (flotante, L y R variables). Todos subóptimos.
- **Fuerza bruta:** $\binom{d}{p}$ subconjuntos (inviable).

### Fórmulas
- Nº subconjuntos: $n_p=\frac{d!}{(d-p)!p!}$
- Criterio tipo Fisher: $J=\frac{\text{Tr}\{S_B\}}{\text{Tr}\{S_W\}}$
- Info mutua: $I(X_m;c)=H(c)-H(c|X_m)$ (capta relaciones no lineales)

### Errores comunes
- Creer que analizar cada variable individualmente basta (pueden complementarse).
- Confundir selección (originales) con extracción (nuevas).
- Pensar que SFS/SBS dan el óptimo (son heurísticos).

### Preguntas tipo examen
1. **¿Por qué no basta analizar variables individualmente?** Una "mala" sola puede ser buena combinada.
2. **Filtro vs wrapper.** Intrínseco/rápido/general vs con modelo/exacto/lento.
3. **Limitación de SFS.** No puede remover variables que se vuelven redundantes.
4. **Ventaja de SFFS sobre LRS.** L y R se determinan de los datos (no fijos).
5. **¿Subconjuntos de 10 sobre 25?** $\binom{25}{10}=3.268.760$.

---

## CLASE 19 — LASSO y Redes Elásticas

### Idea central
Regresión con penalización **L1** → coeficientes **exactamente cero** → regresión + selección **embebida**. $\lambda$ por validación cruzada.

### Conceptos clave
- **LASSO (L1):** lleva coeficientes a cero (sparse). **Ridge (L2):** solo los encoge.
- **Elastic Net:** L1+L2; mejor con features correlacionadas o $d\gg N$.
- **LASSO path:** coeficientes vs $\lambda$. Algoritmos: LARS, coordenada descendente.

### Fórmulas
- LASSO: $\min\sum_i(y_i-\sum_j w_j x_{ij})^2+\lambda\sum_{j\ge1}|w_j|$ (no penaliza $w_0$)
- Restricción: $\sum_j|w_j|\le s$
- Elastic Net: $+\lambda_1\sum|w_j|+\lambda_2\sum w_j^2$
- $\lambda=0$ → mínimos cuadrados; $\lambda\to\infty$ → todos los $w_j\to0$.

### Errores comunes
- Confundir Ridge (encoge) con LASSO (anula).
- Creer que se penaliza $w_0$ (no).
- Olvidar que Elastic Net tiene 2 hiperparámetros (`alpha`, `l1_ratio`).

### Preguntas tipo examen
1. **¿Por qué L1 da sparsidad?** Región diamante; vértices sobre los ejes (algunos $w_j=0$).
2. **Efecto de $\lambda=0$ y $\lambda\to\infty$.** Mínimos cuadrados / todos a cero.
3. **¿Por qué Elastic Net > LASSO con correlación?** L2 da coeficientes similares a correlacionadas; LASSO elige una arbitraria.
4. **¿Cómo se elige $\lambda$?** Validación cruzada.
5. **¿Por qué LASSO es embebido?** La selección ocurre durante el entrenamiento.

---

## CLASE 20 — Análisis de Componentes Principales (PCA)

### Idea central
Extracción **no supervisada** que maximiza **varianza**. Componentes = autovectores de la covarianza, ordenados por autovalor.

### Conceptos clave
- Varianza del $k$-ésimo componente = autovalor $\lambda_k$.
- **Centrar los datos** antes (media 0).
- Limitación: máxima varianza ≠ mejor separación de clases.
- **EigenFaces:** PCA en reconocimiento facial.

### Fórmulas
- $\mathbf{S}=\frac1N\mathbf{X}^T\mathbf{X}$ · problema: $\mathbf{S}\mathbf{u}=\lambda\mathbf{u}$
- Proyección: $z_i=\mathbf{u}_1^T\mathbf{x}_i$ · varianza acumulada $\frac{\sum_{k\le p}\lambda_k}{\sum_k\lambda_k}$

### Algoritmo
Centrar → covarianza $\mathbf{S}$ → autovalores/autovectores → ordenar → tomar $p$ primeros → proyectar $\mathbf{Z}=\mathbf{X}\mathbf{U}_p$.

### Selección de $p$
Codo de la curva de $\lambda_i$ · varianza acumulada (80/90/95%) · descartar $\lambda_i$ < media.

### Errores comunes
- No centrar antes de PCA.
- Confundir $\lambda_k$ con otra cosa: **es** la varianza del componente.
- Pensar que PCA es óptimo para clasificación (es no supervisado).

### Preguntas tipo examen
1. **¿Por qué $\mathbf{u}_1$ = autovector del mayor $\lambda$?** Maximizar $\mathbf{u}^T\mathbf{S}\mathbf{u}$ s.a. $\|\mathbf{u}\|=1$ → $\mathbf{S}\mathbf{u}=\lambda\mathbf{u}$, varianza $=\lambda$.
2. **¿Condición sobre los datos?** Centrados (media 0).
3. **¿Cómo se elige $p$?** Varianza acumulada / codo.
4. **¿PCA óptimo para clasificar?** No; maximiza varianza, no separabilidad.
5. **Relación varianza proyectada ↔ autovalores.** Es exactamente $\lambda_k$.

---

## CLASE 21 — Análisis Discriminante de Fisher (LDA)

### Idea central
Extracción **supervisada**: proyecta maximizando separabilidad entre clases ($S_B$) sobre dispersión interna ($S_W$).

### Conceptos clave
- $S_W$ (intra-clase) = compacidad; $S_B$ (entre-clases) = separación de medias.
- Máximo $C-1$ componentes (con $C$ clases).
- Asume Gaussianas con covarianzas similares; requiere $S_W$ invertible.

### Fórmulas
- $S_W=S_1+S_2$, $S_k=\sum_{\mathbf{x}\in C_k}(\mathbf{x}-\mu_k)(\mathbf{x}-\mu_k)^T$
- $S_B=(\mu_1-\mu_2)(\mu_1-\mu_2)^T$
- Criterio: $J(\mathbf{v})=\frac{\mathbf{v}^T S_B\mathbf{v}}{\mathbf{v}^T S_W\mathbf{v}}$ → $S_W^{-1}S_B\mathbf{v}=\lambda\mathbf{v}$
- Bi-clase: $\mathbf{v}=S_W^{-1}(\mu_1-\mu_2)$

### Tabla PCA vs LDA
| | PCA | LDA |
|---|---|---|
| Supervisado | No | Sí |
| Maximiza | varianza | separabilidad |
| Nº máx. comp. | $\min(d,N)$ | $\min(d,C-1)$ |

### Errores comunes
- Creer que basta maximizar $|\tilde\mu_1-\tilde\mu_2|$ (hay que normalizar por dispersión interna).
- Creer que LDA da más de $C-1$ componentes.
- Usar LDA con $S_W$ singular o clases muy no Gaussianas.

### Preguntas tipo examen
1. **¿Por qué no basta separar medias?** Ignora la dispersión interna.
2. **¿Qué problema resuelve Fisher?** Maximizar cociente de Rayleigh → $S_W^{-1}S_B\mathbf{v}=\lambda\mathbf{v}$.
3. **Solución bi-clase.** $\mathbf{v}=S_W^{-1}(\mu_1-\mu_2)$.
4. **¿Nº máx. de discriminantes?** $\min(d,C-1)$.
5. **PCA vs LDA.** No supervisado/varianza vs supervisado/separabilidad.

---

## CLASE 22 — Extracción no lineal / Aprendizaje de variedades

### Idea central
Cuando los datos yacen sobre una **variedad (manifold)** no lineal, los métodos lineales fallan. Métodos: **Kernel PCA, t-SNE, UMAP, LLE**.

### Conceptos clave
- **Kernel PCA:** PCA en espacio $\phi(\mathbf{x})$ vía kernel; trabaja con matriz Gram $N\times N$.
- **t-SNE:** solo **visualización** (no generaliza). Gaussiana en origen, t-Student en reducido. Preserva estructura local.
- **UMAP:** rápido, **generaliza**, preserva local + global. Basado en grafos de k-vecinos.
- **LLE:** reconstrucción lineal local en 2 pasos.

### Fórmulas
- Kernel PCA: $N\lambda\mathbf{u}=\mathbf{K}\mathbf{u}$, $K_{ij}=\langle\phi(\mathbf{x}_i),\phi(\mathbf{x}_j)\rangle$
- t-SNE: $\min KL(P\|Q)=\sum_{ij}p_{ij}\log\frac{p_{ij}}{q_{ij}}$ ; $q_{ij}$ con t-Student (colas pesadas → crowding)
- LLE: paso 1 pesos $\min\sum_i\|\mathbf{x}_i-\sum_j w_{ji}\mathbf{x}_j\|^2$ (s.a. $\sum_j w_{ji}=1$); paso 2 embedding con mismos pesos.

### Hiperparámetros
t-SNE: `perplexity` (5-50) · UMAP: `n_neighbors` (~15), `min_dist` · LLE: `n_neighbors`.

### Errores comunes
- Usar t-SNE en producción (no generaliza).
- Interpretar distancias globales en t-SNE (solo lo local es confiable).
- Confundir matriz Gram $N\times N$ (Kernel PCA) con covarianza $d\times d$ (PCA).

### Preguntas tipo examen
1. **¿Por qué Kernel PCA usa $N\times N$?** Opera en $\phi(\mathbf{x})$ (dim. alta/infinita); kernel trick sobre la Gram.
2. **¿Por qué t-Student en t-SNE?** Colas pesadas alivian el crowding problem.
3. **¿Por qué t-SNE no sirve en producción?** No proyecta muestras nuevas; cómputo $O(N^2)$.
4. **Los 2 pasos de LLE.** Estimar pesos locales → embedding que los preserva.
5. **¿Cuándo PCA antes de t-SNE?** Con muchas dimensiones, reducir a ~50 (ruido + velocidad).

---

## CLASE 23 — Explicabilidad de Modelos

### Idea central
Modelos complejos (RF, NN, SVM-kernel) son precisos pero opacos. La explicabilidad responde por qué decidió, qué variables importaron y qué cambiar.

### Conceptos clave
- **Global** (comportamiento general) vs **local** (una predicción).
- **Específico** (depende del modelo, ej. `feature_importances_`) vs **agnóstico** (cualquier caja negra).
- **SHAP:** teoría de juegos (valores de Shapley); contribución marginal promediada sobre coaliciones.
- **LIME:** modelo lineal local sobre perturbaciones ponderadas.
- **Contrafactual:** ¿qué cambiar para otro resultado?

### Fórmulas
- SHAP: $f(\mathbf{x})=\phi_0+\sum_i\phi_i$, $\phi_0=\mathbb{E}[f(X)]$ (baseline)
- $\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!(d-|S|-1)!}{d!}[f_{S\cup\{i\}}-f_S]$

### LIME paso a paso
Perturbar la muestra → ponderar por similitud → predecir con el modelo → entrenar regresión lineal ponderada → coeficientes = explicación local.

### Errores comunes
- Confundir explicabilidad global con local.
- Creer que SHAP siempre es exacto (exponencial; TreeExplainer aprovecha la estructura).
- Creer que LIME es determinista (usa muestreo aleatorio).
- Confundir $\phi_0$ con la predicción de la muestra ($\phi_0=\mathbb{E}[f(X)]$, baseline global).

### Preguntas tipo examen
1. **Global vs local.** Comportamiento general vs predicción concreta.
2. **¿En qué se basa SHAP?** Valores de Shapley (teoría de juegos cooperativos).
3. **¿Qué responde un contrafactual vs SHAP?** "Qué cambiar" vs "por qué se decidió".
4. **Procedimiento de LIME.** (perturbar → ponderar → predecir → regresión lineal local)
5. **¿Qué es $\phi_0$?** Predicción media del modelo (baseline).
