# Unidad 3: Complejidad de Modelos, Métricas de Evaluación y Sobreajuste

---

## Introducción

Esta unidad aborda tres temas fundamentales e interconectados del ciclo de vida de cualquier proyecto de Machine Learning: cómo medir la complejidad de un modelo y su impacto en el desempeño, cómo evaluar correctamente ese desempeño mediante métricas apropiadas, y cómo combatir el fenómeno del sobreajuste para construir modelos que generalicen bien.

Los tres documentos que componen la unidad corresponden a las Clases 05, 06 y 07 del curso *Introducción al Machine Learning* (repositorio `jdariasl/Intro_ML_2025`) y forman una secuencia pedagógica coherente:

1. **Métricas de evaluación** (Clase 05): Qué herramientas usar para medir el desempeño de un modelo.
2. **Complejidad de modelos** (Clase 06): Cómo la complejidad del modelo afecta el error, y cómo seleccionar hiperparámetros con metodologías de validación robustas.
3. **Sobreajuste y Regularización** (Clase 07): Qué es el sobreajuste, por qué ocurre y cómo combatirlo matemáticamente.

Los objetivos principales de la unidad son:

- Distinguir entre función de costo y métrica de evaluación.
- Dominar las métricas para clasificación (matriz de confusión, ROC/AUC, F1, MCC, BACC) y regresión (MSE, MAE, R²).
- Comprender el trade-off sesgo-varianza y su relación con la complejidad del modelo.
- Aplicar metodologías de validación (k-fold, bootstrapping, leave-one-out, por grupos, estratificada) para seleccionar hiperparámetros de forma robusta.
- Entender e implementar la regularización L1 y L2 como técnica para combatir el sobreajuste.
- Interpretar curvas de aprendizaje para diagnosticar sesgo y varianza.

---

## Contenido de la Unidad

| Archivo | Clase | Tema |
|---|---|---|
| `Métricas_de_evaluación.pdf` | Clase 05 | Métricas de evaluación para clasificación y regresión |
| `Complejidad_de_modelos.pdf` | Clase 06 | Bias-Variance tradeoff, metodologías de validación, curva de aprendizaje |
| `Sobreajuste_y_Regularización.pdf` | Clase 07 | Overfitting, regularización L1/L2, maldición de la dimensionalidad |

---

## Estructura de Archivos

---

### Archivo 1: `Métricas_de_evaluación___Introducción_al_Machine_Learning.pdf`
**Clase 05 – Métricas de error**

#### Propósito

Establecer la diferencia entre las funciones de costo usadas durante el entrenamiento y las métricas de evaluación usadas para interpretar el desempeño real de un modelo frente al problema que se desea resolver. Cubre métricas para clasificación (incluyendo problemas desbalanceados) y regresión.

---

#### Sección 1 – Diferencia entre Función de Costo y Métrica de Evaluación

Al definir una tarea de ML se necesitan tres elementos: el modelo, el criterio de entrenamiento (función de costo) y el algoritmo de entrenamiento. La **función de costo** sirve para optimizar el modelo y debe cumplir requisitos matemáticos como continuidad y diferenciabilidad, pero su valor absoluto no tiene una interpretación directa del problema.

Por eso se necesitan **métricas de evaluación**: miden el desempeño del modelo en términos comprensibles para el dominio del problema.

**Ejemplo del documento**: La cross-entropía (función de costo para clasificación) vs. el error de clasificación (proporción de muestras mal clasificadas). Aunque se optimiza la cross-entropía, el indicador de interés práctico es el error de clasificación.

---

#### Sección 2 – Métricas para Clasificación

##### 2.1 Error de Clasificación

La métrica más simple: fracción de muestras clasificadas incorrectamente.

```
E = (1/N) · Σ(i=1 a N) [[f(xi) ≠ yi]]
```

Donde `[[·]]` es la función indicador (vale 1 si la condición se cumple, 0 si no).

**Limitación:** En problemas desbalanceados, un clasificador que siempre predice la clase mayoritaria puede obtener un error muy bajo sin ser útil.

##### 2.2 Matriz de Confusión

La idea central: **en clasificación, no todos los errores tienen el mismo costo**. Para un problema binario (positivo/negativo), la matriz de confusión descompone las predicciones en cuatro categorías:

| | Predicho Positivo | Predicho Negativo |
|---|---|---|
| **Real Positivo** | True Positive (TP) | False Negative (FN) |
| **Real Negativo** | False Positive (FP) | True Negative (TN) |

- **TP**: El modelo predijo positivo y era positivo. ✅
- **TN**: El modelo predijo negativo y era negativo. ✅
- **FP (Error Tipo I)**: El modelo predijo positivo pero era negativo. ❌
- **FN (Error Tipo II)**: El modelo predijo negativo pero era positivo. ❌

**Ejemplo en el documento**: Se entrena un `QuadraticDiscriminantAnalysis` sobre 1000 muestras de dos distribuciones gaussianas bivariadas. Se visualiza la distribución de los scores logarítmicos de cada categoría (TP, TN, FP, FN) y se grafican las matrices de confusión con y sin normalización. Resultado: TP=469, TN=473, FP=31, FN=27 (con normalización: 0.94 y 0.95 en la diagonal).

**Código Python clave:**
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(Y, predictions, labels=[1,0], normalize=normalize)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
```

**Para múltiples clases**: La matriz se extiende a `C×C` donde `C` es el número de clases. El documento muestra un ejemplo con 6 clases donde es evidente que la clase 0 es la más fácil de clasificar (0.99 de acierto) mientras que la clase 4 es la más difícil (0.25 de acierto máximo).

##### 2.3 Medidas Derivadas de la Matriz de Confusión

| Métrica | Fórmula | Interpretación |
|---|---|---|
| **Sensibilidad / Recall / TPR** | `TP / (TP + FN)` | De todos los positivos reales, cuántos detectó el modelo |
| **Precisión / PPV** | `TP / (TP + FP)` | De todo lo que predijo como positivo, cuántos lo eran realmente |
| **Especificidad** | `TN / (TN + FP)` | De todos los negativos reales, cuántos clasificó correctamente |
| **False Positive Rate (FPR)** | `1 - Especificidad` | Fracción de negativos mal clasificados como positivos |
| **Exactitud / Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Fracción de predicciones correctas totales |

**Métrica F-beta:**

Combina Precisión y Recall en un único número ponderado:

```
F_β = (1 + β²) · (Precision · Recall) / (β² · Precision + Recall)
```

- Con `β = 1` (F1-Score): Da **igual importancia** a Precisión y Recall. Es la más usada.
- Con `β > 1`: Pondera más el Recall (casos donde los FN son más costosos, como en diagnóstico médico).
- Con `β < 1`: Pondera más la Precisión.

##### 2.4 Problemas Desbalanceados

El desbalance ocurre cuando una clase tiene significativamente más muestras que las otras. El accuracy puede ser engañoso en estos casos.

**Ejemplo del documento (desbalance extremo):**
- 5 muestras de clase positiva (1%) vs. 500 de clase negativa (99%).
- El modelo obtiene: `Accuracy = 0.99`, `Precision = 1.0` (¡parece perfecto!)
- Pero: `Recall = 0.2`, `F1 = 0.33` (¡el modelo ignora casi todos los positivos!)

**Métricas recomendadas para desbalance:**

| Métrica | Fórmula | Ventaja |
|---|---|---|
| **MCC** (Matthews Correlation Coefficient) | `(TP·TN - FP·FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]` | Considera los cuatro cuadrantes de la matriz; robusto al desbalance |
| **BACC** (Balanced Accuracy) | `TP/[2(TP+FN)] + TN/[2(TN+FP)]` | Promedio de sensibilidad y especificidad |
| **G-mean** | `√(Sensibilidad · Especificidad)` | Media geométrica; penaliza fuertemente si una clase es ignorada. Para q clases: raíz q del producto de todos los Recalls |

**Resultado del ejemplo:** `MCC = 0.445`, `BACC = 0.6` — mucho más honestos que el Accuracy de 0.99.

**Estrategias adicionales para desbalance:**
- Submuestreo inteligente de la clase mayoritaria (eliminar datos atípicos y redundantes).
- Sobremuestreo inteligente: **SMOTE** (Synthetic Minority Oversampling Technique) genera muestras artificiales de la clase minoritaria.
- Validación estratificada para preservar proporciones.
- Pesos diferenciales por clase en la función de costo.
- Librería recomendada: `imbalanced-learn`.

**Código Python:**
```python
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
print('MCC = ' + str(matthews_corrcoef(Y, y_pred)))
print('BACC = ' + str(balanced_accuracy_score(Y, y_pred)))
```

##### 2.5 Curva ROC y AUC

La **curva ROC** (Receiver Operating Characteristic) muestra el desempeño del clasificador para todos los posibles umbrales de decisión, graficando `TPR vs FPR`.

- **Umbral bajo**: Se detectan más positivos (↑ TPR) pero también más falsos positivos (↑ FPR).
- **Umbral alto**: Se detectan menos positivos (↓ TPR) y menos falsos positivos (↓ FPR).
- **Clasificador perfecto**: Curva que pasa por el punto (0, 1) — TPR=1, FPR=0.
- **Clasificador aleatorio**: Diagonal `y = x`.
- **AUC** (Area Under the Curve): Escalar entre 0 y 1. AUC=1 → clasificador perfecto; AUC=0.5 → clasificador aleatorio.

**Efecto de la separabilidad de scores:**
El documento ilustra con 4 escenarios de separación creciente entre distribuciones de scores positivos y negativos:
- Distribuciones completamente superpuestas → AUC ≈ 0.47 (aleatorio)
- Ligera separación → AUC ≈ 0.77
- Buena separación → AUC ≈ 0.91
- Separación perfecta → AUC = 1.00

**Código Python:**
```python
from sklearn.metrics import roc_curve, auc
y_pred2 = clf.predict_log_proba(X)
score = y_pred2[:,0] - y_pred2[:,1]
tpr, fpr, _ = roc_curve(Y, score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
```

**Resultado:** AUC = 0.99 con el clasificador QDA sobre datos bien separados.

---

#### Sección 3 – Métricas para Regresión

| Métrica | Fórmula | Característica |
|---|---|---|
| **MSE** (Mean Squared Error) | `(1/N) · Σ(yi - f(xi))²` | Penaliza errores grandes cuadráticamente; sensible a outliers |
| **MAE** (Mean Absolute Error) | `(1/N) · Σ|yi - f(xi)|` | Más robusto que MSE a outliers; interpretación directa |
| **MedAE** (Median Absolute Error) | `mediana(|y1-f(x1)|, ..., |yN-f(xN)|)` | Muy robusto a outliers; no influenciado por valores extremos |
| **MAPE** (Mean Absolute Percentage Error) | `(1/N) · Σ|(yi - f(xi)) / yi|` | Error relativo; útil cuando la magnitud varía mucho |
| **R²** (Coeficiente de determinación) | `1 - Σ(yi - f(xi))² / Σ(yi - ȳ)²` | Proporción de varianza explicada; 1=perfecto, 0=igual que la media |

**Ejemplo en el documento:** Regresión polinomial de grado 8 sobre datos sinusoidales con ruido (σ=0.3).

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_absolute_percentage_error, r2_score
print('MSE = ' + str(mean_squared_error(y, p_y)))      # 0.065
print('MAE = ' + str(mean_absolute_error(y, p_y)))     # 0.201
print('MedAE = ' + str(median_absolute_error(y, p_y))) # 0.159
print('MAPE = ' + str(mean_absolute_percentage_error(y, p_y))) # 0.344
print('R^2 = ' + str(r2_score(y, p_y)))                # 0.966
```

**Nota importante del documento:** Aunque se usen varias métricas para evaluar distintos aspectos, **siempre debe definirse una sola métrica global** que sea el objetivo de optimización para la selección del mejor modelo y sus hiperparámetros. En problemas de múltiples salidas, se puede usar la suma o promedio de las métricas de cada salida.

---

### Archivo 2: `Complejidad_de_modelos___Introducción_al_Machine_Learning.pdf`
**Clase 06 – Complejidad de modelos, sobreajuste y metodologías de validación**

#### Propósito

Estudiar cómo la complejidad del modelo afecta el error de predicción (trade-off sesgo-varianza), y cómo las metodologías de validación permiten seleccionar hiperparámetros y estimar el desempeño de forma confiable.

---

#### Sección 1 – Motivación: El Problema de la Complejidad

El documento abre mostrando un dataset de clasificación generado con dos distribuciones gaussianas bivariadas. Se plantea cómo varía la frontera de decisión según el tipo de modelo:

**Con QDA (Análisis Discriminante Cuadrático):**
```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis()
clf.fit(X, Y.flatten())
```
La frontera es una curva cuadrática suave que separa bien las dos clases.

**Con Kernel Classifier (ventana de Parzen):**
```python
clf = Kernel_classifier(bandwidth=0.5)
```
La frontera es muy irregular y se adapta localmente a cada muestra. Con pocos datos (N=100), la frontera varía enormemente entre corridas; con más datos (N=1000), se estabiliza.

**Observación clave:** La variabilidad de la frontera entre corridas con distintas muestras de entrenamiento es precisamente la **varianza** del modelo.

**Criterios formales para selección por complejidad** (mencionados como referencias):
- **AIC** (Akaike Information Criterion)
- **BIC** (Bayesian Information Criterion)
- **MDL** (Minimum Description Length)

---

#### Sección 2 – Bias vs. Variance (Sesgo vs. Varianza)

##### Definiciones

**Bias (Sesgo):** Diferencia entre la predicción promedio del modelo y el valor real. Un modelo con alto bias no captura las relaciones relevantes entre entradas y salida → **subajuste**.

**Variance:** Sensibilidad del modelo a las fluctuaciones en el conjunto de entrenamiento. Alta varianza → el modelo aprende el ruido de los datos → **sobreajuste**.

##### Descomposición Formal del Error

Para un sistema `y = f(x) + e` donde `e ~ N(0, σ²_e)`, el error de predicción esperado se descompone como:

```
Err(x) = E[(y - f̂(x))²]
Err(x) = (E[f̂(x)] - f(x))² + E[(f̂(x) - E[f̂(x)])²] + σ²_e
Err(x) = Bias²           + Variance                   + Error Irreducible
```

Donde:
- `Bias² = (E[f̂(x)] - f(x))²`: cuánto se desvía la predicción promedio del modelo de la función real.
- `Variance = E[(f̂(x) - E[f̂(x)])²]`: cuánto varía la predicción del modelo entre distintos conjuntos de entrenamiento.
- `σ²_e`: ruido intrínseco del problema; no puede reducirse con ningún modelo.

##### Diagrama de Diana (Bias-Variance)

El documento incluye un diagrama de diana que ilustra las cuatro combinaciones posibles:

| | Baja Varianza | Alta Varianza |
|---|---|---|
| **Bajo Bias** | Modelo ideal (tiros en el centro) | Overfitting (tiros dispersos pero promedio cercano al centro) |
| **Alto Bias** | Underfitting (tiros consistentemente fuera del centro) | Peor caso (tiros dispersos y lejos del centro) |

##### Gráfica del Trade-off

La gráfica `Error vs. Complejidad del Modelo` muestra:
- A medida que aumenta la complejidad: Bias² **decrece** monotónicamente.
- A medida que aumenta la complejidad: Variance **crece** monotónicamente.
- El error total tiene forma de **U**: existe una complejidad óptima que minimiza `Bias² + Variance`.

---

#### Sección 3 – Ejemplo Práctico: Regresión Polinomial y Bias-Variance

##### Funciones Auxiliares

```python
def f(size):
    """Función verdadera sin ruido: y = 2·sin(1.5x)"""
    x = np.linspace(0, 4.5, size)
    y = 2 * np.sin(x * 1.5)
    return (x, y)

def sample(size):
    """Muestras con ruido gaussiano añadido"""
    x = np.linspace(0, 4.5, size)
    y = 2 * np.sin(x * 1.5) + np.random.randn(x.size)
    return (x, y)
```

##### Visualización del Bias-Variance por Grado

```python
for k, degree in enumerate([3, 5, 10, 18]):
    for i in range(n_models):     # 20 modelos por grado
        (x, y) = sample(n_samples)
        model = PolynomialLinearRegression(degree=degree)
        model.fit(x, y)
        p_y = model.predict(x)
        avg_y += p_y
    avg_y /= n_models
```

**Interpretación de los 4 gráficos:**
- **Grado 3**: Todas las curvas son similares entre sí (baja varianza) pero el promedio no sigue bien la función real (sesgo moderado).
- **Grado 5**: Buen equilibrio; el promedio se ajusta bien a la función real y las curvas individuales no varían demasiado.
- **Grado 10**: Las curvas individuales empiezan a divergir significativamente (alta varianza), aunque el promedio es aún razonable.
- **Grado 18**: Las curvas individuales son muy erráticas (varianza muy alta); sobreajuste evidente.

##### Cálculo Explícito de Bias y Varianza

```python
for degree in range(1, max_degree):
    # Entrenar 100 modelos con muestras diferentes
    for i in range(n_models):
        (x, y) = sample(n_samples)
        model = PolynomialLinearRegression(degree=degree)
        model.fit(x, y)
        p_y = model.predict(x)
        avg_y += p_y
    avg_y /= n_models
    
    # Bias²: distancia entre predicción promedio y función real
    bias_2 = norm(avg_y - f_y) / f_y.size
    
    # Variance: dispersión de predicciones individuales respecto al promedio
    for p_y in models:
        variance += norm(avg_y - p_y)
    variance /= f_y.size * n_models
```

**Resultado del gráfico:**
- Para grados 1-4: Bias alto (0.25→0.05), varianza moderada (0.07→0.11).
- Alrededor de grado 5-6: Mínimo del error total (≈0.15).
- Para grados 7-14: Bias casi constante (~0.02), varianza creciente (0.13→0.19), error total creciendo.

---

#### Sección 4 – Metodologías de Validación

Cuando se tiene un único conjunto de datos `D = {(xi, yi)}`, las metodologías de validación permiten usar ese conjunto de manera óptima para: (a) seleccionar hiperparámetros y (b) estimar medidas de desempeño confiables.

##### 4.1 Validación Cruzada (k-Fold Cross-Validation)

**Procedimiento:**
1. Dividir aleatoriamente los datos en Training (80%) y Test (20%).
2. Dividir Training en `k` subconjuntos disyuntos (folds).
3. En cada iteración, usar `k-1` folds para entrenar y el fold restante para validar. Repetir `k` veces.
4. Usar Training+Validación para seleccionar hiperparámetros.
5. Usar Test (nunca visto) para evaluación final.

**Diagrama (5-fold):**
```
All Data → [Training data (80%)] | [Test data (20%)]

Split 1: [Fold1*] [Fold2] [Fold3] [Fold4] [Fold5]  → Finding Parameters
Split 2: [Fold1] [Fold2*] [Fold3] [Fold4] [Fold5]
Split 3: [Fold1] [Fold2] [Fold3*] [Fold4] [Fold5]
Split 4: [Fold1] [Fold2] [Fold3] [Fold4*] [Fold5]
Split 5: [Fold1] [Fold2] [Fold3] [Fold4] [Fold5*]
                                    ↓
                             Final evaluation → [Test data]
(* = fold usado como validación en esa iteración)
```

**Código Python completo:**
```python
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
kf = KFold(n_splits=5)
Acc = FronterasCV(X_train, y_train, kf)
print('Accuracy promedio = ' + str(np.mean(Acc, axis=1)))
# Resultado: [0.9169 0.9269 0.9356 0.9369] para k=1,3,5,7
```

**El mejor modelo fue `k=7`** (mayor accuracy promedio: 0.9369).

```python
clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train, y_train.flatten())
y_pred = clf.predict(X_test)
print('Accuracy en test = ' + str(accuracy_score(y_test, y_pred)))
# Accuracy en test = 0.915
```

**Nota:** El error en test es típicamente ligeramente mayor que en validación, porque la validación tiene cierto grado de optimismo al haber sido usada para seleccionar el modelo.

##### 4.2 Leave-One-Out (LOO)

Caso particular de k-fold donde `k = N` (número de muestras). En cada iteración se entrena con `N-1` muestras y se valida con la muestra restante.

**Cuándo usarlo:** Conjuntos de datos muy pequeños donde se necesita maximizar el número de muestras de entrenamiento.

```python
from sklearn.model_selection import LeaveOneOut
# equivalente a:
kf = KFold(n_splits=N)
```

##### 4.3 Leave-P-Out

Generalización donde se reservan exactamente `p` muestras para validación. El número de repeticiones es `C(N, p)` (combinaciones).

```python
from sklearn.model_selection import LeavePOut
lpo = LeavePOut(2)
lpo.get_n_splits(X)  # Para N=10 → 45 combinaciones
```

##### 4.4 Bootstrapping (Shuffle-Split)

La partición entrenamiento/validación se realiza aleatoriamente, definiendo un porcentaje y un número de repeticiones. **Diferencia clave con k-fold:** Una misma muestra puede repetirse en distintos subconjuntos de validación.

```python
from sklearn.model_selection import ShuffleSplit
rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
Acc = FronterasCV(X_train, y_train, rs)
# Accuracy promedio: [0.918 0.9255 0.933 0.9375] → mejor modelo: k=7
```

**Resultado:** Aunque se usa un 5% menos de muestras por fold, los resultados son muy similares al k-fold. El mejor hiperparámetro sigue siendo `k=7`.

##### 4.5 Validación Estratificada (Problemas Desbalanceados)

En conjuntos desbalanceados, la partición aleatoria puede producir subconjuntos que no representen bien la clase minoritaria.

**Sin estratificación:** La clase 1 (minoritaria, 9.1% del total) puede quedar representada como 8.4% en training (distorsión).

**Con estratificación:** La proporción de cada clase se preserva exactamente en todos los splits.

```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
# Versión estratificada de validación cruzada:
from sklearn.model_selection import StratifiedKFold
# Versión estratificada de bootstrapping:
from sklearn.model_selection import StratifiedShuffleSplit
```

**Resultado visual:** Con `stratify=Y`, la distribución del conjunto de training es idéntica (0.909/0.091) a la del dataset original. Sin estratificación, hay ligeras diferencias.

##### 4.6 Validación por Grupos (Datos Dependientes)

Cuando las muestras **no son independientes entre sí** porque provienen de la misma fuente, la validación estándar produce resultados optimistamente sesgados.

**Casos típicos:**
- Varias grabaciones del mismo paciente en diferentes sesiones.
- Múltiples mediciones del mismo sensor en distintos momentos.
- Aprendizaje multi-instancia.

**Ejemplo del documento: Dataset de Parkinson (UCI)**
```python
import pandas as pd
df = pd.read_csv("local/data/parkinsons.data", delimiter=',')
# 195 muestras, 24 características, 32 pacientes únicos
```

El nombre de cada muestra tiene formato `phon_R01_S01_1` (paciente S01, sesión 1). Si no se tiene en cuenta que varias muestras pertenecen al mismo paciente, muestras del mismo paciente pueden aparecer tanto en entrenamiento como en validación.

**Comparación de resultados (k-NN con GridSearchCV):**

| Metodología | Accuracy Validación | Accuracy Test | Discrepancia |
|---|---|---|---|
| Sin considerar grupos (ShuffleSplit) | 0.947 | 0.814 | **13.3 puntos** |
| Considerando grupos (GroupShuffleSplit) | 0.870 | 0.860 | **1.0 puntos** |

**Conclusión crítica:** La metodología sin grupos era optimistamente sesgada por casi 13 puntos porcentuales. Al separar correctamente por pacientes, la estimación de validación fue mucho más honesta y el test mejoró 5 puntos.

```python
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2).split(X, Y, groups=Pacientes))
rs = GroupShuffleSplit(test_size=.25, n_splits=5).split(Xtrain, Y[train_inds], groups=Pacientes[train_inds])
```

---

#### Sección 5 – GridSearchCV: Búsqueda de Hiperparámetros

```python
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xtrain = scaler.fit_transform(X[train_inds, :])

tuned_parameters = [
    {'n_neighbors': [1,3,5,7,9], 'metric': ['minkowski'], 'p': [1, 2]},
    {'n_neighbors': [1,3,5,7,9], 'metric': ['chebyshev']}
]
clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=rs, scoring='accuracy')
clf.fit(Xtrain, Y[train_inds])
print(clf.best_params_)   # {'metric': 'minkowski', 'n_neighbors': 9, 'p': 1}
print(clf.best_score_)    # 0.8695
```

**Nota:** La normalización se aplica usando `fit_transform` solo sobre training y luego `transform` sobre test, para evitar fuga de información.

---

#### Sección 6 – Curva de Aprendizaje

La curva de aprendizaje muestra cómo varía el error de entrenamiento y validación en función del número de muestras de entrenamiento. Es útil para:
- Diagnosticar si el modelo tiene alto sesgo o alta varianza.
- Determinar cuántas muestras son necesarias para que el modelo converja.
- Decidir si vale la pena recopilar más datos.

**Interpretación de patrones:**

| Patrón | Diagnóstico | Solución |
|---|---|---|
| Error train bajo, error val alto, brecha grande | Alta varianza (sobreajuste) | Más datos, regularización, modelo más simple |
| Error train y val altos, brecha pequeña | Alto sesgo (subajuste) | Modelo más complejo, más features |
| Error train y val convergen rápidamente | El modelo ya no mejora con más datos | Cambiar modelo |
| Error val sigue mejorando al aumentar datos | El modelo puede beneficiarse de más datos | Recopilar más datos |

**Código Python:**
```python
from sklearn.model_selection import ShuffleSplit
from local.library.learning_curve import plot_learning_curve

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = RandomForestClassifier(n_estimators=20, max_depth=3)
plot_learning_curve(estimator, title, Xtrain, Y, cv=cv, n_jobs=4)
```

**Resultados del documento:**
- **Random Forest (n=20, max_depth=3) en Parkinson**: Alta varianza visible — gran brecha entre training score (~0.99) y validación (~0.88). Se requieren más datos.
- **SVM (RBF, γ=0.001) en Parkinson**: Similar patrón de alta varianza.
- **Naive Bayes en Digits**: Bajo bias pero alta varianza; los scores convergen hacia 0.85.
- **SVM (RBF, γ=0.001) en Digits**: Mejor caso — baja varianza y bajo bias. El modelo alcanza ~0.99 de accuracy en validación con suficientes muestras (>1200).

---

### Archivo 3: `Sobreajuste_y_Regularización___Introducción_al_Machine_Learning.pdf`
**Clase 07 – Regularización**

#### Propósito

Definir formalmente el sobreajuste y el subajuste, identificar sus causas, y presentar las técnicas matemáticas (regularización L2/L1) y estratégicas (parada anticipada, partición train/val/test) para combatirlos. Se introduce también la maldición de la dimensionalidad como causa estructural de sobreajuste.

---

#### Sección 1 – Sobreajuste y Subajuste

##### Definiciones

**Sobreajuste (Overfitting):** El modelo memoriza las muestras de entrenamiento (incluyendo su ruido) pero no captura la estructura subyacente. Resultado: error de entrenamiento muy bajo (~0) pero error de test muy alto.

**Subajuste (Underfitting):** El modelo no es suficientemente flexible para ajustarse a los datos. Resultado: error alto tanto en entrenamiento como en test.

**Ninguno de los dos es deseable:** En ambos casos el modelo no puede hacer predicciones apropiadas sobre datos nuevos.

##### Diagnóstico

| Condición | Diagnóstico |
|---|---|
| Error entrenamiento ≈ 0, error test >> 0 | Sobreajuste |
| Error entrenamiento alto ≈ Error test alto | Subajuste |
| Error entrenamiento ≈ Error test bajo | Modelo bien ajustado |

##### Visualización en el documento

**Regresión (3 grados de polinomio):**
- Grado 1: `MSE = 0.408 ± 0.425` → Subajuste (línea recta no sigue la curva sinusoidal).
- Grado 4: `MSE = 0.043 ± 0.071` → Buen ajuste (sigue la curva razonablemente).
- Grado 15: `MSE = 1.83×10⁸ ± 5.48×10⁸` → Sobreajuste severo (curva extremadamente irregular).

**Clasificación (SVM con diferentes gamma):**
- `gamma=10.00`: Frontera muy irregular, sobreajusta cada punto → overfitting.
- `gamma=1.00`: Frontera suave y razonable → buen ajuste.
- `gamma=0.01`: Frontera lineal simple → underfitting.

---

#### Sección 2 – Causas del Sobreajuste

El documento lista cuatro causas principales:

1. **Muestras de entrenamiento muy ruidosas**: El ruido puede provenir de sensores (acelerómetros, micrófonos), datos desactualizados en bases de datos estructuradas, o etiquetas incorrectas (captchas, crowdsourcing).

2. **Modelo demasiado complejo**: Un modelo con alta flexibilidad (muchos parámetros) puede ajustarse al ruido en lugar de a la señal.

3. **Conjunto de entrenamiento pequeño**: Con pocos datos, el modelo puede memorizar las respuestas sin aprender la estructura.

4. **Espacio de características de alta dimensión**: Demasiadas variables con pocas muestras (ver Maldición de la Dimensionalidad).

**Nota importante:** El sobreajuste requiere **dos condiciones simultáneas**: ruido en los datos **y** un modelo suficientemente flexible para ajustarse a ese ruido. Una regresión logística de grado 1 (`d+1` parámetros) **nunca puede sobreajustarse** por su limitada flexibilidad.

---

#### Sección 3 – Medidas Contra el Sobreajuste

##### 3.1 Partición en Tres Conjuntos

Cuando se seleccionan hiperparámetros, es necesario una partición en **tres subconjuntos**:

| Subconjunto | Uso | Garantía |
|---|---|---|
| **Entrenamiento** | Ajustar parámetros del modelo | Los pesos `w` se optimizan aquí |
| **Validación** (Development) | Seleccionar hiperparámetros; decidir cuándo detener el entrenamiento | No contamina la selección de parámetros |
| **Test** | Evaluar capacidad de generalización final | Nunca visto durante ajuste ni selección |

**Con esta partición:** El conjunto de test garantiza una estimación honesta del desempeño real, no influenciada ni por el ajuste de parámetros ni por la selección de hiperparámetros. Siempre se deben hacer varias repeticiones (k-fold o bootstrapping) para obtener medidas `desempeño ± IC`.

##### 3.2 Parada Anticipada (Early Stopping)

En modelos con muchos parámetros (especialmente redes neuronales), se monitorea simultáneamente el error de entrenamiento y el error de validación durante cada época:

- Mientras ambos decreccen → continuar entrenando.
- Cuando el error de validación **deja de decrecer** (aunque el de entrenamiento siga bajando) → **detener el entrenamiento**.

Esto previene que el modelo continúe memorizando el ruido de los datos de entrenamiento.

**En scikit-learn:**
```python
# Disponible en modelos como MLPClassifier, SGDClassifier:
early_stopping=True  # parámetro del constructor
```

---

#### Sección 4 – Regularización (L2 y L1)

##### 4.1 Regularización L2 (Ridge)

La regularización introduce un **término de penalización** en la función objetivo que crece con la magnitud de los pesos del modelo:

```
E(w) = (1/2) · Σ(yi - f(xi,w))² + (λ/2) · ||w||²
```

Donde:
- `(1/2) · Σ(yi - f(xi,w))²`: error cuadrático medio (término de ajuste a los datos).
- `(λ/2) · ||w||²`: penalización L2 (norma cuadrada del vector de pesos).
- `λ`: hiperparámetro de regularización. **Mayor λ → mayor penalización → modelo más simple → menos sobreajuste pero más sesgo potencial**.

**Intuición:** Cuando hay sobreajuste, los coeficientes del polinomio tienden a crecer en magnitud. La regularización restringe ese crecimiento, forzando al modelo a encontrar una solución más suave.

##### Regla de Actualización para Gradiente Descendente (Regresión Lineal)

```
w_j = w_j - η · (λ·w_j + Σ(i=1 a N) (f(xi,w) - yi) · xij)
```

El término `λ·w_j` actúa como un "decaimiento de pesos" (weight decay) que empuja los pesos hacia cero en cada iteración.

##### Solución Analítica Regularizada

```
w = (X^T·X + λI)^(-1) · X^T·Y
```

La adición de `λI` a la matriz `X^T·X` garantiza que sea invertible incluso cuando los datos son colineales (problema de matriz singular), haciendo la solución numéricamente estable.

##### Para Regresión Logística

La función de costo se deriva de máxima verosimilitud sobre una distribución Bernoulli. La regla de actualización regularizada es:

```
w_j = w_j - η · (λ·w_j + Σ(i=1 a N) (g(f(xi,w)) - yi) · xij)
```

Donde `g(·)` es la función sigmoide.

##### Ejemplo Visual: Efecto de λ en la Frontera de Decisión

Dataset: `ex2data2.txt` — problema binario no linealmente separable (datos en forma de anillo). Se usa regresión logística con polinomio de grado 6.

| λ | Error Entrenamiento | Descripción de la Frontera |
|---|---|---|
| 0 (sin regularizar) | 0.179 | Frontera muy irregular, se ajusta perfectamente a los datos → overfitting |
| 0.1 | 0.251 | Frontera más suave, algunas irregularidades |
| 0.5 | 0.336 | Frontera casi elíptica, bien suavizada |
| 1.0 | 0.393 | Frontera elíptica simple → subajuste |

**Observación**: A medida que `λ` aumenta, el error de entrenamiento sube (el modelo se simplifica) pero la frontera se vuelve más suave y generalizable. El valor óptimo de `λ` se selecciona mediante validación cruzada.

##### 4.2 Regularización L1 (LASSO)

La norma L1 penaliza la **suma de valores absolutos** de los pesos:

```
E(w) = (1/2) · Σ(yi - f(xi,w))² + λ · ||w||₁
```

**Propiedad especial del LASSO:** Además de limitar la magnitud de los pesos, causa que algunos pesos tomen exactamente valor **cero**, realizando automáticamente **selección de características**.

```python
from sklearn import linear_model, datasets
diabetes = datasets.load_diabetes()
clf = linear_model.Lasso(alpha=0.5)
clf.fit(diabetes.data, diabetes.target)
print(clf.coef_)
# [ 0.  -0.  471.04  136.50  -0.  -0.  -58.32  0.  408.02  0. ]
```

**Resultado:** De 10 características del dataset de diabetes, LASSO con `alpha=0.5` elimina 6 (les asigna coeficiente exactamente 0), conservando solo las 4 más relevantes.

| Regularización | Nombre | Efecto en pesos | Selección features |
|---|---|---|---|
| L2: `λ·||w||²` | Ridge | Reduce magnitud de todos los pesos | No (pesos pequeños pero ≠ 0) |
| L1: `λ·||w||₁` | LASSO | Reduce magnitud; puede llevar a exactamente 0 | Sí (automática) |

---

#### Sección 5 – La Maldición de la Dimensionalidad

Se presenta cuando el espacio de características tiene **dimensión muy alta** pero el número de muestras es **pequeño**:

- El espacio entre muestras crece exponencialmente con la dimensión.
- Los modelos tienden a memorizar las pocas muestras disponibles → overfitting.
- Entre mayor sea el número de variables, mayor debe ser el número de muestras.

**Contexto actual:** Con el big data, este problema es menos frecuente en aplicaciones masivas. Sin embargo, sigue siendo relevante en dominios donde los datos son escasos, como en sistemas de apoyo diagnóstico médico donde ciertos tipos de enfermedades no tienen bases de datos grandes.

**Solución:** Reducción de dimensionalidad — seleccionar un subconjunto de características con máxima capacidad predictiva. El documento anuncia que se abordará en detalle en la clase sobre "Selección de características".

---

## Conceptos Clave

| Concepto | Definición |
|---|---|
| **Función de costo** | Medida optimizable matemáticamente durante el entrenamiento (ej: cross-entropía, MSE) |
| **Métrica de evaluación** | Medida interpretable del desempeño real del modelo (ej: accuracy, F1, R²) |
| **Matriz de confusión** | Tabla que descompone predicciones en TP, TN, FP, FN |
| **Recall / Sensibilidad / TPR** | `TP / (TP+FN)`: qué fracción de positivos reales fueron detectados |
| **Precisión / PPV** | `TP / (TP+FP)`: qué fracción de las predicciones positivas fueron correctas |
| **F1-Score** | Media armónica de Precisión y Recall |
| **MCC** | Coeficiente de correlación de Matthews: métrica balanceada para desbalance |
| **BACC** | Balanced Accuracy: promedio de sensibilidad y especificidad |
| **Curva ROC** | TPR vs FPR para distintos umbrales de decisión |
| **AUC** | Área bajo la curva ROC; 1=perfecto, 0.5=aleatorio |
| **Bias (Sesgo)** | Error sistemático del modelo; diferencia entre predicción promedio y valor real |
| **Variance (Varianza)** | Sensibilidad del modelo a las fluctuaciones del conjunto de entrenamiento |
| **Overfitting (Sobreajuste)** | El modelo memorizó el ruido; error train≈0, error test alto |
| **Underfitting (Subajuste)** | El modelo es muy simple; error train y test ambos altos |
| **Trade-off Sesgo-Varianza** | Al aumentar complejidad: bias baja, varianza sube; existe complejidad óptima |
| **k-Fold Cross-Validation** | El conjunto de training se divide en k folds; cada fold es validación una vez |
| **Leave-One-Out** | Caso extremo de k-fold con k=N |
| **Bootstrapping / ShuffleSplit** | Partición aleatoria repetida; permite repetición de muestras |
| **Validación estratificada** | Preserva proporciones de clases en cada fold |
| **Validación por grupos** | Garantiza que muestras del mismo grupo no estén en train y test |
| **GridSearchCV** | Búsqueda exhaustiva de hiperparámetros con validación cruzada |
| **Curva de aprendizaje** | Error train y val vs. número de muestras; diagnostica bias/varianza |
| **Regularización L2 (Ridge)** | Penaliza `||w||²`; reduce magnitud de pesos sin llegar a cero |
| **Regularización L1 (LASSO)** | Penaliza `||w||₁`; puede llevar pesos exactamente a cero → selección de features |
| **λ (lambda)** | Hiperparámetro de regularización; controla el balance ajuste-penalización |
| **Parada anticipada** | Detener entrenamiento cuando el error de validación deja de mejorar |
| **Maldición de la dimensionalidad** | Alta dimensión con pocas muestras → overfitting estructural |
| **SMOTE** | Synthetic Minority Oversampling Technique: genera muestras artificiales de la clase minoritaria |

---

## Flujo General de la Unidad

```
PROBLEMA DE ML
      │
      ├─── ¿Cómo medir el desempeño? (Clase 05)
      │         ├─── Clasificación
      │         │     ├─── Matriz de Confusión → TP, TN, FP, FN
      │         │     ├─── Recall, Precisión, F1, Accuracy
      │         │     ├─── Desbalance → MCC, BACC, G-mean, SMOTE
      │         │     └─── Curva ROC / AUC
      │         └─── Regresión
      │               └─── MSE, MAE, MedAE, MAPE, R²
      │
      ├─── ¿Cómo seleccionar el modelo correcto? (Clase 06)
      │         ├─── Trade-off Bias-Variance
      │         │     └─── Err = Bias² + Varianza + Error Irreducible
      │         ├─── Metodologías de Validación
      │         │     ├─── k-Fold Cross-Validation
      │         │     ├─── Leave-One-Out / Leave-P-Out
      │         │     ├─── Bootstrapping (ShuffleSplit)
      │         │     ├─── Estratificada (datos desbalanceados)
      │         │     └─── Por grupos (datos dependientes)
      │         └─── Curva de Aprendizaje
      │               └─── Diagnosticar sesgo/varianza y necesidad de más datos
      │
      └─── ¿Cómo evitar el sobreajuste? (Clase 07)
                ├─── Diagnóstico: train≈0 y test alto → overfitting
                ├─── Partición Train / Validación / Test
                ├─── Parada Anticipada (Early Stopping)
                ├─── Regularización
                │     ├─── L2 (Ridge): reduce magnitud de pesos
                │     └─── L1 (LASSO): elimina features irrelevantes (peso=0)
                └─── Maldición de la Dimensionalidad
                      └─── Solución: Selección de características (próxima clase)
```

---

## Conceptos Aprendidos

1. La función de costo y la métrica de evaluación sirven propósitos distintos y no deben confundirse.
2. El error de clasificación es insuficiente; la matriz de confusión revela el tipo de errores cometidos.
3. En problemas desbalanceados, el accuracy puede ser completamente engañoso.
4. El F1-Score, MCC y BACC son métricas más robustas para datos desbalanceados.
5. La curva ROC y el AUC permiten evaluar un clasificador independientemente del umbral de decisión.
6. El error total de un modelo se descompone en Bias², Varianza y Error Irreducible.
7. Modelos más complejos reducen el sesgo pero aumentan la varianza.
8. La complejidad óptima minimiza el error total (punto mínimo de la curva en U).
9. La validación cruzada k-fold es el estándar para selección de hiperparámetros.
10. El conjunto de test debe reservarse exclusivamente para la evaluación final y nunca usarse para seleccionar parámetros.
11. El error de validación suele ser ligeramente optimista respecto al error de test real.
12. En datos con dependencias (múltiples muestras por sujeto), la validación por grupos es obligatoria.
13. Ignorar la dependencia entre muestras puede inflar el desempeño estimado en validación en más de 10 puntos porcentuales.
14. La curva de aprendizaje permite diagnosticar si el modelo sufre de alta varianza, alto sesgo, o si necesita más datos.
15. El sobreajuste requiere simultáneamente: ruido en los datos y un modelo suficientemente flexible.
16. La partición en tres conjuntos (train/val/test) garantiza estimaciones honestas cuando se seleccionan hiperparámetros.
17. La regularización L2 (Ridge) limita la magnitud de los pesos mediante una penalización cuadrática.
18. La regularización L1 (LASSO) puede llevar pesos exactamente a cero, realizando selección automática de características.
19. El hiperparámetro λ controla el balance entre ajuste a los datos y penalización por complejidad.
20. La maldición de la dimensionalidad surge cuando la dimensión del espacio es grande y las muestras son escasas.
21. La solución a la maldición de la dimensionalidad es la selección o reducción de características.
22. La normalización de datos (StandardScaler) debe aplicarse solo sobre training y luego transformarse sobre test.

---

## Conclusiones

### Conclusiones Técnicas

- No existe una única métrica que sea adecuada para todos los problemas; la elección depende del tipo de problema (clasificación/regresión), del balance de clases y de los costos relativos de los distintos tipos de error.
- El AUC-ROC es particularmente útil cuando el umbral de decisión no está fijo, ya que evalúa el clasificador en todo el espacio de umbrales posibles.
- La validación cruzada no es una sola metodología sino una familia de técnicas; la elección correcta depende del tamaño del dataset, el balance de clases y la independencia entre muestras.
- La regularización no es solo una técnica para evitar el sobreajuste, sino también una herramienta para selección de características (L1) y para estabilizar soluciones en matrices mal condicionadas (L2).
- El hiperparámetro λ de la regularización debe seleccionarse mediante validación cruzada, exactamente igual que cualquier otro hiperparámetro del modelo.

### Conclusiones Académicas

- El trade-off sesgo-varianza es uno de los conceptos más fundamentales del ML y aplica a todos los tipos de modelos, no solo a los polinomiales.
- Las metodologías de validación son tan importantes como los algoritmos de aprendizaje: un modelo excelente evaluado con una metodología incorrecta producirá conclusiones falsas.
- El caso del dataset de Parkinson ilustra claramente cómo la metodología incorrecta (sin grupos) puede llevar a conclusiones clínicas erróneas con consecuencias reales.
- La maldición de la dimensionalidad conecta esta unidad con la siguiente (selección de características), mostrando que el pipeline de ML es un proceso integral donde cada etapa está interconectada.

---

## Recursos y Referencias

### Librerías Python Utilizadas

| Librería | Módulos usados | Función |
|---|---|---|
| `numpy` | `linalg.norm`, operaciones matriciales | Cálculo de bias y varianza |
| `matplotlib` | `pyplot`, `cm` | Visualizaciones y curvas |
| `sklearn.discriminant_analysis` | `QuadraticDiscriminantAnalysis` | Clasificador QDA |
| `sklearn.neighbors` | `KNeighborsClassifier` | Clasificador k-NN |
| `sklearn.svm` | `SVC` | Support Vector Classifier |
| `sklearn.ensemble` | `RandomForestClassifier` | Random Forest |
| `sklearn.naive_bayes` | `GaussianNB` | Naive Bayes |
| `sklearn.metrics` | `accuracy_score`, `f1_score`, `precision_score`, `recall_score`, `confusion_matrix`, `ConfusionMatrixDisplay`, `roc_curve`, `auc`, `matthews_corrcoef`, `balanced_accuracy_score`, `mean_squared_error`, `mean_absolute_error`, `median_absolute_error`, `mean_absolute_percentage_error`, `r2_score` | Todas las métricas |
| `sklearn.model_selection` | `train_test_split`, `KFold`, `LeaveOneOut`, `LeavePOut`, `ShuffleSplit`, `StratifiedKFold`, `StratifiedShuffleSplit`, `GroupKFold`, `GroupShuffleSplit`, `GridSearchCV`, `learning_curve` | Metodologías de validación |
| `sklearn.preprocessing` | `StandardScaler` | Normalización Z-Score |
| `sklearn.linear_model` | `Lasso` | Regresión LASSO (L1) |
| `sklearn.datasets` | `load_digits`, `load_diabetes` | Datasets de prueba |
| `pandas` | `read_csv`, operaciones de DataFrame | Carga y procesamiento del dataset de Parkinson |
| `local.library.regularization` | `PolynomialLinearRegression`, `Fronteras`, `Kernel_classifier`, `plot_ellipse`, `StandardLogisticRegression` | Implementaciones locales del curso |
| `local.library.learning_curve` | `plot_learning_curve` | Curvas de aprendizaje personalizadas |

### Datasets Utilizados

| Dataset | Fuente | Características | Muestras | Uso en el documento |
|---|---|---|---|---|
| Gaussianas Bivariadas | Generado (numpy) | 2 | 200-2000 | Clasificación sintética para ilustrar fronteras y bias-variance |
| Sinusoidal con ruido | Generado (numpy) | 1 | 20-50 | Regresión polinomial y bias-variance |
| Parkinson | UCI Machine Learning Repository | 22 (acústicas) | 195 (32 pacientes) | Validación por grupos |
| Digits | sklearn.datasets | 64 (píxeles) | 1797 | Curvas de aprendizaje |
| Diabetes | sklearn.datasets | 10 | 442 | Regularización LASSO |
| ex2data2.txt | Local del curso | 2 | ~118 | Regularización en regresión logística |

### Herramientas y Frameworks

- **scikit-learn**: Librería principal de Machine Learning en Python.
- **imbalanced-learn**: Librería para manejo de datos desbalanceados (SMOTE y variantes).
- **Jupyter Book**: Entorno de ejecución de los notebooks.

### Referencias para Profundizar

- **AIC y BIC**: Criterios de información de Akaike y Bayesiano para selección de modelos por complejidad.
- **SMOTE**: Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research.
- **Bias-Variance Tradeoff**: Geman, S., Bienenstock, E. & Doursat, R. (1992). "Neural Networks and the Bias/Variance Dilemma."
- **LASSO**: Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." Journal of the Royal Statistical Society.
- **Documentación oficial scikit-learn**: `https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics`

### Repositorio del Curso

- **URL**: `https://jdariasl.github.io/Intro_ML_2025/`
- **Clase 05**: Métricas de error
- **Clase 06**: Complejidad de modelos, sobreajuste y metodologías de validación
- **Clase 07**: Regularización

---

*Documentación generada para la Unidad 3 del curso Introducción al Machine Learning.*
