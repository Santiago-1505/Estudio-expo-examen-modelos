# Clase Guiada: Machine Learning para Predicción de Intención de Compra en E-Commerce

---

## 1. INTRODUCCIÓN Y CONTEXTO DEL PROBLEMA

### ¿Qué problema de negocio se intenta resolver?

Imagina que tienes una tienda en línea. Cada día entran miles de personas: navegan por los productos, visitan páginas de descripción, revisan precios, miran fotografías... y la gran mayoría se va sin comprar. Este fenómeno es conocido en la industria como **abandono de sesión**, y es uno de los problemas más costosos del comercio electrónico.

Las tasas de conversión promedio en e-commerce oscilan entre el 1% y el 4%. Eso significa que por cada 100 personas que entran a una tienda virtual, entre 96 y 99 se van sin comprar. El dataset de este proyecto refleja exactamente esa realidad: de 12.330 sesiones registradas, solo el **15.5% terminaron en compra** (1.908 sesiones). El 84.5% restante no generó ingresos.

Ahora bien, ¿qué pasaría si pudieras saber, mientras un usuario está navegando, si tiene intención real de comprar o no? Esa capacidad de predicción en tiempo real cambiaría por completo la estrategia de negocio.

### ¿Qué significa predecir intención de compra?

Predecir la intención de compra no es adivinar el futuro: es encontrar **patrones en el comportamiento de navegación** que distinguen a los usuarios que comprarán de los que no. Algunos comportamientos intuitivos que podrían indicar mayor intención de compra son:

- Pasar mucho tiempo en páginas de producto (alta `ProductRelated_Duration`).
- Visitar páginas con alto valor estimado para la transacción (alta `PageValues`).
- No rebotar inmediatamente de la primera página visitada (bajo `BounceRates`).
- Volver al sitio como usuario recurrente (`VisitorType = Returning_Visitor`).

El objetivo del modelo es precisamente cuantificar esos patrones y convertirlos en una predicción binaria: **¿esta sesión terminará en compra? Sí o No.**

### ¿Por qué Machine Learning y no reglas manuales?

Una alternativa más simple sería escribir reglas manualmente: "si el usuario visita más de 10 páginas de producto Y su tiempo en el sitio supera 5 minutos, entonces es probable que compre". El problema es que:

1. Las interacciones entre variables son complejas y no lineales. No basta con mirar una variable sola.
2. El comportamiento cambia con el mes, el tipo de visitante, el tráfico, el dispositivo.
3. Son 65 variables (después del encoding). Es humanamente imposible escribir reglas para todas sus combinaciones.

El Machine Learning resuelve esto aprendiendo automáticamente esas combinaciones y pesos desde los datos históricos. Un modelo como XGBoost, por ejemplo, puede descubrir que la combinación de `PageValues` alto + `Month = Noviembre` + `VisitorType = Returning` es mucho más predictiva que cualquiera de esas variables individualmente.

### ¿Qué utilidad tendría este modelo en escenarios reales?

Si el modelo predice en tiempo real que una sesión tiene alta probabilidad de compra, la empresa podría:
- **No intervenir** y dejar que el usuario compre naturalmente (ahorrando costos de promoción).
- **Priorizar atención personalizada** vía chat o recomendaciones relevantes.

Si el modelo predice que la sesión probablemente NO terminará en compra:
- Mostrar un **pop-up de descuento personalizado** justo antes de que el usuario cierre la ventana.
- Activar una **recomendación de productos** basada en el historial de navegación.
- Enviar un **email de seguimiento** horas después si el usuario tiene cuenta registrada.

El valor económico de este tipo de sistema es directo: aumentar la tasa de conversión aunque sea del 15.5% al 17% representa millones en ingresos adicionales para una tienda con alto volumen.

---

## 2. DATASET Y ANÁLISIS INICIAL

### ¿Qué contiene el dataset?

El dataset es el **Online Shoppers Purchasing Intention Dataset** de la UCI Machine Learning Repository. Contiene 12.330 registros, donde cada fila representa **una sesión de navegación de un usuario diferente** en un sitio de e-commerce, registrada a lo largo de un año.

Un detalle metodológico importante: el hecho de que cada sesión corresponda a un usuario diferente elimina el sesgo de un mismo usuario repetido, evitando que el modelo aprenda los hábitos de una persona específica en lugar de patrones generales.

### ¿Qué representa cada tipo de variable?

El dataset tiene 18 variables originales que se pueden agrupar en cuatro categorías:

**Variables de comportamiento de navegación (cuantitativas):**

| Variable | Interpretación práctica |
|---|---|
| `Administrative` | ¿Cuántas páginas de cuenta/perfil visitó? |
| `Administrative_Duration` | ¿Cuánto tiempo pasó en esas páginas? |
| `Informational` | ¿Cuántas páginas de ayuda/contacto visitó? |
| `Informational_Duration` | ¿Cuánto tiempo pasó en esas páginas? |
| `ProductRelated` | ¿Cuántas páginas de producto visitó? Esta es clave. |
| `ProductRelated_Duration` | ¿Cuánto tiempo total estuvo viendo productos? |

**Variables de calidad de la sesión (cuantitativas):**

| Variable | Interpretación práctica |
|---|---|
| `BounceRates` | Porcentaje de sesiones donde el usuario entró y salió de esa página sin interactuar con ningún otro elemento del sitio. Un valor alto indica desinterés. |
| `ExitRates` | Porcentaje de visitas donde esa página fue la última antes de irse. Es similar a BounceRates pero para cualquier página, no solo la de entrada. |
| `PageValues` | Valor promedio estimado de una página en términos de su contribución a completar una transacción. Es calculado por Google Analytics: páginas más cercanas al checkout tienen valores más altos. |
| `SpecialDay` | Qué tan cercana está la sesión a una fecha especial (0 = lejos, 1 = el día exacto). Captura el efecto de Black Friday, San Valentín, Navidad, etc. |

**Variables contextuales (categóricas):**

| Variable | Significado |
|---|---|
| `Month` | Mes del año en que ocurrió la sesión. |
| `OperatingSystems` | Sistema operativo del visitante. |
| `Browser` | Navegador usado. |
| `Region` | Región geográfica. |
| `TrafficType` | Canal de origen (búsqueda orgánica, publicidad, directo, etc.). |
| `VisitorType` | Si el usuario es nuevo, recurrente, u otro. |

**Variables binarias:**

| Variable | Significado |
|---|---|
| `Weekend` | ¿La sesión ocurrió en fin de semana? |
| `Revenue` | **Variable objetivo.** ¿La sesión terminó en compra? |

### Interpretación intuitiva de variables importantes

**PageValues** es la variable más discriminativa del dataset. Piénsalo así: si alguien llega a la página del carrito de compras o a la página de pago, esa página tiene un valor altísimo porque está directamente en el camino de la transacción. Un usuario con `PageValues` alto visitó páginas que estadísticamente preceden a las compras. Es casi una señal directa de intención.

**ProductRelated_Duration** mide el tiempo total invertido mirando productos. Un usuario que pasó 20 minutos leyendo descripciones, comparando opciones y revisando reseñas está más comprometido que uno que pasó 30 segundos.

**BounceRates** es una señal de desinterés. Si el usuario entró al sitio y se fue inmediatamente sin hacer clic en nada, la tasa de rebote de esa página es 100% para esa sesión. Valores altos correlacionan negativamente con la compra.

**ExitRates** captura en qué parte del flujo el usuario abandonó. Si la página de mayor tasa de salida es la de "envío y costos", podría indicar que el costo de envío desmotiva la compra.

**VisitorType** es conceptualmente importante. Los usuarios recurrentes ya conocen el sitio, confían en él y posiblemente volvieron porque encontraron algo interesante en su visita anterior. Tienen tasas de conversión sistemáticamente más altas que los nuevos visitantes.

**Month** captura efectos estacionales. Noviembre (Black Friday, Cyber Monday) tiene comportamientos de compra radicalmente distintos a julio. El modelo necesita aprender estas diferencias temporales.

### El problema del desbalance de clases

Esta es una de las decisiones metodológicas más importantes del proyecto. El dataset tiene:
- 10.422 sesiones sin compra (84.5%)
- 1.908 sesiones con compra (15.5%)

Esta proporción de aproximadamente 5.5 a 1 es un **desbalance moderado-severo**. El problema que esto genera es el siguiente: si un modelo predice "No compra" para TODAS las sesiones, obtiene una exactitud del 84.5%. Sin escribir ningún modelo de Machine Learning, simplemente prediciendo siempre la misma clase, "acertarías" en 8 de cada 10 casos.

Esto hace que la exactitud tradicional (accuracy) sea una métrica completamente engañosa aquí. Un modelo así sería inútil para el negocio, porque nunca detectaría a los compradores reales (la clase que más interesa).

Por esta razón el proyecto usa:
1. **SMOTE** durante el entrenamiento para generar muestras sintéticas de compradores.
2. **Balanced Accuracy** como métrica de optimización durante el GridSearchCV.
3. **ROC-AUC y F1-score** para evaluar el rendimiento final.

---

## 3. PARADIGMA DE APRENDIZAJE Y MODELOS UTILIZADOS

### ¿Por qué se combinan enfoques supervisado y no supervisado?

El proyecto usa **aprendizaje supervisado** como eje principal porque se tiene la variable objetivo `Revenue` para todas las muestras. El objetivo es aprender un mapeo `f(X) → Revenue` que generalice bien a nuevas sesiones.

Se incluye **aprendizaje no supervisado** (K-Means y GMM) con un propósito exploratorio: ¿existen grupos naturales de usuarios en el espacio de características? ¿Esos grupos coinciden con compradores vs. no compradores? La respuesta resultó ser no, pero ese hallazgo en sí mismo es informativo.

### Random Forest

**¿Cómo funciona?** Random Forest construye cientos de árboles de decisión, cada uno entrenado sobre una muestra aleatoria del conjunto de entrenamiento (con reemplazo, técnica llamada *bagging*) y usando solo un subconjunto aleatorio de variables en cada nodo de decisión. La predicción final es la votación mayoritaria de todos los árboles.

**Intuición:** Imagina 200 analistas de negocio, cada uno con información ligeramente diferente sobre los mismos clientes. Cada analista llega a su propia conclusión. La decisión final se toma por mayoría. Ningún analista individual puede memorizar todos los casos, y la diversidad entre ellos captura diferentes aspectos del problema.

**¿Cuándo funciona bien?** Con variables de distinto tipo y escala, cuando hay interacciones no lineales entre variables, con datos tabulares de tamaño mediano, y cuando el overfitting es una preocupación (el bagging lo controla naturalmente).

**Ventajas:** Robusto al ruido, maneja bien variables irrelevantes, difícil de sobreajustar agresivamente, da importancia de variables gratis.

**Limitaciones:** No extrapolación (no puede predecir más allá del rango de entrenamiento), lento con datasets muy grandes, difícil de interpretar individualmente.

**Por qué tiene sentido aquí:** El dataset tiene 65 variables con comportamientos muy distintos (algunas continuas muy sesgadas, otras binarias del OHE). Random Forest maneja esa heterogeneidad naturalmente.

### XGBoost

**¿Cómo funciona?** XGBoost usa *gradient boosting*: construye árboles secuencialmente donde cada nuevo árbol aprende a corregir los errores residuales del conjunto de árboles anterior. Matemáticamente, minimiza una función de pérdida usando gradiente descendente en el espacio funcional (de ahí "gradient boosting").

**Intuición:** Piénsalo como un equipo donde cada nuevo miembro se especializa en los casos que el equipo anterior falló. El primer árbol predice lo mejor que puede. El segundo árbol se entrena sobre los errores del primero. El tercero sobre los errores del conjunto de los dos primeros. Y así sucesivamente, construyendo un sistema cada vez más preciso.

**¿Cuándo funciona bien?** Es el estado del arte en datos tabulares estructurados. Especialmente poderoso cuando hay relaciones complejas y no lineales, y cuando el dataset es de tamaño mediano (miles a millones de filas).

**Ventajas sobre Random Forest:** Generalmente más preciso, más flexible en la función de pérdida, puede manejar valores faltantes nativamente (aunque aquí no los hay), mejor regularización incorporada.

**Limitaciones:** Más hiperparámetros que tunear, más propenso a overfitting si no se regulariza bien, más costoso computacionalmente.

**Por qué resultó el mejor modelo:** XGBoost capturó mejor las interacciones no lineales complejas entre variables como `PageValues`, `Month` y `VisitorType`, que son las más predictivas.

### SVM (Support Vector Machine)

**¿Cómo funciona?** SVM busca el hiperplano de separación que maximiza el margen entre las dos clases. Con el kernel RBF (Radial Basis Function), proyecta implícitamente los datos a un espacio de dimensiones superiores donde las clases pueden ser linealmente separables.

**Intuición:** Imagina que tienes puntos rojos y azules en un plano 2D mezclados. No se pueden separar con una línea recta. El kernel RBF "eleva" esos puntos a un espacio 3D donde sí se pueden separar con un plano. La magia matemática del *kernel trick* hace esto sin calcular explícitamente esa transformación.

**¿Cuándo funciona bien?** Con datos de dimensión media-alta, cuando las clases son separables (aunque sea en dimensiones altas), y especialmente con el kernel lineal cuando hay linealidad en los datos.

**Resultado en el proyecto:** SVM con kernel lineal obtuvo ROC-AUC de 0.9132, comparable a Random Forest (0.9103) y casi igual a XGBoost (0.9175). Esto sugiere que el problema tiene componentes tanto lineales como no lineales.

**Limitación práctica:** SVM es computacionalmente costoso con datasets grandes. Con 12.330 muestras y 65 variables ya requiere tiempo considerable, por eso la malla de hiperparámetros fue más reducida que la de los otros modelos.

### MLP (Multilayer Perceptron)

**¿Cómo funciona?** Es una red neuronal feedforward: capas de neuronas interconectadas donde cada neurona aplica una transformación no lineal a la combinación lineal de sus entradas. El aprendizaje ocurre mediante backpropagation, que ajusta los pesos de las conexiones para minimizar el error.

**Intuición:** Cada capa aprende representaciones progresivamente más abstractas de los datos. La primera capa podría aprender si el usuario visitó muchas páginas de producto. La segunda capa podría combinar eso con el tiempo en el sitio. La salida da la probabilidad de compra.

**Resultado problemático:** El MLP mostró **overfitting severo**. La exactitud balanceada en entrenamiento fue 0.9799–0.9990, pero en validación cruzada bajó a 0.8028–0.8068. Esta brecha de ~0.18 puntos es una señal clara de que la red memorizó el conjunto de entrenamiento en lugar de aprender patrones generalizables.

**Por qué ocurrió el overfitting:** Las redes neuronales tienen muchísimos parámetros. Con solo 9.864 muestras de entrenamiento y 65 variables, hay relativamente pocos ejemplos por parámetro. Aunque se usó regularización L2 (`alpha`), no fue suficiente para evitar la memorización.

### Logistic Regression

**¿Cómo funciona?** Aprende una combinación lineal de las variables de entrada, aplica la función sigmoide para convertirla en probabilidad, y clasifica según un umbral (generalmente 0.5). Es el modelo lineal paramétrico por excelencia para clasificación binaria.

**Su rol en el proyecto:** Sirve como **línea base lineal**. Si los modelos no lineales (Random Forest, XGBoost) superan a la regresión logística, confirma que el problema tiene componentes no lineales que vale la pena capturar. La diferencia de ROC-AUC entre Logistic Regression (0.9136) y XGBoost (0.9175) es pequeña (~0.004 puntos), lo cual indica que el problema tiene una fuerte componente lineal, aunque los métodos de árboles la capturan algo mejor.

**Ventajas:** Extremadamente interpretable (los coeficientes tienen significado directo), rápido, probabilidades bien calibradas, sin hiperparámetros críticos.

### KNN (K-Nearest Neighbors)

**¿Cómo funciona?** Para clasificar un nuevo punto, busca los `k` puntos más cercanos en el conjunto de entrenamiento y asigna la clase más frecuente entre ellos. No hay fase de "entrenamiento" como tal: el modelo simplemente memoriza todos los ejemplos.

**El fracaso del KNN en este proyecto:** KNN obtuvo ROC-AUC de apenas 0.63 en prueba, con un overfitting brutal (accuracy train = 1.0 con k=3 o k=5). Hay dos razones:

1. **La maldición de la dimensionalidad:** Con 65 variables, la distancia euclidiana pierde significado. Todos los puntos tienden a ser igualmente "cercanos" entre sí en alta dimensión. El concepto de "vecino cercano" se vuelve difuso.

2. **El desbalance de clases:** Como la clase mayoritaria (No compra) domina, los vecinos más cercanos de casi cualquier punto son No compradores, independientemente de la realidad.

### K-Means

**¿Cómo funciona?** K-Means divide el dataset en `k` grupos (clusters) minimizando la varianza intra-cluster. Asigna cada punto al centroide más cercano, recalcula los centroides, y repite hasta convergencia.

**Por qué se usó como clasificador:** Se implementó `KMeansClassifier`, una clase personalizada que agrupa los datos y luego asigna a cada cluster la etiqueta de su clase mayoritaria. Es un puente entre el paradigma no supervisado y la tarea de clasificación.

**Por qué fracasó:** K-Means asume que las clases forman clusters geométricamente compactos y separables. En este dataset, los compradores y no compradores no forman grupos separados en el espacio de características: están mezclados. El hecho de comprar no se explica por una región geométrica del espacio de características, sino por combinaciones complejas de variables. ROC-AUC de 0.5 es equivalente a una predicción aleatoria.

### GMM (Gaussian Mixture Model)

**¿Cómo funciona?** Un GMM modela la distribución de los datos como una mezcla de distribuciones gaussianas. Para clasificación, se ajusta un GMM independiente para cada clase (compradores y no compradores). Al clasificar, se calcula la verosimilitud de que un punto pertenezca a cada GMM y se asigna la clase de mayor probabilidad posterior (clasificación bayesiana generativa).

**Diferencia con K-Means:** K-Means tiene asignaciones duras (un punto pertenece a exactamente un cluster). GMM tiene asignaciones suaves (un punto puede pertenecer a varios clusters con diferentes probabilidades). GMM es más flexible porque las gaussianas pueden tener diferente forma y orientación según el parámetro `covariance_type`.

**Resultado:** GMM obtuvo ROC-AUC de 0.8423, mejor que K-Means pero inferior a los modelos supervisados bien ajustados. Su ventaja sobre K-Means es que puede modelar distribuciones más complejas, pero sigue siendo un modelo generativo que lucha con la superposición de clases.

---

## 4. CONFIGURACIÓN EXPERIMENTAL Y PIPELINE DE ENTRENAMIENTO

Esta sección es el corazón metodológico del proyecto. Cada decisión tiene una justificación precisa.

### División Train/Test 80/20

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

Lo primero que se hace es separar el 20% de los datos como **conjunto de prueba**, que no se toca durante todo el proceso de entrenamiento y selección de modelos. Este conjunto simula el "futuro" y permite una estimación imparcial del rendimiento real del modelo.

**¿Por qué 80/20?** Es la proporción estándar para datasets de este tamaño. Con 12.330 muestras, el 20% da 2.466 ejemplos de prueba, suficientes para estimaciones estables de métricas.

**¿Por qué `stratify=y`?** Sin estratificación, la división aleatoria podría poner desproporcionadamente más compradores en train que en test o viceversa. Con `stratify=y`, ambos subconjuntos mantienen exactamente la misma proporción de clases (~15.5% de `Revenue=1`) que el dataset completo.

**La regla de oro:** Una vez separado el test set, **nunca más se toca hasta la evaluación final**. Ni para escalar, ni para ajustar, ni para explorar. Si se usa el test set para cualquier decisión, los resultados estarán artificialmente inflados.

### Stratified K-Fold Cross Validation

```python
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

La validación cruzada resuelve un problema crítico: con una sola división train/validación, el rendimiento estimado depende de qué ejemplos cayeron en cada partición. Podría ser que por azar todos los casos "fáciles" quedaron en validación, dando una estimación optimista. O al revés.

**¿Cómo funciona K-Fold?** Divide el conjunto de entrenamiento en 10 partes iguales. Itera 10 veces: en cada iteración, 9 partes son para entrenar y 1 para validar, rotando cuál es la parte de validación. El score final es el promedio de los 10 scores. Esto garantiza que cada ejemplo de entrenamiento se usa exactamente una vez para validación.

**¿Por qué estratificado?** Con desbalance de clases, algunas particiones podrían tener pocas o ninguna muestra de la clase minoritaria. `StratifiedKFold` garantiza que cada fold mantiene la misma proporción de clases (~15.5% de compradores) que el conjunto completo. Esto da estimaciones más estables y representativas.

**¿Por qué 10 folds?** Es el estándar en la literatura de ML (recomendado por Kohavi 1995). Con 5 folds hay más sesgo (menos datos para entrenar en cada iteración); con 20 folds hay más varianza (mayor costo computacional). 10 es el balance empíricamente validado para datasets de este tamaño.

### StandardScaler

```python
('scaler', StandardScaler())
```

`StandardScaler` transforma cada variable para que tenga **media 0 y desviación estándar 1**:

```
x_normalizada = (x - media) / desviacion_estandar
```

**¿Por qué es necesario?** Porque las variables tienen escalas muy diferentes. `ProductRelated_Duration` puede llegar a miles de segundos; `BounceRates` está entre 0 y 1; `SpecialDay` entre 0 y 1; `ProductRelated` puede ser decenas de páginas. Sin normalización:

- SVM y KNN (basados en distancias) serían dominados por las variables de mayor magnitud.
- MLP tendría gradientes inestables durante el entrenamiento.
- SMOTE generaría muestras sintéticas en un espacio sesgado.

**¿Es necesario para Random Forest y XGBoost?** Técnicamente no, porque los árboles son invariantes a escala (trabajan con comparaciones ordinales de valores, no distancias). Sin embargo, se incluye para mantener consistencia en todos los pipelines y porque SMOTE sí se beneficia de los datos escalados.

**Precaución clave:** El scaler se ajusta (`fit`) solo sobre los datos de entrenamiento de cada fold, y luego se aplica (`transform`) al fold de validación. Nunca se ajusta sobre los datos de validación. Si se hiciera al revés, estarías "filtrando" información del futuro al modelo (data leakage).

### SMOTE (Synthetic Minority Oversampling Technique)

```python
('smote', SMOTE(random_state=42, sampling_strategy=0.3))
```

SMOTE genera muestras sintéticas de la clase minoritaria (compradores) para reducir el desbalance. El proceso es:

1. Para cada muestra de la clase minoritaria, encuentra sus `k` vecinos más cercanos en el espacio de características (también de la clase minoritaria).
2. Genera un nuevo punto sintético en el segmento de línea entre la muestra y uno de sus vecinos, eligiendo una posición aleatoria en ese segmento.

**¿Por qué no simplemente duplicar muestras existentes?** Duplicar (oversampling básico) no agrega información nueva: el modelo simplemente ve el mismo punto más veces. SMOTE genera puntos en regiones del espacio entre muestras reales existentes, creando una distribución más rica de la clase minoritaria.

**¿Por qué `sampling_strategy=0.3`?** Este valor indica que después de SMOTE, la clase minoritaria tendrá el 30% de la magnitud de la clase mayoritaria. Antes del SMOTE, la proporción era ~0.183 (18.3%). Llevarlo al 30% es un balance entre representar mejor a los compradores y no distorsionar tanto la distribución que el modelo aprenda a partir de datos completamente artificiales.

**Valores evaluados: 0.22, 0.30, 0.40.** Se incluyeron los tres en la malla de GridSearchCV para que el algoritmo seleccionara empíricamente cuál da mejor rendimiento para cada modelo. El más frecuentemente seleccionado fue 0.30.

### El Data Leakage: el error más peligroso en ML

**Data leakage** ocurre cuando información del conjunto de validación o prueba "contamina" el proceso de entrenamiento, produciendo estimaciones artificialmente optimistas del rendimiento real.

**El error clásico con SMOTE:** Aplicar SMOTE a todo el dataset antes de dividir en folds. Si se hace así:

1. Se generan muestras sintéticas basadas en todos los datos (incluyendo los que después serán de validación).
2. Los puntos sintéticos generados pueden ser "copias suavizadas" de muestras de validación.
3. Cuando el modelo se evalúa en validación, ya "conoce" indirectamente esas muestras.
4. El score de validación es artificialmente alto.

**La solución correcta: SMOTE dentro del pipeline.** Al incluir SMOTE dentro del pipeline de sklearn/imblearn:

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE()),
    ('classifier', RandomForestClassifier())
])
```

Durante cada fold de la validación cruzada:
- El scaler se ajusta solo sobre el fold de entrenamiento.
- SMOTE solo genera muestras sintéticas basadas en el fold de entrenamiento.
- El fold de validación nunca influye en ninguna transformación.
- Las métricas de validación reflejan rendimiento real en datos nunca vistos.

**El error con el scaler:** Si se hiciera `StandardScaler().fit_transform(X)` antes del GridSearchCV, el scaler usaría la media y desviación estándar de todos los datos (incluyendo validación) para escalar. Esto es data leakage más sutil pero igualmente dañino.

**El pipeline garantiza el orden correcto automáticamente.** Es la razón por la que se usa Pipeline en lugar de aplicar cada transformación manualmente.

### GridSearchCV

```python
grid_rf = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid_rf,
    cv=cv_strategy,
    scoring='balanced_accuracy',
    n_jobs=-1,
    refit=True
)
grid_rf.fit(X_train, y_train)
```

`GridSearchCV` realiza una **búsqueda exhaustiva** sobre todas las combinaciones posibles de hiperparámetros definidas en `param_grid`. Para cada combinación, ejecuta la validación cruzada estratificada completa (10 folds) y registra el score promedio.

**¿Cuántas combinaciones se evaluaron para Random Forest?** La malla era:
- `smote__sampling_strategy`: 3 valores
- `smote__k_neighbors`: 3 valores
- `classifier__max_depth`: 2 valores
- `classifier__min_samples_leaf`: 2 valores
- `classifier__n_estimators`: 1 valor
- `classifier__max_features`: 1 valor

Total: 3 × 3 × 2 × 2 = 36 combinaciones. Con 10 folds cada una: **360 modelos entrenados** solo para Random Forest.

**`n_jobs=-1`** usa todos los núcleos del procesador en paralelo, reduciendo significativamente el tiempo de ejecución.

**`refit=True`** indica que, una vez encontrados los mejores hiperparámetros, GridSearchCV entrena automáticamente un modelo final con esos parámetros usando todos los datos de entrenamiento disponibles. Este es el modelo que queda en `grid.best_estimator_`.

---

## 5. HIPERPARÁMETROS Y BÚSQUEDA DE CONFIGURACIONES

### ¿Qué son los hiperparámetros?

Los parámetros de un modelo son los valores que el algoritmo aprende de los datos (los pesos de una red neuronal, los coeficientes de la regresión logística, las divisiones de un árbol). Los **hiperparámetros** son los valores que se fijan antes del entrenamiento y que controlan cómo ocurre ese aprendizaje.

Piénsalo así: los parámetros son las decisiones que el modelo toma durante el entrenamiento; los hiperparámetros son las reglas bajo las cuales puede tomar esas decisiones.

### Hiperparámetros de Random Forest

**`n_estimators = 200`** (fijo): Número de árboles en el bosque. Más árboles = predicciones más estables y menos varianza, pero más tiempo de cómputo. A partir de cierto número (generalmente 100-200), el beneficio marginal es mínimo. Se fijó en 200 como un valor suficientemente robusto.

**`max_depth` ∈ {10, 20}**: Profundidad máxima de cada árbol. Un árbol sin límite de profundidad crecería hasta que cada hoja tenga una sola muestra, memorizando perfectamente el entrenamiento (overfitting). `max_depth=10` crea modelos más simples y generalizables; `max_depth=20` permite más complejidad. El mejor fue `max_depth=10`, indicando que la regularización adicional es beneficiosa.

**`min_samples_leaf` ∈ {1, 4}**: Número mínimo de muestras que debe tener una hoja para existir. Con `min_samples_leaf=1`, las hojas pueden tener una sola muestra (alta complejidad, posible overfitting). Con `min_samples_leaf=4`, cada hoja necesita al menos 4 muestras, creando fronteras de decisión más suaves. El mejor fue `min_samples_leaf=4`.

**`max_features = 'sqrt'`** (fijo): En cada nodo, evaluar solo la raíz cuadrada del número total de features (≈8 de las 65). Esto introduce aleatoriedad que decorrelaciona los árboles entre sí, que es precisamente lo que hace que el Random Forest funcione mejor que un solo árbol de decisión.

### Hiperparámetros de XGBoost

**`learning_rate` (eta) ∈ {0.05, 0.10}**: La tasa de aprendizaje escala la contribución de cada árbol al ensemble. Una tasa menor (0.05) requiere más árboles para el mismo rendimiento pero generalmente generaliza mejor porque cada árbol "da pasos pequeños" hacia el mínimo de la función de pérdida. Con `n_estimators=200` y `learning_rate=0.05`, el modelo fue más conservador y generalizó mejor.

**`max_depth` ∈ {3, 6}**: En boosting, la profundidad recomendada es mucho menor que en Random Forest (3-6 vs. 10-20+). Esto porque en boosting, la complejidad acumulada de cientos de árboles poco profundos puede ser enorme. El mejor fue `max_depth=3`, árbol de decisión muy simple pero efectivo en conjunto.

**`subsample = 0.8`** (fijo): Cada árbol usa el 80% de las muestras de entrenamiento, seleccionadas aleatoriamente. Esto reduce el overfitting e introduce variedad entre los árboles.

**`colsample_bytree = 0.8`** (fijo): Cada árbol usa el 80% de las features. Similar al efecto de `max_features` en Random Forest. Reduce correlación entre árboles y mejora generalización.

### Hiperparámetros del MLP

**`hidden_layer_sizes` ∈ {(50,), (100,)}**: Define la arquitectura de la red. `(50,)` significa una capa oculta con 50 neuronas; `(100,)` una capa con 100. La entrada tiene 65 neuronas (una por feature) y la salida 1 (probabilidad de compra). Con una sola capa oculta, la red puede aproximar funciones continuas arbitrarias (teorema de aproximación universal), pero puede necesitar muchas neuronas para funciones muy complejas.

**`alpha` ∈ {0.0001, 0.001}**: Parámetro de regularización L2. Penaliza los pesos grandes de la red, forzando a que la solución sea más simple. Valores más altos de `alpha` = más regularización = modelo más simple. Como el MLP sufría overfitting, `alpha=0.001` fue preferido.

**`learning_rate_init` ∈ {0.001, 0.01}**: La tasa de aprendizaje inicial del optimizador Adam. Valores muy altos hacen que el entrenamiento sea inestable (oscila alrededor del mínimo); valores muy bajos hacen que converja lentamente. `0.01` fue seleccionado, aunque el overfitting persistió independientemente de estos ajustes.

### Hiperparámetros de SVM

**`kernel` ∈ {linear, rbf}**: El kernel define el tipo de frontera de decisión. El kernel lineal crea hiperplanos de separación lineales en el espacio original. El kernel RBF proyecta los datos a un espacio infinito-dimensional donde fronteras lineales pueden capturar relaciones no lineales. El kernel lineal resultó ganador (ROC-AUC ≈ 0.91), confirming la componente lineal del problema.

**`C` ∈ {0.01, 0.1, 1}**: El parámetro de margen blando. `C` grande = penalización alta por clasificaciones incorrectas = el modelo intentará clasificar bien todos los puntos, incluso a costa de reducir el margen (puede overfit). `C` pequeño = margen grande, más puntos mal clasificados aceptados (regularización). El óptimo fue `C=1`.

**`gamma` ∈ {scale, auto, 0.01, 0.1}** (solo para rbf): Controla el radio de influencia de cada muestra de entrenamiento. `gamma` alto = radio pequeño, cada punto solo influye en sus vecinos muy cercanos (modelo complejo). `gamma` bajo = influencia amplia (modelo más suave). `gamma='scale'` usa `1/(n_features * var(X))` como valor por defecto.

---

## 6. MÉTRICAS DE EVALUACIÓN

### Por qué accuracy sola es engañosa

Con 84.5% de "No compra", un clasificador que siempre predice "No compra" obtiene accuracy del 84.5% sin aprender nada. Ese modelo es completamente inútil para el negocio.

Se necesitan métricas que evalúen el comportamiento del modelo en ambas clases.

### Conceptos base: TP, TN, FP, FN

Para entender las métricas, hay que entender la **matriz de confusión**:

```
                  Predicho: No compra    Predicho: Compra
Real: No compra        TN (Verdadero Neg)   FP (Falso Pos)
Real: Compra           FN (Falso Neg)       TP (Verdadero Pos)
```

- **TP (Verdadero Positivo):** El modelo predijo "compra" y sí compró. ✓
- **TN (Verdadero Negativo):** El modelo predijo "no compra" y no compró. ✓
- **FP (Falso Positivo):** El modelo predijo "compra" pero no compró. Error Tipo I.
- **FN (Falso Negativo):** El modelo predijo "no compra" pero sí compró. Error Tipo II.

Para el negocio, **los FN son más costosos**: son compradores potenciales a los que no se les ofreció ninguna intervención. Los FP son menos graves: se ofrece un descuento innecesario a alguien que no iba a comprar.

### Recall (Sensibilidad)

```
Recall = TP / (TP + FN)
```

¿De todos los compradores reales, qué fracción detecté? Es la métrica que captura cuántos compradores el modelo "rescata". Un recall alto en la clase positiva es crucial para el negocio.

### Precision (Precisión)

```
Precision = TP / (TP + FP)
```

¿De todos los que predije como compradores, qué fracción realmente compró? Un precision alto significa que cuando el modelo dice "este va a comprar", es porque tiene buena evidencia. Relevante para controlar el costo de las intervenciones (si se dan muchos descuentos innecesarios, el costo sube).

### F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Es la media armónica entre Precision y Recall. La media harmónica penaliza desbalances: si uno de los dos es muy bajo, el F1 baja aunque el otro sea perfecto. Es la métrica estándar para problemas desbalanceados cuando importan tanto los FP como los FN.

El proyecto usó F1 sobre la clase positiva (compradores) como métrica secundaria de evaluación en el conjunto de prueba. El mejor resultado fue XGBoost con F1 = 0.7827.

### Balanced Accuracy

```
Balanced Accuracy = (Recall_clase_0 + Recall_clase_1) / 2
```

Es el promedio del recall por clase. Si el modelo clasifica bien el 90% de los no compradores y el 80% de los compradores, la Balanced Accuracy es 0.85. Trata ambas clases con igual importancia independientemente de su frecuencia.

Esta fue la **métrica de optimización durante el GridSearchCV**, porque es directamente sensible al desbalance sin importar el tamaño relativo de las clases.

### ROC-AUC (Área Bajo la Curva ROC)

La curva ROC (Receiver Operating Characteristic) grafica la **Tasa de Verdaderos Positivos (Recall) vs. la Tasa de Falsos Positivos (1 - Especificidad)** para todos los posibles umbrales de clasificación.

El **AUC (Área Bajo la Curva)** resume esa curva en un solo número:
- AUC = 1.0: clasificador perfecto.
- AUC = 0.5: clasificador aleatorio (la diagonal).
- AUC < 0.5: peor que aleatorio (raro, indica un bug o inversión de etiquetas).

**¿Por qué es tan útil?** Porque es **independiente del umbral de clasificación**. En lugar de evaluar el modelo a un solo umbral (0.5 por defecto), evalúa su capacidad discriminativa en todos los umbrales posibles. Esto es especialmente valioso con clases desbalanceadas, donde el umbral óptimo puede no ser 0.5.

El mejor resultado fue XGBoost con ROC-AUC = 0.9175 en el conjunto de prueba.

### Classification Report

El `classification_report` de sklearn produce una tabla con Precision, Recall y F1 por clase:

```
              precision    recall   f1-score   support
No compra         0.96      0.94      0.95      2085
Compra            0.75      0.84      0.79       381
accuracy                              0.93      2466
macro avg         0.86      0.89      0.87      2466
weighted avg      0.93      0.93      0.93      2466
```

La fila "Compra" es la más importante: muestra cómo se desempeña el modelo específicamente en detectar compradores.

### Matriz de Confusión Normalizada

En el proyecto se usó `normalize='true'`, que divide cada fila por el total real de esa clase:

```python
cm = confusion_matrix(y_test, y_pred, normalize='true')
```

Esto da proporciones: la diagonal muestra la tasa de acierto por clase (recall). Si el valor en la celda [Compra, Compra] es 0.78, el modelo detectó el 78% de los compradores reales (Recall = 0.78).

---

## 7. INTERPRETACIÓN DE RESULTADOS DEL ENTRENAMIENTO

### La tabla de resultados y qué nos dice

Los resultados del top 3 para cada modelo mostraron un patrón claro:

| Modelo | BA Train | BA Val (CV) | ROC-AUC Test | F1 Test |
|---|---|---|---|---|
| Random Forest 1 | 0.8979 | 0.8543 | 0.9103 | 0.7785 |
| XGBoost 1 | 0.8427 | 0.8345 | 0.9175 | 0.7827 |
| MLP 1 | **0.9799** | 0.8068 | 0.8530 | 0.6777 |
| K-Means 1 | 0.5000 | 0.5136 | 0.5000 | 0.0000 |
| Logistic Reg 1 | 0.8491 | 0.8466 | 0.9136 | 0.7712 |
| KNN 1 | **1.0000** | 0.5970 | 0.6321 | 0.3311 |
| SVM 1 | 0.8364 | 0.8339 | 0.9132 | 0.7785 |
| GMM 1 | 0.8311 | 0.8099 | 0.8423 | 0.6667 |

### Por qué XGBoost y Random Forest lideraron

Ambos son métodos de ensamble basados en árboles. Su ventaja sobre los demás modelos proviene de varios factores que se alinean bien con este problema:

1. **Captura de interacciones no lineales:** `PageValues` no actúa en forma lineal. Un `PageValues` moderado puede combinarse con una `ProductRelated_Duration` alta para predecir compra, pero ninguno de los dos individualmente es tan predictivo. Los árboles capturan esas interacciones naturalmente.

2. **Robustez al ruido y variables irrelevantes:** De las 65 variables, muchas (especialmente las del OHE de Browser o Region) tienen poca capacidad discriminativa. Los árboles las ignoran naturalmente porque no ofrecen splits informativos.

3. **Sin sensibilidad a la escala:** No necesitan normalización (aunque se incluyó por consistencia), son robustos a outliers y a distribuciones sesgadas como las variables de duración (muchos ceros y pocos valores muy altos).

4. **La diferencia entre los dos:** XGBoost supera a Random Forest marginalmente (0.9175 vs. 0.9103 en ROC-AUC) porque el boosting puede aprender relaciones sutiles que el bagging pasa por alto. XGBoost construye cada árbol mirando los errores del árbol anterior, lo cual es una forma más informada de aprender que el bagging aleatorio.

**El gap entre train y validación es pequeño** en ambos (RF: 0.8979 vs. 0.8543; XGBoost: 0.8427 vs. 0.8345), indicando buena generalización sin overfitting severo.

### Por qué MLP muestra overfitting

El patrón del MLP es alarmante: BA en train entre 0.98 y 0.999, pero BA en validación solo 0.80-0.81. Esto es un gap de ~0.18-0.19 puntos.

**¿Por qué ocurre esto específicamente con el MLP?**

Las redes neuronales tienen capacidad de memorización muy alta. Con 65 variables de entrada, una capa oculta de 100 neuronas y una de salida, el modelo tiene: 65×100 + 100 + 100×1 + 1 = **6.601 parámetros**. Con solo ~9.864 muestras de entrenamiento, la proporción parámetros/muestras es alta. La red tiene suficiente capacidad para memorizar el conjunto de entrenamiento.

La regularización L2 (`alpha=0.001`) no fue suficiente. Posibles mejoras incluirían Dropout, más datos, o una arquitectura más pequeña.

**¿Por qué el ROC-AUC del MLP en test (0.8530) es razonablemente bueno si hay overfitting?** Porque ROC-AUC mide la capacidad discriminativa en términos de ranking, no de calibración exacta. El MLP aprendió suficiente sobre la estructura de los datos para ordenar las muestras razonablemente bien, aunque no tan bien como si no hubiera overfitting.

### Por qué KNN tuvo bajo desempeño

KNN con `n_neighbors=3` obtiene BA en train = 1.0 (memorización perfecta) pero BA en validación de solo 0.597. Este es el **overfitting más extremo** del proyecto.

Con k=3, si los 3 vecinos más cercanos en el conjunto de entrenamiento son la misma muestra (o copias muy similares), el modelo simplemente memorizó. En test, los vecinos son diferentes y el modelo falla.

El ROC-AUC en test de 0.6321 indica que, aunque el modelo puede separar las clases mejor que aleatoriamente, es muy inferior a los métodos basados en árboles. La maldición de la dimensionalidad con 65 variables degrada seriamente la calidad de las distancias.

### Por qué K-Means falla como clasificador

K-Means con ROC-AUC = 0.5000 y F1 = 0.0000 es estadísticamente indistinguible de adivinar al azar. El F1 = 0 indica que el modelo nunca predice la clase positiva (compradores), o predice siempre la misma clase.

La razón fundamental: **los compradores no forman un cluster geométrico separado en el espacio de características**. La intención de compra es un fenómeno de alta complejidad que emerge de la combinación de muchas variables. No hay una región del espacio 65-dimensional que "contenga" a los compradores y excluya a los no compradores.

K-Means fue útil en su rol original (exploración no supervisada), pero adaptarlo a clasificación forzó un paradigma para el que no está diseñado.

### El concepto de generalización

**Generalización** es la capacidad del modelo de rendir bien en datos no vistos durante el entrenamiento. Un modelo que generaliza bien tiene:
- Score de entrenamiento alto pero no perfecto.
- Score de validación cercano al de entrenamiento.
- Score de test similar al de validación.

**Overfitting** (sobreajuste): el modelo memoriza el entrenamiento pero no generaliza. Se manifiesta como un gap grande entre train y validación. KNN y MLP fueron los casos más claros.

**Underfitting** (subajuste): el modelo es demasiado simple para capturar los patrones. Todos los scores son bajos. No hubo casos claros de underfitting en este proyecto, excepto K-Means que simplemente no es adecuado para la tarea.

---

## 8. CURVAS DE APRENDIZAJE

### ¿Qué son y para qué sirven?

Las curvas de aprendizaje muestran cómo evoluciona el rendimiento del modelo (ROC-AUC en este caso) en función del tamaño del conjunto de entrenamiento. Se generan entrenando el modelo repetidamente con subconjuntos de tamaño creciente (10%, 20%, ..., 100% del conjunto de entrenamiento) y evaluando tanto en el subconjunto de entrenamiento como en validación cruzada.

```python
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model, X=X_train, y=y_train,
    cv=cv_strategy, scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
```

### ¿Cómo se leen las curvas?

**La curva azul (entrenamiento)** muestra el ROC-AUC del modelo sobre sus propios datos de entrenamiento. Casi siempre empieza alta (con pocos datos, el modelo los memoriza fácilmente) y va bajando ligeramente al aumentar el tamaño.

**La curva naranja (validación)** muestra el ROC-AUC promedio de los 10 folds de validación cruzada. Generalmente empieza baja (con pocos datos, el modelo aprende poco) y sube al aumentar el tamaño.

**Gap pequeño entre las curvas = buena generalización.** Si ambas curvas convergen conforme aumenta el tamaño del conjunto, el modelo generaliza bien. Más datos sería beneficioso pero no crítico.

**Gap grande entre las curvas = overfitting.** Si la curva de entrenamiento está muy por encima de la de validación, el modelo está memorizando el entrenamiento. Se necesita más regularización, más datos, o un modelo más simple.

**Ambas curvas planas y bajas = underfitting.** Si la validación deja de mejorar aunque se agreguen más datos, el modelo es demasiado simple para capturar los patrones. Se necesita un modelo más complejo.

### Interpretación para los modelos del proyecto

**Random Forest:** La curva de validación sube de forma estable y converge hacia la de entrenamiento. El gap es pequeño (~0.05 puntos a máximo tamaño), indicando buena generalización. La curva de validación todavía tiene pendiente positiva al final, lo que sugiere que más datos podrían mejorar ligeramente el rendimiento.

**XGBoost:** Patrón similar a Random Forest. La curva de validación es más suave (menor varianza entre folds), lo que indica un modelo más estable. El gap entre entrenamiento y validación es menor que en Random Forest, lo cual es consistente con su mejor rendimiento en test.

**MLP:** La curva de entrenamiento se mantiene muy alta (cercana a 1.0) mientras la de validación sube moderadamente y se estabiliza alrededor de 0.85. El gap persistente y grande es la señal de overfitting. Agregar más datos podría ayudar, pero la brecha no desaparece completamente.

**KNN:** La curva de entrenamiento es 1.0 para todo el rango (memorización perfecta). La curva de validación permanece baja (~0.65-0.70) y casi no sube con más datos. Esto indica que el problema no es la cantidad de datos sino el modelo: KNN simplemente no es adecuado para este espacio de alta dimensión.

**Regresión Logística:** Ambas curvas convergen rápidamente y se mantienen muy cercanas entre sí. Esto es el patrón ideal: no hay overfitting, y el modelo usa bien cada dato adicional. La diferencia pequeña entre LR y XGBoost confirma que gran parte del problema es capturado por relaciones lineales.

---

## 9. REDUCCIÓN DE DIMENSIÓN

### ¿Por qué se estudia la reducción de dimensión?

El dataset tiene 65 variables después del encoding (OHE generó muchas columnas binarias para Browser, Region, TrafficType, etc.). Muchas de esas columnas tienen muy poca varianza o están correlacionadas entre sí. La reducción de dimensión busca:

1. **Eliminar ruido y redundancia**, simplificando el espacio de entrada.
2. **Reducir la complejidad computacional**, acelerando el entrenamiento e inferencia.
3. **Potencialmente mejorar la generalización**, al eliminar variables que solo causan overfitting.

Sin embargo, la reducción tiene un costo: si se eliminan variables que contienen información discriminativa, el rendimiento cae.

### 9.1 Análisis Individual de Variables

Antes de aplicar técnicas de reducción, se analiza cuáles variables tienen baja capacidad discriminativa.

**Correlación de Pearson absoluta con Revenue:**

```python
pearson_corr = X_train_df.corrwith(y_series).abs().sort_values(ascending=False)
```

Mide la fuerza de la relación **lineal** entre cada variable y la variable objetivo. Un valor de 0.3 indica correlación lineal moderada; por debajo de 0.05 es prácticamente nula.

**Limitación:** Pearson solo captura relaciones lineales. Una variable puede tener correlación de Pearson cercana a cero con `Revenue` pero aun así ser útil para un modelo no lineal como Random Forest.

**Information Value (IV):**

```python
iv = sum((dist_pos - dist_neg) * log(dist_pos / dist_neg))
```

El IV cuantifica **cuánto ayuda cada variable a separar las dos clases**. Para calcularlo, la variable se divide en bins (cuantiles para variables continuas) y se mide la diferencia entre la distribución de positivos y negativos en cada bin.

La escala de referencia es:
- IV < 0.02: poder discriminativo nulo o despreciable.
- 0.02 ≤ IV < 0.10: discriminativo débil.
- 0.10 ≤ IV < 0.30: discriminativo medio.
- IV ≥ 0.30: discriminativo fuerte.

**¿Por qué `PageValues` tiene IV = 4.43?** Este valor extraordinariamente alto (más de 10 veces el umbral de "fuerte") se explica porque PageValues es casi un proxy directo de la intención de compra. Las páginas de alto valor son las que están directamente antes del checkout: quien visita esas páginas tiene alta probabilidad de comprar. La distribución de PageValues es completamente diferente entre compradores y no compradores: los compradores tienen PageValues promedio mucho más alto.

**¿Por qué no se eliminaron las 61 variables candidatas?** Aunque el análisis univariado las señala como débilmente discriminativas, los modelos basados en árboles pueden capturar **interacciones** entre múltiples variables débiles que en conjunto predicen bien. Eliminar variables basándose solo en análisis univariado puede ser subóptimo para modelos no lineales.

### 9.2 PCA (Extracción Lineal de Características)

**¿Qué es PCA?**

PCA (Principal Component Analysis) encuentra las direcciones en el espacio de características que capturan la mayor varianza de los datos. El primer componente principal es la dirección de máxima varianza; el segundo es la dirección de máxima varianza ortogonal al primero; y así sucesivamente.

**Intuición:** Imagina una nube de puntos en 3D que tiene forma de elipsoide alargado. PCA encuentra el eje largo del elipsoide (primer componente: captura más varianza), el eje mediano (segundo componente), y el eje corto (tercer componente). Si el elipsoide es muy plano (los puntos están casi en un plano 2D), los primeros dos componentes capturan casi toda la varianza y el tercero es ruido.

**¿Por qué normalizar antes de PCA?**

PCA maximiza varianza. Si una variable tiene escala 0-10.000 y otra 0-1, la primera domina el cálculo de varianza artificialmente. `StandardScaler` garantiza que todas las variables contribuyen equitativamente a los componentes principales.

**Selección del número de componentes:**

```python
varianza_acum = np.cumsum(pca_full.explained_variance_ratio_)
n_componentes_pca = np.argmax(varianza_acum >= 0.95) + 1  # = 54
```

Con el criterio del 95% de varianza, se necesitan **54 de los 65 componentes**. Esto ya es una señal: la reducción es pequeña (16.9%) porque la información del dataset está distribuida a lo largo de muchas dimensiones, no concentrada en pocas.

**El orden correcto del pipeline: Scaler → PCA → SMOTE → Modelo**

Este orden es crítico. PCA debe aplicarse después del scaler (para que la varianza sea comparable entre variables) pero antes de SMOTE (para que las muestras sintéticas se generen en el espacio reducido). Si SMOTE se aplicara antes de PCA, generaría muestras sintéticas en las 65 dimensiones originales, y luego PCA proyectaría ese espacio aumentado, lo cual es metodológicamente más complejo y propenso a inconsistencias.

**¿Por qué PCA redujo el rendimiento?**

Los resultados fueron claros:
- RF baseline: ROC-AUC = 0.9096 → RF + PCA: ROC-AUC = 0.8155 (caída de ~0.09 puntos)
- XGBoost baseline: ROC-AUC = 0.9167 → XGBoost + PCA: ROC-AUC = 0.8292 (caída de ~0.09 puntos)

Esto ocurrió porque el 5% de varianza descartada (los últimos ~11 componentes) contenía **información discriminativa relevante**. En este dataset, las variables con baja varianza no son necesariamente irrelevantes para predecir `Revenue`. Por ejemplo, una variable binaria de OHE puede tener varianza muy baja (porque el 98% de las sesiones tienen el mismo valor) pero esa pequeña fracción del 2% podría estar altamente correlacionada con compradores.

PCA optimiza para preservar varianza, no para preservar capacidad discriminativa. Esos dos objetivos no siempre se alinean.

### 9.3 UMAP (Extracción No Lineal de Características)

**¿Qué es UMAP?**

UMAP (Uniform Manifold Approximation and Projection) es un algoritmo de reducción de dimensión no lineal que busca preservar la **estructura topológica local** del dataset en el espacio reducido. Se basa en teoría de variedades: asume que los datos de alta dimensión yacen sobre una variedad de dimensión mucho menor, y aproxima esa variedad matemáticamente.

**Diferencia fundamental con PCA:**

PCA es una transformación **lineal**: proyecta los datos sobre los eigenvectores de la matriz de covarianza. Solo captura relaciones lineales. Si la estructura real de los datos es curva o no lineal, PCA la distorsiona al proyectarla.

UMAP es una transformación **no lineal**: puede "doblar y estirar" el espacio de forma no uniforme para preservar la estructura local. Si un cluster de puntos tiene una forma de banana en alta dimensión, UMAP puede preservar esa forma en el espacio reducido.

**Selección de componentes:**

Se evaluaron 2, 5, 10, 15 y 20 componentes con validación cruzada para ambos modelos. El óptimo fue **15 componentes** (reducción del 76.9%), donde el ROC-AUC promedio en CV era más alto.

**¿Por qué UMAP devastó el rendimiento?**

Los resultados fueron dramáticos:
- RF + UMAP: ROC-AUC = 0.5301 (caída de ~0.38 puntos, casi aleatorio)
- XGBoost + UMAP: ROC-AUC = 0.4985 (peor que aleatorio)

Hay varias explicaciones:

1. **UMAP está optimizado para visualización, no para clasificación.** Sus hiperparámetros predeterminados preservan la estructura local para que los clusters sean visibles en 2D, pero esto no garantiza que la información discriminativa supervisada sea preservada.

2. **UMAP no usa la etiqueta de clase.** A diferencia de métodos supervisados de reducción (como LDA), UMAP ignora `Revenue` durante la proyección. Puede compactar regiones del espacio donde compradores y no compradores están mezclados, perdiendo exactamente la información más valiosa.

3. **La estructura del dataset no es un manifold simple.** UMAP funciona mejor cuando los datos tienen una estructura geométrica clara (clusters compactos, filamentos, etc.). En este dataset, la distribución de datos no parece tener esa estructura limpia.

4. **Con 76.9% de compresión, se descarta demasiada información.** Pasar de 65 a 15 dimensiones es muy agresivo para un problema de clasificación supervisada.

La visualización 2D del espacio UMAP mostró superposición significativa entre las dos clases, confirmando que 2 dimensiones no son suficientes para separar compradores de no compradores.

---

## 10. DISCUSIÓN DE RESULTADOS Y ESTADO DEL ARTE

### Comparación con trabajos previos

El dataset Online Shoppers Purchasing Intention ha sido estudiado en varios trabajos académicos que el informe referencia. Esta comparación es valiosa porque permite evaluar si los resultados son razonables y qué decisiones metodológicas marcan la diferencia.

**Sakar et al. (2018):** Reportaron exactitud del 98.7% y F1-score de 0.895 con MLP. El resultado parece muy superior al obtenido en este proyecto (F1 ≈ 0.68 para MLP). Sin embargo, **no aplicaron SMOTE ni usaron Balanced Accuracy**. En un dataset con 84.5% de clase negativa, un modelo que predice bien la clase mayoritaria puede obtener exactitud del 98% con recall bajo en la clase minoritaria. Sus métricas están infladas por el desbalance no corregido.

**Abdullah-All-Tanvir et al. (2023):** Reportaron ROC-AUC de 0.937 con XGBoost usando partición 70/30 y SMOTE. El proyecto obtuvo 0.9175. La diferencia de ~0.02 puntos se explica principalmente por la **estrategia de validación más conservadora**: el proyecto usa K-Fold estratificado con 10 folds, que da estimaciones más estables pero generalmente ligeramente más bajas que una sola partición 70/30.

**Alamsyah y Wahyuni (2024):** Reportaron ROC-AUC de 0.94 con Random Forest. El proyecto obtuvo 0.9103. La diferencia parcialmente se explica por la incorporación de SMOTE: SMOTE mejora el recall de la clase minoritaria pero puede reducir ligeramente el ROC-AUC global porque al aumentar la clase minoritaria sintéticamente, el clasificador puede confundir algunos no compradores con compradores.

**Adhikari (2023):** Reportó XGBoost con exactitud del 93.54% y Random Forest con AUC de 0.98. El AUC de 0.98 para RF es notablemente más alto que el obtenido en este proyecto (0.9103). Esto probablemente se debe a diferencias en el preprocessing, la malla de hiperparámetros, o cómo calcularon la métrica.

### La importancia de la estrategia de validación

Esta es una de las lecciones más importantes del proyecto: **los resultados comparados entre trabajos solo son válidos si se usan las mismas estrategias de validación**.

Una partición simple 70/30 puede dar ROC-AUC de 0.95 por azar favorable en esa partición específica. K-Fold con 10 iteraciones da una estimación más conservadora pero más confiable porque promedia sobre 10 particiones diferentes.

Los resultados del proyecto (con K-Fold estratificado) son **más robustos y comparativamente más confiables**, aunque numéricamente parezcan ligeramente inferiores a algunos trabajos previos.

### El efecto de SMOTE en los resultados

SMOTE tuvo un efecto positivo y medible en el proyecto:
- Mejoró el **recall de la clase positiva** (compradores), que es el objetivo principal del negocio.
- El parámetro óptimo fue `sampling_strategy=0.3` para la mayoría de modelos: llevar la proporción al 30% fue el balance óptimo.
- SMOTE dentro del pipeline evitó data leakage, dando estimaciones de rendimiento más honestas.

---

## 11. CONCLUSIONES FINALES

### ¿Qué se aprendió realmente del proyecto?

Este proyecto es una demostración completa de cómo abordar un problema de clasificación binaria con clases desbalanceadas en un dataset tabular del mundo real. Los aprendizajes principales son:

**1. El problema del desbalance es central, no secundario.**
El 84.5% vs 15.5% de desbalance de clases exige decisiones específicas en métricas (Balanced Accuracy, F1, ROC-AUC), en el sampling (SMOTE) y en la validación (estratificada). Ignorar el desbalance produce modelos que parecen buenos pero son inútiles para el negocio.

**2. Los pipelines son la arquitectura correcta para ML reproducible.**
Incluir scaler, SMOTE y modelo en un Pipeline garantiza que no haya data leakage, que el código sea mantenible, y que los hiperparámetros de preprocesamiento puedan optimizarse junto con los del modelo.

**3. La validación cruzada estratificada da estimaciones más confiables que un split único.**
La variabilidad entre folds da información sobre la estabilidad del modelo que un split único no proporciona.

### Por qué XGBoost fue el mejor modelo

XGBoost obtuvo el mejor balance de todas las métricas evaluadas:
- ROC-AUC Test = 0.9175 (más alto de todos)
- F1 Test = 0.7827 (más alto de todos)
- Balanced Accuracy CV = 0.8345 ± 0.0097 (alta y estable)
- Gap train-validación pequeño (0.8427 vs. 0.8345): buena generalización

La razón técnica es que XGBoost combina lo mejor del aprendizaje secuencial (corrige errores del modelo anterior) con regularización integrada (L1, L2, subsampling) que evita overfitting. Para datos tabulares con variables heterogéneas y relaciones no lineales moderadas, es empíricamente el estado del arte.

### La variable `PageValues`: el hallazgo más importante

Con un Information Value de 4.43 (clasificado como "fuerte" desde 0.30), `PageValues` es por lejos la variable más discriminativa. Esto tiene una interpretación de negocio directa: **las páginas que los usuarios visitan revelan más sobre su intención de compra que cualquier característica demográfica o técnica**.

Un usuario que navega por páginas de alto valor (`PageValues` alto) está casi en el funnel de conversión. Una empresa que quiera implementar este modelo podría usar `PageValues` en tiempo real para disparar intervenciones (chat, descuento, recomendación) exactamente cuando el usuario está más cerca de comprar.

### La reducción de dimensión no mejoró el rendimiento

Este es un hallazgo contraintuitivo pero importante: más variables no siempre es peor. La información discriminativa estaba distribuida a lo largo de muchas dimensiones del dataset. Las técnicas de reducción (especialmente UMAP) eliminaron demasiada información relevante al comprimir agresivamente. PCA fue menos dañino pero tampoco mejoró el baseline.

La conclusión práctica: para este dataset y estos modelos, **usar las 65 variables originales es la mejor opción** si los recursos computacionales lo permiten.

### Implicaciones prácticas del trabajo

Si este modelo se desplegara en producción, el flujo sería:
1. El usuario llega al sitio → se registran sus acciones de navegación en tiempo real.
2. Cada ciertos segundos, se calcula un vector de features con las variables del dataset.
3. El modelo XGBoost entrenado predice la probabilidad de compra (`predict_proba`).
4. Si la probabilidad supera un umbral (que podría ser optimizado según el costo de intervención), se activa una acción: descuento, recomendación, chat, etc.

El umbral de clasificación no tiene por qué ser 0.5. Dependiendo del costo de un FP (descuento innecesario) vs. el costo de un FN (comprador perdido), el negocio puede ajustar el umbral para maximizar el valor económico de las intervenciones.

### Posibles mejoras futuras

1. **Feature engineering:** Crear variables derivadas como la tasa de páginas de producto sobre el total de páginas, o el tiempo promedio por página de producto.

2. **Modelos más recientes:** LightGBM, CatBoost, o TabNet (red neuronal diseñada específicamente para datos tabulares) podrían superar a XGBoost.

3. **Reducción supervisada:** Linear Discriminant Analysis (LDA) o t-SNE supervisado preservarían mejor la información discriminativa que PCA o UMAP.

4. **Calibración de probabilidades:** Isotonic Regression o Platt Scaling calibrarían las probabilidades de XGBoost para que sean más precisas como estimaciones reales de probabilidad de compra.

5. **Optimización del umbral de decisión:** En lugar de 0.5, optimizar el umbral sobre la curva ROC para maximizar una función de ganancia que refleje el valor económico de cada tipo de predicción.

6. **Datos adicionales:** Incluir historial de sesiones anteriores del usuario (no solo la sesión actual) transformaría el problema en aprendizaje secuencial y probablemente mejoraría el rendimiento significativamente.

---

*Este documento cubre el proyecto completo de Machine Learning para predicción de intención de compra, siguiendo el orden exacto del notebook y conectando cada decisión técnica con su justificación metodológica y su impacto en los resultados finales.*
