# Preguntas de Sustentación — Proyecto de Clasificación de Intención de Compra

Preguntas simuladas por el profesor y respuestas del equipo durante la preparación de la sustentación del proyecto.

---

## Dataset y Variables

**¿Cómo se obtuvieron las variables que se están procesando?**

Las variables provienen del dataset *Online Shoppers Purchasing Intention* del repositorio UCI Machine Learning, descargado mediante la librería `ucimlrepo`. El dataset contiene 18 variables que describen el comportamiento de sesiones de usuarios en un sitio de e-commerce: páginas visitadas, tiempos de permanencia, tasas de rebote, tipo de visitante, mes, región, entre otras. Ninguna variable fue construida manualmente; todas vienen directamente del dataset original.

**¿La base de datos está balanceada?**

No. El 84.5% de las sesiones no terminaron en compra (clase 0, 10.422 sesiones) y solo el 15.5% sí terminaron en compra (clase 1, 1.908 sesiones). Por eso se aplicó SMOTE para generar muestras sintéticas de la clase minoritaria y reducir ese desbalance durante el entrenamiento.

**¿Toda la evaluación del proyecto se usó con los usuarios que estaban en el dataset?**

Sí. Todo el proyecto se desarrolló con las 12.330 sesiones del dataset, donde cada sesión corresponde a un usuario diferente. No se recolectaron datos externos ni se generaron datos reales adicionales; las únicas muestras adicionales fueron las sintéticas generadas por SMOTE durante el entrenamiento para balancear las clases.

---

## Procesamiento de Datos

**En la normalización, normalizaron toda la base de datos y luego hicieron la partición. ¿Es eso correcto?**

No, fue un error en ese paso. Lo correcto era primero hacer la partición train/test y luego ajustar el `StandardScaler` únicamente con los datos de entrenamiento. Al normalizar todo el dataset antes de partir, los datos de prueba influyeron en los parámetros de escalado, lo que técnicamente constituye **data leakage**.

> **Nota:** El impacto práctico depende de la escala de las variables y de qué tan similares son las distribuciones entre train y test. Con datasets grandes y distribuciones similares el efecto suele ser pequeño, pero técnicamente invalida la separación entre entrenamiento y prueba. El pipeline de `imblearn` sí garantiza que `StandardScaler` y SMOTE se ajusten solo con los datos de entrenamiento de cada fold en la validación cruzada; el error estuvo en el paso de normalización inicial, previo al pipeline.

---

## Partición del Conjunto de Datos

**¿Qué partición hicieron: entrenamiento/validación/test, o solo entrenamiento/test?**

Solo hicimos partición en dos conjuntos: entrenamiento (80%) y prueba (20%). No hubo un conjunto de validación separado de forma explícita. La validación se realizó mediante Stratified K-Fold de 10 folds sobre el conjunto de entrenamiento, donde en cada iteración un fold actuaba como validación y los 9 restantes como entrenamiento.

**¿Por qué la validación cruzada se aplicó únicamente sobre el conjunto de entrenamiento?**

Porque el conjunto de prueba debe permanecer completamente aislado durante todo el proceso de entrenamiento y selección de hiperparámetros. Si se usara en la validación cruzada, el modelo tendría acceso indirecto a esos datos durante el ajuste, lo que generaría estimaciones artificialmente optimistas del desempeño real (data leakage).

**La partición en tres conjuntos es necesaria para conocer el desempeño real. ¿Ustedes la hicieron?**

No. Solo dividimos en entrenamiento (80%) y prueba (20%). La validación cruzada de 10 folds sobre el entrenamiento cumplió el rol del conjunto de validación para la selección de hiperparámetros, pero no tuvimos un conjunto de validación independiente separado desde el inicio.

---

## Hiperparámetros y Mallas de Búsqueda

**¿Cómo determinaron el número óptimo de árboles en Random Forest y XGBoost?**

El número de árboles (`n_estimators=200`) se fijó como valor constante en la malla de hiperparámetros, no se buscó el óptimo mediante GridSearchCV. Se eligió 200 como un valor estándar que generalmente ofrece buen desempeño sin costo computacional excesivo.

**¿Qué mallas de valores usaron para los modelos y para SMOTE?**

Para **SMOTE**:
- `sampling_strategy`: 0.22, 0.30, 0.40
- `k_neighbors`: 3, 5, 7 (según el modelo)

Para los **modelos principales**:

| Modelo | Parámetros evaluados |
|--------|----------------------|
| Random Forest | `max_depth`: (10, 20); `min_samples_leaf`: (1, 4) |
| XGBoost | `max_depth`: (3, 6); `learning_rate`: (0.05, 0.10) |
| MLP | `hidden_layer_sizes`: ((50,), (100,)); `alpha`: (0.0001, 0.001); `learning_rate_init`: (0.001, 0.01) |
| SVM | `kernel`: (linear, rbf); `C`: (0.01, 0.1, 1); `gamma`: (scale, auto, 0.01, 0.1) |
| KNN | `n_neighbors`: (3, 5, 11); `weights`: (uniform, distance); `metric`: (euclidean, manhattan) |
| GMM | `n_components`: (1, 2, 3, 4); `covariance_type`: (full, diag, tied, spherical) |

Todo evaluado mediante **GridSearchCV con Stratified K-Fold de 10 folds**.

**¿Cómo se escogieron los porcentajes de `sampling_strategy`?**

Los valores 0.22, 0.30 y 0.40 se definieron como una malla de hiperparámetros y se seleccionó el óptimo mediante GridSearchCV usando Balanced Accuracy como métrica. El valor **0.30** fue el más frecuentemente seleccionado, lo que indica que aumentar la clase minoritaria al 30% del tamaño de la mayoritaria fue el mejor balance para la mayoría de los modelos.

---

## Reducción de Dimensionalidad

**¿Cuál fue el resultado de aplicar PCA o técnicas no lineales?**

Se aplicaron dos técnicas sobre los dos mejores modelos (RF y XGBoost):

- **PCA (lineal):** conservando el 95% de la varianza acumulada, se redujeron las 65 variables a 54 componentes (reducción del 16.9%). El desempeño bajó aproximadamente 0.09 puntos de ROC-AUC en ambos modelos.
- **UMAP (no lineal):** se evaluaron 2, 5, 10, 15 y 20 componentes, seleccionando 15 como óptimo maximizando el ROC-AUC promedio en validación cruzada (reducción del 76.9%). El desempeño se degradó severamente, cayendo por debajo de 0.55 en ambos modelos.

La conclusión fue que **ninguna técnica mejoró el desempeño respecto al baseline original**.

**¿Cómo se llegó a conservar 54 componentes para el 95% de varianza con 65 variables originales?**

Con PCA se calcularon todos los componentes principales ordenados de mayor a menor varianza explicada. Luego se fue acumulando la varianza de cada componente hasta alcanzar el 95% del total. El punto de corte quedó en 54 componentes, lo que significa que esos 54 componentes explican el 95% de la varianza de las 65 variables originales, descartando los 11 componentes restantes que aportaban solo el 5%. Eso representa una reducción del 16.9% en dimensionalidad (11/65).

---

## Paradigma de Múltiples Instancias

**¿Implementaron el esquema de múltiples instancias?**

No. Ese tema no fue abordado en el proyecto. Trabajamos con un esquema de clasificación binaria estándar donde cada sesión de usuario corresponde a una única instancia con su respectiva etiqueta (compra o no compra).

**¿Qué significa el paradigma de múltiples instancias?**

El paradigma de *Multiple Instance Learning* (MIL) consiste en que, en lugar de etiquetar instancias individuales, se etiquetan **bolsas (bags)** de instancias. Una bolsa es positiva si al menos una instancia dentro de ella lo es. En nuestro proyecto no lo utilizamos; trabajamos con aprendizaje supervisado estándar donde cada sesión es una instancia individual con su propia etiqueta directa en la variable `Revenue`.
