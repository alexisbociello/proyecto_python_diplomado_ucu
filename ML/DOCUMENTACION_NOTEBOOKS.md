# Guía de las notebooks de Machine Learning

## Propósito

Este documento resume las tres notebooks del trabajo final del Grupo F. Está pensado como punto de entrada para quienes necesiten entender qué problema resuelve cada análisis, cómo se preparan los datos, qué modelos se comparan, qué resultados se obtienen y qué precauciones deben tenerse al interpretar los hallazgos.

Las notebooks cubren los tres tipos principales de aprendizaje trabajados en el proyecto:

| Notebook | Tipo de aprendizaje | Objetivo | Modelo seleccionado |
|---|---|---|---|
| `01_regresion_hormigon.ipynb` | Regresión supervisada | Predecir resistencia del hormigón en MPa | Random Forest Regressor |
| `02_clasificacion_bank_marketing.ipynb` | Clasificación supervisada | Predecir la contratación de un depósito a plazo | Balanced Random Forest |
| `03_clustering_wholesale.ipynb` | Aprendizaje no supervisado | Segmentar clientes según su patrón de gasto | K-Means con K = 2 |

## Requisitos y ejecución

Las notebooks esperan que la carpeta `datasets/` se encuentre en el mismo directorio desde el cual se inicia Jupyter. Los archivos utilizados son:

- `datasets/Concrete_Data.xls`
- `datasets/bank-full.csv`
- `datasets/Wholesale_customers_data.csv`

Las principales dependencias son Python, Jupyter, NumPy, pandas, Matplotlib, seaborn, scikit-learn, SciPy, imbalanced-learn y un lector de archivos Excel compatible con `.xls`.

Se recomienda ejecutar las notebooks en orden numérico y usar **Restart Kernel and Run All**. Todas emplean `random_state = 42` en las operaciones relevantes para facilitar la reproducibilidad. Los resultados documentados corresponden a la versión actual de las notebooks y de los datasets.

---

## 1. Regresión: resistencia a la compresión del hormigón

**Archivo:** `01_regresion_hormigon.ipynb`

### Objetivo y datos

La notebook estima la resistencia a la compresión de una mezcla de hormigón, expresada en MPa. El dataset contiene 1.030 mezclas y ocho predictores cuantitativos:

- cemento (`cement`);
- escoria (`slag`);
- ceniza volante (`fly_ash`);
- agua (`water`);
- superplastificante (`superplasticizer`);
- agregado grueso (`coarse_aggregate`);
- agregado fino (`fine_aggregate`);
- edad del hormigón en días (`age_days`).

La variable objetivo es `strength_mpa`. La pregunta práctica es cuánto resistirá una mezcla dadas su composición y su edad.

### Contenido del análisis

1. Carga y normalización de nombres de columnas.
2. Estadística descriptiva, faltantes, duplicados, asimetría y posibles valores extremos mediante IQR.
3. Histogramas, relaciones entre cada predictor y la resistencia, matriz de correlaciones y análisis por grupos de edad.
4. Separación aleatoria en entrenamiento (80 %) y prueba (20 %).
5. Entrenamiento y comparación mediante MAE, RMSE y R².
6. Validación cruzada de 7 folds y validación repetida de 7 folds por 3 repeticiones (21 evaluaciones).
7. Comparación estadística con Friedman y Wilcoxon.
8. Diagnóstico de residuos e importancia de variables, incluida importancia por permutación.

### Modelos utilizados

| Modelo | Configuración principal | Motivo de inclusión |
|---|---|---|
| Regresión lineal | Estandarización dentro de un pipeline | Línea base simple e interpretable |
| Random Forest Regressor | 350 árboles; `min_samples_leaf=2` | Captura interacciones y relaciones no lineales |
| Gradient Boosting Regressor | 300 árboles; `learning_rate=0.04`; profundidad 2 | Corrige errores de forma secuencial y modela patrones sutiles |

La estandarización de la regresión lineal se realiza dentro de un pipeline para evitar fuga de información durante la validación.

### Resultados principales

| Modelo | RMSE promedio (CV repetida) | MAE promedio | R² promedio | Lectura |
|---|---:|---:|---:|---|
| Random Forest | 5,01 MPa | 3,50 MPa | 0,908 | Mejor desempeño promedio y mayor estabilidad |
| Gradient Boosting | 5,57 MPa | — | 0,887 | Alternativa competitiva; gana en una única partición de prueba |
| Regresión lineal | 10,47 MPa | — | 0,601 | Insuficiente para la estructura no lineal del problema |

En el conjunto de prueba, Random Forest obtiene RMSE de 5,67 MPa y R² de 0,875. Aunque Gradient Boosting alcanza el menor RMSE en esa partición aislada, Random Forest se selecciona por su mejor resultado promedio en las evaluaciones repetidas.

### Hallazgos

- La resistencia media es 35,82 MPa y el máximo observado es 82,60 MPa; la distribución tiene una ligera cola hacia valores altos.
- Cemento y edad son los predictores con las relaciones positivas más visibles. El agua presenta una asociación negativa.
- La ganancia de resistencia con la edad es no lineal: la media pasa de 22,31 MPa hasta 7 días a 35,73 MPa entre 8 y 28 días, 47,64 MPa entre 29 y 90 días y 49,55 MPa después de 90 días.
- Edad y cemento concentran la mayor utilidad predictiva del modelo. Esto no implica causalidad ni permite interpretar cada variable de forma aislada.
- Los residuos se concentran alrededor de cero, pero existen colas y errores grandes en algunos casos extremos.

### Conclusión y uso recomendado

Random Forest es el modelo recomendado porque explica aproximadamente el 91 % de la variación en validación cruzada y representa mejor las relaciones no lineales entre ingredientes, edad y resistencia. Puede ayudar a priorizar formulaciones y señalar ensayos que merecen revisión, pero no reemplaza ensayos físicos, validación profesional ni criterios de seguridad de ingeniería.

---

## 2. Clasificación: contratación de un depósito a plazo

**Archivo:** `02_clasificacion_bank_marketing.ipynb`

### Objetivo y datos

La notebook predice si un cliente bancario contratará un depósito a plazo (`y`). Utiliza `bank-full.csv`, con información demográfica, financiera y del historial de contactos de marketing.

La clase positiva es poco frecuente: el 11,70 % contrató y el 88,30 % no lo hizo. Por eso la accuracy no se usa como criterio aislado. El objetivo operativo es encontrar clientes interesados sin generar una cantidad desproporcionada de contactos improductivos.

### Preparación y decisiones de diseño

- Se eliminan `duration` y `contact` antes del modelado. `duration` no está disponible antes de completar la llamada y produciría fuga de información; retirar `contact` evita apoyar la decisión en el canal de contacto.
- La división entrenamiento/prueba es 80/20 y está estratificada para conservar la proporción de clases.
- Las variables numéricas se imputan con la mediana y se estandarizan.
- Las variables categóricas se imputan con la moda y se codifican con one-hot encoding, ignorando categorías nuevas.
- Todo el preprocesamiento se integra en pipelines para evitar fuga de información.
- La validación cruzada es estratificada y utiliza 7 folds.
- SMOTE se aplica exclusivamente dentro de cada fold en las variantes que lo usan.

### Modelos utilizados

| Modelo | Configuración o función |
|---|---|
| Regresión logística | Línea base probabilística; `class_weight='balanced'` |
| Random Forest | 250 árboles; pesos balanceados; `min_samples_leaf=2` |
| Gradient Boosting | 200 árboles; `learning_rate=0.05`; profundidad 3 |
| Extra Trees | 400 árboles; pesos balanceados; `min_samples_leaf=2` |
| Balanced Random Forest | 300 árboles; balanceo de clases dentro de cada árbol |
| Variantes con SMOTE | Regresión logística, Random Forest y Gradient Boosting |

Las métricas son accuracy, precision, recall, F1, ROC-AUC y balanced accuracy. F1 es el criterio principal porque combina la capacidad de encontrar positivos con la proporción de predicciones positivas correctas.

### Resultados principales

Balanced Random Forest es el modelo seleccionado:

- **Validación cruzada:** F1 0,446; balanced accuracy 0,720; recall 0,572; ROC-AUC 0,776.
- **Prueba:** F1 0,458; recall 0,587; precision 0,375; accuracy 0,837; ROC-AUC 0,787.

Comparaciones relevantes en prueba:

| Modelo | Resultado destacado | Interpretación |
|---|---|---|
| Balanced Random Forest | F1 0,458; recall 0,587 | Encuentra más clientes que efectivamente contratan, con más falsos positivos |
| Extra Trees | F1 0,442 | Segunda alternativa adicional más equilibrada |
| Random Forest | F1 0,439; precision 0,513; recall 0,383 | Menos contactos improductivos, pero omite más positivos |
| Gradient Boosting | precision 0,657; recall 0,201 | Muy conservador: alta precisión, baja cobertura de interesados |
| Regresión logística | ROC-AUC de prueba 0,752 en la comparación inicial | Referencia interpretable, inferior a los ensambles |

Random Forest obtiene ROC-AUC 0,787 en prueba y Gradient Boosting 0,784; la cercanía muestra que el umbral y el costo de los errores son más importantes que una diferencia mínima de AUC.

### Hallazgos

- El desbalance es central: predecir siempre “no” daría una accuracy alta pero ningún valor comercial.
- Quienes contrataron presentan un saldo mediano mayor (733 frente a 417) y, en general, menos contactos durante la campaña.
- `poutcome`, que resume el resultado de una campaña anterior, es el diferenciador descriptivo más fuerte.
- Mayo concentra llamadas, pero no la mayor tasa de contratación. Marzo, septiembre, octubre y diciembre muestran tasas proporcionalmente superiores, aunque esto no demuestra que el mes cause el resultado.
- SMOTE mejora Gradient Boosting: el F1 promedio pasa de 0,311 a 0,419 y el recall de 0,205 a 0,370. No supera a Balanced Random Forest y no mejora el F1 de todos los algoritmos.
- Friedman detecta diferencias globales entre modelos (p = 0,000158). McNemar indica que Balanced Random Forest comete un patrón de errores significativamente distinto del de los competidores (p < 0,001), pero la significación estadística no sustituye la evaluación del impacto comercial.

### Conclusión y uso recomendado

Balanced Random Forest ofrece el mejor equilibrio si el negocio valora recuperar más clientes interesados aun a costa de contactos adicionales. Si cada contacto improductivo fuera muy costoso, Random Forest o Gradient Boosting podrían ser preferibles por su mayor precisión.

Antes de usar el modelo en producción deben definirse costos reales de falsos positivos y falsos negativos, ajustar el umbral, calibrar probabilidades, hacer una validación temporal y revisar sesgos. Las asociaciones encontradas no son relaciones causales y no deberían emplearse para decisiones discriminatorias.

---

## 3. Clustering: perfiles de clientes mayoristas

**Archivo:** `03_clustering_wholesale.ipynb`

### Objetivo y datos

La notebook segmenta 440 clientes mayoristas según su gasto anual en seis categorías:

- `Fresh`;
- `Milk`;
- `Grocery`;
- `Frozen`;
- `Detergents_Paper`;
- `Delicassen`.

`Channel` y `Region` se excluyen de la construcción de los clusters para descubrir patrones de compra sin reproducir categorías administrativas preexistentes. Pueden servir posteriormente como contraste externo.

### Contenido y preparación

1. Estadística descriptiva, asimetría, ceros, valores extremos, gasto total y correlaciones.
2. Transformación `log1p` para reducir la influencia de compradores de volumen extremo.
3. Estandarización de las seis variables para hacer comparables las distancias.
4. Selección de K mediante codo, silhouette y Davies-Bouldin.
5. Búsqueda de parámetros de DBSCAN y análisis de sensibilidad.
6. Visualización PCA, usada solamente para representar los grupos; las métricas se calculan en el espacio transformado completo.
7. Comparación mediante silhouette, Davies-Bouldin, Calinski-Harabasz, cobertura y estabilidad.
8. Caracterización por medianas, ANOVA, Kruskal-Wallis y eta cuadrado.
9. Comparación adicional con Gaussian Mixture y clustering aglomerativo Ward.

### Modelos utilizados

| Modelo | Configuración o aporte |
|---|---|
| K-Means | K evaluado entre 2 y 8; solución final K = 2; 30 inicializaciones |
| DBSCAN | Grilla de `eps` y `min_samples`; solución documentada `eps=0.9`, `min_samples=10` |
| Gaussian Mixture | 2 componentes, covarianza completa y asignación probabilística |
| Aglomerativo Ward | 2 clusters y dendrograma jerárquico |

No se usan accuracy, precision ni ROC porque no existe una clase verdadera. Las métricas internas describen separación y compactación, pero no prueban que los segmentos sean comercialmente válidos.

### Comparación de modelos

| Modelo | Cobertura | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|---:|
| K-Means, K = 2 | 100 % | 0,290 | 1,352 | 189,0 |
| Gaussian Mixture | 100 % | 0,266 | 1,391 | 165,4 |
| Aglomerativo Ward | 100 % | 0,258 | 1,600 | 134,6 |
| DBSCAN, solo núcleos asignados | 43,6 % | 0,461 | 0,858 | 236,7 |

Las métricas de DBSCAN parecen mejores porque se calculan después de retirar el 56,4 % marcado como ruido. Por tanto, no son directamente comparables con las de los métodos que asignan a toda la muestra.

### Segmentos seleccionados

| Segmento | Clientes | Participación | Perfil típico | Posibles acciones |
|---|---:|---:|---|---|
| Frescos y congelados | 252 | 57,3 % | Mayor gasto relativo en `Fresh` y `Frozen`; menor en lácteos, almacén y limpieza | Reposición frecuente, cadena de frío, descuentos por volumen y paquetes combinados |
| Almacén, lácteos y limpieza | 188 | 42,7 % | Mayor gasto relativo en `Milk`, `Grocery` y `Detergents_Paper`; menor orientación a frescos y congelados | Promociones cruzadas, contratos de abastecimiento y beneficios por recurrencia |

Los nombres son interpretaciones comerciales y no tipos de negocio observados directamente.

### Hallazgos

- Todas las categorías tienen asimetría positiva y compradores extremos. El gasto total tiene mediana 27.492 y máximo 199.891, lo que justifica la transformación logarítmica.
- `Grocery`, `Milk` y `Detergents_Paper` forman el bloque de correlaciones positivas más fuerte; `Fresh` muestra un comportamiento más independiente.
- K = 2 obtiene el silhouette máximo entre K = 2 y K = 8. El valor 0,290 indica grupos reconocibles, pero con solapamiento y fronteras imperfectas.
- La estabilidad de K-Means es muy alta: ARI promedio 0,996 frente a la solución de referencia en 30 inicializaciones.
- DBSCAN encuentra dos núcleos de 88 y 104 clientes, pero marca 248 clientes como ruido. Es más útil para revisar casos inusuales que para segmentar toda la cartera.
- Gaussian Mixture identifica 25 clientes con incertidumbre de pertenencia superior a 0,25. Su ARI frente a K-Means es 0,683; Ward alcanza 0,639. Ambos recuperan el mismo contraste general, aunque con peores métricas de separación.
- ANOVA, Kruskal-Wallis y eta cuadrado ayudan a describir qué categorías diferencian los clusters; no son validación externa porque se aplican a las mismas variables usadas para construirlos.

### Conclusión y uso recomendado

K-Means con K = 2 es la solución principal por el mejor equilibrio entre cobertura, separación, estabilidad e interpretación. La separación es moderada, por lo que los segmentos deben validarse con el equipo comercial, datos de otro período y resultados reales de campañas o decisiones de inventario. Los clientes cercanos a la frontera, con alta incertidumbre o marcados como ruido requieren revisión individual.

---

## Lectura conjunta del proyecto

Las tres notebooks siguen una lógica común:

1. entienden el problema y la calidad de los datos;
2. preparan los atributos evitando fuga de información;
3. comparan modelos con supuestos distintos;
4. seleccionan con validación y no solamente con una partición o un gráfico;
5. traducen métricas a consecuencias prácticas;
6. separan utilidad predictiva de causalidad;
7. explicitan las limitaciones antes de recomendar un uso.

La calidad de los resultados no se interpreta igual en los tres ejercicios. En regresión interesa minimizar el error continuo; en clasificación importa el equilibrio entre oportunidades detectadas y contactos innecesarios; en clustering se busca una segmentación estable, completa e interpretable sin disponer de etiquetas verdaderas.

## Recomendaciones para continuar el trabajo

- Guardar los modelos y transformadores finales en artefactos versionados si se planea reutilizarlos.
- Registrar versiones de Python y dependencias para asegurar reproducibilidad.
- Incorporar validación temporal o con datos nuevos en los ejercicios supervisados.
- Definir costos de negocio y ajustar el umbral en la clasificación bancaria.
- Validar externamente los segmentos mayoristas y medir su respuesta a acciones concretas.
- Añadir controles de calidad, monitoreo de deriva y criterios claros para reentrenamiento antes de cualquier uso productivo.

