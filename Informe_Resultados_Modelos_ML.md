# Informe Detallado: Análisis de Modelos de Machine Learning para Predicción de Threat Level

## 1. Resumen Ejecutivo

Este informe presenta un análisis exhaustivo de dos modelos de Machine Learning aplicados al dataset de tiroteos fatales por policía en Estados Unidos. El objetivo es predecir el nivel de amenaza (`threat_level`) que incluye tres clases: **attack**, **other** y **undetermined**.

### Dataset Utilizado
- **Total de registros**: 5,416 casos de tiroteos fatales
- **Período**: Enero 2015 - Junio 2020
- **Variable objetivo**: `threat_level` (clasificación multiclase)
- **División de datos**: 80% entrenamiento / 20% prueba (con estratificación)
- **Fuente del Dataset**: [Washington Post - Fatal Police Shootings](https://www.kaggle.com/datasets/washingtonpost/police-shootings)

---

## 2. Modelos Evaluados

### 2.1 Random Forest Classifier

#### Configuración del Modelo Optimizado
- **Número de árboles (n_estimators)**: 200
- **Profundidad máxima (max_depth)**: 20
- **Mínimo de muestras para dividir (min_samples_split)**: 5
- **Mínimo de muestras en hoja (min_samples_leaf)**: 1
- **Balanceo de clases**: Activado (`class_weight='balanced'`)
- **Score de validación cruzada (CV)**: 0.6014

#### Resultados en el Conjunto de Prueba

**Métricas Generales:**
- **Accuracy (Precisión Global)**: 68.36%
- **Precision (macro)**: 59.40%
- **Recall (macro)**: 57.07%
- **F1-Score (macro)**: 57.97%

**Análisis por Clase:**

| Clase | Precision | Recall | F1-Score | Muestras |
|-------|-----------|--------|----------|----------|
| **attack** | 0.79 | 0.75 | 0.77 | 699 |
| **other** | 0.52 | 0.58 | 0.55 | 337 |
| **undetermined** | 0.47 | 0.38 | 0.42 | 48 |

**Interpretación de Resultados:**
- La clase **attack** es la mejor predicha, con un F1-score de 0.77. Esto es esperado dado que es la clase mayoritaria (64.5% del dataset).
- La clase **other** muestra un rendimiento moderado (F1-score: 0.55), con mejor recall que precision, indicando que el modelo tiende a sobre-predecir esta clase.
- La clase **undetermined** tiene el peor rendimiento (F1-score: 0.42), principalmente debido a su escasez (solo 48 muestras en el conjunto de prueba, 4.4% del total).

### 2.2 XGBoost Classifier

#### Configuración del Modelo Optimizado
- **Número de estimadores (n_estimators)**: 100
- **Profundidad máxima (max_depth)**: 3
- **Tasa de aprendizaje (learning_rate)**: 0.01
- **Submuestreo (subsample)**: 1.0 (100% de las muestras)
- **Score de validación cruzada (CV)**: 0.6041

#### Resultados en el Conjunto de Prueba

**Métricas Generales:**
- **Accuracy (Precisión Global)**: 68.54%
- **Precision (macro)**: 60.24%
- **Recall (macro)**: 56.42%
- **F1-Score (macro)**: 57.86%

**Análisis por Clase:**

| Clase | Precision | Recall | F1-Score | Muestras |
|-------|-----------|--------|----------|----------|
| **attack** | 0.78 | 0.76 | 0.77 | 699 |
| **other** | 0.52 | 0.58 | 0.55 | 337 |
| **undetermined** | 0.50 | 0.35 | 0.41 | 48 |

**Interpretación de Resultados:**
- XGBoost muestra un rendimiento muy similar a Random Forest para la clase **attack** (F1-score: 0.77).
- Para la clase **other**, el rendimiento es idéntico a Random Forest (F1-score: 0.55).
- La clase **undetermined** tiene un rendimiento ligeramente inferior (F1-score: 0.41 vs 0.42 de Random Forest).

---

## 3. Comparación de Modelos

### Tabla Comparativa

| Métrica | Random Forest | XGBoost | Diferencia |
|---------|---------------|---------|------------|
| **Accuracy** | 68.36% | **68.54%** | +0.18% |
| **Precision (macro)** | 59.40% | **60.24%** | +0.84% |
| **Recall (macro)** | **57.07%** | 56.42% | -0.65% |
| **F1-Score (macro)** | **57.97%** | 57.86% | -0.11% |

### Análisis Comparativo

1. **Accuracy**: XGBoost obtiene una ligera ventaja (0.18 puntos porcentuales), pero la diferencia es prácticamente despreciable desde el punto de vista práctico.

2. **Precision (macro)**: XGBoost supera a Random Forest en 0.84 puntos porcentuales, indicando que tiene un mejor balance en la precisión promedio entre todas las clases.

3. **Recall (macro)**: Random Forest obtiene un recall ligeramente superior (0.65 puntos porcentuales), lo que indica una mejor capacidad para identificar correctamente las clases verdaderas.

4. **F1-Score (macro)**: Random Forest tiene un F1-Score macro ligeramente superior (0.11 puntos porcentuales), que es la métrica más equilibrada para evaluar el rendimiento en problemas multiclase desbalanceados.

### Conclusión de la Comparación

**Los dos modelos tienen un rendimiento prácticamente idéntico**, con diferencias menores a 1 punto porcentual en todas las métricas. Esto sugiere que:

- Ambos modelos capturan patrones similares en los datos
- El dataset puede tener limitaciones inherentes que impiden mejoras significativas
- La elección entre uno u otro podría depender más de aspectos prácticos (velocidad de entrenamiento, interpretabilidad) que de rendimiento puro

---

## 4. Análisis de Matrices de Confusión

### Random Forest

La matriz de confusión muestra que:
- **attack**: El modelo predice correctamente la mayoría de los casos de esta clase (525 de 699 correctos), con algunas confusiones principalmente hacia "other".
- **other**: Tiene más dificultades (195 de 337 correctos), confundiéndose frecuentemente con "attack".
- **undetermined**: Presenta el mayor desafío (18 de 48 correctos), siendo frecuentemente confundida con "other" o "attack".

### XGBoost

La matriz de confusión es similar a Random Forest:
- Patrones de confusión prácticamente idénticos entre clases
- La dificultad principal sigue siendo la clasificación de "undetermined"
- La clase "attack" mantiene su buen rendimiento

---

## 5. Limitaciones y Desafíos

### 5.1 Desbalance de Clases

El dataset presenta un **desbalance significativo**:
- **attack**: ~64.5% de las muestras
- **other**: ~31.1% de las muestras  
- **undetermined**: ~4.4% de las muestras

Este desbalance afecta especialmente el rendimiento en la clase "undetermined", que tiene muy pocas muestras para entrenar el modelo adecuadamente.

### 5.2 Complejidad del Problema

- La clasificación de "threat_level" puede depender de factores contextuales no capturados en el dataset
- La clase "undetermined" por su naturaleza ambigua es inherentemente difícil de predecir
- Las características disponibles pueden no ser suficientes para distinguir perfectamente entre todas las clases

### 5.3 Métricas de Rendimiento

- Con un F1-Score macro del ~58%, los modelos tienen margen de mejora
- El accuracy del ~68% es aceptable pero no óptimo
- La diferencia entre precision y recall en algunas clases indica que los modelos tienen sesgos hacia ciertas predicciones

---

## 6. Importancia de Características

Aunque no se muestran los detalles específicos en este resumen, el análisis de importancia de características (disponible en el notebook) revela que las variables más relevantes para la predicción incluyen:

1. Variables relacionadas con el tipo de arma (`armed_grouped`)
2. Características demográficas (`age`, `race`, `gender`)
3. Variables contextuales (`flee`, `signs_of_mental_illness`, `body_camera`)
4. Variables temporales (`year`, `month`, `day_of_week`)

---

## 7. Recomendaciones y Conclusiones

### 7.1 Selección del Modelo

**Recomendación: Random Forest**

Aunque ambos modelos tienen rendimientos muy similares, Random Forest es recomendado por:

1. **Ligera ventaja en F1-Score macro**: La métrica más importante para problemas desbalanceados
2. **Mejor recall**: Importante para no perder casos verdaderos
3. **Interpretabilidad**: Random Forest proporciona una visualización más clara de la importancia de características
4. **Estabilidad**: Menos hiperparámetros que ajustar

### 7.2 Mejoras Potenciales

1. **Técnicas de balanceo de clases**:
   - Oversampling de la clase "undetermined" (SMOTE)
   - Undersampling de las clases mayoritarias
   - Pesos personalizados por clase

2. **Ingeniería de características**:
   - Creación de características de interacción
   - Transformaciones de variables categóricas
   - Agrupación de estados/ciudades por similitud

3. **Modelos adicionales**:
   - Prueba de modelos de ensemble más complejos
   - Modelos de deep learning para capturar patrones no lineales complejos

4. **Recolección de datos**:
   - Más muestras de la clase "undetermined" para mejorar el entrenamiento
   - Variables adicionales que capturen mejor el contexto del incidente

### 7.3 Aplicación Práctica

Los modelos actuales tienen un **rendimiento aceptable para aplicaciones prácticas**, especialmente considerando:

- La naturaleza compleja y a menudo subjetiva de la clasificación de "threat_level"
- El desbalance inherente del dataset
- El rendimiento consistente entre modelos diferentes

Un accuracy del ~68% con F1-Score macro de ~58% es **razonable para un problema de clasificación multiclase desbalanceado** en el dominio de análisis de incidentes policiales.

---

## 8. Métricas de Validación Cruzada

- **Random Forest CV Score**: 0.6014
- **XGBoost CV Score**: 0.6041

Los scores de validación cruzada (5-fold) son consistentes con los resultados del conjunto de prueba, indicando que los modelos **no presentan overfitting significativo** y que los resultados son generalizables.

---

## Anexo: Distribución de Clases en el Dataset

- **attack**: 3,491 muestras (64.5%)
- **other**: 1,686 muestras (31.1%)
- **undetermined**: 239 muestras (4.4%)

**Total**: 5,416 muestras

---

## 9. Referencias

### Dataset

**Fatal Police Shootings Dataset**
- **Fuente**: Washington Post
- **Plataforma**: Kaggle
- **URL**: https://www.kaggle.com/datasets/washingtonpost/police-shootings
- **Descripción**: Dataset que contiene registros de personas que fueron baleadas y asesinadas por oficiales de policía en los Estados Unidos desde 2015.

---

*Este informe fue generado basado en los resultados del notebook `threat_level_prediction.ipynb` ejecutado con el dataset `fatal-police-shootings-data.csv`.*
