# 🔫 Predicción de Threat Level en Tiroteos Fatales por Policía

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-blue.svg)](https://xgboost.readthedocs.io/)

Proyecto de análisis predictivo que utiliza modelos de Machine Learning para predecir el nivel de amenaza (`threat_level`) en casos de tiroteos fatales por policía en Estados Unidos. Este proyecto implementa modelos avanzados de clasificación multiclase y proporciona un análisis exhaustivo de los factores que influyen en la determinación del nivel de amenaza.

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Resultados](#-resultados)
- [Metodología](#-metodología)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Dataset](#-dataset)
- [Referencias](#-referencias)
- [Licencia](#-licencia)

## 🎯 Descripción

Este proyecto analiza un dataset de **5,416 casos** de tiroteos fatales por policía ocurridos en Estados Unidos entre enero de 2015 y junio de 2020. El objetivo principal es desarrollar modelos de Machine Learning capaces de predecir el nivel de amenaza (`threat_level`) clasificado en tres categorías:

- **attack**: Amenaza de ataque
- **other**: Otras circunstancias
- **undetermined**: Indeterminado

El proyecto incluye un análisis exploratorio completo (EDA), preprocesamiento de datos, feature engineering, optimización de hiperparámetros y evaluación comparativa de múltiples modelos de Machine Learning.

## ✨ Características

- ✅ **Análisis Exploratorio de Datos (EDA)** completo con visualizaciones
- ✅ **Preprocesamiento avanzado** con manejo de valores faltantes
- ✅ **Feature Engineering** incluyendo características temporales
- ✅ **Optimización de hiperparámetros** mediante GridSearchCV
- ✅ **Comparación de modelos**: Random Forest vs XGBoost
- ✅ **Evaluación exhaustiva** con múltiples métricas de rendimiento
- ✅ **Análisis de importancia** de características
- ✅ **Informe detallado** de resultados y conclusiones
- ✅ **Documentación completa** en español

## 📁 Estructura del Proyecto

```
Tiroteo_USA/
│
├── README.md                           # Este archivo
├── threat_level_prediction.ipynb       # Notebook principal con todo el análisis
├── Informe_Resultados_Modelos_ML.md    # Informe detallado de resultados
├── fatal-police-shootings-data.csv     # Dataset original
├── referencia_dataset_kaggle.txt       # Referencia del dataset
└── archive.zip                         # Archivo comprimido de respaldo
```

### Descripción de Archivos

- **`threat_level_prediction.ipynb`**: Notebook Jupyter que contiene todo el pipeline de Machine Learning:
  - Carga y exploración de datos
  - Preprocesamiento y limpieza
  - Feature engineering
  - Entrenamiento de modelos (Random Forest y XGBoost)
  - Evaluación y comparación
  - Visualizaciones y análisis de importancia

- **`Informe_Resultados_Modelos_ML.md`**: Informe técnico detallado con:
  - Análisis de resultados por modelo
  - Comparación de métricas
  - Interpretación de matrices de confusión
  - Recomendaciones y conclusiones

- **`fatal-police-shootings-data.csv`**: Dataset principal con 5,416 registros y 14 características

## 🔧 Requisitos

### Requisitos del Sistema
- Python 3.7 o superior
- Jupyter Notebook o JupyterLab

### Librerías Python

Las siguientes librerías son necesarias para ejecutar el proyecto:

```
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
```

## 📦 Instalación

1. **Clonar el repositorio** (o descargar los archivos)

```bash
git clone https://github.com/tu-usuario/Tiroteo_USA.git
cd Tiroteo_USA
```

2. **Crear un entorno virtual** (recomendado)

```bash
python -m venv venv

# En Windows
venv\Scripts\activate

# En Linux/Mac
source venv/bin/activate
```

3. **Instalar las dependencias**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

O usar el archivo `requirements.txt` si está disponible:

```bash
pip install -r requirements.txt
```

4. **Abrir Jupyter Notebook**

```bash
jupyter notebook threat_level_prediction.ipynb
```

## 🚀 Uso

### Ejecución Básica

1. Asegúrate de que el archivo `fatal-police-shootings-data.csv` esté en el mismo directorio que el notebook
2. Abre el notebook `threat_level_prediction.ipynb` en Jupyter
3. Ejecuta todas las celdas secuencialmente (Cell → Run All)

### Ejecución por Secciones

El notebook está organizado en secciones que puedes ejecutar de forma independiente:

1. **Importación de librerías y carga de datos**
2. **Análisis Exploratorio de Datos (EDA)**
3. **Preprocesamiento de datos**
4. **Feature Engineering**
5. **División de datos (Train/Test)**
6. **Entrenamiento de modelos**
   - Random Forest Classifier
   - XGBoost Classifier
7. **Evaluación y comparación de modelos**
8. **Análisis de importancia de características**

### Tiempo Estimado de Ejecución

- **Ejecución completa**: ~10-15 minutos (dependiendo del hardware)
- **GridSearchCV**: ~5-8 minutos por modelo (puede variar significativamente)

## 📊 Resultados

### Resumen de Rendimiento

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
|--------|----------|-------------------|----------------|------------------|
| **Random Forest** | 68.36% | 59.40% | 57.07% | **57.97%** |
| **XGBoost** | 68.54% | 60.24% | 56.42% | 57.86% |

### Resultados por Clase (Random Forest)

| Clase | Precision | Recall | F1-Score | Muestras |
|-------|-----------|--------|----------|----------|
| **attack** | 0.79 | 0.75 | 0.77 | 699 |
| **other** | 0.52 | 0.58 | 0.55 | 337 |
| **undetermined** | 0.47 | 0.38 | 0.42 | 48 |

### Conclusiones Principales

- ✅ **Random Forest** obtiene el mejor F1-Score macro (57.97%), siendo el modelo recomendado
- ✅ La clase **attack** es la mejor predicha (F1-Score: 0.77) debido a su mayor representación
- ⚠️ La clase **undetermined** presenta mayores desafíos (F1-Score: 0.42) por su escasez en el dataset
- 📈 Ambos modelos muestran rendimiento similar, validando la robustez del análisis

Para más detalles, consulta el [Informe de Resultados](Informe_Resultados_Modelos_ML.md).

## 🔬 Metodología

### Pipeline de Machine Learning

1. **Análisis Exploratorio (EDA)**
   - Estadísticas descriptivas
   - Análisis de valores faltantes
   - Distribuciones y correlaciones
   - Visualizaciones interactivas

2. **Preprocesamiento**
   - Manejo de valores faltantes (imputación y categorías "Unknown")
   - Codificación de variables categóricas (One-Hot Encoding)
   - Normalización de variables numéricas (StandardScaler)
   - Conversión de variables booleanas

3. **Feature Engineering**
   - Extracción de características temporales (año, mes, día de la semana)
   - Agrupación de categorías raras en variables categóricas
   - Creación de features derivadas

4. **Modelado**
   - División estratificada de datos (80/20)
   - Optimización de hiperparámetros con GridSearchCV (5-fold CV)
   - Entrenamiento de modelos optimizados
   - Evaluación con múltiples métricas

5. **Evaluación**
   - Matrices de confusión
   - Métricas por clase y promedio
   - Análisis de importancia de características
   - Comparación de modelos

### Hiperparámetros Optimizados

#### Random Forest
- `n_estimators`: 200
- `max_depth`: 20
- `min_samples_split`: 5
- `min_samples_leaf`: 1
- `class_weight`: 'balanced'

#### XGBoost
- `n_estimators`: 100
- `max_depth`: 3
- `learning_rate`: 0.01
- `subsample`: 1.0

## 🛠️ Tecnologías Utilizadas

### Librerías Principales

- **pandas**: Manipulación y análisis de datos
- **numpy**: Operaciones numéricas
- **matplotlib**: Visualizaciones básicas
- **seaborn**: Visualizaciones estadísticas avanzadas
- **scikit-learn**: Preprocesamiento, modelado y evaluación
- **xgboost**: Modelo avanzado de gradient boosting

### Herramientas

- **Jupyter Notebook**: Entorno de desarrollo interactivo
- **Git**: Control de versiones

## 📚 Dataset

### Información General

- **Nombre**: Fatal Police Shootings Dataset
- **Fuente**: Washington Post
- **Plataforma**: Kaggle
- **URL**: https://www.kaggle.com/datasets/washingtonpost/police-shootings
- **Registros**: 5,416 casos
- **Período**: Enero 2015 - Junio 2020
- **Características**: 14 variables (demográficas, contextuales y temporales)

### Variables Principales

- `threat_level`: Variable objetivo (attack, other, undetermined)
- `armed`: Tipo de arma
- `age`: Edad
- `gender`: Género
- `race`: Raza
- `signs_of_mental_illness`: Signos de enfermedad mental
- `flee`: Comportamiento de huida
- `body_camera`: Presencia de cámara corporal
- `date`: Fecha del incidente
- Y más...

### Distribución de Clases

- **attack**: 3,491 muestras (64.5%)
- **other**: 1,686 muestras (31.1%)
- **undetermined**: 239 muestras (4.4%)

*Nota: El dataset presenta un desbalance de clases que es abordado mediante técnicas de balanceo en el modelo.*

## 📖 Referencias

### Dataset

- **Washington Post - Fatal Police Shootings**
  - Plataforma: Kaggle
  - URL: https://www.kaggle.com/datasets/washingtonpost/police-shootings

### Documentación de Librerías

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [pandas Documentation](https://pandas.pydata.org/docs/)

## 📝 Notas Adicionales

### Limitaciones del Proyecto

- El dataset presenta desbalance de clases, especialmente en la categoría "undetermined"
- La naturaleza subjetiva de algunas clasificaciones puede afectar el rendimiento
- Los modelos capturan patrones estadísticos pero no pueden explicar causalidades

### Posibles Mejoras Futuras

- Implementación de técnicas avanzadas de balanceo (SMOTE)
- Prueba de modelos de Deep Learning
- Feature engineering adicional con interacciones
- Análisis de importancia de características más detallado
- Despliegue del modelo como API

## 👤 Autor

Proyecto desarrollado como parte de un análisis de Machine Learning para predicción de niveles de amenaza en incidentes policiales.

## 📄 Licencia

Este proyecto es de código abierto y está disponible para fines educativos y de investigación.

---

**⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub**

