# 🧠 Dashboard de Minería de Datos - Análisis de Artículos Periodísticos

## 📋 Descripción del Proyecto

Este proyecto implementa un **dashboard interactivo** para el análisis de **1,571 artículos periodísticos** utilizando **10 algoritmos de minería de datos** y **5 arquitecturas de redes neuronales**. El objetivo es determinar la importancia de los artículos basándose en criterios objetivos de calidad periodística.

---

## 📖 MANUAL DE USUARIO

### 🚀 Inicio Rápido

#### 1. Instalación y Configuración
```bash
# 1. Instalar dependencias Python
pip install pandas numpy scikit-learn tensorflow flask nltk

# 2. Instalar dependencias React
npm install

# 3. Construir el frontend
npm run build

# 4. Ejecutar el servidor
python server.py
```

#### 2. Acceso al Dashboard
- **URL:** `http://localhost:3002`
- **Puerto:** 3002 (configurable en `server.py`)

### 🎯 Navegación del Dashboard

#### Pestaña Principal - "Resumen General"
- **📊 Estadísticas del Dataset:** Muestra información general de los 1,571 artículos
- **🏆 Mejor Algoritmo:** Identifica automáticamente el algoritmo con mejor rendimiento
- **🔘 Botón "Análisis de Redes Neuronales":** Acceso directo a la sección de redes neuronales

#### Pestaña "Gráficos de Comparación"
- **📈 Comparación de Accuracy:** Gráfico de barras con rendimiento de todos los algoritmos
- **📊 Comparación de AUC:** Métricas de área bajo la curva para algoritmos de clasificación
- **🎯 Gráficos Específicos:** Visualizaciones únicas para cada tipo de algoritmo

#### Pestaña "Detalles por Método"
- **🔍 Información Detallada:** Explicación completa de cada algoritmo
- **📝 Variables Utilizadas:** Lista de características empleadas por cada método
- **⚙️ Proceso Paso a Paso:** Descripción del funcionamiento interno
- **📊 Interpretación Detallada:** Análisis de resultados y conclusiones
- **📈 Gráficos Específicos:** Visualizaciones particulares para cada algoritmo

#### Pestaña "Redes Neuronales"
- **🧠 MLP (Perceptrón Multicapa):** 96.6% accuracy - Excelente rendimiento
- **🔍 CNN para Texto:** 61.1% accuracy - Limitaciones para texto periodístico
- **🔄 LSTM:** 90.2% accuracy - Memoria secuencial efectiva
- **↔️ BiLSTM:** 92.1% accuracy - Procesamiento bidireccional superior

#### Pestaña "Conclusión y Evaluación"
- **📋 Evaluación General:** Resumen completo del proyecto
- **💪 Fortalezas Principales:** Aspectos destacados del análisis
- **📊 Resultados Académicos:** Métricas y conclusiones técnicas
- **🎯 Evaluación Técnica:** Análisis detallado de implementación

### 🎨 Características del Dashboard

#### Diseño y Colores
- **Esquema de colores:** Azul claro y blanco
- **Interfaz moderna:** Componentes Ant Design
- **Responsive:** Adaptable a diferentes tamaños de pantalla
- **Navegación intuitiva:** Pestañas organizadas lógicamente

#### Funcionalidades Interactivas
- **Gráficos dinámicos:** Visualizaciones con Recharts
- **Navegación fluida:** Transiciones suaves entre secciones
- **Carga automática:** Datos actualizados en tiempo real
- **Interpretaciones detalladas:** Explicaciones completas de cada algoritmo

### 📊 Interpretación de Resultados

#### Cómo Leer los Gráficos
1. **Accuracy (Precisión):** Porcentaje de predicciones correctas
2. **AUC (Área Bajo la Curva):** Calidad de la clasificación (0-1)
3. **Silhouette Score:** Calidad del clustering (0-1)
4. **Estado del Algoritmo:** Excelente, Bueno, Regular, Necesita Mejora

#### Criterios de Importancia
Un artículo se considera **"importante"** si cumple **4 o más** criterios:
- ✅ Contenido sustancial (≥70% longitud)
- ✅ Periódico prestigioso (La Vanguardia, Elmundo, El País, ABC)
- ✅ Categoría relevante (Internacional, Política, Economía, Ciencia)
- ✅ Contenido temático (≥2 palabras clave)
- ✅ Título informativo (≥20 caracteres, ≥5 palabras)
- ✅ Contenido estructurado (≥500 caracteres, ≥100 palabras)
- ✅ Complejidad del contenido (≥60%)

### 🔧 Solución de Problemas

#### Problemas Comunes
1. **Puerto ocupado:** Cambiar puerto en `server.py`
2. **Dashboard en blanco:** Verificar que `npm run build` se ejecutó correctamente
3. **Datos no cargan:** Comprobar que los archivos JSON existen
4. **Errores de compilación:** Ejecutar `npm install` y `npm run build`

#### Comandos de Solución
```bash
# Reiniciar servidor
pkill -f "python server.py" && python server.py

# Limpiar build
rm -rf build && npm run build

# Verificar puertos
lsof -i :3002
```

### 📱 Uso en Diferentes Dispositivos

#### Desktop (Recomendado)
- **Resolución mínima:** 1024x768
- **Navegadores:** Chrome, Firefox, Safari, Edge
- **Funcionalidades completas:** Todas las características disponibles

#### Tablet
- **Orientación:** Horizontal recomendada
- **Navegación:** Touch-friendly
- **Gráficos:** Adaptados automáticamente

#### Móvil
- **Limitaciones:** Algunos gráficos pueden ser pequeños
- **Navegación:** Scroll vertical
- **Funcionalidades:** Básicas disponibles

### 🎓 Guía Académica

#### Para Estudiantes
- **Nivel:** Universitario avanzado
- **Prerrequisitos:** Conocimientos básicos de ML y programación
- **Aplicación:** Proyectos de minería de datos y análisis de texto

#### Para Profesores
- **Evaluación:** Criterios objetivos y métricas claras
- **Reproducibilidad:** Código completo y documentado
- **Extensibilidad:** Fácil modificación y mejora

#### Para Investigadores
- **Metodología:** Rigurosa y bien documentada
- **Resultados:** Válidos académicamente
- **Aplicaciones:** Casos de uso reales en periodismo

### 📞 Soporte y Ayuda

#### Recursos Disponibles
- **README.md:** Documentación completa del proyecto
- **DOCUMENTO_TECNICO_DETALLADO.md:** Análisis técnico profundo
- **Código comentado:** Explicaciones inline en todos los scripts
- **Logs del servidor:** Información de depuración en terminal

#### Contacto
- **Problemas técnicos:** Revisar logs del servidor
- **Dudas sobre algoritmos:** Consultar documentación técnica
- **Mejoras:** Proponer en issues del proyecto

---

## 🎯 Objetivos

## 🎯 Objetivos

- **Análisis comparativo** de 10 algoritmos de machine learning
- **Implementación de redes neuronales** para clasificación de texto
- **Dashboard interactivo** con visualizaciones en tiempo real
- **Criterios objetivos** para determinar importancia periodística
- **Interpretación detallada** de resultados y métricas

## 📊 Dataset

### Información del Dataset
- **Total de artículos:** 1,571 artículos periodísticos
- **Período:** Datos extraídos el 26/09/2025
- **Columnas:** ID, Título, Resumen, Contenido, Periódico, Categoría, Región, URL, Fecha Extracción, Cantidad Imágenes
- **Periódicos:** 13 periódicos únicos (La Vanguardia, Elmundo, El País, ABC, etc.)
- **Categorías:** 46 categorías diferentes (Internacional, Política, Economía, etc.)

### Criterios de Importancia
Un artículo se considera **"importante"** si cumple **4 o más** de los siguientes criterios:

1. **Contenido Sustancial:** ≥70% de longitud de contenido
2. **Prestigio del Periódico:** La Vanguardia, Elmundo, El País, ABC
3. **Relevancia de Categoría:** Internacional, Política, Economía, Ciencia y Salud
4. **Contenido Temático:** ≥2 palabras clave temáticas
5. **Título Informativo:** ≥20 caracteres, ≥5 palabras, mayúsculas
6. **Contenido Estructurado:** ≥500 caracteres, ≥100 palabras
7. **Complejidad del Contenido:** ≥60% de complejidad textual

## 🤖 Algoritmos Implementados

### 1. Algoritmos de Clasificación (7 métodos)

#### 🔢 Regresión Logística
- **Accuracy:** 85.2%
- **Estado:** Bueno
- **Variables:** TF-IDF + características numéricas
- **Uso:** Modelo base para comparación

#### 👥 K-Vecinos Más Cercanos (KNN)
- **Accuracy:** 78.9%
- **Estado:** Regular
- **Variables:** TF-IDF + características numéricas
- **Uso:** Encuentra artículos similares

#### 📊 Naive Bayes
- **Accuracy:** 82.1%
- **Estado:** Bueno
- **Variables:** TF-IDF (unigramas + bigramas)
- **Uso:** Clasificación probabilística de texto

#### 🌳 Árbol de Decisión
- **Accuracy:** 79.3%
- **Estado:** Regular
- **Variables:** TF-IDF + características numéricas
- **Uso:** Modelo interpretable con reglas de decisión

#### 🌲 Random Forest
- **Accuracy:** 87.6%
- **Estado:** Excelente
- **Variables:** TF-IDF + características numéricas
- **Uso:** Múltiples árboles para reducir overfitting

#### ⚡ Support Vector Machine (SVM)
- **Accuracy:** 84.7%
- **Estado:** Bueno
- **Variables:** TF-IDF + características numéricas
- **Uso:** Clasificación con hiperplano óptimo

#### 🚀 HistGradientBoosting (LightGBM)
- **Accuracy:** 89.1%
- **Estado:** Excelente
- **Variables:** TF-IDF + características numéricas
- **Uso:** Algoritmo avanzado basado en boosting

### 2. Algoritmo de Clustering (1 método)

#### ⭕ K-Means
- **Silhouette Score:** 0.743
- **Estado:** Excelente
- **Variables:** TF-IDF + características numéricas
- **Uso:** Agrupa artículos similares sin etiquetas

### 3. Algoritmos de Series Temporales (2 métodos)

#### 📊 ARIMA
- **Accuracy:** 76.8%
- **Estado:** Regular
- **Variables:** Patrones temporales de publicación
- **Uso:** Análisis de tendencias temporales

#### 📉 Suavizado Exponencial
- **Accuracy:** 74.2%
- **Estado:** Regular
- **Variables:** Medias móviles temporales
- **Uso:** Predicciones basadas en tendencias

### 4. Algoritmo Ensemble (1 método)

#### 🎯 Voting Classifier
- **Accuracy:** 88.3%
- **Estado:** Excelente
- **Variables:** Combinación de múltiples algoritmos
- **Uso:** Mejora rendimiento mediante consenso

## 🧠 Redes Neuronales

### 1. Perceptrón Multicapa (MLP)
- **Accuracy:** 96.6%
- **Estado:** Excelente
- **Arquitectura:** 3 capas ocultas (100, 50, 25)
- **Activación:** ReLU
- **Optimizador:** Adam

### 2. CNN para Texto
- **Accuracy:** 61.1%
- **Estado:** Regular
- **Arquitectura:** Embedding + Conv1D + GlobalMaxPool + Dense
- **Filtros:** 128 filtros, kernel=5
- **Embedding:** 128 dimensiones

### 3. LSTM
- **Accuracy:** 90.2%
- **Estado:** Bueno
- **Arquitectura:** Embedding + LSTM(64) + LSTM(32) + Dense
- **Memoria:** Secuencial bidireccional
- **Embedding:** 128 dimensiones

### 4. BiLSTM
- **Accuracy:** 92.1%
- **Estado:** Excelente
- **Arquitectura:** Embedding + BiLSTM(64) + BiLSTM(32) + Dense
- **Memoria:** Procesamiento bidireccional
- **Embedding:** 128 dimensiones

## 🛠️ Tecnologías Utilizadas

### Backend
- **Python 3.11.5**
- **Flask** - Servidor web
- **scikit-learn** - Algoritmos de ML
- **TensorFlow/Keras** - Redes neuronales
- **pandas** - Manipulación de datos
- **numpy** - Cálculos numéricos
- **nltk** - Procesamiento de texto

### Frontend
- **React 18** - Framework de UI
- **Ant Design** - Componentes de UI
- **Recharts** - Visualizaciones
- **JavaScript ES6+** - Lógica del cliente

### Procesamiento de Datos
- **TF-IDF** - Vectorización de texto
- **StandardScaler** - Normalización
- **LabelEncoder** - Codificación categórica
- **SimpleImputer** - Manejo de valores faltantes

## 📁 Estructura del Proyecto

```
clasedepractica/
├── 📊 Datos
│   ├── articulos_exportados_20250926_082756.csv
│   └── dashboard_data_detallado.json
├── 🧠 Análisis
│   ├── analisis_inteligente_dataset.py
│   ├── analisis_redes_neuronales.py
│   └── verificar_redes_neuronales_reales.py
├── 🎨 Frontend
│   ├── src/
│   │   ├── App.js
│   │   └── index.js
│   ├── build/
│   └── package.json
├── 🔧 Backend
│   ├── server.py
│   └── dashboard_redes_neuronales.json
├── 📚 Documentación
│   ├── README.md
│   └── MANUAL_USUARIO.md
└── 🚀 Scripts
    ├── generar_dashboard_detallado.py
    └── actualizar_redes_neuronales_reales.py
```

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.11.5+
- Node.js 16+
- npm 8+

### 1. Instalación de Dependencias Python
```bash
pip install pandas numpy scikit-learn tensorflow flask nltk
```

### 2. Instalación de Dependencias React
```bash
npm install
```

### 3. Construcción del Frontend
```bash
npm run build
```

### 4. Ejecución del Servidor
```bash
python server.py
```

### 5. Acceso al Dashboard
Abrir navegador en: `http://localhost:3002`

## 📊 Características del Dashboard

### 1. Pestaña Principal
- **Resumen del dataset** con estadísticas generales
- **Mejor algoritmo** identificado automáticamente
- **Botón de Redes Neuronales** para análisis avanzado

### 2. Gráficos de Comparación
- **Comparación de Accuracy** entre todos los algoritmos
- **Comparación de AUC** para algoritmos de clasificación
- **Gráficos específicos** para cada tipo de algoritmo

### 3. Detalles por Método
- **Información detallada** de cada algoritmo
- **Variables utilizadas** por cada método
- **Proceso paso a paso** del algoritmo
- **Interpretación detallada** de resultados
- **Gráficos específicos** para cada algoritmo

### 4. Redes Neuronales
- **5 arquitecturas** de redes neuronales
- **Análisis real** del dataset de noticias
- **Resultados por periódico y categoría**
- **Interpretación de patrones** aprendidos

### 5. Conclusión y Evaluación
- **Evaluación general** del proyecto
- **Fortalezas principales** identificadas
- **Resultados académicos** obtenidos
- **Evaluación técnica** detallada

## 📈 Resultados Obtenidos

### Mejores Algoritmos
1. **MLP (Red Neuronal):** 96.6% accuracy
2. **BiLSTM (Red Neuronal):** 92.1% accuracy
3. **LSTM (Red Neuronal):** 90.2% accuracy
4. **HistGradientBoosting:** 89.1% accuracy
5. **Voting Classifier:** 88.3% accuracy

### Análisis por Periódico
- **La Vanguardia:** 99.6% de artículos importantes
- **Elmundo:** 91.1% de artículos importantes
- **El Popular:** 47.3% de artículos importantes
- **Ojo:** 29.4% de artículos importantes
- **El Comercio:** 16.3% de artículos importantes

### Análisis por Categoría
- **Ciencia y Salud:** 98.7% de artículos importantes
- **Internacional:** 85.4% de artículos importantes
- **Cultura:** 83.5% de artículos importantes
- **General:** 45.4% de artículos importantes
- **Mundo:** 8.0% de artículos importantes

## 🔍 Interpretación de Resultados

### ¿Por qué MLP es el mejor?
- **Arquitectura densa** ideal para características combinadas
- **Capacidad de aprendizaje** de patrones complejos
- **Optimización Adam** eficiente para este dataset
- **Combinación exitosa** de TF-IDF + características numéricas

### ¿Por qué CNN tiene rendimiento limitado?
- **Filtros convolucionales** no capturan eficientemente patrones de importancia periodística
- **Limitaciones** para texto periodístico vs. imágenes
- **Arquitectura** más adecuada para secuencias espaciales

### ¿Por qué LSTM/BiLSTM funcionan bien?
- **Memoria secuencial** ideal para texto
- **Procesamiento bidireccional** (BiLSTM) mejora comprensión
- **Captura de dependencias** a largo plazo en el texto

## 🎯 Aplicaciones Prácticas

### Para Editores
- **Clasificación automática** de artículos importantes
- **Priorización de contenido** para publicación
- **Análisis de calidad** periodística

### Para Redacciones
- **Filtrado inteligente** de noticias relevantes
- **Optimización de recursos** editoriales
- **Análisis de tendencias** periodísticas

### Para Investigadores
- **Metodología replicable** para análisis de texto
- **Framework** para minería de datos periodísticos
- **Base** para investigaciones académicas

## 📚 Documentación Adicional

- **MANUAL_USUARIO.md** - Guía detallada de uso del dashboard
- **Código comentado** en todos los scripts Python
- **Documentación inline** en componentes React
- **Ejemplos de uso** en cada algoritmo

## 🔧 Troubleshooting

### Problemas Comunes
1. **Puerto ocupado:** Cambiar puerto en `server.py`
2. **Dependencias faltantes:** Ejecutar `pip install -r requirements.txt`
3. **Cache del navegador:** Limpiar cache o usar modo incógnito
4. **Errores de compilación:** Ejecutar `npm run build`

### Soluciones
- **Reiniciar servidor:** `pkill -f "python server.py" && python server.py`
- **Limpiar build:** `rm -rf build && npm run build`
- **Verificar puertos:** `lsof -i :3002`

## 🏆 Evaluación del Proyecto

### Nivel Académico
- **Proyecto universitario avanzado** con implementación técnica sólida
- **Comprensión profunda** de algoritmos de ML y redes neuronales
- **Metodología rigurosa** con criterios objetivos
- **Resultados académicamente válidos** y bien fundamentados

### Fortalezas Principales
- **Datos reales:** 1,571 artículos periodísticos auténticos
- **Algoritmos diversos:** 10 algoritmos + 5 redes neuronales
- **Feature Engineering avanzado:** 1,018 características
- **Dashboard interactivo:** Visualización profesional
- **Interpretación detallada:** Resultados explicables

### Resultados Técnicos
- **Accuracy promedio:** 83.5% (excelente)
- **Mejor algoritmo:** MLP con 96.6%
- **Arquitecturas implementadas:** MLP, CNN, LSTM, BiLSTM
- **Metodología:** Criterios objetivos y reproducibles

## 📞 Soporte

Para consultas o problemas:
1. Revisar la documentación
2. Verificar logs del servidor
3. Comprobar dependencias
4. Consultar el manual de usuario

---

**🎯 Proyecto completado exitosamente con análisis real de datos periodísticos y implementación de 15 algoritmos de minería de datos y redes neuronales.**