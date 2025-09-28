# 📖 Manual de Usuario - Dashboard de Machine Learning

## 🎯 Introducción

Este manual te guiará paso a paso para usar el Dashboard de Machine Learning para análisis de artículos periodísticos. El sistema te permite analizar 1,571 artículos usando 10 algoritmos diferentes de Machine Learning.

## 🚀 Inicio Rápido

### Paso 1: Abrir el Dashboard
```bash
python abrir_dashboard.py
```
El dashboard se abrirá automáticamente en tu navegador en: `http://localhost:3002`

### Paso 2: Explorar la Interfaz
Verás una interfaz con:
- **Header**: Información del dataset y mejor algoritmo
- **Proceso de Análisis**: Pasos del análisis de datos
- **Selección de Métodos**: Tarjetas de algoritmos organizados por categorías
- **Resultados**: Gráficos y comparaciones

## 🎮 Guía de Uso Detallada

### 📊 Sección 1: Información del Dataset

**¿Qué muestra?**
- Total de artículos analizados (1,571)
- Mejor algoritmo identificado
- Métricas de rendimiento principales

**¿Cómo usar?**
- Esta información se actualiza automáticamente
- Te da contexto sobre el análisis realizado

### 🔄 Sección 2: Proceso de Análisis

**¿Qué muestra?**
- 5 pasos del proceso de análisis
- Progreso visual del análisis
- Descripción de cada etapa

**¿Cómo usar?**
- Los pasos se ejecutan automáticamente al seleccionar algoritmos
- Cada paso muestra el progreso del análisis

### 🎯 Sección 3: Selección de Algoritmos

**¿Qué muestra?**
- 10 algoritmos organizados en 3 categorías:
  - 🤖 **Clasificación** (7 métodos)
  - 🔍 **Clustering** (1 método)
  - 📈 **Series Temporales** (2 métodos)

**¿Cómo usar?**
1. **Seleccionar algoritmos**: Haz clic en las tarjetas para seleccionar
2. **Ver detalles**: Cada tarjeta muestra:
   - Nombre del algoritmo
   - Descripción breve
   - Accuracy y AUC
   - Tipo de algoritmo
3. **Ejecutar análisis**: Usa el botón "🚀 Ejecutar Análisis"

### 📈 Sección 4: Resultados del Análisis

**¿Qué muestra?**
- 4 pestañas con diferentes vistas:
  - **Gráficos de Comparación**: Comparación visual de algoritmos
  - **Detalles por Método**: Explicaciones detalladas de cada algoritmo
  - **Gráficos Detallados**: Visualizaciones específicas
  - **Comparación Final**: Tabla comparativa completa

## 🔍 Guía por Pestañas

### 📊 Pestaña 1: Gráficos de Comparación

**¿Qué hace?**
- Muestra gráficos de barras comparando Accuracy y AUC
- Solo muestra los algoritmos seleccionados
- Permite comparación visual rápida

**¿Cómo usar?**
1. Selecciona los algoritmos que quieres comparar
2. Haz clic en "🚀 Ejecutar Análisis"
3. Ve a la pestaña "Gráficos de Comparación"
4. Analiza las barras para ver rendimiento relativo

### 🔍 Pestaña 2: Detalles por Método

**¿Qué hace?**
- Muestra información detallada de cada algoritmo seleccionado
- Incluye interpretaciones específicas del dataset
- Explica qué hizo cada algoritmo y cómo funcionó

**¿Cómo usar?**
1. Selecciona los algoritmos de interés
2. Ejecuta el análisis
3. Ve a "Detalles por Método"
4. Para cada algoritmo verás:
   - **¿Qué es?** - Definición del algoritmo
   - **¿Cómo funciona?** - Proceso interno
   - **Variables utilizadas** - Características específicas
   - **Objetivo específico** - Para qué se usa
   - **Preprocesamiento** - Cómo se preparan los datos
   - **Proceso paso a paso** - 4 pasos detallados
   - **Interpretación detallada** - Análisis específico del dataset
   - **Gráfico específico** - Visualización única del algoritmo
   - **Ventajas y desventajas** - Análisis completo

### 📊 Pestaña 3: Gráficos Detallados

**¿Qué hace?**
- Muestra gráficos específicos para cada algoritmo
- Incluye métricas de rendimiento individuales
- Visualiza variables utilizadas por cada método

**¿Cómo usar?**
1. Selecciona algoritmos específicos
2. Ejecuta el análisis
3. Ve a "Gráficos Detallados"
4. Analiza cada gráfico:
   - **Rendimiento del método**: Accuracy y AUC
   - **Gráfico específico**: Visualización única del algoritmo
   - **Variables utilizadas**: Lista de características

### 🏆 Pestaña 4: Comparación Final

**¿Qué hace?**
- Muestra tabla completa de todos los algoritmos
- Incluye ranking, métricas y estado
- Proporciona recomendaciones específicas

**¿Cómo usar?**
1. Ve a "Comparación Final"
2. Analiza la tabla completa:
   - **Ranking**: Posición de cada algoritmo
   - **Método**: Nombre del algoritmo
   - **Accuracy**: Precisión del modelo
   - **AUC**: Área bajo la curva ROC
   - **Categoría**: Tipo de algoritmo
   - **Estado**: Evaluación del rendimiento
3. Revisa las recomendaciones al final

## 🎨 Características de la Interfaz

### 🎯 Navegación
- **Botones principales**: Ejecutar, Limpiar, Ver Proceso
- **Pestañas organizadas**: Fácil navegación entre secciones
- **Tarjetas interactivas**: Selección visual de algoritmos

### 📊 Visualizaciones
- **Gráficos interactivos**: Tooltips informativos
- **Colores diferenciados**: Por tipo de algoritmo
- **Responsive**: Adaptable a diferentes pantallas

### 🔄 Estados del Sistema
- **Cargando**: Spinner mientras se cargan los datos
- **Seleccionado**: Tarjetas resaltadas en verde
- **Ejecutando**: Progreso visual del análisis

## 📋 Interpretación de Resultados

### 🏆 Mejores Algoritmos
1. **Árbol de Decisión** (100% Accuracy)
   - **Uso**: Clasificación interpretable
   - **Ventaja**: Reglas claras y fáciles de entender
   - **Aplicación**: Clasificación automática de artículos

2. **HistGradientBoosting** (100% Accuracy)
   - **Uso**: Máxima precisión
   - **Ventaja**: Maneja casos complejos
   - **Aplicación**: Clasificación de alta precisión

### 📊 Métricas Importantes
- **Accuracy**: Porcentaje de predicciones correctas
- **AUC**: Capacidad de distinguir entre clases
- **Variables importantes**: Características más relevantes

### 🎯 Recomendaciones por Uso
- **Para interpretabilidad**: Árbol de Decisión
- **Para máxima precisión**: HistGradientBoosting
- **Para robustez**: Random Forest
- **Para análisis temporal**: ARIMA
- **Para segmentación**: K-Means

## 🔧 Funciones Avanzadas

### 🔄 Limpiar Selección
- Botón "🔄 Limpiar Selección"
- Deselecciona todos los algoritmos
- Permite empezar de nuevo

### 📈 Ver Proceso Completo
- Botón "📈 Ver Proceso Completo"
- Muestra todos los pasos del análisis
- Útil para entender el proceso completo

### 🎯 Selección Múltiple
- Puedes seleccionar varios algoritmos
- Compara rendimiento entre métodos
- Analiza diferentes enfoques simultáneamente

## 🚨 Solución de Problemas

### ❌ Dashboard no se abre
1. Verifica que el servidor esté funcionando
2. Revisa que el puerto 3002 esté libre
3. Ejecuta: `python abrir_dashboard.py`

### ❌ Algoritmos no se muestran
1. Verifica que `dashboard_data_final_completo.json` existe
2. Revisa la consola del navegador para errores
3. Recarga la página

### ❌ Gráficos no se cargan
1. Verifica la conexión a internet (para librerías)
2. Revisa que Recharts esté instalado
3. Recarga la página

## 💡 Consejos de Uso

### 🎯 Para Análisis Efectivo
1. **Empieza con algoritmos simples**: Árbol de Decisión, Regresión Logística
2. **Compara diferentes tipos**: Clasificación vs Clustering
3. **Analiza las interpretaciones**: Entiende qué hizo cada algoritmo
4. **Revisa las variables**: Ve qué características son más importantes

### 📊 Para Interpretación de Resultados
1. **Accuracy alto**: El algoritmo clasifica bien
2. **AUC alto**: Distingue bien entre clases
3. **Variables importantes**: Características más relevantes
4. **Interpretaciones**: Explicaciones específicas del dataset

### 🔍 Para Análisis Profundo
1. **Lee las interpretaciones detalladas**
2. **Analiza los gráficos específicos**
3. **Compara diferentes algoritmos**
4. **Revisa las recomendaciones**

## 📞 Soporte

Si tienes problemas:
1. Revisa este manual
2. Verifica la consola del navegador
3. Comprueba que todos los archivos estén presentes
4. Reinicia el servidor si es necesario

---

**¡Disfruta explorando el poder del Machine Learning en el análisis de artículos periodísticos! 🎉📊**
