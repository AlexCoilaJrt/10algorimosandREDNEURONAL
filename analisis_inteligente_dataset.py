#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Inteligente del Dataset de Artículos Periodísticos
Usando variables realmente útiles y relevantes para el análisis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, silhouette_score
from scipy.sparse import hstack
import json
import warnings
from datetime import datetime
import re
warnings.filterwarnings('ignore')

def cargar_y_analizar_dataset():
    """Cargar y analizar el dataset para entender las variables útiles"""
    print("📊 Cargando y analizando dataset...")
    
    # Cargar datos
    df = pd.read_csv('articulos_exportados_20250926_082756.csv', sep=';')
    print(f"✅ Dataset cargado: {df.shape[0]} artículos, {df.shape[1]} columnas")
    
    # Análisis exploratorio
    print(f"\n🔍 ANÁLISIS EXPLORATORIO:")
    print(f"   • Periódicos únicos: {df['Periódico'].nunique()}")
    print(f"   • Categorías únicas: {df['Categoría'].nunique()}")
    print(f"   • Regiones únicas: {df['Región'].nunique()}")
    print(f"   • Artículos con imágenes: {df['Cantidad Imágenes'].sum()}")
    
    # Limpiar datos
    df = df.dropna(subset=['Título', 'Contenido'])
    df['Título'] = df['Título'].fillna('')
    df['Resumen'] = df['Resumen'].fillna('')
    df['Contenido'] = df['Contenido'].fillna('')
    
    print(f"✅ Datos limpiados: {df.shape[0]} artículos")
    return df

def crear_variables_inteligentes(df):
    """
    Crear variables realmente útiles para análisis periodístico
    """
    print("🧠 Creando variables inteligentes...")
    
    # 1. ANÁLISIS DE CONTENIDO TEXTUAL
    df['longitud_titulo'] = df['Título'].str.len()
    df['longitud_resumen'] = df['Resumen'].str.len()
    df['longitud_contenido'] = df['Contenido'].str.len()
    df['palabras_titulo'] = df['Título'].str.split().str.len()
    df['palabras_contenido'] = df['Contenido'].str.split().str.len()
    
    # 2. ANÁLISIS DE COMPLEJIDAD DEL TEXTO
    def calcular_complejidad_texto(texto):
        if pd.isna(texto) or texto == '':
            return 0
        # Palabras por oración (complejidad)
        oraciones = len(re.split(r'[.!?]+', str(texto)))
        palabras = len(str(texto).split())
        return palabras / max(oraciones, 1)
    
    df['complejidad_titulo'] = df['Título'].apply(calcular_complejidad_texto)
    df['complejidad_contenido'] = df['Contenido'].apply(calcular_complejidad_texto)
    
    # 3. ANÁLISIS DE TEMAS IMPORTANTES (palabras clave periodísticas)
    palabras_importantes = {
        'politica': ['gobierno', 'política', 'elecciones', 'parlamento', 'ministro', 'presidente', 'congreso'],
        'economia': ['economía', 'financiero', 'mercado', 'empresa', 'negocio', 'inversión', 'crisis'],
        'internacional': ['internacional', 'mundo', 'global', 'país', 'nación', 'extranjero', 'diplomacia'],
        'social': ['sociedad', 'social', 'comunidad', 'público', 'ciudadano', 'derechos', 'justicia'],
        'tecnologia': ['tecnología', 'digital', 'innovación', 'ciencia', 'investigación', 'desarrollo'],
        'cultura': ['cultura', 'arte', 'literatura', 'música', 'cine', 'teatro', 'exposición']
    }
    
    for tema, palabras in palabras_importantes.items():
        def contar_palabras_tema(texto):
            if pd.isna(texto):
                return 0
            texto_lower = str(texto).lower()
            return sum(1 for palabra in palabras if palabra in texto_lower)
        
        df[f'conteo_{tema}'] = df['Contenido'].apply(contar_palabras_tema)
    
    # 4. ANÁLISIS DE ESTRUCTURA PERIODÍSTICA
    # Títulos que siguen formato periodístico (más informativos)
    df['titulo_informativo'] = (
        (df['longitud_titulo'] > df['longitud_titulo'].quantile(0.6)) &
        (df['palabras_titulo'] >= 5) &
        (df['Título'].str.contains(r'[A-Z][a-z]+.*[A-Z][a-z]+', regex=True))  # Múltiples palabras capitalizadas
    )
    
    # Contenido con estructura periodística
    df['contenido_estructurado'] = (
        (df['longitud_contenido'] > df['longitud_contenido'].quantile(0.7)) &
        (df['palabras_contenido'] >= 100) &
        (df['complejidad_contenido'] > df['complejidad_contenido'].quantile(0.5))
    )
    
    # 5. ANÁLISIS DE PERIÓDICOS (prestigio periodístico real)
    # Basado en reconocimiento periodístico y calidad editorial
    periodicos_prestigio = {
        'alto': ['La Vanguardia', 'Elmundo', 'Nytimes'],
        'medio': ['El Comercio', 'Peru21', 'El Peruano'],
        'bajo': ['Trome', 'Ojo', 'El Popular']
    }
    
    def asignar_prestigio(periodico):
        for nivel, periodicos in periodicos_prestigio.items():
            if periodico in periodicos:
                return nivel
        return 'medio'  # Por defecto
    
    df['prestigio_periodico'] = df['Periódico'].apply(asignar_prestigio)
    
    # 6. ANÁLISIS DE CATEGORÍAS (relevancia periodística)
    categorias_relevantes = {
        'alta': ['Internacional', 'Política', 'Economía', 'Ciencia y Salud'],
        'media': ['Cultura', 'Sociedad', 'Actualidad', 'Mundo'],
        'baja': ['Deportes', 'Espectáculos', 'Horóscopo', 'Vida']
    }
    
    def asignar_relevancia_categoria(categoria):
        for nivel, categorias in categorias_relevantes.items():
            if categoria in categorias:
                return nivel
        return 'media'
    
    df['relevancia_categoria'] = df['Categoría'].apply(asignar_relevancia_categoria)
    
    # 7. ANÁLISIS TEMPORAL (fecha de extracción)
    df['fecha_extraccion'] = pd.to_datetime(df['Fecha Extracción'])
    df['dia_semana'] = df['fecha_extraccion'].dt.day_name()
    df['es_fin_semana'] = df['dia_semana'].isin(['Saturday', 'Sunday'])
    
    # 8. CRITERIO DE IMPORTANCIA INTELIGENTE
    # Un artículo es importante si cumple múltiples criterios de calidad periodística
    criterios_importancia = (
        # Contenido sustancial
        (df['longitud_contenido'] > df['longitud_contenido'].quantile(0.7)).astype(int) +
        (df['palabras_contenido'] > df['palabras_contenido'].quantile(0.7)).astype(int) +
        
        # Estructura periodística
        df['titulo_informativo'].astype(int) +
        df['contenido_estructurado'].astype(int) +
        
        # Prestigio del medio
        (df['prestigio_periodico'] == 'alto').astype(int) +
        
        # Relevancia temática
        (df['relevancia_categoria'] == 'alta').astype(int) +
        
        # Contenido temático importante
        (df['conteo_politica'] >= 2).astype(int) +
        (df['conteo_economia'] >= 2).astype(int) +
        (df['conteo_internacional'] >= 2).astype(int) +
        
        # Complejidad del contenido
        (df['complejidad_contenido'] > df['complejidad_contenido'].quantile(0.6)).astype(int)
    )
    
    # Artículo importante si cumple 4 o más criterios (más estricto)
    df['es_importante'] = (criterios_importancia >= 4)
    
    # Estadísticas del criterio inteligente
    total_importantes = df['es_importante'].sum()
    porcentaje_importantes = (total_importantes / len(df)) * 100
    
    print(f"\n📊 CRITERIOS INTELIGENTES APLICADOS:")
    print(f"   • Contenido sustancial (longitud + palabras)")
    print(f"   • Estructura periodística (título + contenido)")
    print(f"   • Prestigio del medio (La Vanguardia, Elmundo, Nytimes)")
    print(f"   • Relevancia temática (Internacional, Política, Economía)")
    print(f"   • Contenido temático (política, economía, internacional)")
    print(f"   • Complejidad del contenido")
    print(f"   • Un artículo es importante si cumple 4+ criterios")
    
    print(f"\n📈 RESULTADO DEL CRITERIO INTELIGENTE:")
    print(f"   • Artículos importantes: {total_importantes} ({porcentaje_importantes:.1f}%)")
    print(f"   • Artículos no importantes: {len(df) - total_importantes} ({100-porcentaje_importantes:.1f}%)")
    
    # Análisis por criterios
    print(f"\n🔍 ANÁLISIS POR CRITERIOS:")
    print(f"   • Contenido extenso: {df['longitud_contenido'] > df['longitud_contenido'].quantile(0.7)}")
    print(f"   • Título informativo: {df['titulo_informativo'].sum()} artículos")
    print(f"   • Contenido estructurado: {df['contenido_estructurado'].sum()} artículos")
    print(f"   • Periódicos de alto prestigio: {df['prestigio_periodico'] == 'alto'} artículos")
    print(f"   • Categorías de alta relevancia: {df['relevancia_categoria'] == 'alta'} artículos")
    
    return df

def preparar_caracteristicas_avanzadas(df):
    """Preparar características avanzadas para ML"""
    print("🔧 Preparando características avanzadas...")
    
    # TF-IDF del texto combinado
    tfidf = TfidfVectorizer(
        max_features=2000, 
        stop_words=None,
        ngram_range=(1, 2),  # Unigramas y bigramas
        min_df=2,  # Mínimo 2 documentos
        max_df=0.95  # Máximo 95% de documentos
    )
    texto_combinado = df['Título'] + ' ' + df['Resumen'] + ' ' + df['Contenido']
    X_tfidf = tfidf.fit_transform(texto_combinado)
    
    # Características numéricas avanzadas
    caracteristicas_numericas = [
        'longitud_titulo', 'longitud_resumen', 'longitud_contenido',
        'palabras_titulo', 'palabras_contenido',
        'complejidad_titulo', 'complejidad_contenido',
        'conteo_politica', 'conteo_economia', 'conteo_internacional',
        'conteo_social', 'conteo_tecnologia', 'conteo_cultura'
    ]
    X_numericas = df[caracteristicas_numericas].values
    
    # Codificar variables categóricas
    le_periodico = LabelEncoder()
    le_categoria = LabelEncoder()
    le_region = LabelEncoder()
    le_prestigio = LabelEncoder()
    le_relevancia = LabelEncoder()
    
    df['periodico_encoded'] = le_periodico.fit_transform(df['Periódico'])
    df['categoria_encoded'] = le_categoria.fit_transform(df['Categoría'])
    df['region_encoded'] = le_region.fit_transform(df['Región'])
    df['prestigio_encoded'] = le_prestigio.fit_transform(df['prestigio_periodico'])
    df['relevancia_encoded'] = le_relevancia.fit_transform(df['relevancia_categoria'])
    
    # Características categóricas codificadas
    X_categoricas = df[[
        'periodico_encoded', 'categoria_encoded', 'region_encoded',
        'prestigio_encoded', 'relevancia_encoded'
    ]].values
    
    # Características binarias
    X_binarias = df[[
        'titulo_informativo', 'contenido_estructurado', 'es_fin_semana'
    ]].astype(int).values
    
    # Imputar valores faltantes
    imputer = SimpleImputer(strategy='median')
    X_numericas = imputer.fit_transform(X_numericas)
    
    # Combinar todas las características
    X_combined = hstack([X_tfidf, X_numericas, X_categoricas, X_binarias])
    
    # Target
    y = df['es_importante'].astype(int)
    
    print(f"✅ Características preparadas: {X_combined.shape}")
    print(f"   • TF-IDF: {X_tfidf.shape[1]} características")
    print(f"   • Numéricas: {len(caracteristicas_numericas)} características")
    print(f"   • Categóricas: 5 características")
    print(f"   • Binarias: 3 características")
    print(f"   • Total: {X_combined.shape[1]} características")
    
    return X_combined, y, df

def entrenar_algoritmos_avanzados(X, y):
    """Entrenar algoritmos con características avanzadas"""
    print("🤖 Entrenando algoritmos avanzados...")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convertir a denso para algunos algoritmos
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled = scaler.transform(X_test_dense)
    
    algoritmos = {
        'Regresión Logística': LogisticRegression(random_state=42, max_iter=1000),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': MultinomialNB(),
        'Árbol de Decisión': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
        'SVM': SVC(random_state=42, probability=True, kernel='rbf'),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42, max_depth=10),
        'Ensemble': VotingClassifier([
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(random_state=42, n_estimators=50)),
            ('dt', DecisionTreeClassifier(random_state=42, max_depth=10))
        ])
    }
    
    resultados = []
    
    for nombre, modelo in algoritmos.items():
        try:
            print(f"   🔄 Entrenando {nombre}...")
            
            # Seleccionar datos apropiados
            if nombre == 'Naive Bayes':
                X_train_use = np.abs(X_train_dense)
                X_test_use = np.abs(X_test_dense)
            elif nombre in ['Regresión Logística', 'SVM']:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train_dense
                X_test_use = X_test_dense
            
            # Entrenar
            modelo.fit(X_train_use, y_train)
            
            # Predecir
            y_pred = modelo.predict(X_test_use)
            y_pred_proba = modelo.predict_proba(X_test_use)[:, 1] if hasattr(modelo, 'predict_proba') else y_pred
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            resultados.append({
                'nombre': nombre,
                'accuracy': float(accuracy),
                'auc': float(auc),
                'categoria': 'Clasificación'
            })
            
            print(f"      ✅ {nombre}: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
            
        except Exception as e:
            print(f"      ❌ Error en {nombre}: {str(e)}")
            resultados.append({
                'nombre': nombre,
                'accuracy': 0.0,
                'auc': 0.5,
                'categoria': 'Clasificación',
                'error': str(e)
            })
    
    # Clustering (K-Means)
    try:
        print("   🔄 Entrenando K-Means...")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_train_dense)
        
        silhouette = silhouette_score(X_train_dense, clusters)
        
        resultados.append({
            'nombre': 'K-Means',
            'accuracy': 0.0,
            'auc': 0.0,
            'silhouette': float(silhouette),
            'categoria': 'Clustering'
        })
        
        print(f"      ✅ K-Means: Silhouette={silhouette:.3f}")
        
    except Exception as e:
        print(f"      ❌ Error en K-Means: {str(e)}")
        resultados.append({
            'nombre': 'K-Means',
            'accuracy': 0.0,
            'auc': 0.0,
            'categoria': 'Clustering',
            'error': str(e)
        })
    
    return resultados, X_test, y_test

def generar_dashboard_inteligente(df, resultados):
    """Generar datos del dashboard con análisis inteligente"""
    print("📊 Generando dashboard inteligente...")
    
    # Ordenar resultados por accuracy
    resultados_ordenados = sorted([r for r in resultados if 'error' not in r], 
                                 key=lambda x: x['accuracy'], reverse=True)
    
    # Agregar ranking y estado
    for i, resultado in enumerate(resultados_ordenados):
        resultado['ranking'] = i + 1
        resultado['id'] = i + 1
        
        if resultado['accuracy'] >= 0.90:
            resultado['estado'] = 'Excelente'
        elif resultado['accuracy'] >= 0.80:
            resultado['estado'] = 'Muy Bueno'
        elif resultado['accuracy'] >= 0.70:
            resultado['estado'] = 'Bueno'
        elif resultado['accuracy'] >= 0.60:
            resultado['estado'] = 'Regular'
        else:
            resultado['estado'] = 'Necesita Mejora'
    
    # Información del dataset
    dataset_info = {
        'total_articulos': len(df),
        'columnas': list(df.columns),
        'periodicos': df['Periódico'].value_counts().to_dict(),
        'categorias': df['Categoría'].value_counts().to_dict(),
        'regiones': df['Región'].value_counts().to_dict(),
        'prestigio_distribucion': df['prestigio_periodico'].value_counts().to_dict(),
        'relevancia_distribucion': df['relevancia_categoria'].value_counts().to_dict()
    }
    
    # Resumen del análisis
    mejor_algoritmo = resultados_ordenados[0] if resultados_ordenados else None
    
    resumen_analisis = {
        'mejor_algoritmo': mejor_algoritmo,
        'metodologia': 'Análisis inteligente con criterios periodísticos avanzados',
        'criterio_clasificacion': 'Múltiples criterios de calidad periodística: contenido sustancial, estructura periodística, prestigio del medio, relevancia temática, contenido temático, complejidad',
        'algoritmos_evaluados': len(resultados),
        'variables_utilizadas': [
            'TF-IDF del texto (títulos, resúmenes, contenido)',
            'Longitud y complejidad del contenido',
            'Análisis temático (política, economía, internacional, etc.)',
            'Prestigio del periódico',
            'Relevancia de la categoría',
            'Estructura periodística (títulos informativos, contenido estructurado)',
            'Características temporales'
        ]
    }
    
    # Crear estructura completa
    dashboard_data = {
        'dataset_info': dataset_info,
        'resumen_analisis': resumen_analisis,
        'resultados': resultados_ordenados
    }
    
    # Guardar JSON
    with open('dashboard_data_inteligente.json', 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Dashboard inteligente generado: dashboard_data_inteligente.json")
    
    return dashboard_data

def main():
    """Función principal"""
    print("🚀 INICIANDO ANÁLISIS INTELIGENTE DEL DATASET")
    print("=" * 70)
    
    # 1. Cargar y analizar dataset
    df = cargar_y_analizar_dataset()
    
    # 2. Crear variables inteligentes
    df = crear_variables_inteligentes(df)
    
    # 3. Preparar características avanzadas
    X, y, df = preparar_caracteristicas_avanzadas(df)
    
    # 4. Entrenar algoritmos avanzados
    resultados, X_test, y_test = entrenar_algoritmos_avanzados(X, y)
    
    # 5. Generar dashboard inteligente
    dashboard_data = generar_dashboard_inteligente(df, resultados)
    
    print("\n" + "=" * 70)
    print("🎉 ANÁLISIS INTELIGENTE COMPLETADO")
    print("=" * 70)
    
    # Mostrar resumen
    print(f"\n📊 RESUMEN DE RESULTADOS:")
    print(f"   • Total de artículos analizados: {len(df)}")
    print(f"   • Algoritmos evaluados: {len(resultados)}")
    print(f"   • Mejor algoritmo: {dashboard_data['resumen_analisis']['mejor_algoritmo']['nombre']}")
    print(f"   • Accuracy del mejor: {dashboard_data['resumen_analisis']['mejor_algoritmo']['accuracy']:.3f}")
    
    print(f"\n🧠 CRITERIOS INTELIGENTES APLICADOS:")
    print(f"   • Contenido sustancial (longitud + palabras)")
    print(f"   • Estructura periodística (títulos informativos + contenido estructurado)")
    print(f"   • Prestigio del medio (La Vanguardia, Elmundo, Nytimes)")
    print(f"   • Relevancia temática (Internacional, Política, Economía)")
    print(f"   • Contenido temático (análisis de palabras clave)")
    print(f"   • Complejidad del contenido")
    print(f"   • Un artículo es importante si cumple 4+ criterios")
    
    print(f"\n✅ Archivo generado: dashboard_data_inteligente.json")

if __name__ == "__main__":
    main()
