#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para mostrar los artículos más importantes identificados por el análisis
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def cargar_y_analizar_articulos():
    """Cargar dataset y mostrar artículos más importantes"""
    print("📊 Cargando dataset de noticias...")
    
    # Cargar datos
    df = pd.read_csv('articulos_exportados_20250926_082756.csv', sep=';')
    print(f"✅ Dataset cargado: {df.shape[0]} artículos, {df.shape[1]} columnas")
    
    # Limpieza básica
    df = df.dropna(subset=['Título', 'Contenido'])
    df['Título'] = df['Título'].fillna('')
    df['Resumen'] = df['Resumen'].fillna('')
    df['Contenido'] = df['Contenido'].fillna('')
    
    # Feature Engineering (mismo que en el análisis)
    df['longitud_titulo'] = df['Título'].str.len()
    df['longitud_resumen'] = df['Resumen'].str.len()
    df['longitud_contenido'] = df['Contenido'].str.len()
    df['palabras_titulo'] = df['Título'].str.split().str.len()
    df['palabras_contenido'] = df['Contenido'].str.split().str.len()
    
    # Complejidad textual
    df['complejidad_titulo'] = df['palabras_titulo'] / (df['longitud_titulo'] + 1)
    df['complejidad_contenido'] = df['palabras_contenido'] / (df['longitud_contenido'] + 1)
    
    # Conteo temático
    palabras_clave = {
        'politica': ['política', 'gobierno', 'elecciones', 'presidente', 'congreso', 'senado'],
        'economia': ['economía', 'finanzas', 'mercado', 'empresa', 'negocio', 'dinero'],
        'internacional': ['internacional', 'mundo', 'global', 'país', 'nación', 'extranjero'],
        'social': ['sociedad', 'comunidad', 'gente', 'personas', 'vida', 'social'],
        'tecnologia': ['tecnología', 'digital', 'internet', 'software', 'app', 'online'],
        'cultura': ['cultura', 'arte', 'música', 'literatura', 'teatro', 'cine']
    }
    
    for tema, palabras in palabras_clave.items():
        df[f'conteo_{tema}'] = df['Contenido'].str.lower().str.count('|'.join(palabras))
    
    # Criterios de importancia
    df['conteo_tematico'] = df[['conteo_politica', 'conteo_economia', 'conteo_internacional', 
                               'conteo_social', 'conteo_tecnologia', 'conteo_cultura']].sum(axis=1)
    
    # Título informativo
    df['titulo_informativo'] = (
        (df['longitud_titulo'] >= 20) & 
        (df['palabras_titulo'] >= 5) & 
        (df['Título'].str.contains('[A-Z]', regex=True))
    ).astype(int)
    
    # Contenido estructurado
    df['contenido_estructurado'] = (
        (df['longitud_contenido'] >= 500) & 
        (df['palabras_contenido'] >= 100) & 
        (df['complejidad_contenido'] >= 0.15)
    ).astype(int)
    
    # Prestigio del periódico
    periodicos_prestigiosos = ['La Vanguardia', 'Elmundo', 'El País', 'ABC']
    df['prestigio_periodico'] = df['Periódico'].isin(periodicos_prestigiosos).astype(int)
    
    # Relevancia de categoría
    categorias_relevantes = ['Internacional', 'Política', 'Economía', 'Ciencia y Salud']
    df['relevancia_categoria'] = df['Categoría'].isin(categorias_relevantes).astype(int)
    
    # Variable objetivo: es_importante
    criterios = [
        df['longitud_contenido'] >= df['longitud_contenido'].quantile(0.7),
        df['prestigio_periodico'] == 1,
        df['relevancia_categoria'] == 1,
        df['conteo_tematico'] >= 2,
        df['titulo_informativo'] == 1,
        df['contenido_estructurado'] == 1,
        df['complejidad_contenido'] >= df['complejidad_contenido'].quantile(0.6)
    ]
    
    df['es_importante'] = (sum(criterios) >= 4).astype(int)
    
    # Mostrar estadísticas
    print(f"\n📊 ESTADÍSTICAS DEL ANÁLISIS:")
    print(f"Total de artículos: {len(df)}")
    print(f"Artículos importantes: {df['es_importante'].sum()} ({df['es_importante'].mean()*100:.1f}%)")
    print(f"Artículos regulares: {(1-df['es_importante']).sum()} ({(1-df['es_importante']).mean()*100:.1f}%)")
    
    # Mostrar artículos más importantes
    articulos_importantes = df[df['es_importante'] == 1].copy()
    articulos_importantes = articulos_importantes.sort_values('longitud_contenido', ascending=False)
    
    print(f"\n🏆 TOP 10 ARTÍCULOS MÁS IMPORTANTES:")
    print("="*80)
    
    for i, (idx, articulo) in enumerate(articulos_importantes.head(10).iterrows()):
        print(f"\n{i+1}. {articulo['Título'][:80]}...")
        print(f"   📰 Periódico: {articulo['Periódico']}")
        print(f"   📂 Categoría: {articulo['Categoría']}")
        print(f"   📏 Longitud: {articulo['longitud_contenido']} caracteres")
        print(f"   🔤 Palabras: {articulo['palabras_contenido']}")
        criterios_cumplidos = sum([
            articulo['longitud_contenido'] >= df['longitud_contenido'].quantile(0.7),
            articulo['prestigio_periodico'] == 1,
            articulo['relevancia_categoria'] == 1,
            articulo['conteo_tematico'] >= 2,
            articulo['titulo_informativo'] == 1,
            articulo['contenido_estructurado'] == 1,
            articulo['complejidad_contenido'] >= df['complejidad_contenido'].quantile(0.6)
        ])
        print(f"   🏆 Criterios cumplidos: {criterios_cumplidos}/7")
        print(f"   📝 Resumen: {articulo['Resumen'][:150]}...")
        print("-" * 80)
    
    # Análisis por periódico
    print(f"\n📰 ANÁLISIS POR PERIÓDICO:")
    print("="*50)
    for periodico in df['Periódico'].value_counts().head(5).index:
        subset = df[df['Periódico'] == periodico]
        importantes = subset[subset['es_importante'] == 1]
        print(f"{periodico}: {len(importantes)}/{len(subset)} importantes ({len(importantes)/len(subset)*100:.1f}%)")
    
    # Análisis por categoría
    print(f"\n📂 ANÁLISIS POR CATEGORÍA:")
    print("="*50)
    for categoria in df['Categoría'].value_counts().head(5).index:
        subset = df[df['Categoría'] == categoria]
        importantes = subset[subset['es_importante'] == 1]
        print(f"{categoria}: {len(importantes)}/{len(subset)} importantes ({len(importantes)/len(subset)*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    df = cargar_y_analizar_articulos()
    print(f"\n✅ Análisis completado. Dataset con {len(df)} artículos procesados.")
