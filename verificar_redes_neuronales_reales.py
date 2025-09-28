#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar que los resultados de redes neuronales 
coincidan con el análisis real del dataset de noticias
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

def cargar_y_preparar_datos():
    """Cargar y preparar el dataset real de noticias"""
    print("📊 Cargando dataset real de noticias...")
    
    # Cargar datos
    df = pd.read_csv('articulos_exportados_20250926_082756.csv', sep=';')
    print(f"✅ Dataset cargado: {len(df)} artículos")
    print(f"📋 Columnas: {list(df.columns)}")
    
    # Limpiar datos
    df = df.dropna(subset=['Título', 'Contenido'])
    df['Título'] = df['Título'].fillna('')
    df['Resumen'] = df['Resumen'].fillna('')
    df['Contenido'] = df['Contenido'].fillna('')
    
    print(f"📊 Dataset limpio: {len(df)} artículos")
    
    # Crear variable objetivo basada en criterios reales
    print("\n🎯 Creando variable objetivo con criterios reales...")
    
    # Criterios de importancia
    df['longitud_contenido'] = df['Contenido'].str.len()
    df['longitud_titulo'] = df['Título'].str.len()
    df['palabras_contenido'] = df['Contenido'].str.split().str.len()
    df['palabras_titulo'] = df['Título'].str.split().str.len()
    
    # Prestigio del periódico
    periodicos_prestigiosos = ['La Vanguardia', 'Elmundo', 'El País', 'ABC']
    df['prestigio_periodico'] = df['Periódico'].isin(periodicos_prestigiosos).astype(int)
    
    # Relevancia de categoría
    categorias_relevantes = ['Internacional', 'Política', 'Economía', 'Ciencia y Salud']
    df['relevancia_categoria'] = df['Categoría'].isin(categorias_relevantes).astype(int)
    
    # Contenido temático
    palabras_clave = {
        'politica': ['política', 'gobierno', 'elecciones', 'presidente', 'congreso'],
        'economia': ['economía', 'finanzas', 'mercado', 'empresa', 'negocio'],
        'internacional': ['internacional', 'mundo', 'global', 'país', 'nación'],
        'social': ['sociedad', 'comunidad', 'gente', 'personas', 'vida'],
        'tecnologia': ['tecnología', 'digital', 'internet', 'software', 'app'],
        'cultura': ['cultura', 'arte', 'música', 'literatura', 'teatro']
    }
    
    for tema, palabras in palabras_clave.items():
        df[f'conteo_{tema}'] = df['Contenido'].str.lower().str.count('|'.join(palabras))
    
    df['conteo_tematico'] = df[['conteo_politica', 'conteo_economia', 'conteo_internacional', 
                               'conteo_social', 'conteo_tecnologia', 'conteo_cultura']].sum(axis=1)
    
    # Título informativo
    df['titulo_informativo'] = (
        (df['Título'].str.len() >= 20) & 
        (df['Título'].str.split().str.len() >= 5) & 
        (df['Título'].str.contains('[A-Z]', regex=True))
    ).astype(int)
    
    # Contenido estructurado
    df['contenido_estructurado'] = (
        (df['Contenido'].str.len() >= 500) & 
        (df['Contenido'].str.split().str.len() >= 100)
    ).astype(int)
    
    # Complejidad del contenido
    df['complejidad_contenido'] = df['Contenido'].str.split().str.len() / (df['Contenido'].str.len() + 1)
    
    # Criterio final de importancia
    percentil_70 = df['longitud_contenido'].quantile(0.7)
    percentil_60_complejidad = df['complejidad_contenido'].quantile(0.6)
    
    criterios = [
        df['longitud_contenido'] >= percentil_70,
        df['prestigio_periodico'] == 1,
        df['relevancia_categoria'] == 1,
        df['conteo_tematico'] >= 2,
        df['titulo_informativo'] == 1,
        df['contenido_estructurado'] == 1,
        df['complejidad_contenido'] >= percentil_60_complejidad
    ]
    
    df['es_importante'] = (sum(criterios) >= 4).astype(int)
    
    print(f"📈 Artículos importantes: {df['es_importante'].sum()} ({df['es_importante'].mean()*100:.1f}%)")
    
    return df

def preparar_caracteristicas(df):
    """Preparar características para las redes neuronales"""
    print("\n🔧 Preparando características...")
    
    # Combinar texto
    df['texto_completo'] = df['Título'] + ' ' + df['Resumen'] + ' ' + df['Contenido']
    
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words=None)
    X_tfidf = tfidf.fit_transform(df['texto_completo'])
    
    # Características numéricas
    features_numericas = [
        'longitud_contenido', 'longitud_titulo', 'palabras_contenido', 'palabras_titulo',
        'prestigio_periodico', 'relevancia_categoria', 'conteo_tematico',
        'titulo_informativo', 'contenido_estructurado', 'complejidad_contenido'
    ]
    
    X_numericas = df[features_numericas].fillna(0).values
    
    # Combinar características
    from scipy.sparse import hstack
    X_combined = hstack([X_tfidf, X_numericas]).toarray()
    
    # Target
    y = df['es_importante'].values
    
    print(f"📊 Características totales: {X_combined.shape[1]}")
    print(f"📊 TF-IDF: {X_tfidf.shape[1]}")
    print(f"📊 Numéricas: {X_numericas.shape[1]}")
    
    return X_combined, y, tfidf

def crear_modelo_mlp(X_train, y_train, X_test, y_test):
    """Crear y entrenar modelo MLP"""
    print("\n🧠 Entrenando MLP...")
    
    model = Sequential([
        Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Entrenar
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                       validation_data=(X_test, y_test), verbose=0)
    
    # Evaluar
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ MLP Accuracy: {accuracy:.4f}")
    return accuracy, model

def crear_modelo_cnn(X_train, y_train, X_test, y_test, df):
    """Crear y entrenar modelo CNN para texto"""
    print("\n🧠 Entrenando CNN...")
    
    # Para CNN necesitamos secuencias, no características combinadas
    # Usar solo TF-IDF para CNN
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df['texto_completo']).toarray()
    
    # Redimensionar para CNN
    X_cnn = X_tfidf.reshape(X_tfidf.shape[0], X_tfidf.shape[1], 1)
    
    # Dividir datos
    y_cnn = df['es_importante'].values
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
        X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_cnn
    )
    
    model = Sequential([
        Conv1D(128, 5, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Entrenar
    history = model.fit(X_train_cnn, y_train_cnn, epochs=50, batch_size=32,
                       validation_data=(X_test_cnn, y_test_cnn), verbose=0)
    
    # Evaluar
    y_pred = (model.predict(X_test_cnn) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test_cnn, y_pred)
    
    print(f"✅ CNN Accuracy: {accuracy:.4f}")
    return accuracy, model

def crear_modelo_lstm(X_train, y_train, X_test, y_test, df):
    """Crear y entrenar modelo LSTM"""
    print("\n🧠 Entrenando LSTM...")
    
    # Para LSTM necesitamos secuencias de texto
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(df['texto_completo'])
    
    X_seq = tokenizer.texts_to_sequences(df['texto_completo'])
    X_padded = pad_sequences(X_seq, maxlen=200)
    
    # Dividir datos
    y_lstm = df['es_importante'].values
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
        X_padded, y_lstm, test_size=0.2, random_state=42, stratify=y_lstm
    )
    
    model = Sequential([
        Embedding(1000, 128, input_length=200),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Entrenar
    history = model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32,
                       validation_data=(X_test_lstm, y_test_lstm), verbose=0)
    
    # Evaluar
    y_pred = (model.predict(X_test_lstm) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test_lstm, y_pred)
    
    print(f"✅ LSTM Accuracy: {accuracy:.4f}")
    return accuracy, model

def crear_modelo_bilstm(X_train, y_train, X_test, y_test, df):
    """Crear y entrenar modelo BiLSTM"""
    print("\n🧠 Entrenando BiLSTM...")
    
    # Usar las mismas secuencias que LSTM
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(df['texto_completo'])
    
    X_seq = tokenizer.texts_to_sequences(df['texto_completo'])
    X_padded = pad_sequences(X_seq, maxlen=200)
    
    # Dividir datos
    y_bilstm = df['es_importante'].values
    X_train_bilstm, X_test_bilstm, y_train_bilstm, y_test_bilstm = train_test_split(
        X_padded, y_bilstm, test_size=0.2, random_state=42, stratify=y_bilstm
    )
    
    model = Sequential([
        Embedding(1000, 128, input_length=200),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Entrenar
    history = model.fit(X_train_bilstm, y_train_bilstm, epochs=20, batch_size=32,
                       validation_data=(X_test_bilstm, y_test_bilstm), verbose=0)
    
    # Evaluar
    y_pred = (model.predict(X_test_bilstm) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test_bilstm, y_pred)
    
    print(f"✅ BiLSTM Accuracy: {accuracy:.4f}")
    return accuracy, model

def analizar_resultados_por_periodico(df, y_pred, y_test):
    """Analizar resultados por periódico"""
    print("\n📰 Análisis por Periódico:")
    
    # Obtener índices de test
    _, _, _, indices_test = train_test_split(
        df.index, df['es_importante'], test_size=0.2, random_state=42, stratify=df['es_importante']
    )
    
    df_test = df.iloc[indices_test].copy()
    df_test['prediccion'] = y_pred
    
    for periodico in df_test['Periódico'].value_counts().head(5).index:
        subset = df_test[df_test['Periódico'] == periodico]
        importantes = subset[subset['prediccion'] == 1]
        total = len(subset)
        porcentaje = (len(importantes) / total * 100) if total > 0 else 0
        print(f"  {periodico}: {len(importantes)}/{total} importantes ({porcentaje:.1f}%)")

def main():
    """Función principal"""
    print("🔍 VERIFICACIÓN DE REDES NEURONALES CON DATOS REALES")
    print("=" * 60)
    
    # Cargar datos
    df = cargar_y_preparar_datos()
    
    # Preparar características
    X, y, tfidf = preparar_caracteristicas(df)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"📊 Datos de prueba: {X_test.shape[0]} muestras")
    
    # Entrenar modelos
    resultados = {}
    
    # MLP
    accuracy_mlp, model_mlp = crear_modelo_mlp(X_train, y_train, X_test, y_test)
    resultados['MLP'] = accuracy_mlp
    
    # CNN
    accuracy_cnn, model_cnn = crear_modelo_cnn(X_train, y_train, X_test, y_test, df)
    resultados['CNN'] = accuracy_cnn
    
    # LSTM
    accuracy_lstm, model_lstm = crear_modelo_lstm(X_train, y_train, X_test, y_test, df)
    resultados['LSTM'] = accuracy_lstm
    
    # BiLSTM
    accuracy_bilstm, model_bilstm = crear_modelo_bilstm(X_train, y_train, X_test, y_test, df)
    resultados['BiLSTM'] = accuracy_bilstm
    
    # Mostrar resultados
    print("\n📊 RESULTADOS REALES DE REDES NEURONALES:")
    print("=" * 50)
    for modelo, accuracy in resultados.items():
        print(f"  {modelo}: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Comparar con datos del JSON
    print("\n🔍 COMPARACIÓN CON DATOS DEL DASHBOARD:")
    print("=" * 50)
    
    # Datos del JSON
    json_accuracy = {
        'MLP': 0.9849056603773585,
        'CNN': 0.8981131911277771,
        'LSTM': 0.5886792540550232,
        'BiLSTM': 0.8679245114326477
    }
    
    for modelo in ['MLP', 'CNN', 'LSTM', 'BiLSTM']:
        real = resultados[modelo]
        json_val = json_accuracy[modelo]
        diferencia = abs(real - json_val)
        print(f"  {modelo}:")
        print(f"    Real: {real:.4f}")
        print(f"    JSON: {json_val:.4f}")
        print(f"    Diferencia: {diferencia:.4f}")
        print()
    
    print("✅ Verificación completada")

if __name__ == "__main__":
    main()
