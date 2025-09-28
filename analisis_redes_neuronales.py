#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de Redes Neuronales para Dataset de Noticias
Implementa 5 tipos de redes neuronales para clasificación de artículos periodísticos
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Importaciones para redes neuronales avanzadas
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Attention
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow no disponible. Solo se ejecutarán redes neuronales básicas.")

def cargar_y_preparar_datos():
    """Cargar y preparar el dataset de noticias"""
    print("📊 Cargando dataset de noticias...")
    
    # Cargar datos
    df = pd.read_csv('articulos_exportados_20250926_082756.csv', sep=';')
    print(f"✅ Dataset cargado: {df.shape[0]} artículos, {df.shape[1]} columnas")
    
    # Limpieza básica
    df = df.dropna(subset=['Título', 'Contenido'])
    df['Título'] = df['Título'].fillna('')
    df['Resumen'] = df['Resumen'].fillna('')
    df['Contenido'] = df['Contenido'].fillna('')
    
    # Feature Engineering
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
    
    print(f"✅ Feature Engineering completado")
    print(f"📊 Distribución de importancia: {df['es_importante'].value_counts().to_dict()}")
    
    return df

def preparar_caracteristicas(df):
    """Preparar características para redes neuronales"""
    print("🔧 Preparando características...")
    
    # TF-IDF para texto
    textos_combinados = df['Título'] + ' ' + df['Resumen'] + ' ' + df['Contenido']
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_tfidf = vectorizer.fit_transform(textos_combinados)
    
    # Características numéricas
    features_numericas = [
        'longitud_titulo', 'longitud_resumen', 'longitud_contenido',
        'palabras_titulo', 'palabras_contenido', 'complejidad_titulo', 'complejidad_contenido',
        'conteo_politica', 'conteo_economia', 'conteo_internacional',
        'conteo_social', 'conteo_tecnologia', 'conteo_cultura', 'conteo_tematico',
        'titulo_informativo', 'contenido_estructurado', 'prestigio_periodico', 'relevancia_categoria'
    ]
    
    X_numericas = df[features_numericas].fillna(0)
    
    # Escalado
    scaler = StandardScaler()
    X_numericas_scaled = scaler.fit_transform(X_numericas)
    
    # Combinar características
    from scipy.sparse import hstack
    X_combined = hstack([X_tfidf, X_numericas_scaled])
    
    # Target
    y = df['es_importante']
    
    print(f"✅ Características preparadas: {X_combined.shape}")
    return X_combined, y, vectorizer, scaler

def red_neuronal_mlp(X_train, X_test, y_train, y_test):
    """Perceptrón Multicapa (MLP)"""
    print("🧠 Entrenando Perceptrón Multicapa...")
    
    # Convertir a denso si es necesario
    if hasattr(X_train, 'toarray'):
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
    else:
        X_train_dense = X_train
        X_test_dense = X_test
    
    # MLP con múltiples capas
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    )
    
    mlp.fit(X_train_dense, y_train)
    y_pred = mlp.predict(X_test_dense)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'nombre': 'Perceptrón Multicapa (MLP)',
        'accuracy': accuracy,
        'modelo': mlp,
        'tipo': 'Red Neuronal Básica',
        'capas': '3 capas ocultas (100, 50, 25)',
        'activacion': 'ReLU',
        'optimizador': 'Adam'
    }

def red_neuronal_cnn_texto(textos_train, textos_test, y_train, y_test):
    """Red Neuronal Convolucional para texto"""
    if not TENSORFLOW_AVAILABLE:
        return {
            'nombre': 'CNN para Texto',
            'accuracy': 0.0,
            'error': 'TensorFlow no disponible'
        }
    
    print("🧠 Entrenando CNN para texto...")
    
    # Tokenización
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(textos_train)
    
    X_train_seq = tokenizer.texts_to_sequences(textos_train)
    X_test_seq = tokenizer.texts_to_sequences(textos_test)
    
    # Padding
    max_len = 200
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
    
    # Modelo CNN
    model = Sequential([
        Embedding(5000, 128, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenamiento
    history = model.fit(
        X_train_padded, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluación
    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    
    return {
        'nombre': 'CNN para Texto',
        'accuracy': accuracy,
        'modelo': model,
        'tipo': 'Red Convolucional',
        'capas': 'Embedding + Conv1D + GlobalMaxPool + Dense',
        'embedding_dim': 128,
        'filtros': 128,
        'kernel_size': 5
    }

def red_neuronal_lstm(textos_train, textos_test, y_train, y_test):
    """Red Neuronal Recurrente (LSTM)"""
    if not TENSORFLOW_AVAILABLE:
        return {
            'nombre': 'LSTM',
            'accuracy': 0.0,
            'error': 'TensorFlow no disponible'
        }
    
    print("🧠 Entrenando LSTM...")
    
    # Tokenización
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(textos_train)
    
    X_train_seq = tokenizer.texts_to_sequences(textos_train)
    X_test_seq = tokenizer.texts_to_sequences(textos_test)
    
    # Padding
    max_len = 200
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
    
    # Modelo LSTM
    model = Sequential([
        Embedding(5000, 128, input_length=max_len),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenamiento
    history = model.fit(
        X_train_padded, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluación
    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    
    return {
        'nombre': 'LSTM',
        'accuracy': accuracy,
        'modelo': model,
        'tipo': 'Red Recurrente',
        'capas': 'Embedding + LSTM(64) + LSTM(32) + Dense',
        'lstm_units': [64, 32],
        'embedding_dim': 128
    }

def red_neuronal_bilstm(textos_train, textos_test, y_train, y_test):
    """Red Neuronal Bidireccional (BiLSTM)"""
    if not TENSORFLOW_AVAILABLE:
        return {
            'nombre': 'BiLSTM',
            'accuracy': 0.0,
            'error': 'TensorFlow no disponible'
        }
    
    print("🧠 Entrenando BiLSTM...")
    
    # Tokenización
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(textos_train)
    
    X_train_seq = tokenizer.texts_to_sequences(textos_train)
    X_test_seq = tokenizer.texts_to_sequences(textos_test)
    
    # Padding
    max_len = 200
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
    
    # Modelo BiLSTM
    model = Sequential([
        Embedding(5000, 128, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenamiento
    history = model.fit(
        X_train_padded, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluación
    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    
    return {
        'nombre': 'BiLSTM',
        'accuracy': accuracy,
        'modelo': model,
        'tipo': 'Red Bidireccional',
        'capas': 'Embedding + BiLSTM(64) + BiLSTM(32) + Dense',
        'bilstm_units': [64, 32],
        'embedding_dim': 128
    }

def red_neuronal_attention(textos_train, textos_test, y_train, y_test):
    """Red Neuronal con Mecanismo de Atención"""
    if not TENSORFLOW_AVAILABLE:
        return {
            'nombre': 'Red con Atención',
            'accuracy': 0.0,
            'error': 'TensorFlow no disponible'
        }
    
    print("🧠 Entrenando Red con Atención...")
    
    # Tokenización
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(textos_train)
    
    X_train_seq = tokenizer.texts_to_sequences(textos_train)
    X_test_seq = tokenizer.texts_to_sequences(textos_test)
    
    # Padding
    max_len = 200
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
    
    # Modelo con Atención
    model = Sequential([
        Embedding(5000, 128, input_length=max_len),
        LSTM(64, return_sequences=True),
        Attention(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenamiento
    history = model.fit(
        X_train_padded, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluación
    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    
    return {
        'nombre': 'Red con Atención',
        'accuracy': accuracy,
        'modelo': model,
        'tipo': 'Red con Atención',
        'capas': 'Embedding + LSTM + Attention + Dense',
        'attention_mechanism': True,
        'embedding_dim': 128
    }

def ejecutar_analisis_redes_neuronales():
    """Ejecutar análisis completo de redes neuronales"""
    print("🚀 INICIANDO ANÁLISIS DE REDES NEURONALES")
    print("=" * 60)
    
    # Cargar y preparar datos
    df = cargar_y_preparar_datos()
    X, y, vectorizer, scaler = preparar_caracteristicas(df)
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 División de datos: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
    
    # Preparar textos para redes neuronales avanzadas
    textos_train = df.loc[y_train.index, 'Título'] + ' ' + df.loc[y_train.index, 'Contenido']
    textos_test = df.loc[y_test.index, 'Título'] + ' ' + df.loc[y_test.index, 'Contenido']
    
    resultados = []
    
    # 1. Perceptrón Multicapa
    try:
        resultado_mlp = red_neuronal_mlp(X_train, X_test, y_train, y_test)
        resultados.append(resultado_mlp)
        print(f"✅ MLP: {resultado_mlp['accuracy']:.4f}")
    except Exception as e:
        print(f"❌ Error en MLP: {e}")
    
    # 2. CNN para Texto
    if TENSORFLOW_AVAILABLE:
        try:
            resultado_cnn = red_neuronal_cnn_texto(textos_train, textos_test, y_train, y_test)
            resultados.append(resultado_cnn)
            print(f"✅ CNN: {resultado_cnn['accuracy']:.4f}")
        except Exception as e:
            print(f"❌ Error en CNN: {e}")
    
    # 3. LSTM
    if TENSORFLOW_AVAILABLE:
        try:
            resultado_lstm = red_neuronal_lstm(textos_train, textos_test, y_train, y_test)
            resultados.append(resultado_lstm)
            print(f"✅ LSTM: {resultado_lstm['accuracy']:.4f}")
        except Exception as e:
            print(f"❌ Error en LSTM: {e}")
    
    # 4. BiLSTM
    if TENSORFLOW_AVAILABLE:
        try:
            resultado_bilstm = red_neuronal_bilstm(textos_train, textos_test, y_train, y_test)
            resultados.append(resultado_bilstm)
            print(f"✅ BiLSTM: {resultado_bilstm['accuracy']:.4f}")
        except Exception as e:
            print(f"❌ Error en BiLSTM: {e}")
    
    # 5. Red con Atención
    if TENSORFLOW_AVAILABLE:
        try:
            resultado_attention = red_neuronal_attention(textos_train, textos_test, y_train, y_test)
            resultados.append(resultado_attention)
            print(f"✅ Atención: {resultado_attention['accuracy']:.4f}")
        except Exception as e:
            print(f"❌ Error en Atención: {e}")
    
    # Generar resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE REDES NEURONALES")
    print("=" * 60)
    
    for i, resultado in enumerate(resultados, 1):
        print(f"{i}. {resultado['nombre']}: {resultado['accuracy']:.4f}")
    
    # Mejor modelo
    if resultados:
        mejor_modelo = max(resultados, key=lambda x: x['accuracy'])
        print(f"\n🏆 MEJOR RED NEURONAL: {mejor_modelo['nombre']} ({mejor_modelo['accuracy']:.4f})")
    
    return resultados

if __name__ == "__main__":
    resultados = ejecutar_analisis_redes_neuronales()
    
    # Guardar resultados
    with open('redes_neuronales_resultados.json', 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n💾 Resultados guardados en: redes_neuronales_resultados.json")
