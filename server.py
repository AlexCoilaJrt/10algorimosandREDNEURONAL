#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servidor simple para servir los datos del dashboard
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import json
import os

app = Flask(__name__, static_folder='build', static_url_path='/')
CORS(app)  # Permitir CORS para React

@app.route('/api/dashboard-data')
def get_dashboard_data():
    """Endpoint para obtener los datos del dashboard"""
    try:
        with open('dashboard_data_detallado.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "Datos no encontrados"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/redes-neuronales-data')
def get_redes_neuronales_data():
    """Endpoint para obtener los datos de redes neuronales"""
    try:
        with open('dashboard_redes_neuronales.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "Datos de redes neuronales no encontrados"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    """Endpoint de salud del servidor"""
    return jsonify({"status": "ok", "message": "Servidor funcionando"})

@app.route('/')
def serve_react():
    """Servir la aplicación React"""
    return send_from_directory('build', 'index.html')

@app.route('/simple')
def serve_simple_dashboard():
    """Servir dashboard simple HTML"""
    return send_from_directory('.', 'dashboard_simple.html')

@app.route('/<path:path>')
def serve_static(path):
    """Servir archivos estáticos de React"""
    return send_from_directory('build', path)

if __name__ == '__main__':
    print("🚀 Iniciando servidor del dashboard...")
    print("📊 Dashboard disponible en: http://localhost:3002")
    print("🔗 API disponible en: http://localhost:3002/api/dashboard-data")
    app.run(debug=True, host='0.0.0.0', port=3002)
