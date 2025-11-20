#!/usr/bin/env python3
"""
Backend API Server for FL-IoT-Threat Detection Dashboard
Provides REST API endpoints for the Vue.js frontend
"""

import os
import json
import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import torch
from model import NeuralNetwork

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("APIServer")

app = Flask(__name__, static_folder='frontend/dist')
CORS(app)  # Enable CORS for Vue.js frontend

# Paths
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "Logs"
RESULTS_DIR = BASE_DIR / "Results"
MODELS_DIR = BASE_DIR / "SavedGlobalModel"
DETECTION_RESULTS_DIR = BASE_DIR / "RealTimeService" / "detection_results"


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of saved models with metadata"""
    try:
        models = []
        if MODELS_DIR.exists():
            for model_file in sorted(MODELS_DIR.glob('*.pth'), reverse=True):
                stat = model_file.stat()
                models.append({
                    'name': model_file.name,
                    'path': str(model_file.relative_to(BASE_DIR)),
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'is_final': 'final' in model_file.name.lower()
                })
        
        return jsonify({
            'models': models,
            'count': len(models)
        })
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<model_name>/info', methods=['GET'])
def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    try:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            return jsonify({'error': 'Model not found'}), 404
        
        # Load model to get architecture info
        model = NeuralNetwork(17, 10)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # Get parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        stat = model_path.stat()
        
        return jsonify({
            'name': model_name,
            'architecture': {
                'input_size': 17,
                'hidden_layers': [128, 64],
                'output_size': 10,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            },
            'file_info': {
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Error fetching model info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get logs with filtering"""
    try:
        log_type = request.args.get('type', 'server')  # server, client
        level = request.args.get('level', 'all')  # all, error, warn, info
        limit = int(request.args.get('limit', 100))
        
        log_file = LOGS_DIR / f"{log_type}_logs.log"
        
        if not log_file.exists():
            return jsonify({'logs': [], 'count': 0})
        
        logs = []
        with open(log_file, 'r') as f:
            for line in f.readlines()[-limit:]:
                if level == 'all' or level.upper() in line:
                    # Parse log line
                    parts = line.split(' - ', 3)
                    if len(parts) >= 4:
                        logs.append({
                            'timestamp': parts[0],
                            'logger': parts[1],
                            'level': parts[2],
                            'message': parts[3].strip()
                        })
                    else:
                        logs.append({
                            'timestamp': '',
                            'logger': '',
                            'level': 'INFO',
                            'message': line.strip()
                        })
        
        return jsonify({
            'logs': logs,
            'count': len(logs),
            'type': log_type,
            'level': level
        })
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics/training', methods=['GET'])
def get_training_metrics():
    """Get training metrics from server logs"""
    try:
        metrics = {
            'rounds': [],
            'accuracy': [],
            'loss': [],
            'clients': []
        }
        
        log_file = LOGS_DIR / "server_logs.log"
        if log_file.exists():
            with open(log_file, 'r') as f:
                content = f.read()
                # Extract round information
                import re
                round_pattern = r'Round (\d+):'
                for match in re.finditer(round_pattern, content):
                    round_num = int(match.group(1))
                    if round_num not in metrics['rounds']:
                        metrics['rounds'].append(round_num)
        
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error fetching training metrics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics/detection', methods=['GET'])
def get_detection_metrics():
    """Get real-time detection metrics"""
    try:
        summary_file = DETECTION_RESULTS_DIR / "summary_report.json"
        
        if not summary_file.exists():
            return jsonify({
                'total_files': 0,
                'threats_detected': 0,
                'benign_files': 0,
                'results': []
            })
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error fetching detection metrics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/detection/results', methods=['GET'])
def get_detection_results():
    """Get detailed detection results"""
    try:
        results = []
        if DETECTION_RESULTS_DIR.exists():
            for result_file in sorted(DETECTION_RESULTS_DIR.glob('*.json'), reverse=True):
                if result_file.name != 'summary_report.json':
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        data['filename'] = result_file.name
                        results.append(data)
        
        return jsonify({
            'results': results[:50],  # Return latest 50
            'count': len(results)
        })
    except Exception as e:
        logger.error(f"Error fetching detection results: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/events', methods=['GET'])
def get_events():
    """Get system events timeline"""
    try:
        events = []
        
        # Parse server logs for events
        log_file = LOGS_DIR / "server_logs.log"
        if log_file.exists():
            with open(log_file, 'r') as f:
                for line in f.readlines()[-100:]:
                    if any(keyword in line for keyword in ['Round', 'Started', 'Saved', 'completed']):
                        parts = line.split(' - ', 3)
                        if len(parts) >= 4:
                            events.append({
                                'timestamp': parts[0],
                                'type': 'training',
                                'message': parts[3].strip(),
                                'level': parts[2]
                            })
        
        # Parse client logs
        client_log = LOGS_DIR / "client_logs.log"
        if client_log.exists():
            with open(client_log, 'r') as f:
                for line in f.readlines()[-100:]:
                    if any(keyword in line for keyword in ['Connected', 'Training', 'Evaluation']):
                        parts = line.split(' - ', 3)
                        if len(parts) >= 4:
                            events.append({
                                'timestamp': parts[0],
                                'type': 'client',
                                'message': parts[3].strip(),
                                'level': parts[2]
                            })
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'events': events[:100],
            'count': len(events)
        })
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        config_file = BASE_DIR / "RealTimeService" / "config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            return jsonify(config)
        else:
            return jsonify({'error': 'Configuration file not found'}), 404
    except Exception as e:
        logger.error(f"Error fetching config: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        config_file = BASE_DIR / "RealTimeService" / "config.json"
        new_config = request.json
        
        with open(config_file, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        return jsonify({'status': 'success', 'message': 'Configuration updated'})
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall system statistics"""
    try:
        stats = {
            'models': {
                'total': len(list(MODELS_DIR.glob('*.pth'))) if MODELS_DIR.exists() else 0,
                'latest': None
            },
            'training': {
                'total_rounds': 0,
                'status': 'idle'
            },
            'detection': {
                'total_processed': 0,
                'threats_detected': 0
            },
            'logs': {
                'server_size': 0,
                'client_size': 0
            }
        }
        
        # Get latest model
        if MODELS_DIR.exists():
            models = sorted(MODELS_DIR.glob('*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)
            if models:
                stats['models']['latest'] = models[0].name
        
        # Get log sizes
        server_log = LOGS_DIR / "server_logs.log"
        if server_log.exists():
            stats['logs']['server_size'] = server_log.stat().st_size
        
        client_log = LOGS_DIR / "client_logs.log"
        if client_log.exists():
            stats['logs']['client_size'] = client_log.stat().st_size
        
        # Get detection stats
        summary_file = DETECTION_RESULTS_DIR / "summary_report.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                stats['detection']['total_processed'] = summary.get('total_files', 0)
                stats['detection']['threats_detected'] = summary.get('threats_detected', 0)
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return jsonify({'error': str(e)}), 500


# Serve Vue.js frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve the Vue.js frontend"""
    dist_dir = BASE_DIR / 'frontend' / 'dist'
    if dist_dir.exists():
        if path and (dist_dir / path).exists():
            return send_from_directory(str(dist_dir), path)
        else:
            return send_from_directory(str(dist_dir), 'index.html')
    else:
        return jsonify({
            'message': 'Frontend not built yet. Run: cd frontend && npm run build',
            'api_docs': '/api/health'
        })


if __name__ == '__main__':
    logger.info("Starting FL-IoT-Threat Detection API Server")
    logger.info(f"API available at http://localhost:5000/api/")
    
    # Ensure directories exist
    LOGS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
