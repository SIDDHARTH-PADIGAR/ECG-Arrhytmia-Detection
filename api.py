"""
api.py

Flask API for ECG arrhythmia detection.
Provides REST endpoints for ECG beat classification using trained model.

Endpoints:
- POST /predict: Predict arrhythmia from ECG beat data
- GET /health: Health check
- GET /model_info: Get model information
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from typing import Dict, List, Union
import traceback

# Import our inference module
from inference import ECGPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global predictor instance
predictor = None


def initialize_predictor():
    """Initialize the ECG predictor with trained model."""
    global predictor
    
    try:
        predictor = ECGPredictor("models/")
        predictor.load_model()
        logger.info("ECG predictor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        return False


def validate_ecg_data(data: Union[List, np.ndarray], expected_length: int = 360) -> bool:
    """Validate ECG beat data format and dimensions."""
    try:
        data_array = np.array(data)
        
        # Check if data is numeric
        if not np.issubdtype(data_array.dtype, np.number):
            return False
        
        # Check dimensions
        if data_array.ndim == 1:
            # Single beat
            return len(data_array) == expected_length
        elif data_array.ndim == 2:
            # Multiple beats
            return data_array.shape[1] == expected_length
        else:
            return False
            
    except:
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        status = {
            'status': 'healthy',
            'model_loaded': predictor is not None and predictor.is_loaded,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model."""
    try:
        if predictor is None or not predictor.is_loaded:
            return jsonify({'error': 'Model not loaded'}), 503
        
        model_info = predictor.get_model_info()
        return jsonify(model_info), 200
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_arrhythmia():
    """
    Predict arrhythmia from ECG beat data.
    
    Expected JSON payload:
    {
        "data": [[...], [...], ...],  // ECG beat segments (n_beats x 360 samples)
        "data_type": "beats",         // "beats" or "features"
        "return_probabilities": true  // Optional, default false
    }
    
    Returns:
    {
        "predictions": ["Normal", "Arrhythmia", ...],
        "confidence": [0.95, 0.87, ...],
        "probabilities": [[...], [...], ...],  // If requested
        "n_samples": 2,
        "model_info": {...}
    }
    """
    try:
        # Check if model is loaded
        if predictor is None or not predictor.is_loaded:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Parse request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'data' not in data:
            return jsonify({'error': 'Missing required field: data'}), 400
        
        ecg_data = data['data']
        data_type = data.get('data_type', 'beats')
        return_probabilities = data.get('return_probabilities', False)
        
        # Validate data type
        if data_type not in ['beats', 'features']:
            return jsonify({'error': 'data_type must be "beats" or "features"'}), 400
        
        # Convert to numpy array
        try:
            ecg_array = np.array(ecg_data, dtype=float)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid data format. Data must be numeric.'}), 400
        
        # Validate data dimensions
        if data_type == 'beats':
            expected_length = 360  # 1 second at 360 Hz
            if not validate_ecg_data(ecg_array, expected_length):
                return jsonify({
                    'error': f'Invalid beat data. Expected shape: (n_beats, {expected_length}) or ({expected_length},)'
                }), 400
        
        # Ensure 2D array
        if ecg_array.ndim == 1:
            ecg_array = ecg_array.reshape(1, -1)
        
        logger.info(f"Processing {len(ecg_array)} samples of type '{data_type}'")
        
        # Make predictions
        if data_type == 'beats':
            results = predictor.predict_from_beats(ecg_array)
        else:  # features
            results = predictor.predict_from_features(ecg_array)
        
        # Prepare response
        response = {
            'predictions': results['predictions'],
            'confidence': results['confidence'],
            'n_samples': len(results['predictions']),
            'data_type': data_type,
            'model_info': {
                'classes': results['class_names'],
                'model_type': predictor.metadata['model_type']
            }
        }
        
        # Add probabilities if requested
        if return_probabilities:
            response['probabilities'] = results['probabilities']
        
        logger.info(f"Prediction completed successfully for {len(ecg_array)} samples")
        
        return jsonify(response), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/predict_single', methods=['POST'])
def predict_single_beat():
    """
    Predict arrhythmia for a single ECG beat.
    
    Expected JSON payload:
    {
        "beat": [...],  // Single ECG beat (360 samples)
    }
    
    Returns:
    {
        "prediction": "Normal",
        "confidence": 0.95,
        "probabilities": {"Normal": 0.95, "Arrhythmia": 0.05}
    }
    """
    try:
        if predictor is None or not predictor.is_loaded:
            return jsonify({'error': 'Model not loaded'}), 503
        
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        if 'beat' not in data:
            return jsonify({'error': 'Missing required field: beat'}), 400
        
        beat_data = data['beat']
        
        # Convert to numpy array
        try:
            beat_array = np.array(beat_data, dtype=float)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid beat data. Must be numeric.'}), 400
        
        # Validate dimensions
        if len(beat_array) != 360:
            return jsonify({'error': 'Beat must contain exactly 360 samples'}), 400
        
        logger.info("Processing single beat prediction")
        
        # Make prediction
        result = predictor.predict_single_beat(beat_array)
        
        logger.info(f"Single beat prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Single beat prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({'error': 'Method not allowed'}), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


def create_sample_request_data():
    """Create sample data for testing API endpoints."""
    # Generate sample ECG beat (simplified)
    t = np.linspace(0, 1, 360)  # 1 second at 360 Hz
    # Simple synthetic ECG-like signal for testing
    sample_beat = (np.sin(2 * np.pi * 1.2 * t) * 
                   np.exp(-((t - 0.3) ** 2) / 0.02) * 0.8 +
                   np.random.normal(0, 0.05, 360))
    
    return {
        'single_beat_request': {
            'beat': sample_beat.tolist()
        },
        'batch_request': {
            'data': [sample_beat.tolist(), sample_beat.tolist()],
            'data_type': 'beats',
            'return_probabilities': True
        }
    }


if __name__ == '__main__':
    print("Starting ECG Arrhythmia Detection API...")
    
    # Initialize predictor
    if initialize_predictor():
        print("Model loaded successfully!")
        
        # Print sample request format
        print("\nSample API requests:")
        sample_data = create_sample_request_data()
        print("\nSingle beat prediction (POST /predict_single):")
        print(f"curl -X POST http://localhost:5000/predict_single \\")
        print(f"  -H 'Content-Type: application/json' \\")
        print(f"  -d '{sample_data['single_beat_request']}'")
        
        print("\nBatch prediction (POST /predict):")
        print(f"curl -X POST http://localhost:5000/predict \\")
        print(f"  -H 'Content-Type: application/json' \\")
        print(f"  -d '{sample_data['batch_request']}'")
        
        print("\nHealth check: GET http://localhost:5000/health")
        print("Model info: GET http://localhost:5000/model_info")
        
        # Start Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train a model first by running train_model.py")