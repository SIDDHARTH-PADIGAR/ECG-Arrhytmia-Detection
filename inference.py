"""
inference.py

Inference pipeline for ECG arrhythmia detection.
Loads trained model and provides prediction functionality for new ECG data.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Union, Dict, List, Tuple
from feature_extractor import ECGFeatureExtractor
import warnings
warnings.filterwarnings('ignore')


class ECGPredictor:
    """ECG arrhythmia prediction using trained model."""
    
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.metadata = None
        self.feature_extractor = None
        self.is_loaded = False
        
    def load_model(self) -> None:
        """Load trained model and preprocessing components."""
        print(f"Loading model from {self.model_dir}")
        
        try:
            # Load model components
            self.model = joblib.load(f"{self.model_dir}/rf_model.joblib")
            self.scaler = joblib.load(f"{self.model_dir}/scaler.joblib")
            self.imputer = joblib.load(f"{self.model_dir}/imputer.joblib")
            self.label_encoder = joblib.load(f"{self.model_dir}/label_encoder.joblib")
            self.metadata = joblib.load(f"{self.model_dir}/metadata.joblib")
            
            # Initialize feature extractor
            self.feature_extractor = ECGFeatureExtractor(sampling_rate=360.0)
            
            self.is_loaded = True
            print(f"Model loaded successfully!")
            print(f"Model type: {self.metadata['model_type']}")
            print(f"Classes: {self.metadata['class_names']}")
            print(f"Features: {self.metadata['n_features']}")
            
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
    
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to features."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Handle missing values
        features_imputed = self.imputer.transform(features)
        
        # Standardize features
        features_scaled = self.scaler.transform(features_imputed)
        
        return features_scaled
    
    def predict_from_features(self, features: np.ndarray) -> Dict:
        """
        Make predictions from extracted features.
        
        Args:
            features: Feature array (n_samples x n_features)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Validate feature dimensions
        expected_features = self.metadata['n_features']
        if features.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {features.shape[1]}")
        
        # Preprocess features
        features_processed = self.preprocess_features(features)
        
        # Make predictions
        predictions = self.model.predict(features_processed)
        probabilities = self.model.predict_proba(features_processed)
        
        # Convert to class names
        class_names = [self.label_encoder.classes_[pred] for pred in predictions]
        
        # Prepare results
        results = {
            'predictions': class_names,
            'predictions_encoded': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'class_names': self.metadata['class_names'],
            'confidence': np.max(probabilities, axis=1).tolist()
        }
        
        return results
    
    def predict_from_beats(self, beats: np.ndarray) -> Dict:
        """
        Make predictions from raw ECG beat segments.
        
        Args:
            beats: ECG beat segments (n_beats x beat_length)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"Extracting features from {len(beats)} beats...")
        
        # Extract features
        features_df = self.feature_extractor.extract_features_from_beats(beats)
        features = features_df.values
        
        # Make predictions
        results = self.predict_from_features(features)
        
        return results
    
    def predict_single_beat(self, beat: np.ndarray) -> Dict:
        """
        Make prediction for a single ECG beat.
        
        Args:
            beat: Single ECG beat segment (beat_length,)
            
        Returns:
            Dictionary with prediction details
        """
        if beat.ndim == 1:
            beat = beat.reshape(1, -1)
        
        results = self.predict_from_beats(beat)
        
        # Return single prediction
        single_result = {
            'prediction': results['predictions'][0],
            'confidence': results['confidence'][0],
            'probabilities': dict(zip(results['class_names'], results['probabilities'][0])),
            'prediction_encoded': results['predictions_encoded'][0]
        }
        
        return single_result
    
    def batch_predict(self, data: Union[np.ndarray, str], 
                     data_type: str = 'beats') -> Dict:
        """
        Batch prediction for multiple samples.
        
        Args:
            data: Either numpy array or path to saved data
            data_type: 'beats' for raw ECG beats, 'features' for extracted features
            
        Returns:
            Dictionary with batch predictions
        """
        # Load data if path provided
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"Data file not found: {data}")
            data = np.load(data)
        
        print(f"Processing batch of {len(data)} samples...")
        
        # Make predictions based on data type
        if data_type == 'beats':
            results = self.predict_from_beats(data)
        elif data_type == 'features':
            results = self.predict_from_features(data)
        else:
            raise ValueError("data_type must be 'beats' or 'features'")
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        info = {
            "model_type": self.metadata['model_type'],
            "n_features": self.metadata['n_features'],
            "n_classes": self.metadata['n_classes'],
            "class_names": self.metadata['class_names'],
            "feature_names": self.metadata['feature_names'][:10],  # First 10 features
            "total_features": len(self.metadata['feature_names'])
        }
        
        return info


def load_sample_data(data_path: str = "data/processed/beats.npy", 
                    n_samples: int = 5) -> np.ndarray:
    """Load sample ECG beats for testing."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Sample data not found: {data_path}")
    
    data = np.load(data_path)
    return data[:n_samples]


def main():
    """Example usage of ECG predictor."""
    # Initialize predictor
    predictor = ECGPredictor("models/")
    
    try:
        # Load model
        predictor.load_model()
        
        # Get model info
        model_info = predictor.get_model_info()
        print(f"\nModel Info:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Test with sample data if available
        try:
            sample_beats = load_sample_data(n_samples=3)
            print(f"\nTesting with {len(sample_beats)} sample beats...")
            
            # Single beat prediction
            single_result = predictor.predict_single_beat(sample_beats[0])
            print(f"\nSingle beat prediction:")
            print(f"  Prediction: {single_result['prediction']}")
            print(f"  Confidence: {single_result['confidence']:.3f}")
            print(f"  Probabilities: {single_result['probabilities']}")
            
            # Batch prediction
            batch_results = predictor.predict_from_beats(sample_beats)
            print(f"\nBatch predictions:")
            for i, (pred, conf) in enumerate(zip(batch_results['predictions'], 
                                               batch_results['confidence'])):
                print(f"  Beat {i+1}: {pred} (confidence: {conf:.3f})")
                
        except FileNotFoundError:
            print("\nNo sample data found for testing.")
            print("Run data_loader.py and feature_extractor.py first to generate test data.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to train a model first by running train_model.py")


if __name__ == "__main__":
    main()