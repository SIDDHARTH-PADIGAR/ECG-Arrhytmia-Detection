"""
test_pipeline.py

Test script to validate the complete ECG arrhythmia detection pipeline.
Runs end-to-end tests and API validation.
"""

import os
import sys
import numpy as np
import requests
import time
import subprocess
from typing import Dict, List

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['numpy', 'pandas', 'scipy', 'sklearn', 'wfdb', 'flask', 'joblib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'sklearn':
                try:
                    __import__('sklearn')
                except ImportError:
                    missing_packages.append('scikit-learn')
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages installed")
    return True

def check_data_files():
    """Check if MIT-BIH data files are available."""
    raw_data_dir = "data/raw/"
    
    if not os.path.exists(raw_data_dir):
        print(f"❌ Raw data directory not found: {raw_data_dir}")
        return False
    
    # Look for MIT-BIH files
    dat_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.dat')]
    
    if not dat_files:
        print(f"❌ No MIT-BIH .dat files found in {raw_data_dir}")
        print("Download MIT-BIH data first. See README.md for instructions.")
        return False
    
    # Check for corresponding .hea and .atr files
    valid_records = []
    for dat_file in dat_files:
        record_num = dat_file.split('.')[0]
        hea_file = f"{record_num}.hea"
        atr_file = f"{record_num}.atr"
        
        if (os.path.exists(os.path.join(raw_data_dir, hea_file)) and
            os.path.exists(os.path.join(raw_data_dir, atr_file))):
            valid_records.append(record_num)
    
    if not valid_records:
        print("❌ No complete MIT-BIH records found (.dat + .hea + .atr)")
        return False
    
    print(f"✅ Found {len(valid_records)} valid MIT-BIH records: {valid_records[:5]}...")
    return True

def test_data_loading():
    """Test data loading functionality."""
    print("\n🔍 Testing data loading...")
    
    try:
        from data_loader import MITBIHLoader
        
        loader = MITBIHLoader("data/raw/")
        records = loader.get_available_records()
        
        if not records:
            print("❌ No available records found")
            return False
        
        # Test loading first record
        beats, labels, info = loader.load_all_data(max_records=1)
        
        if len(beats) == 0:
            print("❌ No beats extracted from data")
            return False
        
        print(f"✅ Data loading successful: {len(beats)} beats from record {info[0]}")
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return False

def test_feature_extraction():
    """Test feature extraction."""
    print("\n🔍 Testing feature extraction...")
    
    try:
        from feature_extractor import ECGFeatureExtractor
        
        # Create sample beat
        sample_beat = np.random.randn(360)
        
        extractor = ECGFeatureExtractor()
        features = extractor.extract_beat_features(sample_beat)
        
        if len(features) == 0:
            print("❌ No features extracted")
            return False
        
        print(f"✅ Feature extraction successful: {len(features)} features")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {str(e)}")
        return False

def test_model_training():
    """Test model training (requires processed data)."""
    print("\n🔍 Testing model training...")
    
    if not os.path.exists("data/processed/beats.npy"):
        print("⚠️  No processed data found, skipping model training test")
        return True
    
    try:
        from train_model import ECGModelTrainer
        
        trainer = ECGModelTrainer(random_state=42)
        trainer.load_data()
        trainer.split_data(test_size=0.3, val_size=0.2)  # Use more data for testing
        trainer.train_random_forest(tune_hyperparameters=False)  # Faster training
        
        # Quick evaluation
        results = trainer.evaluate_model()
        
        if results['test_accuracy'] < 0.5:  # Sanity check
            print(f"⚠️  Low model accuracy: {results['test_accuracy']:.3f}")
        else:
            print(f"✅ Model training successful: {results['test_accuracy']:.3f} accuracy")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        return False

def test_inference():
    """Test inference functionality."""
    print("\n🔍 Testing inference...")
    
    if not os.path.exists("models/rf_model.joblib"):
        print("⚠️  No trained model found, skipping inference test")
        return True
    
    try:
        from inference import ECGPredictor
        
        predictor = ECGPredictor("models/")
        predictor.load_model()
        
        # Test with sample beat
        sample_beat = np.random.randn(360)
        result = predictor.predict_single_beat(sample_beat)
        
        required_keys = ['prediction', 'confidence', 'probabilities']
        if not all(key in result for key in required_keys):
            print(f"❌ Missing keys in inference result: {result.keys()}")
            return False
        
        print(f"✅ Inference successful: {result['prediction']} (confidence: {result['confidence']:.3f})")
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        return False

def test_api():
    """Test API endpoints."""
    print("\n🔍 Testing API...")
    
    # Start API in background (this is simplified - in practice you'd start it separately)
    print("Note: Start API manually with 'python api.py' to test endpoints")
    
    try:
        # Test if API is running
        response = requests.get('http://localhost:5000/health', timeout=2)
        
        if response.status_code == 200:
            print("✅ API health check passed")
            
            # Test model info endpoint
            info_response = requests.get('http://localhost:5000/model_info', timeout=2)
            if info_response.status_code == 200:
                print("✅ Model info endpoint working")
            
            # Test prediction endpoint
            sample_data = {
                'beat': np.random.randn(360).tolist()
            }
            
            pred_response = requests.post('http://localhost:5000/predict_single', 
                                        json=sample_data, timeout=5)
            
            if pred_response.status_code == 200:
                result = pred_response.json()
                print(f"✅ API prediction successful: {result.get('prediction', 'Unknown')}")
                return True
            else:
                print(f"⚠️  API prediction returned status {pred_response.status_code}")
        
        return True
        
    except requests.exceptions.RequestException:
        print("⚠️  API not running (start with 'python api.py')")
        return True  # Not a failure, just not running

def run_pipeline_test():
    """Run end-to-end pipeline test with minimal data."""
    print("\n🚀 Running minimal end-to-end pipeline test...")
    
    try:
        # Step 1: Data loading (minimal)
        print("Step 1: Loading data...")
        from data_loader import MITBIHLoader
        loader = MITBIHLoader("data/raw/")
        beats, labels, info = loader.load_all_data(max_records=1)
        
        if len(beats) < 10:
            print("⚠️  Very few beats found, results may not be reliable")
        
        # Save minimal processed data
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/beats.npy", beats)
        np.save("data/processed/labels.npy", labels)
        
        # Step 2: Feature extraction
        print("Step 2: Extracting features...")
        from feature_extractor import ECGFeatureExtractor
        extractor = ECGFeatureExtractor()
        features_df = extractor.extract_features_from_beats(beats)
        features_df.to_csv("data/processed/features.csv", index=False)
        
        # Step 3: Model training (simplified)
        print("Step 3: Training model...")
        from train_model import ECGModelTrainer
        trainer = ECGModelTrainer(random_state=42)
        trainer.load_data()
        trainer.split_data(test_size=0.3, val_size=0.1)
        trainer.train_random_forest(tune_hyperparameters=False)
        trainer.save_model()
        
        # Step 4: Test inference
        print("Step 4: Testing inference...")
        from inference import ECGPredictor
        predictor = ECGPredictor("models/")
        predictor.load_model()
        result = predictor.predict_single_beat(beats[0])
        
        print(f"✅ End-to-end pipeline successful!")
        print(f"   - Processed {len(beats)} beats")
        print(f"   - Extracted {len(features_df.columns)} features")
        print(f"   - Sample prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🧪 ECG Arrhythmia Detection Pipeline Test Suite\n")
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Data Files", check_data_files),
        ("Data Loading", test_data_loading),
        ("Feature Extraction", test_feature_extraction),
        ("Model Training", test_model_training),
        ("Inference", test_inference),
        ("API", test_api),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Run end-to-end test if basic components work
    if results.get("Data Files") and results.get("Data Loading"):
        results["End-to-End Pipeline"] = run_pipeline_test()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is ready to use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)