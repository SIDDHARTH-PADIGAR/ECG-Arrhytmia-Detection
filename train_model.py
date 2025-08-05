"""
train_model.py

Train a machine learning model for ECG arrhythmia detection.
Uses extracted features to train a RandomForest classifier with proper
train/validation/test splits and model evaluation.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class ECGModelTrainer:
    """Train and evaluate ECG arrhythmia detection models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.class_names = None
        
    def load_data(self, features_path: str = "data/processed/features.csv",
                  labels_path: str = "data/processed/labels.npy") -> None:
        """Load features and labels for training."""
        print("Loading training data...")
        
        # Load features
        if features_path.endswith('.csv'):
            self.features_df = pd.read_csv(features_path)
            self.X = self.features_df.values
            self.feature_names = self.features_df.columns.tolist()
        else:
            self.X = np.load(features_path)
            try:
                self.feature_names = np.load("data/processed/feature_names.npy").tolist()
            except:
                self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]
        
        # Load labels
        self.y_raw = np.load(labels_path)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y_raw)
        self.class_names = self.label_encoder.classes_
        
        print(f"Loaded {len(self.X)} samples with {len(self.feature_names)} features")
        print(f"Classes: {self.class_names}")
        print(f"Class distribution:")
        for i, class_name in enumerate(self.class_names):
            count = np.sum(self.y == i)
            print(f"  {class_name}: {count} ({count/len(self.y)*100:.1f}%)")
    
    def preprocess_features(self, X_train: np.ndarray, X_val: np.ndarray = None, 
                          X_test: np.ndarray = None) -> tuple:
        """Standardize features and handle missing values."""
        print("Preprocessing features...")
        
        # Handle missing values (replace with median)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        
        results = [X_train_scaled]
        
        if X_val is not None:
            X_val_imputed = imputer.transform(X_val)
            X_val_scaled = scaler.transform(X_val_imputed)
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_imputed = imputer.transform(X_test)
            X_test_scaled = scaler.transform(X_test_imputed)
            results.append(X_test_scaled)
        
        # Store preprocessors for later use
        self.imputer = imputer
        self.scaler = scaler
        
        return tuple(results) if len(results) > 1 else results[0]
    
    def split_data(self, test_size: float = 0.2, val_size: float = 0.2) -> None:
        """Split data into train/validation/test sets."""
        print("Splitting data...")
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, 
            stratify=self.y, random_state=self.random_state
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for already removed test set
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            stratify=y_temp, random_state=self.random_state
        )
        
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Validation set: {len(self.X_val)} samples") 
        print(f"Test set: {len(self.X_test)} samples")
        
        # Preprocess features
        self.X_train_processed, self.X_val_processed, self.X_test_processed = \
            self.preprocess_features(self.X_train, self.X_val, self.X_test)
    
    def train_random_forest(self, tune_hyperparameters: bool = True) -> None:
        """Train RandomForest classifier with optional hyperparameter tuning."""
        print("Training RandomForest classifier...")
        
        if tune_hyperparameters:
            print("Tuning hyperparameters with GridSearch...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            }
            
            # Grid search with cross-validation
            rf = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='f1_weighted',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train_processed, self.y_train)
            
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters with class balancing
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            self.model.fit(self.X_train_processed, self.y_train)
        
        print("Training completed!")
    
    def evaluate_model(self) -> dict:
        """Evaluate model performance on validation and test sets."""
        print("Evaluating model performance...")
        
        results = {}
        
        # Validation set evaluation
        y_val_pred = self.model.predict(self.X_val_processed)
        val_accuracy = accuracy_score(self.y_val, y_val_pred)
        
        print(f"\nValidation Set Performance:")
        print(f"Accuracy: {val_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_val, y_val_pred, 
                                  target_names=self.class_names))
        
        results['val_accuracy'] = val_accuracy
        results['val_predictions'] = y_val_pred
        
        # Test set evaluation
        y_test_pred = self.model.predict(self.X_test_processed)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        print(f"\nTest Set Performance:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_test_pred, 
                                  target_names=self.class_names))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_test_pred))
        
        results['test_accuracy'] = test_accuracy
        results['test_predictions'] = y_test_pred
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        results['feature_importance'] = feature_importance
        
        return results
    
    def save_model(self, model_dir: str = "models/") -> None:
        """Save trained model and preprocessing components."""
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"Saving model to {model_dir}")
        
        # Save main model
        joblib.dump(self.model, f"{model_dir}/rf_model.joblib")
        
        # Save preprocessing components
        joblib.dump(self.scaler, f"{model_dir}/scaler.joblib")
        joblib.dump(self.imputer, f"{model_dir}/imputer.joblib")
        joblib.dump(self.label_encoder, f"{model_dir}/label_encoder.joblib")
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'class_names': self.class_names.tolist(),
            'n_features': len(self.feature_names),
            'n_classes': len(self.class_names),
            'model_type': 'RandomForestClassifier'
        }
        
        joblib.dump(metadata, f"{model_dir}/metadata.joblib")
        
        print("Model saved successfully!")
    
    def cross_validate(self, cv_folds: int = 5) -> dict:
        """Perform cross-validation on the full dataset."""
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        # Use all data for cross-validation
        X_all_processed = self.preprocess_features(self.X)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X_all_processed, self.y, 
            cv=cv_folds, scoring='f1_weighted', n_jobs=-1
        )
        
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }


def main():
    """Main training pipeline."""
    # Check if processed data exists
    if not os.path.exists("data/processed/features.csv"):
        print("No feature data found. Run feature_extractor.py first.")
        return
    
    # Initialize trainer
    trainer = ECGModelTrainer(random_state=42)
    
    # Load data
    trainer.load_data()
    
    # Split data
    trainer.split_data(test_size=0.2, val_size=0.2)
    
    # Train model (set tune_hyperparameters=False for faster training)
    trainer.train_random_forest(tune_hyperparameters=False)
    
    # Evaluate model
    results = trainer.evaluate_model()
    
    # Cross-validation
    cv_results = trainer.cross_validate()
    
    # Save model
    trainer.save_model()
    
    print("\nTraining pipeline completed!")
    print(f"Final test accuracy: {results['test_accuracy']:.4f}")
    print(f"Cross-validation F1: {cv_results['cv_mean']:.4f}")


if __name__ == "__main__":
    main()