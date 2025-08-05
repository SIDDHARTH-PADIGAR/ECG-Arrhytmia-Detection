"""
feature_extractor.py

Extract features from segmented ECG beats for arrhythmia classification.
Implements time-domain and morphological features commonly used in ECG analysis.

Features extracted:
- Statistical: mean, std, min, max, skewness, kurtosis
- Morphological: peak amplitude, peak location, beat duration
- RR interval features (when available)
- Spectral: dominant frequency components
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, periodogram
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class ECGFeatureExtractor:
    """Extract features from ECG beat segments."""
    
    def __init__(self, sampling_rate: float = 360.0):
        self.sampling_rate = sampling_rate
        
    def extract_statistical_features(self, beat: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features from ECG beat."""
        features = {
            'mean': np.mean(beat),
            'std': np.std(beat),
            'min': np.min(beat),
            'max': np.max(beat),
            'range': np.max(beat) - np.min(beat),
            'median': np.median(beat),
            'skewness': stats.skew(beat),
            'kurtosis': stats.kurtosis(beat),
            'rms': np.sqrt(np.mean(beat**2)),
            'energy': np.sum(beat**2)
        }
        return features
    
    def extract_morphological_features(self, beat: np.ndarray) -> Dict[str, float]:
        """Extract morphological features from ECG beat."""
        # Find peaks and valleys
        peaks, peak_props = find_peaks(beat, height=np.mean(beat))
        valleys, valley_props = find_peaks(-beat, height=-np.mean(beat))
        
        features = {
            'num_peaks': len(peaks),
            'num_valleys': len(valleys),
            'max_peak_height': np.max(beat[peaks]) if len(peaks) > 0 else 0,
            'max_valley_depth': np.min(beat[valleys]) if len(valleys) > 0 else 0,
            'peak_to_peak': np.max(beat) - np.min(beat)
        }
        
        # R-peak location (assuming it's the highest peak around center)
        center = len(beat) // 2
        search_window = len(beat) // 4  # Search within ±25% of center
        r_region = beat[center-search_window:center+search_window]
        r_peak_local = np.argmax(r_region)
        r_peak_global = center - search_window + r_peak_local
        
        features.update({
            'r_peak_amplitude': beat[r_peak_global],
            'r_peak_position': r_peak_global / len(beat),  # Normalized position
            'pre_r_amplitude': np.mean(beat[:r_peak_global]) if r_peak_global > 0 else 0,
            'post_r_amplitude': np.mean(beat[r_peak_global:]) if r_peak_global < len(beat)-1 else 0
        })
        
        return features
    
    def extract_spectral_features(self, beat: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features from ECG beat."""
        # Compute power spectral density
        freqs, psd = periodogram(beat, fs=self.sampling_rate)
        
        # Find dominant frequencies
        dominant_freq_idx = np.argmax(psd)
        dominant_freq = freqs[dominant_freq_idx]
        
        # Spectral statistics
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
        
        features = {
            'dominant_frequency': dominant_freq,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_energy': np.sum(psd),
            'low_freq_energy': np.sum(psd[freqs <= 5]),  # 0-5 Hz
            'mid_freq_energy': np.sum(psd[(freqs > 5) & (freqs <= 15)]),  # 5-15 Hz
            'high_freq_energy': np.sum(psd[freqs > 15])  # >15 Hz
        }
        
        return features
    
    def extract_beat_features(self, beat: np.ndarray) -> Dict[str, float]:
        """Extract all features from a single ECG beat."""
        features = {}
        
        # Statistical features
        features.update(self.extract_statistical_features(beat))
        
        # Morphological features
        features.update(self.extract_morphological_features(beat))
        
        # Spectral features
        features.update(self.extract_spectral_features(beat))
        
        return features
    
    def extract_rr_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Extract RR interval features (heart rate variability).
        
        Args:
            rr_intervals: Array of RR intervals in seconds
        """
        if len(rr_intervals) < 2:
            # Return zeros if insufficient data
            return {
                'rr_mean': 0, 'rr_std': 0, 'rr_min': 0, 'rr_max': 0,
                'rr_range': 0, 'rr_cv': 0, 'rmssd': 0, 'pnn50': 0
            }
        
        # Convert to milliseconds for standard HRV metrics
        rr_ms = rr_intervals * 1000
        
        # Time domain HRV features
        rr_mean = np.mean(rr_ms)
        rr_std = np.std(rr_ms)
        
        # RMSSD: root mean square of successive differences
        rr_diff = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(rr_diff**2))
        
        # pNN50: percentage of successive RR differences > 50ms
        pnn50 = np.sum(np.abs(rr_diff) > 50) / len(rr_diff) * 100
        
        features = {
            'rr_mean': rr_mean,
            'rr_std': rr_std,
            'rr_min': np.min(rr_ms),
            'rr_max': np.max(rr_ms),
            'rr_range': np.max(rr_ms) - np.min(rr_ms),
            'rr_cv': rr_std / rr_mean if rr_mean > 0 else 0,  # Coefficient of variation
            'rmssd': rmssd,
            'pnn50': pnn50
        }
        
        return features
    
    def extract_features_from_beats(self, beats: np.ndarray, 
                                  rr_intervals: np.ndarray = None) -> pd.DataFrame:
        """
        Extract features from multiple ECG beats.
        
        Args:
            beats: Array of beat segments (n_beats x beat_length)
            rr_intervals: Optional RR intervals for HRV features
            
        Returns:
            DataFrame with extracted features
        """
        print(f"Extracting features from {len(beats)} beats...")
        
        all_features = []
        
        for i, beat in enumerate(beats):
            if i % 1000 == 0:
                print(f"  Processing beat {i}/{len(beats)}")
            
            # Extract beat-level features
            beat_features = self.extract_beat_features(beat)
            
            # Add RR interval features if available
            if rr_intervals is not None and i < len(rr_intervals):
                # For simplicity, use a small window of RR intervals around current beat
                start_idx = max(0, i-2)
                end_idx = min(len(rr_intervals), i+3)
                local_rr = rr_intervals[start_idx:end_idx]
                rr_features = self.extract_rr_features(local_rr)
                beat_features.update(rr_features)
            else:
                # Add zero RR features if not available
                rr_features = self.extract_rr_features(np.array([]))
                beat_features.update(rr_features)
            
            all_features.append(beat_features)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(all_features)
        
        print(f"Extracted {len(feature_df.columns)} features per beat")
        print(f"Feature names: {list(feature_df.columns)}")
        
        return feature_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be extracted."""
        # Extract features from a dummy beat to get feature names
        dummy_beat = np.random.randn(360)  # 1 second at 360 Hz
        dummy_features = self.extract_beat_features(dummy_beat)
        dummy_rr_features = self.extract_rr_features(np.array([0.8, 0.9, 1.0]))
        
        all_feature_names = list(dummy_features.keys()) + list(dummy_rr_features.keys())
        return all_feature_names


def load_and_extract_features(beats_path: str = "data/processed/beats.npy",
                            labels_path: str = "data/processed/labels.npy",
                            output_dir: str = "data/processed/") -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load beats and extract features.
    
    Args:
        beats_path: Path to saved beats array
        labels_path: Path to saved labels array
        output_dir: Directory to save extracted features
        
    Returns:
        features_df: DataFrame with extracted features
        labels: Corresponding labels
    """
    print("Loading beat data...")
    beats = np.load(beats_path)
    labels = np.load(labels_path)
    
    print(f"Loaded {len(beats)} beats with {len(beats[0])} samples each")
    
    # Initialize feature extractor
    extractor = ECGFeatureExtractor(sampling_rate=360.0)
    
    # Extract features
    features_df = extractor.extract_features_from_beats(beats)
    
    # Save features
    features_path = f"{output_dir}/features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"Saved features to {features_path}")
    
    # Also save as numpy arrays for faster loading
    np.save(f"{output_dir}/features.npy", features_df.values)
    np.save(f"{output_dir}/feature_names.npy", features_df.columns.values)
    
    return features_df, labels


def main():
    """Example usage of feature extraction."""
    import os
    
    # Ensure processed data exists
    if not os.path.exists("data/processed/beats.npy"):
        print("No processed beat data found. Run data_loader.py first.")
        return
    
    # Extract features
    features_df, labels = load_and_extract_features()
    
    print(f"\nFeature extraction complete!")
    print(f"Features shape: {features_df.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"\nFeature statistics:")
    print(features_df.describe())


if __name__ == "__main__":
    main()