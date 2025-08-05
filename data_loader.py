"""
data_loader.py

Load and preprocess MIT-BIH Arrhythmia Database files.
Handles .dat, .hea, and .atr files from PhysioNet format.

MIT-BIH Database contains:
- .dat: signal data (2 leads: MLII and V1/V2/V4/V5)  
- .hea: header with metadata
- .atr: annotations with beat labels

This script loads the raw signals and annotations, segments into beats,
and prepares data for feature extraction.
"""

import os
import numpy as np
import pandas as pd
import wfdb
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class MITBIHLoader:
    """Load and preprocess MIT-BIH Arrhythmia Database files."""
    
    def __init__(self, data_dir: str = "data/raw/"):
        self.data_dir = data_dir
        
        # MIT-BIH annotation codes to arrhythmia classes
        # Simplified mapping for binary/multi-class classification
        self.annotation_map = {
            'N': 'Normal',          # Normal beat
            'L': 'Normal',          # Left bundle branch block beat  
            'R': 'Normal',          # Right bundle branch block beat
            'A': 'Arrhythmia',      # Atrial premature beat
            'a': 'Arrhythmia',      # Aberrated atrial premature beat
            'J': 'Arrhythmia',      # Nodal (junctional) premature beat
            'S': 'Arrhythmia',      # Supraventricular premature beat
            'V': 'Arrhythmia',      # Premature ventricular contraction
            'F': 'Arrhythmia',      # Fusion of ventricular and normal beat
            'e': 'Arrhythmia',      # Atrial escape beat
            'j': 'Arrhythmia',      # Nodal (junctional) escape beat
            'E': 'Arrhythmia',      # Ventricular escape beat
            '/': 'Arrhythmia',      # Paced beat
            'f': 'Arrhythmia',      # Fusion of paced and normal beat
            'x': 'Arrhythmia',      # Non-conducted P-wave (blocked APB)
            'Q': 'Arrhythmia',      # Unclassifiable beat
            '|': 'Artifact',        # Isolated QRS-like artifact
        }
    
    def get_available_records(self) -> List[str]:
        """Get list of available MIT-BIH record numbers from data directory."""
        records = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.dat'):
                record_num = file.split('.')[0]
                # Verify corresponding .hea and .atr files exist
                if (os.path.exists(os.path.join(self.data_dir, f"{record_num}.hea")) and
                    os.path.exists(os.path.join(self.data_dir, f"{record_num}.atr"))):
                    records.append(record_num)
        return sorted(records)
    
    def load_record(self, record_name: str) -> Tuple[np.ndarray, Dict]:
        """
        Load a single MIT-BIH record.
        
        Args:
            record_name: Record identifier (e.g., '100', '101')
            
        Returns:
            signal: ECG signal data (samples x leads)
            metadata: Record metadata
        """
        record_path = os.path.join(self.data_dir, record_name)
        
        try:
            # Load signal and metadata
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal  # Shape: (samples, leads)
            
            metadata = {
                'fs': record.fs,  # Sampling frequency (usually 360 Hz)
                'sig_len': record.sig_len,
                'sig_name': record.sig_name,
                'units': record.units,
                'record_name': record_name
            }
            
            return signal, metadata
            
        except Exception as e:
            raise ValueError(f"Error loading record {record_name}: {str(e)}")
    
    def load_annotations(self, record_name: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load annotations for a MIT-BIH record.
        
        Args:
            record_name: Record identifier
            
        Returns:
            sample_indices: Sample indices of annotations
            symbols: Annotation symbols
        """
        record_path = os.path.join(self.data_dir, record_name)
        
        try:
            annotation = wfdb.rdann(record_path, 'atr')
            return annotation.sample, annotation.symbol
            
        except Exception as e:
            raise ValueError(f"Error loading annotations for {record_name}: {str(e)}")
    
    def segment_beats(self, signal: np.ndarray, ann_samples: np.ndarray, 
                     ann_symbols: List[str], window_size: int = 360,
                     lead_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment ECG signal into individual beats around R-peaks.
        
        Args:
            signal: ECG signal (samples x leads)
            ann_samples: Sample indices of annotations
            ann_symbols: Annotation symbols
            window_size: Beat window size in samples (default: 360 = 1 second at 360Hz)
            lead_idx: Which lead to use (0 for MLII, 1 for second lead)
            
        Returns:
            beat_segments: Array of beat segments (n_beats x window_size)
            labels: Corresponding labels for each beat
        """
        half_window = window_size // 2
        beats = []
        labels = []
        
        # Use specified lead
        if signal.ndim > 1:
            signal_lead = signal[:, lead_idx]
        else:
            signal_lead = signal
        
        for i, (sample_idx, symbol) in enumerate(zip(ann_samples, ann_symbols)):
            # Skip symbols not in our mapping
            if symbol not in self.annotation_map:
                continue
            
            # Extract beat window around annotation
            start_idx = int(sample_idx - half_window)
            end_idx = int(sample_idx + half_window)
            
            # Skip if window extends beyond signal boundaries
            if start_idx < 0 or end_idx >= len(signal_lead):
                continue
            
            beat_segment = signal_lead[start_idx:end_idx]
            
            # Ensure consistent window size
            if len(beat_segment) == window_size:
                beats.append(beat_segment)
                labels.append(self.annotation_map[symbol])
        
        return np.array(beats), np.array(labels)
    
    def load_all_data(self, max_records: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load and segment all available MIT-BIH records.
        
        Args:
            max_records: Maximum number of records to load (None for all)
            
        Returns:
            all_beats: Combined beat segments from all records
            all_labels: Combined labels
            record_info: List of record names processed
        """
        available_records = self.get_available_records()
        
        if max_records is not None:
            available_records = available_records[:max_records]
        
        print(f"Loading {len(available_records)} records: {available_records}")
        
        all_beats = []
        all_labels = []
        record_info = []
        
        for record_name in available_records:
            try:
                print(f"Processing record {record_name}...")
                
                # Load signal and annotations
                signal, metadata = self.load_record(record_name)
                ann_samples, ann_symbols = self.load_annotations(record_name)
                
                # Segment into beats
                beats, labels = self.segment_beats(signal, ann_samples, ann_symbols)
                
                if len(beats) > 0:
                    all_beats.append(beats)
                    all_labels.append(labels)
                    record_info.append(record_name)
                    print(f"  Extracted {len(beats)} beats")
                else:
                    print(f"  No valid beats found")
                    
            except Exception as e:
                print(f"  Error processing {record_name}: {str(e)}")
                continue
        
        if not all_beats:
            raise ValueError("No valid data found in any records")
        
        # Combine all beats and labels
        combined_beats = np.vstack(all_beats)
        combined_labels = np.hstack(all_labels)
        
        print(f"\nTotal beats loaded: {len(combined_beats)}")
        print(f"Label distribution:")
        unique_labels, counts = np.unique(combined_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count}")
        
        return combined_beats, combined_labels, record_info


def main():
    """Example usage of MITBIHLoader."""
    loader = MITBIHLoader("data/raw/")
    
    # Check available records
    records = loader.get_available_records()
    print(f"Available records: {records}")
    
    if records:
        # Load first few records for testing
        beats, labels, info = loader.load_all_data(max_records=3)
        print(f"Loaded {len(beats)} beats from records: {info}")
        
        # Save processed data
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/beats.npy", beats)
        np.save("data/processed/labels.npy", labels)
        print("Saved processed data to data/processed/")


if __name__ == "__main__":
    main()