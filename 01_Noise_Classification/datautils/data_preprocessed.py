import os
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path


class PreprocessedDataset(Dataset):
    """Dataset for loading preprocessed features from pt files"""
    
    def __init__(self, index_file, is_eval=False):
        """
        Args:
            index_file (str): Path to the index JSON file
            is_eval (bool): Whether this is evaluation mode (no labels)
        """
        self.index_file = Path(index_file)
        self.is_eval = is_eval
        
        # Load index
        with open(self.index_file, 'r') as f:
            self.index_data = json.load(f)
        
        print(f"Loaded {len(self.index_data)} samples from {index_file}")
        
        # Verify files exist
        self._verify_files()
    
    def _verify_files(self):
        """Verify that all feature files exist"""
        missing_files = []
        valid_indices = []
        
        for i, entry in enumerate(self.index_data):
            feature_file = entry['feature_file']
            if os.path.exists(feature_file):
                valid_indices.append(i)
            else:
                missing_files.append(feature_file)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} feature files are missing")
            self.index_data = [self.index_data[i] for i in valid_indices]
            print(f"Using {len(self.index_data)} valid samples")
    
    def __len__(self):
        return len(self.index_data)
    
    def __getitem__(self, idx):
        entry = self.index_data[idx]
        feature_file = entry['feature_file']
        
        # Load features
        features = torch.load(feature_file, map_location='cpu')
        spec_db = features['spec_db']
        mfcc = features['mfcc'] 
        f0 = features['f0']
        
        if self.is_eval:
            # Return features and original path for evaluation
            original_path = entry['original_path']
            return spec_db, mfcc, f0, original_path
        else:
            # Return features and label for training
            label = entry['label']
            return spec_db, mfcc, f0, label
    
    def get_stats(self):
        """Get dataset statistics"""
        if self.is_eval:
            return {
                "total_samples": len(self.index_data),
                "subset": "eval"
            }
        
        # Count labels
        label_counts = {}
        for entry in self.index_data:
            label = entry['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            "total_samples": len(self.index_data),
            "label_distribution": label_counts,
            "subset": self.index_data[0].get('subset', 'unknown')
        }


class PreprocessedDatasetManager:
    """Manager class for handling multiple preprocessed datasets"""
    
    def __init__(self, preprocessed_dir):
        """
        Args:
            preprocessed_dir (str): Directory containing preprocessed data
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.available_subsets = self._find_available_subsets()
    
    def _find_available_subsets(self):
        """Find available dataset subsets"""
        subsets = []
        for subset in ['train', 'dev', 'eval']:
            # Try lowercase first, then capitalized
            index_file = self.preprocessed_dir / f"{subset}_index.json"
            if not index_file.exists():
                index_file = self.preprocessed_dir / f"{subset.capitalize()}_index.json"
            if index_file.exists():
                subsets.append(subset)
        return subsets
    
    def get_dataset(self, subset, is_eval=None):
        """
        Get dataset for a specific subset
        
        Args:
            subset (str): 'train', 'dev', or 'eval'
            is_eval (bool): Override evaluation mode. If None, auto-detect based on subset
        
        Returns:
            PreprocessedDataset
        """
        if subset not in self.available_subsets:
            raise ValueError(f"Subset '{subset}' not available. Available: {self.available_subsets}")
        
        # Try lowercase first, then capitalized
        index_file = self.preprocessed_dir / f"{subset}_index.json"
        if not index_file.exists():
            index_file = self.preprocessed_dir / f"{subset.capitalize()}_index.json"
        
        if is_eval is None:
            is_eval = (subset == 'eval')
        
        return PreprocessedDataset(index_file, is_eval=is_eval)
    
    def get_info(self):
        """Get information about all available datasets"""
        info = {
            "preprocessed_dir": str(self.preprocessed_dir),
            "available_subsets": self.available_subsets,
            "datasets": {}
        }
        
        for subset in self.available_subsets:
            try:
                dataset = self.get_dataset(subset)
                info["datasets"][subset] = dataset.get_stats()
            except Exception as e:
                info["datasets"][subset] = {"error": str(e)}
        
        return info


def verify_preprocessed_data(preprocessed_dir):
    """Utility function to verify preprocessed data integrity"""
    manager = PreprocessedDatasetManager(preprocessed_dir)
    info = manager.get_info()
    
    print("Preprocessed Data Verification")
    print("=" * 50)
    print(f"Directory: {info['preprocessed_dir']}")
    print(f"Available subsets: {info['available_subsets']}")
    
    for subset, stats in info['datasets'].items():
        print(f"\n{subset.upper()} Dataset:")
        if 'error' in stats:
            print(f"  Error: {stats['error']}")
        else:
            print(f"  Total samples: {stats['total_samples']}")
            if 'label_distribution' in stats:
                print(f"  Label distribution: {stats['label_distribution']}")
    
    return info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify preprocessed dataset")
    parser.add_argument("--preprocessed_dir", type=str, required=True,
                       help="Directory containing preprocessed data")
    args = parser.parse_args()
    
    verify_preprocessed_data(args.preprocessed_dir)