import numpy as np
import os
from preprocess import process_record
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

# AAMI Class Mapping
AAMI_MAPPING = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

CLASS_INT_MAPPING = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

def make_dataset(data_dir='data/raw', output_dir='data/processed'):
    """
    Generates the dataset from raw records with PATIENT-LEVEL split.
    This ensures test patients are completely unseen during training.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    records = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.dat')]
    records = sorted(list(set(records))) # Unique records
    
    print(f"Processing {len(records)} records...")
    print(f"Using PATIENT-LEVEL split (not random beat split)")
    
    # Sort records by number for reproducible split
    records = sorted(records, key=lambda x: int(x))
    
    # Split patients: 80% train, 20% test
    split_idx = int(len(records) * 0.8)
    train_records = records[:split_idx]  # First 80% of patients
    test_records = records[split_idx:]   # Last 20% of patients
    
    print(f"Train patients ({len(train_records)}): {train_records[:5]}...{train_records[-2:]}")
    print(f"Test patients ({len(test_records)}): {test_records}")
    
    # Process train patients
    train_beats = []
    train_labels = []
    
    for record in train_records:
        try:
            beats, labels = process_record(record, data_dir)
            
            # Map labels
            mapped_labels = [AAMI_MAPPING.get(l, 'Q') for l in labels]
            
            train_beats.append(beats)
            train_labels.extend(mapped_labels)
            
        except Exception as e:
            print(f"Error processing train record {record}: {e}")
    
    # Process test patients
    test_beats = []
    test_labels = []
    
    for record in test_records:
        try:
            beats, labels = process_record(record, data_dir)
            
            # Map labels
            mapped_labels = [AAMI_MAPPING.get(l, 'Q') for l in labels]
            
            test_beats.append(beats)
            test_labels.extend(mapped_labels)
            
        except Exception as e:
            print(f"Error processing test record {record}: {e}")
            
    # Concatenate
    X_train = np.vstack(train_beats)
    y_train = np.array([CLASS_INT_MAPPING[l] for l in train_labels])
    
    X_test = np.vstack(test_beats)
    y_test = np.array([CLASS_INT_MAPPING[l] for l in test_labels])
    
    print(f"\nTrain set: {len(X_train)} beats from {len(train_records)} patients")
    print(f"Train class distribution: {Counter(y_train)}")
    print(f"\nTest set: {len(X_test)} beats from {len(test_records)} patients")
    print(f"Test class distribution: {Counter(y_test)}")
    
    # Save raw split (imbalanced)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print("\nSaved raw dataset with patient-level split.")
    
    # Optional: SMOTE for training data
    # Only apply SMOTE to training data to avoid data leakage
    print("\nApplying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"Resampled training class distribution: {Counter(y_train_res)}")
    
    np.save(os.path.join(output_dir, 'X_train_res.npy'), X_train_res)
    np.save(os.path.join(output_dir, 'y_train_res.npy'), y_train_res)
    
    print("Saved resampled dataset.")

if __name__ == "__main__":
    make_dataset()
