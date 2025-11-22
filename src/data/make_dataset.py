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
    Generates the dataset from raw records.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    records = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.dat')]
    records = sorted(list(set(records))) # Unique records
    
    all_beats = []
    all_labels = []
    
    print(f"Processing {len(records)} records...")
    
    for record in records:
        try:
            beats, labels = process_record(record, data_dir)
            
            # Map labels
            mapped_labels = [AAMI_MAPPING.get(l, 'Q') for l in labels]
            
            # Filter out beats that are not in our mapping (if any remain, though Q catches most)
            # Actually we mapped everything else to Q, so we keep them or discard Q?
            # Usually Q is 'Unknown', we might want to keep it or drop it. Let's keep it for now.
            
            all_beats.append(beats)
            all_labels.extend(mapped_labels)
            
        except Exception as e:
            print(f"Error processing record {record}: {e}")
            
    # Concatenate
    X = np.vstack(all_beats)
    y = np.array([CLASS_INT_MAPPING[l] for l in all_labels])
    
    print(f"Total beats: {len(X)}")
    print(f"Class distribution: {Counter(y)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save raw split (imbalanced)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print("Saved raw dataset.")
    
    # Optional: SMOTE for training data
    # Only apply SMOTE to training data to avoid data leakage
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"Resampled training class distribution: {Counter(y_train_res)}")
    
    np.save(os.path.join(output_dir, 'X_train_res.npy'), X_train_res)
    np.save(os.path.join(output_dir, 'y_train_res.npy'), y_train_res)
    
    print("Saved resampled dataset.")

if __name__ == "__main__":
    make_dataset()
