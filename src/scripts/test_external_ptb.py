"""
Test classifier on PTB Diagnostic ECG Database
Different from MIT-BIH - tests out-of-distribution generalization
"""

import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append('src')

# Force UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# PTB Database info
# Available at: https://physionet.org/content/ptbdb/1.0.0/
# 549 records from 290 patients

# Sample PTB records to test (diverse conditions)
PTB_TEST_RECORDS = [
    ('patient001/s0010_re', 'Myocardial infarction', 'V'),
    ('patient002/s0015lre', 'Bundle branch block', 'N'),
    ('patient003/s0020are', 'Cardiomyopathy', 'N'),
    ('patient004/s0015lre', 'Dysrhythmia', 'V'),
    ('patient005/s0018_re', 'Myocardial infarction', 'V'),
    ('patient016/s0069lre', 'Healthy control', 'N'),
    ('patient020/s0101lre', 'Healthy control', 'N'),
    ('patient094/s0418lre', 'myocardial infarction', 'V'),
]

def download_ptb_record(record_path, output_dir='external_test_samples/ptb'):
    """Download PTB record from PhysioNet"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Downloading {record_path}...")
        
        # Split path into patient dir and record name
        # e.g. patient001/s0010_re -> patient001, s0010_re
        if '/' in record_path:
            patient_dir, record_name = record_path.split('/')
            pn_dir = f'ptbdb/1.0.0/{patient_dir}'
        else:
            record_name = record_path
            pn_dir = 'ptbdb/1.0.0'
            
        # Download from PhysioNet
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
        
        # PTB database has 15 leads, we'll use lead II (index 1) to match MIT-BIH
        # PTB sampling rate is 1000Hz vs MIT-BIH 360Hz
        signal = record.p_signal[:, 1]  # Lead II
        fs = record.fs  # 1000 Hz
        
        # Downsample to 360Hz to match our training data
        from scipy import signal as sp_signal
        downsample_factor = int(fs / 360)
        signal_360hz = sp_signal.decimate(signal, downsample_factor)
        
        # Take first 30 seconds at 360Hz = 10,800 samples
        signal_30s = signal_360hz[:10800]
        
        # Save as CSV
        record_name = record_path.replace('/', '_')
        output_path = f'{output_dir}/{record_name}.csv'
        
        df = pd.DataFrame(signal_30s, columns=['signal'])
        df.to_csv(output_path, index=False)
        
        print(f"✓ Saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"✗ Error downloading {record_path}: {e}")
        return None

def test_external_record(csv_path, record_name, description, expected_class):
    """Test classifier on external data"""
    from app import process_and_predict
    from collections import Counter
    
    print(f"\n{'='*70}")
    print(f"Testing PTB Record: {record_name}")
    print(f"Condition: {description}")
    print(f"Expected class: {expected_class}")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(csv_path)
    signal_data = df.iloc[:, 0].values
    
    # Process
    results, clean_signal, peaks, quality_metrics, pacemaker_info, rhythm_metrics = process_and_predict(signal_data)
    
    # Analyze
    class_counts = Counter([r['pred_idx'] for r in results])
    total = len(results)
    
    class_names = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
    
    print(f"\n📊 Results ({total} beats):")
    for idx in sorted(class_counts.keys()):
        count = class_counts[idx]
        pct = count/total*100
        marker = "👉" if class_names[idx] == expected_class else "  "
        print(f"  {marker} {class_names[idx]}: {count:4d} ({pct:5.1f}%)")
    
    # Quality
    print(f"\n🔍 Signal Quality: {quality_metrics['rating']} ({quality_metrics['quality_score']:.0f}/100)")
    print(f"   SNR: {quality_metrics['snr_db']:.1f} dB")
    
    # Confidence
    confidences = [r['confidence'] for r in results]
    avg_conf = np.mean(confidences)
    print(f"\n💯 Average Confidence: {avg_conf:.2%}")
    
    # Result
    dominant_class_idx = max(class_counts, key=class_counts.get)
    predicted_class = class_names[dominant_class_idx]
    
    match = "✅ PASS" if predicted_class == expected_class else "❌ FAIL"
    print(f"\n{match} - Predicted: {predicted_class}, Expected: {expected_class}")
    
    return {
        'record': record_name,
        'condition': description,
        'expected': expected_class,
        'predicted': predicted_class,
        'match': predicted_class == expected_class,
        'total_beats': total,
        'avg_confidence': avg_conf,
        'quality': quality_metrics['rating']
    }

if __name__ == "__main__":
    print("="*70)
    print("EXTERNAL VALIDATION - PTB Diagnostic ECG Database")
    print("Testing on completely different data source")
    print("="*70)
    
    print("\n📥 Downloading PTB records from PhysioNet...")
    
    test_files = {}
    for record_path, desc, expected in PTB_TEST_RECORDS:
        csv_path = download_ptb_record(record_path)
        if csv_path:
            test_files[record_path] = (csv_path, desc, expected)
    
    print(f"\n🧪 Testing {len(test_files)} PTB records...\n")
    
    results = []
    for record_path, (csv_path, desc, expected) in test_files.items():
        result = test_external_record(csv_path, record_path, desc, expected)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("EXTERNAL VALIDATION SUMMARY - PTB Database")
    print("="*70)
    
    passed = sum(1 for r in results if r['match'])
    total_tests = len(results)
    
    print(f"\nOverall Accuracy: {passed}/{total_tests} ({passed/total_tests*100:.1f}%)")
    print("\nNote: PTB database has:")
    print("  - Different patient population")
    print("  - Different recording equipment")
    print("  - Different sampling rate (1000Hz downsampled to 360Hz)")
    print("  - Different lead configuration (15-lead)")
    print("\nThis tests TRUE out-of-distribution generalization!")
    
    print("\n📋 Detailed Results:")
    print(f"{'Record':<30} {'Condition':<25} {'Expected':<10} {'Predicted':<10} {'Match'}")
    print("-" * 90)
    
    for r in results:
        match_symbol = "✅" if r['match'] else "❌"
        print(f"{r['record']:<30} {r['condition']:<25} {r['expected']:<10} {r['predicted']:<10} {match_symbol}")
    
    print("\n" + "="*70)
    print(f"External validation complete: {passed}/{total_tests} passed")
    print("="*70)
