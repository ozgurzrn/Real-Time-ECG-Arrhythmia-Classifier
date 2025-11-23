"""
Proper validation test suite using actual test patients (220-234).
These patients were NOT in the training set (100-219).
"""

import pandas as pd
import numpy as np
import wfdb
from pathlib import Path
import sys
sys.path.append('src')

# Force UTF-8 encoding for emoji support on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Test cases from the ACTUAL test set (patients 220-234)
TEST_CASES = [
    (220, "Normal with occasional PVCs", "N"),
    (221, "Normal sinus rhythm", "N"),
    (222, "Normal", "N"),
    (223, "Normal sinus rhythm", "N"),
    (224, "Normal", "N"),
    (228, "Normal", "N"),
    (230, "Fusion beats and PVCs", "F"),
    (231, "Normal sinus rhythm", "N"),
    (232, "Normal", "N"),
    (233, "Ventricular ectopy", "V"),
    (234, "Paced rhythm", "Q"),
]

def create_test_csv(record_num, output_dir='test_samples_correct'):
    """Create CSV from MIT-BIH record"""
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Read record
        record = wfdb.rdrecord(f'data/raw/{record_num}')
        
        # Get first 30 seconds (10,800 samples at 360 Hz)
        signal = record.p_signal[:10800, 0]
        
        # Save as CSV
        df = pd.DataFrame(signal, columns=['signal'])
        output_path = f'{output_dir}/record_{record_num}.csv'
        df.to_csv(output_path, index=False)
        
        print(f"✓ Created {output_path}")
        return output_path
        
    except Exception as e:
        print(f"✗ Error creating record {record_num}: {e}")
        return None

def test_classifier_on_record(csv_path, record_num, description, expected_class):
    """Test classifier on a specific record"""
    from app import process_and_predict
    
    print(f"\n{'='*70}")
    print(f"Testing Record {record_num}: {description}")
    print(f"Expected dominant: {expected_class}")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(csv_path)
    signal_data = df.iloc[:, 0].values
    
    # Process
    results, clean_signal, peaks, quality_metrics, pacemaker_info, rhythm_metrics = process_and_predict(signal_data)
    
    # Analyze results
    from collections import Counter
    class_counts = Counter([r['pred_idx'] for r in results])
    total = len(results)
    
    class_names = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
    
    print(f"\n📊 Results ({total} beats classified):")
    for idx in sorted(class_counts.keys()):
        count = class_counts[idx]
        pct = count/total*100
        marker = "👉" if class_names[idx] == expected_class else "  "
        print(f"  {marker} {class_names[idx]}: {count:4d} ({pct:5.1f}%)")
    
    # Signal quality
    print(f"\n🔍 Signal Quality:")
    print(f"  Overall: {quality_metrics['rating']} ({quality_metrics['quality_score']:.0f}/100)")
    print(f"  SNR: {quality_metrics['snr_db']:.1f} dB")
    print(f"  Artifacts: {quality_metrics['artifact_score']:.0f}/100")
    
    # Pacemaker
    if pacemaker_info['has_pacemaker']:
        print(f"\n🔋 Pacemaker: YES ({pacemaker_info['spike_count']} spikes, {pacemaker_info['confidence']*100:.0f}% confidence)")
    else:
        print(f"\n🔋 Pacemaker: NO")
    
    # Rhythm Analysis
    if rhythm_metrics['status'] == 'Complete':
        print(f"\n💓 Rhythm Analysis:")
        print(f"  Heart Rate: {rhythm_metrics['heart_rate_bpm']:.0f} bpm ({rhythm_metrics['rate_classification']})")
        print(f"  Regularity: {'Irregular' if rhythm_metrics['is_irregular'] else 'Regular'} (CV={rhythm_metrics['rr_variability_cv']:.2f})")
        print(f"  Pattern: {rhythm_metrics['rhythm_pattern']}")
        if rhythm_metrics['interpretation']:
            print(f"  Clinical: {' | '.join(rhythm_metrics['interpretation'])}")
    
    # Confidence analysis
    confidences = [r['confidence'] for r in results]
    avg_conf = np.mean(confidences)
    low_conf = sum(1 for c in confidences if c < 0.6)
    
    print(f"\n💯 Confidence:")
    print(f"  Average: {avg_conf:.2%}")
    print(f"  Low confidence (<60%): {low_conf}/{total}")
    
    # Determine if prediction matches expectation
    dominant_class_idx = max(class_counts, key=class_counts.get)
    predicted_class = class_names[dominant_class_idx]
    
    match = "✅ PASS" if predicted_class == expected_class else "❌ FAIL"
    print(f"\n{match} - Predicted: {predicted_class}, Expected: {expected_class}")
    
    return {
        'record': record_num,
        'description': description,
        'expected': expected_class,
        'predicted': predicted_class,
        'match': predicted_class == expected_class,
        'total_beats': total,
        'class_distribution': dict(class_counts),
        'avg_confidence': avg_conf,
        'signal_quality': quality_metrics['rating'],
        'has_pacemaker': pacemaker_info['has_pacemaker']
    }

if __name__ == "__main__":
    print("="*70)
    print("ECG Classifier - PROPER Validation on Test Patients (220-234)")
    print("These patients were NOT in training set (100-219)")
    print("="*70)
    
    # Create test CSVs
    print("\n📁 Creating test samples from TRUE test set...")
    test_files = {}
    for record_num, desc, expected in TEST_CASES:
        csv_path = create_test_csv(record_num)
        if csv_path:
            test_files[record_num] = (csv_path, desc, expected)
    
    # Run tests
    print(f"\n🧪 Running tests on {len(test_files)} TEST SET records...\n")
    
    results = []
    for record_num, (csv_path, desc, expected) in test_files.items():
        result = test_classifier_on_record(csv_path, record_num, desc, expected)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Test Set Performance (Patients 220-234)")
    print("="*70)
    
    passed = sum(1 for r in results if r['match'])
    total_tests = len(results)
    
    print(f"\nOverall Accuracy: {passed}/{total_tests} ({passed/total_tests*100:.1f}%)")
    
    print("\n📋 Detailed Results:")
    print(f"{'Record':<8} {'Expected':<10} {'Predicted':<10} {'Match':<8} {'Beats':<8} {'Confidence':<12} {'Quality':<12}")
    print("-" * 90)
    
    for r in results:
        match_symbol = "✅" if r['match'] else "❌"
        print(f"{r['record']:<8} {r['expected']:<10} {r['predicted']:<10} {match_symbol:<8} {r['total_beats']:<8} {r['avg_confidence']:>10.1%}  {r['signal_quality']:<12}")
    
    print("\n" + "="*70)
    print(f"TRUE Test Set Validation: {passed}/{total_tests} tests passed ({passed/total_tests*100:.1f}%)")
    print("This represents unseen patients - the model was NOT trained on these!")
    print("="*70)
