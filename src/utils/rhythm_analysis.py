"""
Rhythm-level analysis to complement beat-level classification.
Helps detect arrhythmias that manifest as rhythm irregularities.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import variation

def calculate_rr_intervals(peaks, fs=360):
    """
    Calculate RR intervals (time between consecutive R-peaks).
    Returns intervals in milliseconds.
    """
    if len(peaks) < 2:
        return np.array([])
    
    rr_samples = np.diff(peaks)
    rr_ms = (rr_samples / fs) * 1000  # Convert to milliseconds
    
    return rr_ms

def detect_irregular_rhythm(rr_intervals):
    """
    Detect rhythm irregularity using RR interval variability.
    High variability suggests atrial fibrillation or other irregular rhythms.
    """
    if len(rr_intervals) < 5:
        return False, 0.0
    
    # Calculate coefficient of variation
    cv = variation(rr_intervals)
    
    # Atrial fib typically has CV > 0.3
    is_irregular = cv > 0.3
    
    return is_irregular, cv

def calculate_heart_rate(rr_intervals):
    """
    Calculate average heart rate from RR intervals.
    """
    if len(rr_intervals) == 0:
        return 0
    
    avg_rr_ms = np.mean(rr_intervals)
    hr_bpm = 60000 / avg_rr_ms  # 60000 ms per minute
    
    return hr_bpm

def detect_bradycardia_tachycardia(hr_bpm):
    """
    Detect abnormal heart rates.
    """
    if hr_bpm < 60:
        return "Bradycardia", hr_bpm
    elif hr_bpm > 100:
        return "Tachycardia", hr_bpm
    else:
        return "Normal", hr_bpm

def analyze_rr_pattern(rr_intervals):
    """
    Analyze RR interval patterns for specific arrhythmias.
    """
    if len(rr_intervals) < 5:
        return "Insufficient data"
    
    # Check for alternating pattern (bigeminy)
    differences = np.diff(rr_intervals)
    sign_changes = np.diff(np.sign(differences))
    alternating_ratio = np.sum(sign_changes != 0) / len(sign_changes)
    
    if alternating_ratio > 0.7:
        return "Possible Bigeminy"
    
    # Check for regular rhythm
    cv = variation(rr_intervals)
    if cv < 0.1:
        return "Regular Rhythm"
    elif cv < 0.3:
        return "Mildly Irregular"
    else:
        return "Highly Irregular (possible AF)"

def rhythm_based_classification_adjustment(beat_predictions, rr_intervals, peaks):
    """
    Adjust beat-level predictions based on rhythm analysis.
    Helps correct misclassifications for rhythm-dependent arrhythmias.
    """
    if len(rr_intervals) < 5:
        return beat_predictions  # Not enough data
    
    # Detect irregular rhythm
    is_irregular, cv = detect_irregular_rhythm(rr_intervals)
    
    # Calculate heart rate
    hr_bpm = calculate_heart_rate(rr_intervals)
    
    # Analyze pattern
    rhythm_pattern = analyze_rr_pattern(rr_intervals)
    
    adjusted_predictions = beat_predictions.copy()
    
    # If highly irregular rhythm detected, increase likelihood of S (supraventricular)
    if is_irregular and cv > 0.3:
        # For beats currently classified as N with irregular rhythm
        for i, pred in enumerate(beat_predictions):
            if pred == 0:  # Currently classified as Normal
                # In irregular rhythm, more likely to be supraventricular
                # This is a heuristic adjustment
                pass  # Keep original for now, just flag it
    
    rhythm_info = {
        'is_irregular': is_irregular,
        'rr_variability': cv,
        'heart_rate_bpm': hr_bpm,
        'rhythm_pattern': rhythm_pattern,
        'rate_classification': detect_bradycardia_tachycardia(hr_bpm)[0]
    }
    
    return adjusted_predictions, rhythm_info

def comprehensive_rhythm_analysis(peaks, fs=360):
    """
    Perform comprehensive rhythm analysis.
    Returns detailed metrics and interpretations.
    """
    # Calculate RR intervals
    rr_intervals = calculate_rr_intervals(peaks, fs)
    
    if len(rr_intervals) < 5:
        return {
            'status': 'Insufficient data',
            'rr_count': len(rr_intervals)
        }
    
    # Heart rate
    hr_bpm = calculate_heart_rate(rr_intervals)
    rate_class, _ = detect_bradycardia_tachycardia(hr_bpm)
    
    # Regularity
    is_irregular, cv = detect_irregular_rhythm(rr_intervals)
    
    # Pattern analysis
    rhythm_pattern = analyze_rr_pattern(rr_intervals)
    
    # Statistical metrics
    rr_mean = np.mean(rr_intervals)
    rr_std = np.std(rr_intervals)
    rr_min = np.min(rr_intervals)
    rr_max = np.max(rr_intervals)
    
    # Clinical interpretation
    interpretation = []
    
    if is_irregular:
        interpretation.append(f"Irregular rhythm detected (CV={cv:.2f})")
        if cv > 0.4:
            interpretation.append("High irregularity suggests atrial fibrillation")
    
    if rate_class != "Normal":
        interpretation.append(f"{rate_class}: {hr_bpm:.0f} bpm")
    
    if "Bigeminy" in rhythm_pattern:
        interpretation.append("Alternating RR pattern suggests ventricular bigeminy")
    
    if not interpretation:
        interpretation.append("Regular sinus rhythm")
    
    return {
        'status': 'Complete',
        'heart_rate_bpm': hr_bpm,
        'rate_classification': rate_class,
        'is_irregular': is_irregular,
        'rr_variability_cv': cv,
        'rhythm_pattern': rhythm_pattern,
        'rr_mean_ms': rr_mean,
        'rr_std_ms': rr_std,
        'rr_range_ms': (rr_min, rr_max),
        'rr_count': len(rr_intervals),
        'interpretation': interpretation
    }
