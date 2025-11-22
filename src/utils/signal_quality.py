import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

def calculate_snr(signal_data):
    """
    Calculate Signal-to-Noise Ratio (SNR) for ECG signal.
    Higher SNR = better quality signal.
    """
    # Estimate signal power (using variance of signal)
    signal_power = np.var(signal_data)
    
    # Estimate noise power (high-frequency component)
    # Use high-pass filter to isolate noise
    b, a = scipy_signal.butter(3, 0.5, btype='high', fs=360)
    noise = scipy_signal.filtfilt(b, a, signal_data)
    noise_power = np.var(noise)
    
    # Avoid division by zero
    if noise_power == 0:
        return float('inf')
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

def detect_baseline_wander(signal_data, fs=360):
    """
    Detect baseline wander in ECG signal.
    Returns True if significant baseline wander is detected.
    """
    # Use low-pass filter to extract baseline
    b, a = scipy_signal.butter(1, 0.5, btype='low', fs=fs)
    baseline = scipy_signal.filtfilt(b, a, signal_data)
    
    # Check if baseline deviation is significant
    baseline_std = np.std(baseline)
    signal_std = np.std(signal_data)
    
    # If baseline variation is > 20% of signal variation, flag it
    if baseline_std > 0.2 * signal_std:
        return True
    return False

def detect_artifacts(signal_data):
    """
    Detect artifacts in ECG signal using statistical features.
    Returns artifact score (0-100, higher = more artifacts)
    """
    # Calculate statistical features
    kurt = kurtosis(signal_data)
    skewness = skew(signal_data)
    
    # Check for extreme values (clipping)
    values_range = np.max(signal_data) - np.min(signal_data)
    max_abs = max(abs(np.max(signal_data)), abs(np.min(signal_data)))
    
    # Artifact indicators:
    # 1. High kurtosis (sharp spikes)
    # 2. High skewness (asymmetry)
    # 3. Sudden jumps
    
    artifact_score = 0
    
    # Kurtosis check (normal ECG has kurtosis around 3)
    if abs(kurt) > 10:
        artifact_score += 30
    elif abs(kurt) > 5:
        artifact_score += 15
    
    # Skewness check
    if abs(skewness) > 2:
        artifact_score += 20
    elif abs(skewness) > 1:
        artifact_score += 10
    
    # Sudden jump detection (derivative check)
    diff = np.diff(signal_data)
    max_diff = np.max(np.abs(diff))
    if max_diff > 3 * np.std(diff):
        artifact_score += 30
    
    return min(artifact_score, 100)

def assess_signal_quality(signal_data, fs=360):
    """
    Comprehensive signal quality assessment.
    Returns dict with quality metrics and overall score.
    """
    snr = calculate_snr(signal_data)
    has_baseline_wander = detect_baseline_wander(signal_data, fs)
    artifact_score = detect_artifacts(signal_data)
    
    # Overall quality score (0-100, higher = better)
    quality_score = 100
    
    # Penalize based on SNR
    if snr < 10:
        quality_score -= 30
    elif snr < 15:
        quality_score -= 15
    
    # Penalize baseline wander
    if has_baseline_wander:
        quality_score -= 20
    
    # Penalize artifacts
    quality_score -= artifact_score * 0.5
    
    quality_score = max(0, quality_score)
    
    # Quality rating
    if quality_score >= 80:
        rating = "Excellent"
    elif quality_score >= 60:
        rating = "Good"
    elif quality_score >= 40:
        rating = "Fair"
    else:
        rating = "Poor"
    
    return {
        'snr_db': snr,
        'has_baseline_wander': has_baseline_wander,
        'artifact_score': artifact_score,
        'quality_score': quality_score,
        'rating': rating
    }

def detect_pacemaker_spikes(signal_data, fs=360):
    """
    Detect pacemaker spikes in ECG signal.
    Returns list of spike locations and confidence.
    """
    # Pacemaker spikes are very sharp, narrow, high-amplitude pulses
    # Typically < 2ms duration, high amplitude
    
    # High-pass filter to emphasize spikes
    b, a = scipy_signal.butter(4, 20, btype='high', fs=fs)
    filtered = scipy_signal.filtfilt(b, a, signal_data)
    
    # Find peaks that are:
    # 1. Very sharp (high derivative)
    # 2. Narrow (< 5 samples at 360 Hz = ~14ms)
    # 3. High amplitude relative to signal
    
    threshold = 3 * np.std(filtered)
    peaks, properties = scipy_signal.find_peaks(
        np.abs(filtered),
        height=threshold,
        distance=fs * 0.2,  # At least 200ms apart
        width=(1, 5)  # Very narrow
    )
    
    # Check if peaks are pacemaker-like
    pacemaker_spikes = []
    for peak in peaks:
        # Check sharpness (derivative)
        if peak > 5 and peak < len(signal_data) - 5:
            left_slope = signal_data[peak] - signal_data[peak-3]
            right_slope = signal_data[peak] - signal_data[peak+3]
            
            # Very steep slopes indicate pacemaker
            if abs(left_slope) > threshold and abs(right_slope) > threshold:
                pacemaker_spikes.append(peak)
    
    confidence = len(pacemaker_spikes) / max(len(peaks), 1) if len(peaks) > 0 else 0
    
    return {
        'spike_locations': pacemaker_spikes,
        'spike_count': len(pacemaker_spikes),
        'confidence': confidence,
        'has_pacemaker': len(pacemaker_spikes) > 10  # At least 10 spikes detected
    }
