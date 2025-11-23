import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append('src')

from data.streaming import RealTimePipeline
from data.preprocess import denoise_signal

def simulate_real_time(record_path, chunk_size=1):
    """
    Simulate real-time processing by feeding data chunk-by-chunk.
    """
    print(f"Loading record: {record_path}")
    
    # Load data (simulating a stream source)
    if record_path.endswith('.csv'):
        df = pd.read_csv(record_path)
        raw_signal = df.iloc[:, 0].values if df.shape[1] == 1 else df.iloc[:, 1].values
    else:
        # Generate synthetic if file not found
        t = np.linspace(0, 10, 3600)
        raw_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 50 * t) # 1Hz + 50Hz noise
    
    # 1. Offline Batch Processing (The "Cheating" Way)
    print("Running Offline Batch Processing (filtfilt)...")
    # Note: We manually normalize here to match the logic
    offline_filtered = denoise_signal(raw_signal, 360)
    offline_norm = (offline_filtered - np.mean(offline_filtered)) / (np.std(offline_filtered) + 1e-6)
    
    # 2. Real-Time Stream Processing (The "Honest" Way)
    print(f"Running Real-Time Stream Processing (chunk_size={chunk_size})...")
    pipeline = RealTimePipeline(fs=360)
    online_output = []
    
    # Feed data in chunks
    for i in range(0, len(raw_signal), chunk_size):
        chunk = raw_signal[i:i+chunk_size]
        processed_chunk = pipeline.process(chunk)
        online_output.extend(processed_chunk)
        
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(raw_signal)} samples...", end='\r')
            
    online_output = np.array(online_output)
    print("\nDone.")
    
    # 3. Compare
    # Note: Online filter has phase delay, so peaks will be shifted.
    # Online normalizer takes time to stabilize.
    
    plt.figure(figsize=(12, 6))
    
    # Plot first 5 seconds (Stabilization phase)
    plt.subplot(2, 1, 1)
    plt.title("First 5 Seconds (Stabilization Phase)")
    limit = 360 * 5
    plt.plot(offline_norm[:limit], label='Offline (Non-Causal)', alpha=0.7)
    plt.plot(online_output[:limit], label='Online (Causal)', alpha=0.7)
    plt.legend()
    plt.grid(True)
    
    # Plot seconds 10-15 (Stable phase)
    plt.subplot(2, 1, 2)
    plt.title("Seconds 10-15 (Stable Phase)")
    start = 360 * 10
    end = 360 * 15
    if len(raw_signal) > end:
        plt.plot(range(start, end), offline_norm[start:end], label='Offline', alpha=0.7)
        plt.plot(range(start, end), online_output[start:end], label='Online', alpha=0.7)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('realtime_comparison.png')
    print("Comparison plot saved to 'realtime_comparison.png'")
    
    # Calculate error after stabilization
    if len(raw_signal) > 3600:
        stable_online = online_output[3600:]
        stable_offline = offline_norm[3600:]
        
        # Cross-correlation to find phase delay
        correlation = np.correlate(stable_offline, stable_online, mode='full')
        delay = correlation.argmax() - (len(stable_online) - 1)
        print(f"\nPhase Delay Detected: {delay} samples ({delay/360*1000:.1f} ms)")
        
        # Shift online signal to match for error calc
        if delay > 0:
            stable_online_shifted = stable_online[:-delay]
            stable_offline_shifted = stable_offline[delay:]
            mse = np.mean((stable_online_shifted - stable_offline_shifted)**2)
            print(f"MSE after delay correction: {mse:.4f}")

if __name__ == "__main__":
    # Use sample file if exists
    sample_file = "sample_input.csv"
    if not os.path.exists(sample_file):
        print("Generating synthetic data...")
        sample_file = "synthetic_test"
        
    simulate_real_time(sample_file, chunk_size=1)
