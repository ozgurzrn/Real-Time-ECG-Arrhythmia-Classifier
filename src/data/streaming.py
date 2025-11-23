import numpy as np
from scipy import signal
from collections import deque

class StreamFilter:
    """
    Real-time causal filter using Second-Order Sections (SOS).
    Maintains internal state (zi) to process data chunk-by-chunk.
    """
    def __init__(self, fs=360, lowcut=0.5, highcut=50.0, order=2):
        self.fs = fs
        nyquist = 0.5 * fs
        
        # Design causal filter (Butterworth)
        # Note: We use 'sos' output for numerical stability
        self.sos = signal.butter(order, [lowcut/nyquist, highcut/nyquist], 
                               btype='band', output='sos')
        
        # Initialize filter state
        self.zi = signal.sosfilt_zi(self.sos)
        
    def process(self, chunk):
        """
        Process a new chunk of data.
        Args:
            chunk: 1D numpy array of new samples
        Returns:
            filtered_chunk: 1D numpy array of filtered samples
        """
        # Apply filter and update state
        filtered_chunk, self.zi = signal.sosfilt(self.sos, chunk, zi=self.zi)
        return filtered_chunk

class StreamNormalizer:
    """
    Online normalizer using a running buffer.
    Calculates mean and std from the last N seconds of data.
    """
    def __init__(self, fs=360, window_seconds=5.0):
        self.limit = int(fs * window_seconds)
        self.buffer = deque(maxlen=self.limit)
        
        # Fallback stats (if buffer is empty)
        self.running_mean = 0.0
        self.running_std = 1.0
        
    def process(self, chunk):
        """
        Normalize a chunk based on history.
        Args:
            chunk: 1D numpy array of new samples
        Returns:
            normalized_chunk: 1D numpy array
        """
        normalized_chunk = np.zeros_like(chunk, dtype=float)
        
        # Process sample by sample to be strictly causal
        # (Optimization: could process in small blocks if chunk is large)
        for i, x in enumerate(chunk):
            self.buffer.append(x)
            
            # Calculate stats from known history
            if len(self.buffer) > 10:
                self.running_mean = np.mean(self.buffer)
                self.running_std = np.std(self.buffer) + 1e-6
            
            normalized_chunk[i] = (x - self.running_mean) / self.running_std
            
        return normalized_chunk

class RealTimePipeline:
    """
    Wrapper for the full real-time pipeline.
    """
    def __init__(self, fs=360):
        self.filter = StreamFilter(fs)
        self.normalizer = StreamNormalizer(fs)
        
    def process(self, chunk):
        # 1. Filter
        filtered = self.filter.process(chunk)
        
        # 2. Normalize
        normalized = self.normalizer.process(filtered)
        
        return normalized
