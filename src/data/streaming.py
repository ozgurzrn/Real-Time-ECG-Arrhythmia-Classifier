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
    Online normalizer using Welford's Algorithm for running mean and variance.
    This is O(1) per sample and extremely fast.
    """
    def __init__(self, fs=360, window_seconds=5.0):
        # We use an exponential moving average (EMA) approach which effectively
        # represents a window of 'window_seconds'
        # alpha ~= 2 / (N + 1)
        N = int(fs * window_seconds)
        self.alpha = 2 / (N + 1)
        
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False
        
    def process(self, chunk):
        """
        Normalize a chunk based on running stats.
        Args:
            chunk: 1D numpy array of new samples
        Returns:
            normalized_chunk: 1D numpy array
        """
        normalized_chunk = np.zeros_like(chunk, dtype=float)
        
        # If this is the very first chunk, initialize with its stats to avoid initial spike
        if not self.initialized and len(chunk) > 0:
            self.mean = np.mean(chunk)
            self.var = np.var(chunk)
            self.initialized = True
            
        # Optimization: If chunk is large, we can vectorize the update 
        # but for true simulation we iterate. However, pure Python loop is slow.
        # We will use a hybrid approach:
        # Update stats using the whole chunk (batch update) for speed in this demo,
        # but apply it causally if possible.
        
        # FAST APPROXIMATION for Python Demo:
        # Instead of O(N) loop, we update stats using the chunk's properties
        # This is slightly "batchy" within the chunk but much faster.
        # For a 10-second chunk, it's fine.
        
        # However, to be strictly causal sample-by-sample in Python is too slow.
        # Let's implement a Numba-friendly or vectorized approach.
        
        # Vectorized Exponential Moving Average
        # This is hard to vectorize perfectly without a loop.
        # Let's stick to the loop but optimize the math.
        
        # Actually, for the demo, let's just use the loop but REMOVE the np.mean/std call
        # Welford's algorithm is simple scalar math.
        
        for i, x in enumerate(chunk):
            # EMA Update
            diff = x - self.mean
            incr = self.alpha * diff
            self.mean += incr
            
            # Update variance
            # var_new = (1-alpha) * var_old + alpha * (x - mean_new) * (x - mean_old)
            # Simplified EMA for variance:
            self.var = (1 - self.alpha) * self.var + self.alpha * (diff * (x - self.mean))
            
            std = np.sqrt(self.var) + 1e-6
            normalized_chunk[i] = (x - self.mean) / std
            
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
