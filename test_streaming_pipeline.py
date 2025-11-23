import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from data.streaming import StreamFilter, StreamNormalizer, RealTimePipeline

class TestStreamingPipeline(unittest.TestCase):
    def setUp(self):
        self.fs = 360
        self.duration = 10
        self.t = np.linspace(0, self.duration, self.duration * self.fs)
        # Create a synthetic signal: 1Hz sine wave + noise
        self.signal = np.sin(2 * np.pi * 1 * self.t) + 0.1 * np.random.randn(len(self.t))
        
    def test_stream_filter_shape(self):
        """Test that filter preserves input shape and processes chunks correctly"""
        f = StreamFilter(self.fs)
        chunk_size = 100
        chunk = self.signal[:chunk_size]
        output = f.process(chunk)
        
        self.assertEqual(output.shape, chunk.shape)
        self.assertEqual(output.dtype, float)
        
    def test_stream_filter_state_update(self):
        """Test that filter state updates after processing"""
        f = StreamFilter(self.fs)
        initial_zi = f.zi.copy()
        
        chunk = self.signal[:100]
        _ = f.process(chunk)
        
        # State should change
        self.assertFalse(np.array_equal(f.zi, initial_zi))
        
    def test_stream_normalizer_convergence(self):
        """Test that normalizer converges to 0 mean and 1 std"""
        norm = StreamNormalizer(self.fs, window_seconds=2.0)
        
        # Feed a constant signal (should normalize to 0)
        constant_signal = np.ones(1000) * 5.0
        output = norm.process(constant_signal)
        
        # Last few samples should be close to 0 (since mean approaches 5.0)
        # Note: std of constant signal is 0 (plus epsilon), so (5-5)/eps = 0
        self.assertTrue(np.abs(output[-1]) < 0.1)
        
        # Feed a sine wave
        sine_signal = np.sin(np.linspace(0, 20, 2000))
        output = norm.process(sine_signal)
        
        # Check stats of the last part of output
        # It won't be exactly 0/1 because it's a running buffer, but should be close
        recent_output = output[-500:]
        self.assertTrue(abs(np.mean(recent_output)) < 0.5)
        self.assertTrue(abs(np.std(recent_output) - 1.0) < 0.5)
        
    def test_pipeline_integration(self):
        """Test the full pipeline wrapper"""
        pipeline = RealTimePipeline(self.fs)
        
        # Process in chunks
        chunk_size = 360
        n_chunks = 5
        outputs = []
        
        for i in range(n_chunks):
            chunk = self.signal[i*chunk_size : (i+1)*chunk_size]
            out = pipeline.process(chunk)
            outputs.append(out)
            
        full_output = np.concatenate(outputs)
        self.assertEqual(len(full_output), chunk_size * n_chunks)

if __name__ == '__main__':
    unittest.main()
