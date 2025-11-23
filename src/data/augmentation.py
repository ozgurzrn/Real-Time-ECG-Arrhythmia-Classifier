import torch
import numpy as np

class RandomShift(object):
    """Randomly shift the signal left or right."""
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, sample):
        signal, label = sample
        shift = np.random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return signal.astype(np.float32), label
        
        # Shift and pad with zeros (or edge values)
        shifted = np.roll(signal, shift)
        if shift > 0:
            shifted[:shift] = 0
        else:
            shifted[shift:] = 0
            
        return shifted.astype(np.float32), label

class RandomScale(object):
    """Randomly scale the signal amplitude."""
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        signal, label = sample
        scale = np.random.uniform(self.min_scale, self.max_scale)
        return (signal * scale).astype(np.float32), label

class GaussianNoise(object):
    """Add random Gaussian noise."""
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, sample):
        signal, label = sample
        noise = np.random.normal(0, self.sigma, signal.shape)
        return (signal + noise).astype(np.float32), label

class Compose(object):
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
