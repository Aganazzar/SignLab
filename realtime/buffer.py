from collections import deque
import numpy as np


class FeatureBuffer:
    def __init__(self, max_len, feature_dim):
        self.buffer = deque(maxlen=max_len)
        self.feature_dim = feature_dim

    def add(self, features):
        # Ensure consistent feature dimension
        if len(features) < self.feature_dim:
            features = np.concatenate([features, np.zeros(self.feature_dim - len(features))])
        elif len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        self.buffer.append(features)

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen

    def get(self):
        return np.array(list(self.buffer))