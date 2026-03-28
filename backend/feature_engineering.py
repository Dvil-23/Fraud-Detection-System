import pandas as pd
import numpy as np

def build_feature_vector(amount, time):
    features = np.zeros(30)
    features[0] = time
    features[1] = amount
    return features.reshape(1,-1)
