import numpy as np

def binary_cross_entropy_loss(logits, labels):
    return -(np.multiply(labels, np.log(logits)) + np.multiply(1-labels, np.log(1-logits))).mean()

def d_binary_cross_entropy_loss(logits, labels):
    return np.divide(logits-labels, np.multiply(logits, 1-logits))