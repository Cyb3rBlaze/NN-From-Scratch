import numpy as np

#---------------------
def sigmoid(input):
    return 1/(1+np.exp(-input))

def d_sigmoid(input):
    return sigmoid(input)*(1-sigmoid(input))

#---------------------
def tanh(input):
    return (np.exp(input)-np.exp(-input))/(np.exp(input)+np.exp(-input))

def d_tanh(input):
    return 1-(tanh(input)**2)