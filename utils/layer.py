class Layer():
    def __init__(self, num_prev_neurons, num_neurons, activation="linear"):
        self.num_neurons = num_neurons
        self.num_prev_neurons = num_prev_neurons

        self.activation = activation

        self.weights = np.random.randn(num_prev_neurons, num_neurons)
        self.biases = np.array([Number(i-0.5) for i in np.random.rand(num_neurons)])
    
    def print_params(self):
        print("WEIGHTS:")
        print(self.weights)
        print("BIASES:")
        print(self.biases)

    def forward(self, input):
        lin = input @ self.weights + self.biases

        # storing original input passes for chain rule operations during backprop
        self.original_input = input
        self.lin_pass = lin

        if self.activation == "sigmoid":
            return sigmoid(lin)
        elif self.activation == "tanh":
            return tanh(lin)
        return lin

    def backward(self, prev_chain, lr):
        ''' chain rule:
        dz1_db = 1
        dz1_dw = original_input

        dout_dw = prev_chain*da1_dz1*dz1_dw -- prev_chain = dout_da2*da2_dz2*dz2_da1
        dout_db = prev_chain*da1_dz1*dz1_db -- prev_chain = dout_da2*da2_dz2*dz2_da1

        prev_chain is an array representing the chain rule computed values so far in the subsequent layer
        e.g.
        dout_dw1 = dout_dw3*dw3_dw1 + dout_dw4*dw4_dw1
        dout_db1 = dout_db3*db3_db1 + dout_db4*dw4_db1
        prev_chain would be [[dout_dw3, dout_dw4], [dout_db3, dout_db4]]
        '''
        da = lambda x: x
        if self.activation == "sigmoid":
            da = d_sigmoid
        elif self.activation == "tanh":
            da = d_tanh
        
        prev_chain_dw, prev_chain_db = prev_chain

        dout_dw =  (np.tile(self.original_input, (self.num_neurons, 1)).reshape((self.num_prev_neurons, -1)) @ (prev_chain_dw * da(self.lin_pass))).reshape((self.num_prev_neurons, self.num_neurons))
        dout_db = np.dot(np.tile(da(self.lin_pass), (prev_chain_db.shape[0], 1)).T, prev_chain_db)

        self.weights -= lr * dout_dw
        self.biases -= lr * dout_db

        ''' returning new prev_chain:
        output = a2(z2(a1(z1(input)))) -- e.g. two activation + linear operations

        dout_dw = dout_da2*da2_dz2*dz2_da1*da1_dz1*dz1_dw
        dout_dw = prev_chain*da1_dz1*dz1_dw -- prev_chain = dout_da2*da2_dz2*dz2_da1

        dout_db = dout_da2*da2_dz2*dz2_da1*da1_dz1*dz1_db
        dout_db = prev_chain*da1_dz1*dz1_db -- prev_chain = dout_da2*da2_dz2*dz2_da1
        '''
        return (dout_dw, dout_db)