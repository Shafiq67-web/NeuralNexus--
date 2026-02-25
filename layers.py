"""
Neural Network Layers Module
============================

Implementation of fully-connected (dense) layers with forward and backward
propagation capabilities.

Author: [Your Name]
Date: February 2026
"""

import numpy as np
from activations import Activation, ReLU, Sigmoid


class DenseLayer:
    """
    Fully-connected (dense) neural network layer.
    
    Implements the mathematical operations:
    - Forward: Z = W @ A_prev + b, A = activation(Z)
    - Backward: Computes gradients via chain rule
    
    Mathematical formulation:
    -------------------------
    Forward propagation:
        Z^[l] = W^[l] A^[l-1] + b^[l]
        A^[l] = g(Z^[l])
    
    Backward propagation (chain rule):
        dZ^[l] = dA^[l] * g'(Z^[l])
        dW^[l] = (1/m) dZ^[l] (A^[l-1])^T + (λ/m) W^[l]
        db^[l] = (1/m) sum(dZ^[l], axis=1)
        dA^[l-1] = (W^[l])^T dZ^[l]
    
    Attributes:
    -----------
    n_units : int
        Number of neurons in this layer
    n_inputs : int
        Number of inputs to this layer (neurons in previous layer)
    W : np.ndarray of shape (n_units, n_inputs)
        Weight matrix
    b : np.ndarray of shape (n_units, 1)
        Bias vector
    activation : Activation
        Activation function object
    """
    
    def __init__(self, n_units, n_inputs, activation, initializer='he', name=None):
        """
        Initialize a dense layer.
        
        Parameters:
        -----------
        n_units : int
            Number of neurons in this layer
        n_inputs : int
            Number of input features (from previous layer)
        activation : Activation
            Activation function object (ReLU, Sigmoid, etc.)
        initializer : str
            Weight initialization method ('he', 'xavier', 'random')
            - 'he': He initialization for ReLU (variance = 2/n_inputs)
            - 'xavier': Xavier initialization for tanh/sigmoid (variance = 1/n_inputs)
            - 'random': Small random values
        name : str, optional
            Layer name for debugging
        """
        self.n_units = n_units
        self.n_inputs = n_inputs
        self.activation = activation
        self.name = name or f"Layer_{n_units}"
        self.initializer = initializer
        
        # Initialize parameters
        self._initialize_weights(initializer)
        
        # Cache for backpropagation
        self.cache = {}
    
    def _initialize_weights(self, method):
        """
        Initialize weights using specified method.
        
        Mathematical justification:
        ---------------------------
        He initialization (for ReLU):
            W ~ N(0, 2/n_inputs)
            Justification: Maintains variance through ReLU layers
            Reference: He et al. (2015)
        
        Xavier initialization (for tanh/sigmoid):
            W ~ N(0, 1/n_inputs)
            Justification: Maintains variance through symmetric activations
            Reference: Glorot & Bengio (2010)
        """
        if method == 'he':
            # He et al. (2015) - optimal for ReLU
            # Var(W) = 2 / n_inputs
            std = np.sqrt(2.0 / self.n_inputs)
            self.W = np.random.randn(self.n_units, self.n_inputs) * std
        elif method == 'xavier':
            # Glorot & Bengio (2010) - optimal for tanh/sigmoid
            # Var(W) = 1 / n_inputs
            std = np.sqrt(1.0 / self.n_inputs)
            self.W = np.random.randn(self.n_units, self.n_inputs) * std
        elif method == 'random':
            # Small random values (not recommended for deep networks)
            self.W = np.random.randn(self.n_units, self.n_inputs) * 0.01
        else:
            raise ValueError(f"Unknown initializer: {method}")
        
        # Initialize biases to zero
        # Bias initialization is less critical as gradients flow directly
        self.b = np.zeros((self.n_units, 1))
    
    def forward(self, A_prev):
        """
        Forward propagation through this layer.
        
        Mathematical operations:
            Z = W @ A_prev + b    (affine transformation)
            A = activation(Z)     (non-linear activation)
        
        Parameters:
        -----------
        A_prev : np.ndarray of shape (n_inputs, m)
            Activations from previous layer
            
        Returns:
        --------
        A : np.ndarray of shape (n_units, m)
            Output activations of this layer
        """
        m = A_prev.shape[1]
        
        # Linear transformation: Z = W @ A_prev + b
        # W: (n_units, n_inputs), A_prev: (n_inputs, m)
        # Result Z: (n_units, m)
        Z = np.dot(self.W, A_prev) + self.b
        
        # Apply activation function
        A = self.activation.forward(Z)
        
        # Cache values for backpropagation
        self.cache = {
            'A_prev': A_prev,
            'Z': Z,
            'A': A
        }
        
        return A
    
    def backward(self, dA, lambda_reg=0):
        """
        Backward propagation through this layer.
        
        Computes gradients using the chain rule:
            dZ = dA * activation'(Z)
            dW = (1/m) * dZ @ A_prev.T + (λ/m) * W
            db = (1/m) * sum(dZ, axis=1)
            dA_prev = W.T @ dZ
        
        Parameters:
        -----------
        dA : np.ndarray of shape (n_units, m)
            Gradient of loss with respect to output activations
        lambda_reg : float
            L2 regularization parameter
            
        Returns:
        --------
        dA_prev : np.ndarray of shape (n_inputs, m)
            Gradient with respect to input activations
        grads : dict
            Dictionary containing 'dW' and 'db'
        """
        # Retrieve cached values
        A_prev = self.cache['A_prev']
        Z = self.cache['Z']
        m = A_prev.shape[1]
        
        # Compute dZ = dA * g'(Z) (element-wise multiplication/Hadamard product)
        dZ = dA * self.activation.backward(Z)
        
        # Compute gradients
        # dW = (1/m) * dZ @ A_prev.T
        dW = (1/m) * np.dot(dZ, A_prev.T)
        
        # Add L2 regularization gradient: (λ/m) * W
        # This comes from derivative of (λ/2m) * ||W||^2_F
        if lambda_reg > 0:
            dW += (lambda_reg / m) * self.W
        
        # db = (1/m) * sum(dZ, axis=1, keepdims=True)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # dA_prev = W.T @ dZ (gradient to propagate backward)
        dA_prev = np.dot(self.W.T, dZ)
        
        grads = {'dW': dW, 'db': db}
        
        return dA_prev, grads
    
    def update_params(self, grads, learning_rate):
        """
        Update parameters using gradient descent.
        
        Mathematical update rules:
            W := W - α * dW
            b := b - α * db
        
        where α is the learning rate.
        """
        self.W -= learning_rate * grads['dW']
        self.b -= learning_rate * grads['db']
    
    def get_params(self):
        """Return current parameters."""
        return {'W': self.W, 'b': self.b}
    
    def set_params(self, params):
        """Set parameters (for loading saved models)."""
        self.W = params['W']
        self.b = params['b']
    
    def count_parameters(self):
        """
        Count total number of trainable parameters.
        
        Returns:
        --------
        count : int
            Total number of parameters (weights + biases)
        """
        return self.W.size + self.b.size


class DropoutLayer:
    """
    Dropout regularization layer.
    
    During training: randomly zeroes some elements with probability p
    During inference: scales outputs by (1-p)
    
    Mathematical formulation:
    -------------------------
    Training: A = mask * A_prev / (1-p) where mask ~ Bernoulli(1-p)
    Inference: A = A_prev
    
    Inverted dropout scales during training so inference is unchanged.
    """
    
    def __init__(self, dropout_rate=0.5):
        """
        Initialize dropout layer.
        
        Parameters:
        -----------
        dropout_rate : float
            Probability of dropping a neuron (0 to 1)
        """
        if not 0 <= dropout_rate < 1:
            raise ValueError("Dropout rate must be in [0, 1)")
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, A_prev):
        """
        Forward pass with dropout.
        
        Parameters:
        -----------
        A_prev : np.ndarray
            Input activations
            
        Returns:
        --------
        A : np.ndarray
            Output with dropout applied (training) or scaled (inference)
        """
        if self.training and self.dropout_rate > 0:
            # Generate dropout mask
            self.mask = (np.random.rand(*A_prev.shape) > self.dropout_rate).astype(float)
            # Apply mask and scale
            A = A_prev * self.mask / (1 - self.dropout_rate)
        else:
            A = A_prev
        
        return A
    
    def backward(self, dA, lambda_reg=0):
        """
        Backward pass with dropout.
        
        Parameters:
        -----------
        dA : np.ndarray
            Gradient from next layer
            
        Returns:
        --------
        dA_prev : np.ndarray
            Gradient with dropout mask applied
        grads : dict
            Empty dict (no parameters to update)
        """
        if self.training and self.mask is not None:
            dA_prev = dA * self.mask / (1 - self.dropout_rate)
        else:
            dA_prev = dA
        
        return dA_prev, {}
    
    def set_training_mode(self, training=True):
        """Set training mode (True) or inference mode (False)."""
        self.training = training


class BatchNormalizationLayer:
    """
    Batch Normalization layer.
    
    Normalizes layer inputs to have zero mean and unit variance,
    then applies learnable scale and shift parameters.
    
    Mathematical formulation:
    -------------------------
    μ_B = (1/m) sum(x_i)           # mini-batch mean
    σ²_B = (1/m) sum((x_i - μ_B)²) # mini-batch variance
    x̂_i = (x_i - μ_B) / sqrt(σ²_B + ε)  # normalize
    y_i = γ * x̂_i + β              # scale and shift
    
    Benefits:
    - Reduces internal covariate shift
    - Allows higher learning rates
    - Acts as regularization
    """
    
    def __init__(self, n_features, epsilon=1e-8, momentum=0.9):
        """
        Initialize batch normalization layer.
        
        Parameters:
        -----------
        n_features : int
            Number of features (neurons) in this layer
        epsilon : float
            Small constant for numerical stability
        momentum : float
            Momentum for running mean/variance updates
        """
        self.n_features = n_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones((n_features, 1))
        self.beta = np.zeros((n_features, 1))
        
        # Running statistics for inference
        self.running_mean = np.zeros((n_features, 1))
        self.running_var = np.ones((n_features, 1))
        
        self.training = True
        self.cache = {}
    
    def forward(self, X):
        """Forward pass with batch normalization."""
        if self.training:
            # Compute batch statistics
            batch_mean = np.mean(X, axis=1, keepdims=True)
            batch_var = np.var(X, axis=1, keepdims=True)
            
            # Normalize
            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Update running statistics
            self.running_mean = (self.momentum * self.running_mean + 
                                (1 - self.momentum) * batch_mean)
            self.running_var = (self.momentum * self.running_var + 
                               (1 - self.momentum) * batch_var)
            
            # Cache for backward
            self.cache = {
                'X': X, 'X_norm': X_norm,
                'mean': batch_mean, 'var': batch_var
            }
        else:
            # Use running statistics for inference
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        out = self.gamma * X_norm + self.beta
        return out
    
    def backward(self, dout, lambda_reg=0):
        """Backward pass for batch normalization."""
        X = self.cache['X']
        X_norm = self.cache['X_norm']
        mean = self.cache['mean']
        var = self.cache['var']
        m = X.shape[1]
        
        # Gradients of gamma and beta
        dgamma = np.sum(dout * X_norm, axis=1, keepdims=True)
        dbeta = np.sum(dout, axis=1, keepdims=True)
        
        # Gradient w.r.t. normalized input
        dX_norm = dout * self.gamma
        
        # Gradient w.r.t. variance
        dvar = np.sum(dX_norm * (X - mean) * -0.5 * (var + self.epsilon)**(-1.5), 
                      axis=1, keepdims=True)
        
        # Gradient w.r.t. mean
        dmean = np.sum(dX_norm * -1 / np.sqrt(var + self.epsilon), axis=1, keepdims=True)
        dmean += dvar * np.sum(-2 * (X - mean), axis=1, keepdims=True) / m
        
        # Gradient w.r.t. input
        dX = (dX_norm / np.sqrt(var + self.epsilon) + 
              dvar * 2 * (X - mean) / m + 
              dmean / m)
        
        grads = {'dgamma': dgamma, 'dbeta': dbeta}
        
        return dX, grads
    
    def update_params(self, grads, learning_rate):
        """Update gamma and beta parameters."""
        self.gamma -= learning_rate * grads['dgamma']
        self.beta -= learning_rate * grads['dbeta']
    
    def set_training_mode(self, training=True):
        """Set training mode."""
        self.training = training
