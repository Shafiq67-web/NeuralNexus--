"""
Activation Functions Module
===========================

Mathematical implementations of common neural network activation functions
with forward and backward (gradient) computations.

All activations implement:
- forward(Z): Computes activation A = g(Z)
- backward(Z): Computes derivative g'(Z) for backpropagation

Author: [Your Name]
Date: February 2026
"""

import numpy as np


class Activation:
    """
    Base class for activation functions.
    
    Each activation implements:
    - forward(Z): Computes activation A = g(Z)
    - backward(Z): Computes derivative g'(Z) for backpropagation
    """
    
    def forward(self, Z):
        """Compute activation."""
        raise NotImplementedError
    
    def backward(self, Z):
        """Compute derivative for backpropagation."""
        raise NotImplementedError


class ReLU(Activation):
    r"""
    Rectified Linear Unit: g(z) = max(0, z)
    
    Derivative: g'(z) = 1 if z > 0, else 0
    
    Mathematical properties:
    - Non-linear, non-saturating for z > 0
    - Computationally efficient
    - Mitigates vanishing gradient problem
    - Not zero-centered (outputs are non-negative)
    
    Forward: A = ReLU(Z) = max(0, Z)
    Backward: dZ = dA * (Z > 0)
    """
    
    def forward(self, Z):
        """
        Forward pass: A = ReLU(Z) = max(0, Z)
        
        Parameters:
        -----------
        Z : np.ndarray of shape (n_units, m_samples)
            Pre-activation values
            
        Returns:
        --------
        A : np.ndarray of shape (n_units, m_samples)
            Post-activation values
        """
        return np.maximum(0, Z)
    
    def backward(self, Z):
        """
        Backward pass: Compute dReLU/dZ
        
        Returns:
        --------
        dZ : np.ndarray of shape (n_units, m_samples)
            Element-wise derivative (1 where Z > 0, else 0)
        """
        return (Z > 0).astype(float)


class Sigmoid(Activation):
    r"""
    Sigmoid function: g(z) = 1 / (1 + exp(-z))
    
    Derivative: g'(z) = g(z) * (1 - g(z))
    
    Mathematical properties:
    - Maps any real value to (0, 1)
    - Smooth and differentiable everywhere
    - Suffers from vanishing gradients for |z| >> 0
    - Used for binary classification output
    
    Forward: A = sigmoid(Z)
    Backward: dZ = dA * A * (1 - A)
    """
    
    def forward(self, Z):
        """
        Forward pass: A = sigmoid(Z)
        
        Numerical stability: Clip Z to prevent overflow in exp(-Z)
        """
        Z_clipped = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z_clipped))
    
    def backward(self, Z):
        """
        Backward pass: Compute dsigmoid/dZ = A * (1 - A)
        
        Note: We compute A first, then use the elegant derivative formula
        """
        A = self.forward(Z)
        return A * (1 - A)


class Tanh(Activation):
    r"""
    Hyperbolic tangent: g(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    
    Derivative: g'(z) = 1 - g(z)^2
    
    Mathematical properties:
    - Maps any real value to (-1, 1)
    - Zero-centered (unlike sigmoid)
    - Still suffers from vanishing gradients
    
    Forward: A = tanh(Z)
    Backward: dZ = dA * (1 - A^2)
    """
    
    def forward(self, Z):
        """Forward pass: A = tanh(Z)"""
        return np.tanh(Z)
    
    def backward(self, Z):
        """Backward pass: Compute dtanh/dZ = 1 - A^2"""
        A = self.forward(Z)
        return 1 - A ** 2


class LeakyReLU(Activation):
    r"""
    Leaky ReLU: g(z) = max(αz, z) where α is a small constant (default 0.01)
    
    Derivative: g'(z) = 1 if z > 0, else α
    
    Mathematical properties:
    - Addresses dying ReLU problem
    - Allows small negative gradients
    - α typically set to 0.01
    
    Forward: A = LeakyReLU(Z) = max(αZ, Z)
    Backward: dZ = dA * (1 if Z > 0 else α)
    """
    
    def __init__(self, alpha=0.01):
        """
        Initialize LeakyReLU with slope parameter alpha.
        
        Parameters:
        -----------
        alpha : float
            Slope for negative values (default: 0.01)
        """
        self.alpha = alpha
    
    def forward(self, Z):
        """Forward pass: A = max(αZ, Z)"""
        return np.where(Z > 0, Z, self.alpha * Z)
    
    def backward(self, Z):
        """Backward pass: dZ = dA * (1 if Z > 0 else α)"""
        return np.where(Z > 0, 1.0, self.alpha)


class Softmax(Activation):
    r"""
    Softmax function for multi-class classification.
    
    g(z_i) = exp(z_i) / sum_j(exp(z_j))
    
    Mathematical properties:
    - Outputs sum to 1 (probability distribution)
    - Used for multi-class classification output
    - Numerically unstable without stabilization
    
    Forward: A = softmax(Z)
    Backward: dZ (combined with cross-entropy for efficiency)
    """
    
    def forward(self, Z):
        """
        Forward pass with numerical stability.
        
        Subtract max(Z) before exp to prevent overflow.
        """
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def backward(self, Z):
        """
        Backward pass for softmax.
        
        Note: Usually combined with cross-entropy loss for simplification.
        """
        # For standalone use, returns Jacobian (not commonly needed)
        A = self.forward(Z)
        return A * (1 - A)  # Simplified diagonal approximation


# Factory function for creating activation instances
def get_activation(name, **kwargs):
    """
    Factory function to create activation instances by name.
    
    Parameters:
    -----------
    name : str
        Activation function name ('relu', 'sigmoid', 'tanh', 'leaky_relu')
    **kwargs : dict
        Additional parameters for specific activations
        
    Returns:
    --------
    activation : Activation
        Activation function instance
    """
    activations = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'leaky_relu': LeakyReLU,
        'softmax': Softmax
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. "
                        f"Available: {list(activations.keys())}")
    
    return activations[name.lower()](**kwargs)
