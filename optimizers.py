"""
Optimizers Module
=================

Advanced optimization algorithms for neural network training.

Includes:
- SGD with Momentum
- RMSprop
- Adam (Adaptive Moment Estimation)

Author: [Your Name]
Date: February 2026
"""

import numpy as np


class Optimizer:
    """Base class for optimization algorithms."""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.iterations = 0
    
    def update(self, params, grads):
        """Update parameters based on gradients."""
        raise NotImplementedError
    
    def pre_update(self):
        """Called before parameter updates (e.g., for learning rate decay)."""
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.
    
    Mathematical formulation:
    -------------------------
    With momentum:
        v_t = β * v_{t-1} + (1-β) * ∇J(θ)
        θ_t = θ_{t-1} - α * v_t
    
    Without momentum (β = 0):
        θ_t = θ_{t-1} - α * ∇J(θ)
    
    Momentum helps accelerate convergence in relevant directions
    and dampens oscillations.
    
    Reference:
    ----------
    Polyak, B.T. (1964). Some methods of speeding up the convergence 
    of iteration methods. USSR Computational Mathematics and Mathematical 
    Physics, 4(5), 1-17.
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        Initialize SGD optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate (α)
        momentum : float
            Momentum coefficient (β), typically 0.9
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, layer, grads, layer_idx):
        """
        Update layer parameters using SGD with momentum.
        
        Parameters:
        -----------
        layer : DenseLayer
            Layer to update
        grads : dict
            Gradients {'dW': ..., 'db': ...}
        layer_idx : int
            Layer index for velocity tracking
        """
        # Initialize velocities if not exists
        if layer_idx not in self.velocities:
            self.velocities[layer_idx] = {
                'vW': np.zeros_like(layer.W),
                'vb': np.zeros_like(layer.b)
            }
        
        v = self.velocities[layer_idx]
        
        # Update velocities
        v['vW'] = self.momentum * v['vW'] + grads['dW']
        v['vb'] = self.momentum * v['vb'] + grads['db']
        
        # Update parameters
        layer.W -= self.learning_rate * v['vW']
        layer.b -= self.learning_rate * v['vb']


class RMSprop(Optimizer):
    """
    RMSprop (Root Mean Square Propagation).
    
    Adaptive learning rate method that divides the learning rate
    by an exponentially decaying average of squared gradients.
    
    Mathematical formulation:
    -------------------------
    E[g²]_t = β * E[g²]_{t-1} + (1-β) * g_t²
    θ_t = θ_{t-1} - α * g_t / (sqrt(E[g²]_t) + ε)
    
    Benefits:
    - Adapts learning rate per parameter
    - Handles sparse gradients well
    - Mitigates vanishing/exploding learning rates
    
    Reference:
    ----------
    Tieleman, T. & Hinton, G. (2012). Lecture 6.5 - RMSprop.
    Coursera: Neural Networks for Machine Learning.
    """
    
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        """
        Initialize RMSprop optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate (α)
        beta : float
            Decay rate for moving average (typically 0.9)
        epsilon : float
            Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.squared_grads = {}
    
    def update(self, layer, grads, layer_idx):
        """Update layer parameters using RMSprop."""
        # Initialize squared gradients if not exists
        if layer_idx not in self.squared_grads:
            self.squared_grads[layer_idx] = {
                'sW': np.zeros_like(layer.W),
                'sb': np.zeros_like(layer.b)
            }
        
        s = self.squared_grads[layer_idx]
        
        # Update squared gradients (moving average)
        s['sW'] = self.beta * s['sW'] + (1 - self.beta) * (grads['dW'] ** 2)
        s['sb'] = self.beta * s['sb'] + (1 - self.beta) * (grads['db'] ** 2)
        
        # Update parameters with adaptive learning rate
        layer.W -= self.learning_rate * grads['dW'] / (np.sqrt(s['sW']) + self.epsilon)
        layer.b -= self.learning_rate * grads['db'] / (np.sqrt(s['sb']) + self.epsilon)


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation).
    
    Combines momentum (first moment) with RMSprop (second moment)
    for adaptive learning rates.
    
    Mathematical formulation:
    -------------------------
    m_t = β₁ * m_{t-1} + (1-β₁) * g_t        # First moment (momentum)
    v_t = β₂ * v_{t-1} + (1-β₂) * g_t²       # Second moment (RMSprop)
    
    Bias correction:
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)
    
    Update:
        θ_t = θ_{t-1} - α * m̂_t / (sqrt(v̂_t) + ε)
    
    Benefits:
    - Combines advantages of momentum and adaptive learning rates
    - Bias correction helps in early iterations
    - Generally robust hyperparameter choices
    
    Reference:
    ----------
    Kingma, D.P. & Ba, J. (2015). Adam: A method for stochastic optimization.
    ICLR 2015.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate (α), typically 0.001
        beta1 : float
            Exponential decay rate for first moment (typically 0.9)
        beta2 : float
            Exponential decay rate for second moment (typically 0.999)
        epsilon : float
            Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moments = {}
    
    def update(self, layer, grads, layer_idx):
        """Update layer parameters using Adam."""
        self.iterations += 1
        
        # Initialize moments if not exists
        if layer_idx not in self.moments:
            self.moments[layer_idx] = {
                'mW': np.zeros_like(layer.W),
                'mb': np.zeros_like(layer.b),
                'vW': np.zeros_like(layer.W),
                'vb': np.zeros_like(layer.b)
            }
        
        m = self.moments[layer_idx]
        
        # Update biased first moment (momentum)
        m['mW'] = self.beta1 * m['mW'] + (1 - self.beta1) * grads['dW']
        m['mb'] = self.beta1 * m['mb'] + (1 - self.beta1) * grads['db']
        
        # Update biased second moment (RMSprop)
        m['vW'] = self.beta2 * m['vW'] + (1 - self.beta2) * (grads['dW'] ** 2)
        m['vb'] = self.beta2 * m['vb'] + (1 - self.beta2) * (grads['db'] ** 2)
        
        # Bias correction
        mW_hat = m['mW'] / (1 - self.beta1 ** self.iterations)
        mb_hat = m['mb'] / (1 - self.beta1 ** self.iterations)
        vW_hat = m['vW'] / (1 - self.beta2 ** self.iterations)
        vb_hat = m['vb'] / (1 - self.beta2 ** self.iterations)
        
        # Update parameters
        layer.W -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
        layer.b -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + self.epsilon)


class AdaGrad(Optimizer):
    """
    AdaGrad (Adaptive Gradient).
    
    Adapts learning rate by dividing by the square root of
    accumulated squared gradients.
    
    Mathematical formulation:
    -------------------------
    G_t = G_{t-1} + g_t²
    θ_t = θ_{t-1} - α * g_t / (sqrt(G_t) + ε)
    
    Benefits:
    - Well-suited for sparse data
    - Automatically adapts learning rate per feature
    
    Drawbacks:
    - Learning rate monotonically decreases
    - May become too small over time
    
    Reference:
    ----------
    Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient 
    methods for online learning and stochastic optimization. JMLR.
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        """
        Initialize AdaGrad optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate (α)
        epsilon : float
            Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.accumulated_grads = {}
    
    def update(self, layer, grads, layer_idx):
        """Update layer parameters using AdaGrad."""
        # Initialize accumulated gradients if not exists
        if layer_idx not in self.accumulated_grads:
            self.accumulated_grads[layer_idx] = {
                'GW': np.zeros_like(layer.W),
                'Gb': np.zeros_like(layer.b)
            }
        
        G = self.accumulated_grads[layer_idx]
        
        # Accumulate squared gradients
        G['GW'] += grads['dW'] ** 2
        G['Gb'] += grads['db'] ** 2
        
        # Update parameters
        layer.W -= self.learning_rate * grads['dW'] / (np.sqrt(G['GW']) + self.epsilon)
        layer.b -= self.learning_rate * grads['db'] / (np.sqrt(G['Gb']) + self.epsilon)


# Factory function for creating optimizer instances
def get_optimizer(name, **kwargs):
    """
    Factory function to create optimizer instances by name.
    
    Parameters:
    -----------
    name : str
        Optimizer name ('sgd', 'momentum', 'rmsprop', 'adam', 'adagrad')
    **kwargs : dict
        Additional parameters for specific optimizers
        
    Returns:
    --------
    optimizer : Optimizer
        Optimizer instance
    """
    optimizers = {
        'sgd': SGD,
        'momentum': lambda lr, **kw: SGD(lr, momentum=0.9, **kw),
        'rmsprop': RMSprop,
        'adam': Adam,
        'adagrad': AdaGrad
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. "
                        f"Available: {list(optimizers.keys())}")
    
    return optimizers[name.lower()](**kwargs)
