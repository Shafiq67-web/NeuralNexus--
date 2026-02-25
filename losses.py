"""
Loss Functions Module
=====================

Mathematical implementations of common loss functions for neural networks
with gradient computations for backpropagation.

Author: [Your Name]
Date: February 2026
"""

import numpy as np


class Loss:
    """
    Base class for loss functions.
    
    Each loss function implements:
    - compute(Y_pred, Y_true): Compute the loss value
    - gradient(Y_pred, Y_true): Compute dL/dY_pred for backpropagation
    """
    
    def compute(self, Y_pred, Y_true):
        """Compute the loss value."""
        raise NotImplementedError
    
    def gradient(self, Y_pred, Y_true):
        """
        Compute dL/dY_pred for backpropagation.
        This is the initial gradient passed to the output layer.
        """
        raise NotImplementedError


class BinaryCrossEntropy(Loss):
    r"""
    Binary Cross-Entropy (Logistic) Loss
    
    L = -(1/m) * sum[ y*log(y_hat) + (1-y)*log(1-y_hat) ]
    
    Mathematical derivation from maximum likelihood:
    ------------------------------------------------
    For binary classification with y ∈ {0, 1}:
    
    Likelihood: P(y|x) = ŷ^y * (1-ŷ)^(1-y)
    Log-likelihood: log P(y|x) = y*log(ŷ) + (1-y)*log(1-ŷ)
    Negative log-likelihood (loss): L = -log P(y|x)
    
    For sigmoid output with BCE, the gradient simplifies to (ŷ - y),
    which is why we often combine them in the output layer.
    
    Gradient: dL/dŷ = -y/ŷ + (1-y)/(1-ŷ) = (ŷ - y) / (ŷ(1-ŷ))
    
    When combined with sigmoid: dL/dz = ŷ - y (elegant simplification)
    """
    
    def compute(self, Y_pred, Y_true, epsilon=1e-15):
        """
        Compute binary cross-entropy loss.
        
        Parameters:
        -----------
        Y_pred : np.ndarray of shape (1, m) or (n_classes, m)
            Predicted probabilities (output of sigmoid/softmax)
        Y_true : np.ndarray of shape (1, m) or (n_classes, m)
            True binary labels (0 or 1)
        epsilon : float
            Small constant for numerical stability (prevents log(0))
            
        Returns:
        --------
        loss : float
            Average cross-entropy loss across all samples
        """
        m = Y_true.shape[1]
        
        # Clip predictions to avoid log(0)
        Y_pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)
        
        # Compute loss
        loss = -(1/m) * np.sum(
            Y_true * np.log(Y_pred_clipped) + 
            (1 - Y_true) * np.log(1 - Y_pred_clipped)
        )
        
        return loss
    
    def gradient(self, Y_pred, Y_true, epsilon=1e-15):
        """
        Compute dL/dY_pred.
        
        Note: When combined with sigmoid in the output layer,
        this gradient simplifies significantly due to the conjugate
        relationship between sigmoid and cross-entropy.
        
        dL/dŷ = -y/ŷ + (1-y)/(1-ŷ)
        
        Parameters:
        -----------
        Y_pred : np.ndarray of shape (1, m)
            Predicted probabilities
        Y_true : np.ndarray of shape (1, m)
            True binary labels
            
        Returns:
        --------
        dY_pred : np.ndarray of shape (1, m)
            Gradient of loss with respect to predictions
        """
        Y_pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)
        return -(Y_true / Y_pred_clipped) + (1 - Y_true) / (1 - Y_pred_clipped)


class CategoricalCrossEntropy(Loss):
    r"""
    Categorical Cross-Entropy Loss for multi-class classification.
    
    L = -(1/m) * sum_i sum_c [ y_{i,c} * log(ŷ_{i,c}) ]
    
    where:
    - m is the number of samples
    - C is the number of classes
    - y_{i,c} is 1 if sample i belongs to class c, else 0 (one-hot)
    - ŷ_{i,c} is the predicted probability for class c
    
    When combined with softmax, gradient simplifies to: ŷ - y
    """
    
    def compute(self, Y_pred, Y_true, epsilon=1e-15):
        """
        Compute categorical cross-entropy loss.
        
        Parameters:
        -----------
        Y_pred : np.ndarray of shape (n_classes, m)
            Predicted class probabilities (softmax output)
        Y_true : np.ndarray of shape (n_classes, m)
            True one-hot encoded labels
        epsilon : float
            Small constant for numerical stability
            
        Returns:
        --------
        loss : float
            Average categorical cross-entropy loss
        """
        m = Y_true.shape[1]
        Y_pred_clipped = np.clip(Y_pred, epsilon, 1.0)
        
        # Sum over classes, average over samples
        loss = -(1/m) * np.sum(Y_true * np.log(Y_pred_clipped))
        return loss
    
    def gradient(self, Y_pred, Y_true, epsilon=1e-15):
        """
        Compute dL/dY_pred.
        
        Note: When combined with softmax, this simplifies to ŷ - y
        """
        Y_pred_clipped = np.clip(Y_pred, epsilon, 1.0)
        return -Y_true / Y_pred_clipped


class MeanSquaredError(Loss):
    r"""
    Mean Squared Error (L2 Loss) for regression tasks.
    
    L = (1/m) * sum_i (ŷ_i - y_i)^2
    
    Mathematical properties:
    - Differentiable everywhere
    - Penalizes large errors quadratically
    - Sensitive to outliers
    - Used for regression problems
    
    Gradient: dL/dŷ = (2/m) * (ŷ - y)
    """
    
    def compute(self, Y_pred, Y_true):
        """
        Compute MSE loss.
        
        Parameters:
        -----------
        Y_pred : np.ndarray of shape (n_outputs, m)
            Predicted values
        Y_true : np.ndarray of shape (n_outputs, m)
            True values
            
        Returns:
        --------
        loss : float
            Mean squared error
        """
        m = Y_true.shape[1]
        return (1/m) * np.sum((Y_pred - Y_true) ** 2)
    
    def gradient(self, Y_pred, Y_true):
        """
        Compute dL/dY_pred.
        
        dL/dŷ = (2/m) * (ŷ - y)
        """
        m = Y_true.shape[1]
        return (2/m) * (Y_pred - Y_true)


class HuberLoss(Loss):
    r"""
    Huber Loss - Robust loss function combining MSE and MAE.
    
    L_δ(y, ŷ) = {
        0.5 * (y - ŷ)^2       if |y - ŷ| <= δ
        δ * |y - ŷ| - 0.5*δ^2  if |y - ŷ| > δ
    }
    
    Mathematical properties:
    - Less sensitive to outliers than MSE
    - Differentiable at 0 (unlike MAE)
    - δ controls the transition point
    - Combines benefits of L1 and L2 loss
    
    Gradient:
    dL/dŷ = {
        (ŷ - y)        if |y - ŷ| <= δ
        δ * sign(ŷ-y)  if |y - ŷ| > δ
    }
    """
    
    def __init__(self, delta=1.0):
        """
        Initialize Huber loss with delta parameter.
        
        Parameters:
        -----------
        delta : float
            Transition point between quadratic and linear regions
        """
        self.delta = delta
    
    def compute(self, Y_pred, Y_true):
        """
        Compute Huber loss.
        
        Parameters:
        -----------
        Y_pred : np.ndarray
            Predicted values
        Y_true : np.ndarray
            True values
            
        Returns:
        --------
        loss : float
            Huber loss
        """
        m = Y_true.shape[1]
        error = Y_pred - Y_true
        abs_error = np.abs(error)
        
        # Quadratic region
        quadratic = 0.5 * error ** 2
        # Linear region
        linear = self.delta * abs_error - 0.5 * self.delta ** 2
        
        # Apply piecewise condition
        loss = np.where(abs_error <= self.delta, quadratic, linear)
        return (1/m) * np.sum(loss)
    
    def gradient(self, Y_pred, Y_true):
        """
        Compute dL/dY_pred.
        """
        m = Y_true.shape[1]
        error = Y_pred - Y_true
        abs_error = np.abs(error)
        
        # Piecewise gradient
        grad = np.where(
            abs_error <= self.delta,
            error,  # Linear gradient in quadratic region
            self.delta * np.sign(error)  # Constant gradient in linear region
        )
        return (1/m) * grad


# Factory function for creating loss instances
def get_loss(name, **kwargs):
    """
    Factory function to create loss instances by name.
    
    Parameters:
    -----------
    name : str
        Loss function name ('bce', 'cce', 'mse', 'huber')
    **kwargs : dict
        Additional parameters for specific losses
        
    Returns:
    --------
    loss : Loss
        Loss function instance
    """
    losses = {
        'bce': BinaryCrossEntropy,
        'binary_crossentropy': BinaryCrossEntropy,
        'cce': CategoricalCrossEntropy,
        'categorical_crossentropy': CategoricalCrossEntropy,
        'mse': MeanSquaredError,
        'mean_squared_error': MeanSquaredError,
        'huber': HuberLoss
    }
    
    if name.lower() not in losses:
        raise ValueError(f"Unknown loss: {name}. "
                        f"Available: {list(losses.keys())}")
    
    return losses[name.lower()](**kwargs)
