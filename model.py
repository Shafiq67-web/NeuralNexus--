"""
Neural Network Model Module
===========================

Complete feedforward neural network implementation with training,
evaluation, and prediction capabilities.

Author: [Your Name]
Date: February 2026
"""

import numpy as np
from layers import DenseLayer
from activations import ReLU, Sigmoid
from losses import BinaryCrossEntropy


class NeuralNetwork:
    """
    Feedforward Neural Network with L fully-connected layers.
    
    Architecture:
        Input -> [Linear + ReLU] x (L-1) -> [Linear + Sigmoid] -> Output
    
    Training Algorithm (Gradient Descent):
    --------------------------------------
    For each epoch:
        1. Forward propagation to compute predictions
        2. Compute loss (data loss + regularization)
        3. Backward propagation to compute gradients
        4. Update parameters: θ := θ - α * ∇J(θ)
    
    Mathematical formulation:
    -------------------------
    Forward (layer l):
        Z^[l] = W^[l] A^[l-1] + b^[l]
        A^[l] = g^[l](Z^[l])
    
    Backward (output layer L):
        δ^[L] = A^[L] - Y  (for sigmoid + BCE)
    
    Backward (hidden layer l):
        δ^[l] = (W^[l+1])^T δ^[l+1] ⊙ g'^[l](Z^[l])
    
    Gradients:
        ∇W^[l] = (1/m) δ^[l] (A^[l-1])^T + (λ/m) W^[l]
        ∇b^[l] = (1/m) sum(δ^[l], axis=1)
    
    Attributes:
    -----------
    layers : list of DenseLayer
        Network layers
    loss_fn : Loss
        Loss function
    history : dict
        Training history (loss, accuracy per epoch)
    """
    
    def __init__(self, layer_dims, loss_fn=None, lambda_reg=0):
        """
        Initialize neural network.
        
        Parameters:
        -----------
        layer_dims : list of int
            Dimensions of each layer, including input and output
            Example: [30, 64, 32, 1] for 30 inputs, two hidden layers, 1 output
        loss_fn : Loss, optional
            Loss function (default: BinaryCrossEntropy)
        lambda_reg : float
            L2 regularization strength
        """
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1
        self.loss_fn = loss_fn or BinaryCrossEntropy()
        self.lambda_reg = lambda_reg
        self.layers = []
        self.history = {
            'loss': [], 
            'val_loss': [], 
            'accuracy': [], 
            'val_accuracy': []
        }
        
        # Build network layers
        self._build_network()
    
    def _build_network(self):
        """
        Construct network layers.
        
        Layer configuration:
        - Hidden layers: ReLU activation with He initialization
        - Output layer: Sigmoid activation with Xavier initialization
        """
        for i in range(self.n_layers):
            n_inputs = self.layer_dims[i]
            n_units = self.layer_dims[i + 1]
            
            # Output layer uses sigmoid, hidden layers use ReLU
            if i == self.n_layers - 1:
                activation = Sigmoid()
                initializer = 'xavier'  # Better for sigmoid
                name = 'Output'
            else:
                activation = ReLU()
                initializer = 'he'  # Optimal for ReLU
                name = f'Hidden_{i+1}'
            
            layer = DenseLayer(
                n_units=n_units,
                n_inputs=n_inputs,
                activation=activation,
                initializer=initializer,
                name=name
            )
            self.layers.append(layer)
        
        print(f"[OK] Network built: {self.layer_dims}")
    
    def forward(self, X):
        """
        Forward propagation through entire network.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_features, m)
            Input data
            
        Returns:
        --------
        A : np.ndarray of shape (n_output, m)
            Network output (predictions)
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self, Y):
        """
        Backward propagation through entire network.
        
        Parameters:
        -----------
        Y : np.ndarray of shape (1, m)
            True labels
            
        Returns:
        --------
        all_grads : list of dict
            Gradients for each layer
        """
        # Start with gradient from output layer
        # For sigmoid + BCE, this simplifies to (A - Y)
        output_layer = self.layers[-1]
        A_output = output_layer.cache['A']
        
        # Initial gradient: dL/dA_L = A_L - Y (for sigmoid + BCE)
        dA = A_output - Y
        
        # Backpropagate through layers (reverse order)
        all_grads = []
        for layer in reversed(self.layers):
            dA, grads = layer.backward(dA, self.lambda_reg)
            all_grads.insert(0, grads)  # Insert at beginning to maintain order
        
        return all_grads
    
    def compute_loss(self, Y_pred, Y_true):
        """
        Compute total loss including L2 regularization.
        
        J_total = J_data + J_reg
        J_reg = (λ / 2m) * sum_l ||W^[l]||²_F
        """
        # Data loss
        data_loss = self.loss_fn.compute(Y_pred, Y_true)
        
        # L2 regularization term
        reg_loss = 0
        if self.lambda_reg > 0:
            for layer in self.layers:
                reg_loss += np.sum(layer.W ** 2)
            reg_loss = (self.lambda_reg / (2 * Y_true.shape[1])) * reg_loss
        
        return data_loss + reg_loss
    
    def update_parameters(self, all_grads, learning_rate):
        """
        Update all layer parameters using gradient descent.
        
        θ := θ - α * ∇J(θ)
        """
        for layer, grads in zip(self.layers, all_grads):
            layer.update_params(grads, learning_rate)
    
    def fit(self, X_train, Y_train, X_val=None, Y_val=None, 
            epochs=1000, learning_rate=0.01, verbose=100, patience=None):
        """
        Train the neural network using gradient descent.
        
        Parameters:
        -----------
        X_train : np.ndarray of shape (n_features, m_train)
            Training data
        Y_train : np.ndarray of shape (1, m_train)
            Training labels
        X_val : np.ndarray, optional
            Validation data
        Y_val : np.ndarray, optional
            Validation labels
        epochs : int
            Number of training iterations
        learning_rate : float
            Step size for gradient descent (α)
        verbose : int
            Print progress every verbose epochs
        patience : int, optional
            Early stopping patience (number of epochs without improvement)
        """
        print(f"\n{'='*60}")
        print("TRAINING NEURAL NETWORK")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"L2 regularization: {self.lambda_reg}")
        print(f"Training samples: {Y_train.shape[1]}")
        if X_val is not None:
            print(f"Validation samples: {Y_val.shape[1]}")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Forward propagation
            Y_pred_train = self.forward(X_train)
            
            # Compute training loss
            train_loss = self.compute_loss(Y_pred_train, Y_train)
            
            # Compute training accuracy
            train_acc = self._compute_accuracy(Y_pred_train, Y_train)
            
            # Backward propagation
            all_grads = self.backward(Y_train)
            
            # Update parameters
            self.update_parameters(all_grads, learning_rate)
            
            # Store metrics
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)
            
            # Validation metrics
            if X_val is not None and Y_val is not None:
                Y_pred_val = self.forward(X_val)
                val_loss = self.compute_loss(Y_pred_val, Y_val)
                val_acc = self._compute_accuracy(Y_pred_val, Y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                
                # Early stopping check
                if patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"\nEarly stopping at epoch {epoch}")
                            break
            
            # Print progress
            if verbose > 0 and epoch % verbose == 0:
                if X_val is not None:
                    print(f"Epoch {epoch:4d} | Loss: {train_loss:.4f} | "
                          f"Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Val Acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch:4d} | Loss: {train_loss:.4f} | "
                          f"Acc: {train_acc:.4f}")
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
    
    def _compute_accuracy(self, Y_pred, Y_true, threshold=0.5):
        """Compute binary classification accuracy."""
        predictions = (Y_pred >= threshold).astype(int)
        return np.mean(predictions == Y_true)
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        threshold : float
            Classification threshold
            
        Returns:
        --------
        predictions : np.ndarray
            Binary predictions (0 or 1)
        """
        probabilities = self.forward(X)
        return (probabilities >= threshold).astype(int)
    
    def predict_proba(self, X):
        """Return prediction probabilities."""
        return self.forward(X)
    
    def evaluate(self, X, Y):
        """
        Comprehensive model evaluation.
        
        Returns metrics dictionary with:
        - loss: Total loss (data + regularization)
        - accuracy: Classification accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        """
        from sklearn.metrics import (accuracy_score, precision_score, 
                                     recall_score, f1_score)
        
        Y_pred_proba = self.forward(X)
        Y_pred = self.predict(X)
        
        metrics = {
            'loss': self.compute_loss(Y_pred_proba, Y),
            'accuracy': accuracy_score(Y.flatten(), Y_pred.flatten()),
            'precision': precision_score(Y.flatten(), Y_pred.flatten(), zero_division=0),
            'recall': recall_score(Y.flatten(), Y_pred.flatten(), zero_division=0),
            'f1': f1_score(Y.flatten(), Y_pred.flatten(), zero_division=0)
        }
        
        return metrics
    
    def plot_training_history(self, figsize=(14, 5)):
        """Visualize training history."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss curve
        axes[0].plot(self.history['loss'], label='Train Loss', linewidth=2)
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(self.history['accuracy'], label='Train Acc', linewidth=2)
        if self.history['val_accuracy']:
            axes[1].plot(self.history['val_accuracy'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training Accuracy Curve', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_weights(self):
        """Get all layer weights."""
        return [layer.get_params() for layer in self.layers]
    
    def set_weights(self, weights):
        """Set all layer weights."""
        for layer, w in zip(self.layers, weights):
            layer.set_params(w)
    
    def summary(self):
        """
        Print network architecture summary.
        
        Displays layer dimensions, activations, and parameter counts.
        """
        print("\n" + "="*60)
        print("NETWORK ARCHITECTURE SUMMARY")
        print("="*60)
        
        total_params = 0
        print(f"{'Layer':<15} {'Input':<10} {'Output':<10} {'Params':<15}")
        print("-"*60)
        
        for i, layer in enumerate(self.layers):
            n_params = layer.count_parameters()
            total_params += n_params
            print(f"{layer.name:<15} {layer.n_inputs:<10} {layer.n_units:<10} {n_params:<15,}")
        
        print("-"*60)
        print(f"{'Total Parameters:':<36} {total_params:,}")
        print("="*60)


class NeuralNetworkWithActivation(NeuralNetwork):
    """
    Extended Neural Network with configurable hidden activation.
    
    Allows experimentation with different activation functions
    in hidden layers while keeping sigmoid for output.
    """
    
    def __init__(self, layer_dims, hidden_activation='relu', **kwargs):
        """
        Initialize with specific hidden activation.
        
        Parameters:
        -----------
        layer_dims : list of int
            Layer dimensions
        hidden_activation : str
            Hidden layer activation ('relu', 'tanh', 'sigmoid')
        **kwargs : dict
            Additional arguments passed to NeuralNetwork
        """
        self.hidden_activation_name = hidden_activation
        super().__init__(layer_dims, **kwargs)
    
    def _build_network(self):
        """Build network with specified hidden activation."""
        from activations import Tanh
        
        for i in range(self.n_layers):
            n_inputs = self.layer_dims[i]
            n_units = self.layer_dims[i + 1]
            
            if i == self.n_layers - 1:
                activation = Sigmoid()
                initializer = 'xavier'
                name = 'Output'
            else:
                if self.hidden_activation_name == 'relu':
                    activation = ReLU()
                    initializer = 'he'
                elif self.hidden_activation_name == 'tanh':
                    activation = Tanh()
                    initializer = 'xavier'
                elif self.hidden_activation_name == 'sigmoid':
                    activation = Sigmoid()
                    initializer = 'xavier'
                else:
                    raise ValueError(f"Unknown activation: {self.hidden_activation_name}")
                name = f'Hidden_{i+1}'
            
            layer = DenseLayer(
                n_units=n_units,
                n_inputs=n_inputs,
                activation=activation,
                initializer=initializer,
                name=name
            )
            self.layers.append(layer)
        
        print(f"[OK] Network built with {self.hidden_activation_name}: {self.layer_dims}")
