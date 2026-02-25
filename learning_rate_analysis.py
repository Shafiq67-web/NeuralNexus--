"""
Learning Rate Sensitivity Analysis
==================================

Experiment to analyze the effect of different learning rates on
neural network training dynamics and final performance.

Author: [Your Name]
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import NeuralNetwork
from losses import BinaryCrossEntropy


def run_learning_rate_experiment(learning_rates, epochs=1000, verbose=True):
    """
    Run experiment testing different learning rates.
    
    Parameters:
    -----------
    learning_rates : list of float
        Learning rates to test
    epochs : int
        Number of training epochs
    verbose : bool
        Print progress
        
    Returns:
    --------
    results : dict
        Dictionary mapping learning rate to training results
    """
    # Load and prepare data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Transpose for our implementation
    X_train_T = X_train_scaled.T
    X_val_T = X_val_scaled.T
    X_test_T = X_test_scaled.T
    Y_train = y_train.reshape(1, -1)
    Y_val = y_val.reshape(1, -1)
    Y_test = y_test.reshape(1, -1)
    
    results = {}
    
    for lr in learning_rates:
        if verbose:
            print(f"\nTesting learning rate: {lr}")
        
        # Create fresh model
        model = NeuralNetwork(
            layer_dims=[30, 64, 32, 1],
            loss_fn=BinaryCrossEntropy(),
            lambda_reg=0.01
        )
        
        # Train
        model.fit(
            X_train=X_train_T,
            Y_train=Y_train,
            X_val=X_val_T,
            Y_val=Y_val,
            epochs=epochs,
            learning_rate=lr,
            verbose=0
        )
        
        # Evaluate
        test_metrics = model.evaluate(X_test_T, Y_test)
        
        results[lr] = {
            'history': model.history,
            'test_accuracy': test_metrics['accuracy'],
            'test_loss': test_metrics['loss']
        }
        
        if verbose:
            print(f"  Final test accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Final test loss: {test_metrics['loss']:.4f}")
    
    return results


def plot_learning_rate_results(results, save_path=None):
    """
    Visualize learning rate experiment results.
    
    Parameters:
    -----------
    results : dict
        Results from run_learning_rate_experiment
    save_path : str, optional
        Path to save figure
    """
    learning_rates = sorted(results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    for lr in learning_rates:
        axes[0].plot(results[lr]['history']['loss'], 
                     label=f'LR={lr}', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Learning Rate Comparison - Loss', 
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Accuracy curves
    for lr in learning_rates:
        axes[1].plot(results[lr]['history']['accuracy'], 
                     label=f'LR={lr}', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Training Accuracy', fontsize=12)
    axes[1].set_title('Learning Rate Comparison - Accuracy', 
                      fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def print_summary_table(results):
    """Print summary table of results."""
    learning_rates = sorted(results.keys())
    
    print("\n" + "="*60)
    print("LEARNING RATE EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'LR':<12} {'Test Accuracy':<15} {'Test Loss':<15}")
    print("-"*60)
    
    for lr in learning_rates:
        acc = results[lr]['test_accuracy']
        loss = results[lr]['test_loss']
        print(f"{lr:<12.4f} {acc:<15.4f} {loss:<15.4f}")
    
    print("="*60)
    
    # Find best learning rate
    best_lr = max(learning_rates, 
                  key=lambda lr: results[lr]['test_accuracy'])
    print(f"\nBest learning rate: {best_lr}")
    print(f"Best test accuracy: {results[best_lr]['test_accuracy']:.4f}")


if __name__ == "__main__":
    # Define learning rates to test
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    # Run experiment
    print("="*60)
    print("LEARNING RATE SENSITIVITY ANALYSIS")
    print("="*60)
    
    results = run_learning_rate_experiment(
        learning_rates=learning_rates,
        epochs=1000,
        verbose=True
    )
    
    # Print summary
    print_summary_table(results)
    
    # Plot results
    plot_learning_rate_results(results)
    
    print("\nExperiment complete!")
