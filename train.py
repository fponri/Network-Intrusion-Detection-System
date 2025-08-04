"""
Training module for the Network Intrusion Detection System
Compatible with your existing ids.py code structure
"""

from models_updated import initialize_random_forest, initialize_logistic_regression
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import time

def train_models(features, labels, model_type='random_forest', test_size=0.2):
    """
    Train models with real network intrusion data
    Enhanced version that maintains compatibility with your ids.py
    """
    print(f"Training {model_type} model with {len(features)} samples...")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Initialize model based on type
    if model_type == 'random_forest':
        model = initialize_random_forest()
    elif model_type == 'logistic_regression':
        model = initialize_logistic_regression()
    else:
        model = initialize_random_forest()  # Default
    
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Model accuracy: {accuracy:.4f}")
    
    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'training_time': training_time,
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

def train_random_forest(features, labels):
    """Train Random Forest model"""
    return train_models(features, labels, 'random_forest')

def train_logistic_regression(features, labels):
    """Train Logistic Regression model"""
    return train_models(features, labels, 'logistic_regression')

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model with comprehensive metrics"""
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'predictions': y_pred
    }