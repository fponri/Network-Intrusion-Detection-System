# Network Intrusion Detection System (NIDS)

**A machine learning-powered cybersecurity system that analyzes authentic network traffic to detect cyber attacks with 85-95% accuracy.**

## 🎯 Project Summary

This Network Intrusion Detection System processes genuine UNSW-NB15 network security data to identify malicious activities in real-time. Unlike typical academic projects using simulated data, this system analyzes over 82,000 authentic network traffic samples - the same type of data used by professional cybersecurity teams.

**What makes this special:**
- Uses authentic captured network traffic 
- Achieves professional-grade accuracy (85-95%) on real cyber attacks
- Interactive terminal-based monitoring with batch processing

## 🎯 Project Overview

This project demonstrates professional-grade network intrusion detection capabilities using authentic captured network traffic data. It processes over 82,000 real network samples to identify malicious activities with 85-95% accuracy.

## 🚀 Key Features

- **Authentic Data Processing**: Uses genuine UNSW-NB15 network intrusion dataset (82,332+ samples)
- **Advanced Machine Learning**: Random Forest and Logistic Regression models for threat detection
- **Real-time Monitoring**: Interactive batch-by-batch intrusion detection with user control
- **High Accuracy Results**: Consistently achieves 85-95% detection accuracy on real attacks
- **Professional Terminal Interface**: Clean command-line experience with detailed reporting
- **Educational Ready**: Perfect for cybersecurity demonstrations and academic presentations

## 📊 Dataset Information

- **Source**: UNSW-NB15 Network Intrusion Dataset
- **Samples**: 82,332+ authentic network traffic records
- **Attack Types**: DoS, Exploits, Reconnaissance, Backdoors, and more
- **Features**: 45+ network traffic characteristics (protocols, bytes, duration, etc.)

## 🔧 File Structure

### Core System Files

#### `ids_fixed.py` (Main Execution)
- **Purpose**: Primary system controller and user interface
- **Functionality**: 
  - Orchestrates the entire intrusion detection workflow
  - Handles dataset loading and proper train/test splitting
  - Provides interactive batch processing with user prompts
  - Generates comprehensive detection reports and confusion matrices
  - Creates signature pink/black accuracy visualizations

#### `models_updated.py` (ML Models)
- **Purpose**: Machine learning model initialization
- **Functionality**:
  - Initializes Random Forest Classifier with optimal parameters
  - Configures Logistic Regression for binary classification
  - Provides consistent model configurations across the system
- **Models**: RandomForestClassifier, LogisticRegression

#### `data_handler_updated.py` (Data Processing)
- **Purpose**: Comprehensive data preprocessing and management
- **Functionality**:
  - Loads authentic UNSW-NB15 CSV files with fallback mechanisms
  - Handles categorical encoding (converts protocol names to numerical values)
  - Implements feature scaling and normalization
  - Manages train/test data splitting with proper stratification
  - Simulates real-time data streams for batch processing
- **Key Features**: Automatic categorical data conversion, missing value handling

#### `train_updated.py` (Training Pipeline)
- **Purpose**: Model training and evaluation framework
- **Functionality**:
  - Trains Random Forest and Logistic Regression models
  - Implements proper train/test splitting with stratification
  - Calculates comprehensive performance metrics (accuracy, precision, recall, F1)
  - Generates classification reports and confusion matrices
  - Times training processes for performance analysis
- **Outputs**: Trained models, performance metrics, evaluation reports

#### `graph_updated.py` (Visualization)
- **Purpose**: Creates comparison visualizations
- **Functionality**:
  - Generates signature pink and black themed graphs
  - Compares custom IDS performance against baseline systems
  - Creates professional dark-themed visualizations
  - Exports high-resolution plots for presentations
- **Style**: Dark background with pink (#ff69b4) and light pink (#ffc0cb) color scheme

### Interactive Demo Experience
1. **Data Loading**: Automatic detection and loading of UNSW-NB15 files
2. **Dataset Analysis**: Shows distribution of normal vs. attack traffic
3. **Model Training**: Random Forest training on authentic network data
4. **Interactive Monitoring**: Process batches with user control (press 'y' to continue)
5. **Real-time Classification**: See "SAFE" vs "MALICIOUS" detection in action
6. **Performance Report**: Comprehensive accuracy metrics and confusion matrices
