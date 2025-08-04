import numpy as np
import sys
import os
import signal
import logging
import re
from train_updated import train_random_forest, train_logistic_regression
from data_handler_updated import load_data, preprocess_data, simulate_data_stream
from sklearn.metrics import classification_report, confusion_matrix
from models_updated import initialize_random_forest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

class CleanFormatter(logging.Formatter):
    def format(self, record):
        original = super(CleanFormatter, self).format(record)
        return strip_ansi_codes(original)

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_directory = os.path.expanduser('~/Downloads')
    log_filename = f"ids_system_{timestamp}.log"
    log_file_path = os.path.join(log_directory, log_filename)

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    file_formatter = CleanFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
    logging.info(f"\nLogging to {log_file_path}")

setup_logging()

signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

def run_ids():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for any UNSW file in the directory
    unsw_files = [f for f in os.listdir('.') if 'UNSW' in f and f.endswith('.csv')]
    if not unsw_files:
        print("No UNSW-NB15 CSV files found!")
        return
    
    data_file = unsw_files[0]  # Use the first UNSW file found
    print(f"Using dataset: {data_file}")
    
    logging.info("Loading authentic UNSW-NB15 data: {}".format(data_file))
    full_data = load_data(data_file)
    
    if full_data is None:
        print("Could not load data.")
        return
    
    print(f"Total dataset size: {len(full_data)} samples")
    
    # Check label distribution
    if 'label' in full_data.columns:
        print("Label distribution in full dataset:")
        print(full_data['label'].value_counts())
    elif 'Label' in full_data.columns:
        print("Label distribution in full dataset:")
        print(full_data['Label'].value_counts())
    
    # Preprocess the full dataset
    features, labels = preprocess_data(full_data, fit_scaler=True, is_train=True)
    if features is None or labels is None:
        print("Failed to preprocess data.")
        return
    
    # Split into training and testing sets properly
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"\nDataset split:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    print(f"Training attacks: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"Testing attacks: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
    
    # Train model
    model = initialize_random_forest()
    model.fit(X_train, y_train)
    
    logging.info("Training completed. Starting continuous monitoring...")
    
    # Create test batches from the proper test set
    batch_size = 100
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    batch_count = 0
    batch_accuracies = []
    
    # Process test data in batches
    for i in range(0, len(X_test), batch_size):
        batch_features = X_test[i:i+batch_size]
        batch_labels = y_test[i:i+batch_size]
        
        if len(batch_features) == 0:
            break
            
        predictions = model.predict(batch_features)
        batch_acc = accuracy_score(batch_labels, predictions)
        correct = sum(predictions == batch_labels)
        
        batch_count += 1
        total_correct += correct
        total_samples += len(batch_labels)
        all_preds.extend(predictions)
        all_labels.extend(batch_labels)
        batch_accuracies.append(batch_acc)
        
        print(f"\nBatch {batch_count} accuracy: {batch_acc:.2f}")
        print(f"Correct: {correct} / {len(batch_labels)}")
        print(f"Attacks in batch: {sum(batch_labels)} ({sum(batch_labels)/len(batch_labels)*100:.1f}%)")
        
        print("\nIndex | Feature     | Prediction | Actual | Status")
        print("------------------------------------------------------")
        
        for j, (pred, actual) in enumerate(zip(predictions[:10], batch_labels[:10])):  # Show first 10
            feature_name = f"Sample_{j}"
            status = "MALICIOUS" if pred == 1 else "SAFE"
            print(f"{j:<5} | {feature_name:<11} | {pred:^10} | {actual:^6} | {status}")
        
        if batch_count >= 6:  # Limit to 6 batches for demo
            break
            
        cont = input("Continue to next batch? (y/n): ").strip().lower()
        if cont != 'y':
            break
    
    overall_acc = total_correct / total_samples if total_samples else 0
    print("\n--- Detection Summary ---")
    print(f"Batches processed: {batch_count}")
    print(f"Total samples    : {total_samples}")
    print(f"Correct          : {total_correct}")
    print(f"Incorrect        : {total_samples - total_correct}")
    print(f"Overall accuracy : {overall_acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    if overall_acc > 0.8:
        print("🎉 IDS system is performing well!")
    else:
        print("⚠️  IDS system accuracy could be improved.")
    
    # Plotting pink and black accuracy graph
    if batch_accuracies:
        batch_numbers = list(range(1, len(batch_accuracies) + 1))
        
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 6))
        
        plt.plot(batch_numbers, batch_accuracies, marker='o', color='#ff69b4', linewidth=2, label='Custom IDS Accuracy')
        plt.title('Our IDS Accuracy with Authentic UNSW-NB15 Data', fontsize=16, color='white')
        plt.xlabel('Batch Number', fontsize=12, color='white')
        plt.ylabel('Accuracy', fontsize=12, color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        
        plt.savefig('ids_accuracy_graph.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n🎯 Final Results:")
        print(f"   Average Accuracy: {np.mean(batch_accuracies):.2f}")
        print(f"   Best Batch: {max(batch_accuracies):.2f}")
        print(f"   Graph saved as: ids_accuracy_graph.png")

if __name__ == "__main__":
    run_ids()