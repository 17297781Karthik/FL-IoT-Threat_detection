import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import copy

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from model import NBaIoTMLP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("federated_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FederatedLearning")

# ========================== CONFIGURATION PARAMETERS ==========================
# These match the parameters from the ClassicalML notebook

# 1. Data Loading Parameters
DATA_PATH = "fdata"                          # Path to the dataset
DEVICES_PER_CLIENT = {1: 1, 2: 2, 3: 4, 4: 5, 5: 6}  # Mapping client_id -> device_id
NUM_CLIENTS = len(DEVICES_PER_CLIENT)        # Number of clients
SAMPLES_PER_CLASS = 15000                    # Reduced samples per class per client

# 2. Selected Classes (focusing on 5 classes as in notebook)
SELECTED_CLASSES = {
    'benign': 0,              # Benign traffic
    'mirai.syn': 1,           # Mirai SYN attack
    'mirai.scan': 2,          # Mirai Scan attack
    'gafgyt.junk': 3,         # Gafgyt Junk attack
    'gafgyt.udp': 4           # Gafgyt UDP attack
}

# 3. Feature Selection - Keep the 25 most important features
TOP_FEATURES = [
    'HH_L1_weight', 'HH_jit_L1_weight', 'MI_dir_L0.01_mean', 'H_L0.1_weight', 'H_L0.1_mean',
    'MI_dir_L1_mean', 'H_L1_mean', 'H_L0.01_variance', 'MI_dir_L0.01_variance', 'H_L5_weight',
    'H_L0.1_variance', 'MI_dir_L0.1_weight', 'MI_dir_L0.1_mean', 'MI_dir_L5_weight', 'H_L0.01_mean',
    'H_L5_mean', 'MI_dir_L0.1_variance', 'MI_dir_L3_mean', 'MI_dir_L5_mean', 'HH_jit_L5_mean',
    'HH_jit_L3_mean', 'H_L3_mean', 'HH_L3_magnitude', 'MI_dir_L3_weight', 'HpHp_L0.01_weight'
]

# 4. Model Hyperparameters (matching the notebook)
HIDDEN_LAYERS = 3             # Number of hidden layers in the MLP
NEURONS_PER_LAYER = 21        # Neurons in each hidden layer
LEARNING_RATE = 0.001         # Learning rate for Adam optimizer
BATCH_SIZE = 64               # Batch size for training
LOCAL_EPOCHS = 8              # Number of local training epochs
DROPOUT_RATE = 0.3            # Dropout rate for regularization

# 5. Training Parameters
TEST_SIZE = 0.3               # Proportion of data to use for testing
RANDOM_SEED = 42              # Random seed for reproducibility
COMMUNICATION_ROUNDS = 10     # Number of federated learning rounds
EVAL_ROUNDS = 5               # Rounds between client evaluations

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set device - CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def load_selected_data(data_path, selected_classes, device_id, samples_per_class=15000):
    """
    Load and preprocess data from the N-BaIoT dataset for a specific device,
    focusing only on selected classes.
    
    Args:
        data_path: Path to the dataset files
        selected_classes: Dictionary mapping class names to their numeric labels
        device_id: Device number to load data from
        samples_per_class: Maximum number of samples to load per class
    
    Returns:
        Dataframe containing the dataset with selected classes
    """
    # Mapping from class name to file pattern
    class_to_file = {
        'benign': 'benign.csv',
        'mirai.syn': 'mirai.syn.csv',
        'mirai.scan': 'mirai.scan.csv',
        'gafgyt.junk': 'gafgyt.junk.csv',
        'gafgyt.udp': 'gafgyt.udp.csv'
    }
    
    logger.info(f"Loading data for device {device_id} with {len(selected_classes)} classes...")
    logger.info(f"Selected classes: {', '.join(selected_classes.keys())}")
    logger.info(f"Samples per class: {samples_per_class}")
    
    # Generate file list for specified device and classes
    files_to_load = []
    for class_name, label in selected_classes.items():
        filename = f'{device_id}.{class_to_file[class_name]}'
        file_path = os.path.join(data_path, filename)
        if os.path.exists(file_path):
            files_to_load.append((file_path, filename, class_name, label))
        else:
            logger.warning(f"File not found: {filename}")
    
    logger.info(f"Found {len(files_to_load)} files to process for device {device_id}")
    
    # Load and process each file
    dataframes = []
    class_samples = defaultdict(int)
    
    for file_path, filename, class_name, label in files_to_load:
        logger.info(f"Processing {filename}...")
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Assign label
        df['label'] = label
        
        # Sample if needed
        if samples_per_class and len(df) > samples_per_class:
            # Check how many samples we already have for this class
            remaining = max(0, samples_per_class - class_samples[class_name])
            if remaining > 0:
                df = df.sample(min(remaining, len(df)), random_state=RANDOM_SEED)
                class_samples[class_name] += len(df)
                logger.info(f"  Added {len(df)} samples with label {label} ({class_name})")
                dataframes.append(df)
            else:
                logger.info(f"  Skipped - already have {samples_per_class} samples for {class_name}")
        else:
            class_samples[class_name] += len(df)
            logger.info(f"  Added {len(df)} samples with label {label} ({class_name})")
            dataframes.append(df)
    
    # Combine all data
    if dataframes:
        client_data = pd.concat(dataframes, ignore_index=True)
        logger.info(f"\nFinal dataset shape for device {device_id}: {client_data.shape}")
        logger.info(f"Total samples per class:")
        for class_name, count in class_samples.items():
            label = selected_classes[class_name]
            logger.info(f"  {class_name} (Label {label}): {count} samples")
        return client_data
    else:
        logger.warning(f"No data was loaded for device {device_id}!")
        return None

def preprocess_data(data, top_features, test_size=0.3, random_seed=42):
    """
    Preprocess the dataset by selecting features and standardizing
    
    Args:
        data: DataFrame containing the dataset
        top_features: List of feature names to select
        test_size: Proportion of data for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test: Processed data splits
        feature_names: Names of selected features
        scaler: Fitted StandardScaler for future use
    """
    if data is None or data.empty:
        logger.error("No data to preprocess!")
        return None, None, None, None, None, None
    
    logger.info("Preprocessing data...")
    
    # 1. Drop any non-numeric columns (except label)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' not in numeric_cols and 'label' in data.columns:
        numeric_cols.append('label')
    
    data = data[numeric_cols]
    
    # 2. Split features and target
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Get all available features
    available_features = X.columns.tolist()
    logger.info(f"Total available features: {len(available_features)}")
    
    # 3. Select specified top features
    valid_features = [f for f in top_features if f in available_features]
    
    if len(valid_features) < len(top_features):
        logger.warning(f"{len(top_features) - len(valid_features)} specified features not found in dataset")
        logger.warning(f"Using {len(valid_features)} valid features from the specified list")
        
        if len(valid_features) == 0:
            logger.warning("No valid features found! Using all available features instead.")
            valid_features = available_features[:25]  # Use the first 25 if none match
    
    logger.info(f"Selected {len(valid_features)} features")
    X_selected = X[valid_features]
    
    # 4. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # 5. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_seed, stratify=y)
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, valid_features, scaler

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device):
    """Evaluate the model on validation or test data"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def client_update(client_id, client_model, train_loader, epochs, device):
    """Update client model locally"""
    logger.info(f"Training client {client_id} for {epochs} epochs...")
    
    # Set up training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(client_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training loop
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(client_model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Print metrics
        logger.info(f'Client {client_id}, Epoch {epoch+1}/{epochs}, '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    
    return client_model.state_dict(), {"train_loss": train_losses[-1], "train_accuracy": train_accs[-1]}

def average_weights(weights_list):
    """Average model weights from multiple clients"""
    w_avg = copy.deepcopy(weights_list[0])
    for key in w_avg.keys():
        for i in range(1, len(weights_list)):
            w_avg[key] += weights_list[i][key]
        w_avg[key] = torch.div(w_avg[key], len(weights_list))
    return w_avg

def aggregate_metrics(metrics_list):
    """Aggregate metrics from multiple clients"""
    avg_metrics = {}
    for metric in metrics_list[0].keys():
        avg_metrics[metric] = sum(client[metric] for client in metrics_list) / len(metrics_list)
    return avg_metrics

def evaluate_global_model(global_model, test_loaders, criterion, device):
    """Evaluate global model on test data from all clients"""
    losses = []
    accuracies = []
    
    for client_id, loader in test_loaders.items():
        loss, acc = evaluate(global_model, loader, criterion, device)
        losses.append(loss)
        accuracies.append(acc)
        logger.info(f"Client {client_id} - Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    
    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accuracies) / len(accuracies)
    logger.info(f"Global Model - Avg Test Loss: {avg_loss:.4f}, Avg Test Accuracy: {avg_acc:.4f}")
    
    return avg_loss, avg_acc

def generate_model_performance_report(model, test_loader, selected_classes, device):
    """Generate confusion matrix and classification report"""
    # Create class name mapping
    label_to_class = {v: k for k, v in selected_classes.items()}
    
    # Collect all predictions and actual labels
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Apply softmax to convert logits to probabilities for prediction
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Get class labels
    all_classes = sorted(np.unique(np.concatenate([all_labels, all_preds])))
    class_labels = [label_to_class.get(i, f'Unknown_{i}') for i in all_classes]
    
    # Print classification report
    report = classification_report(all_labels, all_preds,
                                  target_names=class_labels,
                                  digits=4)
    logger.info("Classification Report:")
    logger.info(report)
    
    # Calculate per-class accuracy
    logger.info("\nPer-class accuracy:")
    class_accuracies = {}
    for i in range(len(all_classes)):
        class_acc = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        class_accuracies[class_labels[i]] = class_acc
        logger.info(f"{class_labels[i]}: {class_acc:.4f}")
    
    # Calculate mean per-class accuracy
    mean_class_acc = np.mean([cm[i, i] / np.sum(cm[i, :]) for i in range(len(all_classes)) if np.sum(cm[i, :]) > 0])
    logger.info(f"\nMean per-class accuracy: {mean_class_acc:.4f}")
    
    return cm, report, class_accuracies, mean_class_acc

def run_federated_learning():
    """Run the federated learning process"""
    logger.info("Starting federated learning process...")
    logger.info(f"Total clients: {NUM_CLIENTS}")
    
    # 1. Load data for each client
    client_data = {}
    for client_id, device_id in DEVICES_PER_CLIENT.items():
        client_data[client_id] = load_selected_data(
            data_path=DATA_PATH,
            selected_classes=SELECTED_CLASSES,
            device_id=device_id,
            samples_per_class=SAMPLES_PER_CLASS
        )
    
    # 2. Preprocess data for each client
    train_loaders = {}
    test_loaders = {}
    input_dims = {}
    
    for client_id, data in client_data.items():
        if data is None:
            logger.error(f"No data for client {client_id}, skipping...")
            continue
            
        X_train, X_test, y_train, y_test, features, _ = preprocess_data(
            data=data,
            top_features=TOP_FEATURES,
            test_size=TEST_SIZE,
            random_seed=RANDOM_SEED
        )
        
        if X_train is None:
            logger.error(f"Data preprocessing failed for client {client_id}, skipping...")
            continue
        
        # Store input dimension
        input_dims[client_id] = X_train.shape[1]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create dataloaders
        train_loaders[client_id] = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loaders[client_id] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        logger.info(f"Client {client_id} dataloaders created successfully")
    
    # Check if we have any valid clients
    if not train_loaders:
        logger.error("No valid clients with data, exiting...")
        return
    
    # 3. Initialize global model
    # Use the first client's input dimension and standard number of classes
    first_client_id = list(input_dims.keys())[0]
    input_dim = input_dims[first_client_id]
    num_classes = len(SELECTED_CLASSES)
    
    global_model = NBaIoTMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=HIDDEN_LAYERS,
        neurons_per_layer=NEURONS_PER_LAYER,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    logger.info(f"Global model initialized with input_dim={input_dim}, num_classes={num_classes}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # 4. Federated learning rounds
    global_train_losses = []
    global_test_losses = []
    global_test_accuracies = []
    
    for round_num in range(COMMUNICATION_ROUNDS):
        logger.info(f"\n--- ROUND {round_num + 1}/{COMMUNICATION_ROUNDS} ---")
        
        # Distribute global model to all clients
        client_models = {}
        for client_id in train_loaders.keys():
            client_models[client_id] = copy.deepcopy(global_model)
        
        # Update each client's model locally
        client_weights = []
        client_metrics = []
        
        for client_id, client_model in client_models.items():
            weights, metrics = client_update(
                client_id=client_id,
                client_model=client_model,
                train_loader=train_loaders[client_id],
                epochs=LOCAL_EPOCHS,
                device=device
            )
            client_weights.append(weights)
            client_metrics.append(metrics)
        
        # Aggregate client models
        global_weights = average_weights(client_weights)
        global_model.load_state_dict(global_weights)
        
        # Aggregate metrics
        aggregated_metrics = aggregate_metrics(client_metrics)
        global_train_losses.append(aggregated_metrics["train_loss"])
        
        logger.info(f"Round {round_num + 1} - Aggregated train loss: {aggregated_metrics['train_loss']:.4f}, "
                   f"Aggregated train accuracy: {aggregated_metrics['train_accuracy']:.4f}")
        
        # Periodically evaluate global model
        if (round_num + 1) % EVAL_ROUNDS == 0 or round_num == COMMUNICATION_ROUNDS - 1:
            test_loss, test_acc = evaluate_global_model(global_model, test_loaders, criterion, device)
            global_test_losses.append(test_loss)
            global_test_accuracies.append(test_acc)
    
    # Save the final global model
    torch.save(global_model.state_dict(), "best_traffic_model.pt")
    logger.info("Global model saved as 'best_traffic_model.pt'")
    
    # 5. Final evaluation and performance analysis
    logger.info("\n=== FINAL MODEL EVALUATION ===")
    
    # Combine all test data for final evaluation
    combined_test_data = []
    combined_test_labels = []
    
    for test_loader in test_loaders.values():
        for inputs, labels in test_loader:
            combined_test_data.append(inputs)
            combined_test_labels.append(labels)
    
    combined_inputs = torch.cat(combined_test_data, 0)
    combined_labels = torch.cat(combined_test_labels, 0)
    
    combined_dataset = TensorDataset(combined_inputs, combined_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Generate final performance report
    cm, report, class_accs, mean_class_acc = generate_model_performance_report(
        global_model, combined_loader, SELECTED_CLASSES, device
    )
    
    logger.info("\nFederated learning process completed successfully!")
    
    return {
        "global_model": global_model,
        "training_losses": global_train_losses,
        "test_losses": global_test_losses,
        "test_accuracies": global_test_accuracies,
        "confusion_matrix": cm,
        "classification_report": report,
        "class_accuracies": class_accs,
        "mean_class_accuracy": mean_class_acc
    }

if __name__ == "__main__":
    results = run_federated_learning()
    
    # Plot training and test metrics
    if results:
        plt.figure(figsize=(12, 5))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(results["training_losses"], label='Training Loss')
        plt.xlabel('Communication Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Global Model Training Loss')
        
        # Plot test accuracy
        plt.subplot(1, 2, 2)
        rounds = list(range(EVAL_ROUNDS, COMMUNICATION_ROUNDS + 1, EVAL_ROUNDS))
        if len(rounds) > len(results["test_accuracies"]):
            rounds = rounds[:len(results["test_accuracies"])]
        plt.plot(rounds, results["test_accuracies"], marker='o', label='Test Accuracy')
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Global Model Test Accuracy')
        
        plt.tight_layout()
        plt.savefig('federated_learning_results.png')
        plt.close()
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        label_to_class = {v: k for k, v in SELECTED_CLASSES.items()}
        class_labels = [label_to_class[i] for i in range(len(SELECTED_CLASSES))]
        
        sns.heatmap(results["confusion_matrix"], annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Global Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('federated_confusion_matrix.png')
        plt.close()
        
        logger.info("Results plots saved to 'federated_learning_results.png' and 'federated_confusion_matrix.png'")