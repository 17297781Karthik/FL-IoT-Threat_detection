import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import OrderedDict, defaultdict
import argparse
import glob

import flwr as fl
from flwr.common import NDArrays, FitIns, EvaluateIns, FitRes, EvaluateRes, Parameters
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from model import NBaIoTMLP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("client_logs.log"),
        logging.StreamHandler()
    ]
)

# ========================== CONFIGURATION PARAMETERS ==========================
# These match the parameters from the ClassicalML notebook

# 1. Data Loading Parameters
DATA_PATH = "fdata"                          # Path to the dataset
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

class IoTClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, device: torch.device):
        self.client_id = client_id
        self.device = device
        self.logger = logging.getLogger(f"Client-{client_id}")
        
        # Setup dataset first to determine input size
        self.trainloader, self.testloader, self.input_size = self.load_data()
        
        # Initialize model parameters based on loaded data
        self.num_classes = len(SELECTED_CLASSES)  # 5 classes from the notebook
        
        # Initialize model using the NBaIoTMLP from the notebook
        self.model = NBaIoTMLP(
            input_dim=self.input_size, 
            num_classes=self.num_classes,
            hidden_layers=HIDDEN_LAYERS,
            neurons_per_layer=NEURONS_PER_LAYER,
            dropout_rate=DROPOUT_RATE
        ).to(device)
        
        self.logger.info(f"Client {client_id} initialized with device: {device}")
        self.logger.info(f"Model architecture: NBaIoTMLP with {HIDDEN_LAYERS} hidden layers and {NEURONS_PER_LAYER} neurons per layer")

        # Define loss function and optimizer based on notebook parameters
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=LEARNING_RATE
        )
        
        # Learning rate scheduler matching the notebook
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )

    def load_selected_data(self):
        """
        Load and preprocess data from the N-BaIoT dataset for this client's device,
        focusing only on selected classes from the notebook.
        
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
        
        self.logger.info(f"Loading data for client {self.client_id} with {len(SELECTED_CLASSES)} classes...")
        self.logger.info(f"Selected classes: {', '.join(SELECTED_CLASSES.keys())}")
        self.logger.info(f"Samples per class: {SAMPLES_PER_CLASS}")
        
        # Generate file list for this client's device and the selected classes
        files_to_load = []
        for class_name, label in SELECTED_CLASSES.items():
            filename = f'{self.client_id}.{class_to_file[class_name]}'
            file_path = os.path.join(DATA_PATH, filename)
            if os.path.exists(file_path):
                files_to_load.append((file_path, filename, class_name, label))
            else:
                self.logger.warning(f"File not found: {filename}")
        
        self.logger.info(f"Found {len(files_to_load)} files to process for client {self.client_id}")
        
        # Load and process each file
        dataframes = []
        class_samples = defaultdict(int)
        
        for file_path, filename, class_name, label in files_to_load:
            self.logger.info(f"Processing {filename}...")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Assign label
            df['label'] = label
            
            # Sample if needed
            if SAMPLES_PER_CLASS and len(df) > SAMPLES_PER_CLASS:
                # Check how many samples we already have for this class
                remaining = max(0, SAMPLES_PER_CLASS - class_samples[class_name])
                if remaining > 0:
                    df = df.sample(min(remaining, len(df)), random_state=RANDOM_SEED)
                    class_samples[class_name] += len(df)
                    self.logger.info(f"  Added {len(df)} samples with label {label} ({class_name})")
                    dataframes.append(df)
                else:
                    self.logger.info(f"  Skipped - already have {SAMPLES_PER_CLASS} samples for {class_name}")
            else:
                class_samples[class_name] += len(df)
                self.logger.info(f"  Added {len(df)} samples with label {label} ({class_name})")
                dataframes.append(df)
        
        # Combine all data
        if dataframes:
            client_data = pd.concat(dataframes, ignore_index=True)
            self.logger.info(f"\nFinal dataset shape for client {self.client_id}: {client_data.shape}")
            self.logger.info(f"Total samples per class:")
            for class_name, count in class_samples.items():
                label = SELECTED_CLASSES[class_name]
                self.logger.info(f"  {class_name} (Label {label}): {count} samples")
            return client_data
        else:
            self.logger.warning(f"No data was loaded for client {self.client_id}!")
            return None
    
    def preprocess_data(self, data):
        """
        Preprocess the dataset by selecting features and standardizing,
        following the notebook's approach.
        
        Args:
            data: DataFrame containing the dataset
            
        Returns:
            X_train, X_test, y_train, y_test: Processed data splits
            feature_names: Names of selected features
        """
        if data is None or data.empty:
            self.logger.error("No data to preprocess!")
            return None, None, None, None, None
        
        self.logger.info("Preprocessing data...")
        
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
        self.logger.info(f"Total available features: {len(available_features)}")
        
        # 3. Select specified top features
        valid_features = [f for f in TOP_FEATURES if f in available_features]
        
        if len(valid_features) < len(TOP_FEATURES):
            self.logger.warning(f"{len(TOP_FEATURES) - len(valid_features)} specified features not found in dataset")
            self.logger.warning(f"Using {len(valid_features)} valid features from the specified list")
            
            if len(valid_features) == 0:
                self.logger.warning("No valid features found! Using all available features instead.")
                valid_features = available_features[:25]  # Use the first 25 if none match
        
        self.logger.info(f"Selected {len(valid_features)} features")
        X_selected = X[valid_features]
        
        # 4. Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # 5. Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
        
        self.logger.info(f"Training data shape: {X_train.shape}")
        self.logger.info(f"Testing data shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, valid_features
        
    def load_data(self):
        """
        Load client-specific dataset based on client ID.
        Uses the notebook's approach to load and preprocess data.
        """
        try:
            self.logger.info(f"Client {self.client_id} loading dataset...")
            
            # Load selected data for this client
            client_data = self.load_selected_data()
            
            # Check if data was loaded successfully
            if client_data is None or client_data.empty:
                self.logger.error(f"No data loaded for client {self.client_id}")
                raise FileNotFoundError(f"No data loaded for client {self.client_id}")
            
            # Preprocess data
            X_train, X_test, y_train, y_test, selected_features = self.preprocess_data(client_data)
            
            # Record the input size for model initialization
            input_size = len(selected_features) if selected_features else 25  # Default to 25 features
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)
            
            # Create datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            self.logger.info(f"Number of batches in train_loader: {len(train_loader)}")
            self.logger.info(f"Number of batches in test_loader: {len(test_loader)}")
            
            return train_loader, test_loader, input_size
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def get_parameters(self, config) -> NDArrays:
        """Get model parameters as a list of NumPy arrays."""
        self.logger.info("Getting parameters from the local model...")
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy arrays."""
        self.logger.info("Updating local model with global parameters...")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[NDArrays, int, Dict]:
        """Train the model on the local dataset."""
        self.set_parameters(parameters)
        
        # Get local epochs from config or use default
        local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        self.logger.info(f"Starting local training for {local_epochs} epochs...")

        # Set model to training mode
        self.model.train()
        
        # Initialize tracking metrics
        train_losses = []
        train_accuracies = []
        
        # Training loop
        for epoch in range(local_epochs):
            correct = 0
            total = 0
            running_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                # Calculate statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            # Calculate epoch metrics
            epoch_loss = running_loss / len(self.trainloader)
            epoch_accuracy = correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            
            # Update learning rate scheduler
            self.scheduler.step(epoch_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{local_epochs}: "
                           f"Loss: {epoch_loss:.4f}, "
                           f"Accuracy: {epoch_accuracy:.4f}")
        
        # Return updated model parameters and metrics
        parameters = self.get_parameters(config={})
        metrics = {
            "train_loss": float(train_losses[-1]),
            "train_accuracy": float(train_accuracies[-1]),
            "train_examples": len(self.trainloader.dataset)
        }

        return parameters, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict]:
        """Evaluate the model on the local test dataset."""
        self.set_parameters(parameters)
        
        # Set model to evaluation mode
        self.model.eval()
        
        loss = 0.0
        correct = 0
        total = 0
        
        # No gradient calculation needed for evaluation
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                batch_loss = self.criterion(outputs, targets).item()
                loss += batch_loss * targets.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Calculate average loss and accuracy
        average_loss = loss / total
        accuracy = correct / total
        
        metrics = {
            "test_loss": float(average_loss),
            "test_accuracy": float(accuracy)
        }
        
        self.logger.info(f"Evaluation: Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return float(average_loss), total, metrics


def main():
    """Main function for client initialization and start."""
    parser = argparse.ArgumentParser(description="IoT Client")
    parser.add_argument(
        "--client-id", type=int, required=True, help="Client ID (integer)"
    )
    parser.add_argument(
        "--server-address", type=str, default="127.0.0.1:8080", help="Server address"
    )
    args = parser.parse_args()

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Start client
    client = IoTClient(client_id=args.client_id, device=device)
    fl.client.start_numpy_client(args.server_address, client=client)


if __name__ == "__main__":
    main()
