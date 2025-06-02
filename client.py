import os
import torch
import torch.nn as nn
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import OrderedDict
import argparse
import glob

import flwr as fl
from flwr.common import NDArrays, FitIns, EvaluateIns, FitRes, EvaluateRes, Parameters
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from model import MalwareNetSmall  # Import the new model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("client_logs.log"),
        logging.StreamHandler()
    ]
)

# Attack labels mapping
ATT_LABEL = {
    '1.gafgyt.combo.csv': 1,
    '1.gafgyt.junk.csv': 2,
    '1.gafgyt.tcp.csv': 3,
    '1.gafgyt.udp.csv': 4,
    '1.mirai.ack.csv': 5,
    '1.mirai.scan.csv': 6,
    '1.mirai.syn.csv': 7,
    '1.mirai.udp.csv': 8,
    '1.mirai.udpplain.csv': 9
}

class IoTClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, device: torch.device):
        self.client_id = client_id
        self.device = device
        self.logger = logging.getLogger(f"Client-{client_id}")

        # Initialize model parameters (will be updated after loading data)
        self.input_size = 115  # Default size for IoT data features
        self.num_classes = 10  # 0-9 classes (including benign as 0)
        
        # Initialize model
        self.model = MalwareNetSmall(input_size=self.input_size, num_classes=self.num_classes).to(device)
        self.logger.info(f"Client {client_id} initialized with device: {device}")

        # Setup dataset
        self.trainloader, self.testloader = self.load_data()

        # Define loss function and optimizer with weight decay for regularization
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            verbose=True
        )

    def load_data(self):
        """
        Load client-specific IoT dataset based on client ID.
        Each client loads all files starting with their ID from the fdata/ directory.
        """
        try:
            self.logger.info(f"Client {self.client_id} loading dataset...")
            
            # Path to data directory
            data_dir = "fdata/"
            
            # Get all files for this client
            file_pattern = f"{self.client_id}*.csv"
            client_files = glob.glob(os.path.join(data_dir, file_pattern))
            
            # Filter out gafgytscan.csv
            client_files = [f for f in client_files if "gafgytscan.csv" not in f]
            
            if not client_files:
                self.logger.error(f"No data files found for client {self.client_id}")
                raise FileNotFoundError(f"No data files found for client {self.client_id}")
            
            self.logger.info(f"Found {len(client_files)} files for client {self.client_id}: {client_files}")
            
            # Process all CSV files and combine them
            all_data = []
            all_labels = []
            
            for file_path in client_files:
                file_name = os.path.basename(file_path)
                
                # Determine label from filename
                attack_type = None
                for key in ATT_LABEL:
                    if key in file_name:
                        attack_type = ATT_LABEL[key]
                        break
                
                # If no attack type found, consider it benign (label 0)
                if attack_type is None:
                    attack_type = 0
                
                # Read data
                try:
                    df = pd.read_csv(file_path)
                    self.logger.info(f"Loaded file {file_path} with shape {df.shape}")
                    
                    # Remove any non-feature columns if they exist
                    if 'label' in df.columns:
                        df = df.drop('label', axis=1)
                    
                    # Store the input size for the model
                    if len(all_data) == 0:
                        self.input_size = df.shape[1]
                        self.logger.info(f"Setting input size to {self.input_size}")
                        
                    # Add data and labels
                    all_data.append(df.values)
                    all_labels.extend([attack_type] * len(df))
                    
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
            
            if not all_data:
                self.logger.error("No data could be loaded")
                raise ValueError("No data could be loaded")
              # Combine all data
            X = np.vstack(all_data)
            y = np.array(all_labels)
            
            self.logger.info(f"Combined data shape: {X.shape}, labels shape: {y.shape}")
            
            # Sample only 30% of the data for faster processing while maintaining representativeness
            # Use stratified sampling to maintain class distribution
            _, X_sampled, _, y_sampled = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            self.logger.info(f"Sampled 30% of data. New shape: {X_sampled.shape}, labels shape: {y_sampled.shape}")
            
            # Use the sampled data for further processing
            X = X_sampled
            y = y_sampled
            
            # Normalize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Handle class imbalance
            class_counts = np.bincount(y_train)
            class_weights = 1.0 / np.array(class_counts)
            weights = class_weights[y_train]
            weights = weights / weights.sum() * len(weights)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)
              # Create data loaders with larger batch sizes for faster training
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            
            trainloader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
            testloader = DataLoader(test_dataset, batch_size=512)
            
            # Recreate model with correct input size
            self.model = MalwareNetSmall(input_size=self.input_size, num_classes=self.num_classes).to(self.device)
            
            self.logger.info(
                f"Client {self.client_id} loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples"
            )
            self.logger.info(f"Data has {self.input_size} features and {self.num_classes} classes")
            
            return trainloader, testloader
            
        except Exception as e:
            self.logger.error(f"Error loading data for client {self.client_id}: {e}")
            raise

    def get_parameters(self, config) -> List[np.ndarray]:
        """Get model parameters as a list of NumPy arrays."""
        self.logger.debug(f"Client {self.client_id} getting model parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays."""
        self.logger.debug(f"Client {self.client_id} setting model parameters")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on the local dataset."""
        self.logger.info(f"Client {self.client_id} starting local training")

        # Update local model with global parameters
        self.set_parameters(parameters)        # Get training config from server
        epochs = 1  # Slightly more epochs for better accuracy with less data
        
        # Mixed precision for faster training if available
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        use_amp = scaler is not None

        # Train the model
        self.model.train()
        epoch_losses = []
        best_accuracy = 0
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)  # More efficient
                
                if use_amp:
                    # Use mixed precision for faster training
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    
                    # Scale loss and do backward pass
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # Regular training path
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                # Compute metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Print batch progress every 50 batches (reduced from 100 for quicker feedback)
                if batch_idx % 50 == 0:
                    self.logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            # Log epoch metrics
            accuracy = correct / total
            avg_loss = running_loss / len(self.trainloader)
            self.logger.info(
                f"Client {self.client_id} - Epoch {epoch + 1}/{epochs}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            
            # Update learning rate
            epoch_losses.append(avg_loss)
            self.scheduler.step(avg_loss)

        # Return updated model parameters and metrics
        final_accuracy = correct / total
        return self.get_parameters(config={}), total, {"accuracy": float(final_accuracy), "loss": float(avg_loss)}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """Evaluate the model on the local test dataset."""
        self.logger.info(f"Client {self.client_id} evaluating model")

        # Update local model with global parameters
        self.set_parameters(parameters)

        # Evaluate the model
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        
        # Track per-class metrics
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                batch_loss = self.criterion(outputs, targets).item()
                loss += batch_loss

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Calculate per-class accuracy
                for i in range(len(targets)):
                    label = targets[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

        # Compute metrics
        accuracy = correct / total
        avg_loss = loss / len(self.testloader)
        
        # Log per-class accuracy
        for i in range(self.num_classes):
            if class_total[i] > 0:
                class_acc = 100.0 * class_correct[i] / class_total[i]
                self.logger.info(f"Class {i} accuracy: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
                
        self.logger.info(f"Client {self.client_id} evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return float(avg_loss), total, {"accuracy": float(accuracy)}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flower IoT client")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Server address")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(f"Client-{args.client_id}")
    logger.info(f"Using device: {device}")

    # Create client
    client = IoTClient(client_id=args.client_id, device=device)

    # Start client
    logger.info(f"Client {args.client_id} connecting to server at {args.server_address}")
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()