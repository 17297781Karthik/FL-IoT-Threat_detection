import os
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import flwr as fl
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes, parameters_to_ndarrays, NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Server")

# Make sure directories exist
os.makedirs("SavedGlobalModel", exist_ok=True)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


class SaveModelStrategy(FedAvg):
    """Federated Averaging strategy that saves the final model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters = None
        logger.info("SaveModelStrategy initialized")

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate model weights and metrics from clients."""
        logger.info(f"Round {server_round}: Aggregating updates from {len(results)} clients")

        if not results:
            logger.warning(f"Round {server_round}: No results received from clients")
            return None, {}

        # Call aggregate_fit from parent class (FedAvg)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
            logger.info(f"Round {server_round}: Global model parameters updated")

            # Save checkpoint after each round
            if server_round % 5 == 0:
                save_parameters_to_model(
                    self.final_parameters,
                    f"SavedGlobalModel/model_round_{server_round}.pth"
                )
                logger.info(f"Round {server_round}: Saved checkpoint to model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation metrics from clients."""
        if not results:
            logger.warning(f"Round {server_round}: No evaluation results received")
            return None, {}

        logger.info(f"Round {server_round}: Aggregating evaluation results from {len(results)} clients")

        # Call aggregate_evaluate from parent class
        return super().aggregate_evaluate(server_round, results, failures)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calculate weighted average of metrics."""
    logger.debug(f"Computing weighted average from {len(metrics)} client metrics")

    if not metrics:
        return {}

    # Sum of weights
    total_examples = sum([num_examples for num_examples, _ in metrics])

    # Weighted average for each metric
    weighted_metrics = {}

    for metric_name in metrics[0][1].keys():
        weighted_sum = sum(num_examples * m[metric_name] for num_examples, m in metrics if metric_name in m)
        weighted_metrics[metric_name] = weighted_sum / total_examples

    logger.info(f"Aggregated metrics: {weighted_metrics}")
    return weighted_metrics


def save_parameters_to_model(parameters: Parameters, filename: str) -> None:
    """Save Flower parameters to PyTorch model."""
    model = NeuralNetwork()
    params_dict = zip(model.state_dict().keys(), parameters_to_ndarrays(parameters))
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    torch.save(model.state_dict(), filename)


def get_model_parameters(model: torch.nn.Module) -> NDArrays:
    """Get model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def main():
    # Define strategy
    strategy = SaveModelStrategy(
        evaluate_metrics_aggregation_fn=weighted_average,
        min_available_clients=2,
        min_fit_clients=2,
        min_evaluate_clients=2,
    )

    # Define server configuration
    config = ServerConfig(num_rounds=10)

    # Start server
    server_address = "0.0.0.0:8080"
    logger.info(f"Starting server at {server_address}")

    # Start Flower server
    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )

    # After training is complete, save the final model
    if strategy.final_parameters is not None:
        save_parameters_to_model(strategy.final_parameters, "SavedGlobalModel/final_model.pth")
        logger.info("Final model saved to SavedGlobalModel/final_model.pth")
    else:
        logger.error("No model parameters available to save")


if __name__ == "__main__":
    main()