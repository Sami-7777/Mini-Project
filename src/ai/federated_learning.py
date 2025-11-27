"""
Federated Learning for privacy-preserving collaborative cyberattack detection.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import structlog
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from cryptography.fernet import Fernet

from ..database.models import AttackType, SeverityLevel
from ..database.connection import get_database
from ..security.encryption import encryption_manager

logger = structlog.get_logger(__name__)


class FederatedAlgorithm(str, Enum):
    """Federated learning algorithms."""
    FEDAVG = "federated_averaging"
    FEDPROX = "federated_proximal"
    FEDOPT = "federated_optimization"
    SECURE_AGGREGATION = "secure_aggregation"


@dataclass
class ClientModel:
    """Client model in federated learning."""
    client_id: str
    model_weights: Dict[str, torch.Tensor]
    model_version: str
    training_samples: int
    last_update: datetime
    performance_metrics: Dict[str, float]
    privacy_budget: float


@dataclass
class FederatedRound:
    """Federated learning round."""
    round_id: int
    selected_clients: List[str]
    global_model_weights: Dict[str, torch.Tensor]
    aggregation_method: str
    privacy_budget_used: float
    performance_improvement: float
    timestamp: datetime


class FederatedNeuralNetwork(nn.Module):
    """Neural network for federated learning."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(FederatedNeuralNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward pass."""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class DifferentialPrivacy:
    """Differential privacy for federated learning."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise(self, gradients: Dict[str, torch.Tensor], 
                  sensitivity: float = 1.0) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to gradients."""
        try:
            noisy_gradients = {}
            
            for param_name, gradient in gradients.items():
                # Calculate noise scale
                noise_scale = (2 * sensitivity * np.log(1.25 / self.delta)) / self.epsilon
                
                # Add Gaussian noise
                noise = torch.normal(0, noise_scale, size=gradient.shape)
                noisy_gradients[param_name] = gradient + noise
            
            return noisy_gradients
            
        except Exception as e:
            logger.error("Error adding differential privacy noise", error=str(e))
            raise
    
    def calculate_sensitivity(self, gradients: Dict[str, torch.Tensor]) -> float:
        """Calculate L2 sensitivity of gradients."""
        try:
            total_norm = 0.0
            
            for gradient in gradients.values():
                total_norm += torch.norm(gradient).item() ** 2
            
            return np.sqrt(total_norm)
            
        except Exception as e:
            logger.error("Error calculating sensitivity", error=str(e))
            return 1.0


class SecureAggregation:
    """Secure aggregation for federated learning."""
    
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
    
    def encrypt_model_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, bytes]:
        """Encrypt model weights."""
        try:
            encrypted_weights = {}
            
            for param_name, weight in weights.items():
                # Serialize tensor
                weight_bytes = weight.detach().cpu().numpy().tobytes()
                
                # Encrypt
                encrypted_weight = self.fernet.encrypt(weight_bytes)
                encrypted_weights[param_name] = encrypted_weight
            
            return encrypted_weights
            
        except Exception as e:
            logger.error("Error encrypting model weights", error=str(e))
            raise
    
    def decrypt_model_weights(self, encrypted_weights: Dict[str, bytes], 
                            shape_info: Dict[str, Tuple]) -> Dict[str, torch.Tensor]:
        """Decrypt model weights."""
        try:
            decrypted_weights = {}
            
            for param_name, encrypted_weight in encrypted_weights.items():
                # Decrypt
                weight_bytes = self.fernet.decrypt(encrypted_weight)
                
                # Deserialize tensor
                weight_array = np.frombuffer(weight_bytes, dtype=np.float32)
                weight_tensor = torch.from_numpy(weight_array.reshape(shape_info[param_name]))
                
                decrypted_weights[param_name] = weight_tensor
            
            return decrypted_weights
            
        except Exception as e:
            logger.error("Error decrypting model weights", error=str(e))
            raise
    
    def secure_aggregate(self, client_weights: List[Dict[str, torch.Tensor]], 
                        weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Securely aggregate client model weights."""
        try:
            if weights is None:
                weights = [1.0] * len(client_weights)
            
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Initialize aggregated weights
            aggregated_weights = {}
            for param_name in client_weights[0].keys():
                aggregated_weights[param_name] = torch.zeros_like(client_weights[0][param_name])
            
            # Weighted average
            for client_weight, weight in zip(client_weights, normalized_weights):
                for param_name, param_weight in client_weight.items():
                    aggregated_weights[param_name] += weight * param_weight
            
            return aggregated_weights
            
        except Exception as e:
            logger.error("Error in secure aggregation", error=str(e))
            raise


class FederatedLearningEngine:
    """Main engine for federated learning."""
    
    def __init__(self):
        self.global_model = None
        self.client_models = {}
        self.federated_rounds = []
        self.differential_privacy = DifferentialPrivacy()
        self.secure_aggregation = SecureAggregation()
        self.current_round = 0
        self.is_initialized = False
    
    async def initialize_federated_learning(self, input_size: int, hidden_size: int, 
                                          output_size: int) -> None:
        """Initialize federated learning system."""
        try:
            # Initialize global model
            self.global_model = FederatedNeuralNetwork(input_size, hidden_size, output_size)
            
            # Initialize client models
            await self._initialize_client_models()
            
            self.is_initialized = True
            logger.info("Federated learning system initialized")
            
        except Exception as e:
            logger.error("Error initializing federated learning", error=str(e))
            raise
    
    async def _initialize_client_models(self):
        """Initialize client models."""
        try:
            # Get registered clients from database
            db = await get_database()
            collection = db.get_collection("federated_clients")
            
            cursor = collection.find({"status": "active"})
            async for client_doc in cursor:
                client_id = client_doc["client_id"]
                
                # Initialize client model
                client_model = ClientModel(
                    client_id=client_id,
                    model_weights=self._get_model_weights(self.global_model),
                    model_version="1.0.0",
                    training_samples=0,
                    last_update=datetime.utcnow(),
                    performance_metrics={},
                    privacy_budget=1.0
                )
                
                self.client_models[client_id] = client_model
            
            logger.info("Client models initialized", num_clients=len(self.client_models))
            
        except Exception as e:
            logger.error("Error initializing client models", error=str(e))
    
    async def run_federated_round(self, selected_clients: Optional[List[str]] = None,
                                aggregation_method: str = "fedavg") -> FederatedRound:
        """Run a federated learning round."""
        try:
            if not self.is_initialized:
                raise ValueError("Federated learning not initialized")
            
            # Select clients
            if selected_clients is None:
                selected_clients = await self._select_clients()
            
            # Train on selected clients
            client_updates = await self._train_clients(selected_clients)
            
            # Aggregate updates
            if aggregation_method == "fedavg":
                aggregated_weights = self._federated_averaging(client_updates)
            elif aggregation_method == "fedprox":
                aggregated_weights = self._federated_proximal(client_updates)
            elif aggregation_method == "secure_aggregation":
                aggregated_weights = self._secure_aggregation(client_updates)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            # Update global model
            self._update_global_model(aggregated_weights)
            
            # Calculate performance improvement
            performance_improvement = await self._calculate_performance_improvement()
            
            # Create federated round record
            federated_round = FederatedRound(
                round_id=self.current_round,
                selected_clients=selected_clients,
                global_model_weights=aggregated_weights,
                aggregation_method=aggregation_method,
                privacy_budget_used=0.1,  # Would calculate actual privacy budget used
                performance_improvement=performance_improvement,
                timestamp=datetime.utcnow()
            )
            
            self.federated_rounds.append(federated_round)
            self.current_round += 1
            
            logger.info("Federated round completed", 
                       round_id=federated_round.round_id,
                       selected_clients=len(selected_clients),
                       performance_improvement=performance_improvement)
            
            return federated_round
            
        except Exception as e:
            logger.error("Error running federated round", error=str(e))
            raise
    
    async def _select_clients(self, fraction: float = 0.3) -> List[str]:
        """Select clients for federated round."""
        try:
            # Select clients based on various criteria
            available_clients = list(self.client_models.keys())
            
            # Random selection
            num_selected = max(1, int(len(available_clients) * fraction))
            selected_clients = np.random.choice(
                available_clients, 
                size=num_selected, 
                replace=False
            ).tolist()
            
            return selected_clients
            
        except Exception as e:
            logger.error("Error selecting clients", error=str(e))
            return []
    
    async def _train_clients(self, selected_clients: List[str]) -> List[Dict[str, torch.Tensor]]:
        """Train selected clients."""
        try:
            client_updates = []
            
            for client_id in selected_clients:
                # Simulate client training
                client_update = await self._simulate_client_training(client_id)
                client_updates.append(client_update)
            
            return client_updates
            
        except Exception as e:
            logger.error("Error training clients", error=str(e))
            return []
    
    async def _simulate_client_training(self, client_id: str) -> Dict[str, torch.Tensor]:
        """Simulate client training (placeholder)."""
        try:
            # In a real implementation, this would communicate with actual clients
            # For now, we'll simulate the training process
            
            # Get current global model weights
            global_weights = self._get_model_weights(self.global_model)
            
            # Simulate local training by adding small random updates
            client_weights = {}
            for param_name, weight in global_weights.items():
                # Add small random noise to simulate local training
                noise = torch.randn_like(weight) * 0.01
                client_weights[param_name] = weight + noise
            
            # Apply differential privacy
            client_weights = self.differential_privacy.add_noise(client_weights)
            
            return client_weights
            
        except Exception as e:
            logger.error("Error simulating client training", error=str(e))
            raise
    
    def _federated_averaging(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Federated averaging aggregation."""
        try:
            # Simple averaging
            aggregated_weights = {}
            
            for param_name in client_updates[0].keys():
                param_sum = torch.zeros_like(client_updates[0][param_name])
                
                for client_update in client_updates:
                    param_sum += client_update[param_name]
                
                aggregated_weights[param_name] = param_sum / len(client_updates)
            
            return aggregated_weights
            
        except Exception as e:
            logger.error("Error in federated averaging", error=str(e))
            raise
    
    def _federated_proximal(self, client_updates: List[Dict[str, torch.Tensor]], 
                          mu: float = 0.01) -> Dict[str, torch.Tensor]:
        """Federated proximal aggregation."""
        try:
            # FedProx adds a proximal term to the loss
            global_weights = self._get_model_weights(self.global_model)
            aggregated_weights = {}
            
            for param_name in client_updates[0].keys():
                param_sum = torch.zeros_like(client_updates[0][param_name])
                
                for client_update in client_updates:
                    # Add proximal term
                    proximal_term = mu * (client_update[param_name] - global_weights[param_name])
                    param_sum += client_update[param_name] - proximal_term
                
                aggregated_weights[param_name] = param_sum / len(client_updates)
            
            return aggregated_weights
            
        except Exception as e:
            logger.error("Error in federated proximal", error=str(e))
            raise
    
    def _secure_aggregation(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Secure aggregation with encryption."""
        try:
            # Encrypt client updates
            encrypted_updates = []
            shape_info = {}
            
            for client_update in client_updates:
                encrypted_update = self.secure_aggregation.encrypt_model_weights(client_update)
                encrypted_updates.append(encrypted_update)
                
                # Store shape information
                if not shape_info:
                    for param_name, weight in client_update.items():
                        shape_info[param_name] = weight.shape
            
            # Aggregate encrypted updates
            aggregated_encrypted = {}
            for param_name in encrypted_updates[0].keys():
                param_sum = b""
                
                for encrypted_update in encrypted_updates:
                    param_sum += encrypted_update[param_name]
                
                aggregated_encrypted[param_name] = param_sum
            
            # Decrypt aggregated result
            aggregated_weights = self.secure_aggregation.decrypt_model_weights(
                aggregated_encrypted, shape_info
            )
            
            # Normalize
            for param_name in aggregated_weights.keys():
                aggregated_weights[param_name] /= len(client_updates)
            
            return aggregated_weights
            
        except Exception as e:
            logger.error("Error in secure aggregation", error=str(e))
            raise
    
    def _update_global_model(self, aggregated_weights: Dict[str, torch.Tensor]):
        """Update global model with aggregated weights."""
        try:
            # Update global model parameters
            with torch.no_grad():
                for param_name, param in self.global_model.named_parameters():
                    if param_name in aggregated_weights:
                        param.data = aggregated_weights[param_name].clone()
            
        except Exception as e:
            logger.error("Error updating global model", error=str(e))
            raise
    
    def _get_model_weights(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Get model weights."""
        try:
            weights = {}
            for name, param in model.named_parameters():
                weights[name] = param.data.clone()
            return weights
            
        except Exception as e:
            logger.error("Error getting model weights", error=str(e))
            return {}
    
    async def _calculate_performance_improvement(self) -> float:
        """Calculate performance improvement after federated round."""
        try:
            # This would calculate actual performance improvement
            # For now, return a simulated improvement
            return np.random.uniform(0.01, 0.05)
            
        except Exception as e:
            logger.error("Error calculating performance improvement", error=str(e))
            return 0.0
    
    async def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """Register a new client for federated learning."""
        try:
            db = await get_database()
            collection = db.get_collection("federated_clients")
            
            # Check if client already exists
            existing_client = await collection.find_one({"client_id": client_id})
            
            if existing_client:
                logger.warning("Client already registered", client_id=client_id)
                return False
            
            # Register new client
            client_doc = {
                "client_id": client_id,
                "client_info": client_info,
                "status": "active",
                "registered_at": datetime.utcnow(),
                "last_seen": datetime.utcnow(),
                "privacy_budget": 1.0,
                "performance_metrics": {}
            }
            
            await collection.insert_one(client_doc)
            
            # Add to client models
            client_model = ClientModel(
                client_id=client_id,
                model_weights=self._get_model_weights(self.global_model),
                model_version="1.0.0",
                training_samples=0,
                last_update=datetime.utcnow(),
                performance_metrics={},
                privacy_budget=1.0
            )
            
            self.client_models[client_id] = client_model
            
            logger.info("Client registered successfully", client_id=client_id)
            return True
            
        except Exception as e:
            logger.error("Error registering client", error=str(e))
            return False
    
    async def get_federated_statistics(self) -> Dict[str, Any]:
        """Get federated learning statistics."""
        try:
            return {
                'total_clients': len(self.client_models),
                'active_clients': len([c for c in self.client_models.values() if c.privacy_budget > 0]),
                'completed_rounds': len(self.federated_rounds),
                'current_round': self.current_round,
                'is_initialized': self.is_initialized,
                'privacy_budget_remaining': sum(c.privacy_budget for c in self.client_models.values()),
                'average_performance_improvement': np.mean([r.performance_improvement for r in self.federated_rounds]) if self.federated_rounds else 0.0
            }
            
        except Exception as e:
            logger.error("Error getting federated statistics", error=str(e))
            return {}


# Global federated learning engine instance
federated_learning_engine = FederatedLearningEngine()
