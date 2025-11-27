"""
Graph Neural Networks for advanced relationship analysis in cyberattack detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import structlog
from dataclasses import dataclass
from enum import Enum

from ..database.models import AttackType, SeverityLevel
from ..database.connection import get_database

logger = structlog.get_logger(__name__)


class GraphType(str, Enum):
    """Types of graphs for analysis."""
    URL_RELATIONSHIP = "url_relationship"
    IP_RELATIONSHIP = "ip_relationship"
    DOMAIN_RELATIONSHIP = "domain_relationship"
    USER_BEHAVIOR = "user_behavior"
    THREAT_CORRELATION = "threat_correlation"


@dataclass
class GraphNode:
    """Graph node representation."""
    node_id: str
    node_type: str
    features: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class GraphEdge:
    """Graph edge representation."""
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    features: Dict[str, float]
    timestamp: datetime


@dataclass
class GraphData:
    """Complete graph data structure."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    graph_type: GraphType
    metadata: Dict[str, Any]


class GCNModel(nn.Module):
    """Graph Convolutional Network for cyberattack detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the network."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


class GATModel(nn.Module):
    """Graph Attention Network for cyberattack detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_heads: int = 4, num_layers: int = 3):
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.5))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.5))
        
        # Output layer
        self.convs.append(GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=0.5))
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the network."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


class GraphSAGEModel(nn.Module):
    """GraphSAGE model for cyberattack detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super(GraphSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GraphSAGE(input_dim, hidden_dim, num_layers=1))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim, num_layers=1))
        
        # Output layer
        self.convs.append(GraphSAGE(hidden_dim, output_dim, num_layers=1))
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the network."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


class GraphNeuralNetworkEngine:
    """Engine for graph neural network analysis."""
    
    def __init__(self):
        self.models = {
            'gcn': GCNModel,
            'gat': GATModel,
            'graphsage': GraphSAGEModel
        }
        self.trained_models = {}
        self.graph_cache = {}
        self.node_embeddings = {}
    
    async def build_url_relationship_graph(self, urls: List[str], 
                                         analysis_results: List[Dict]) -> GraphData:
        """Build URL relationship graph."""
        try:
            nodes = []
            edges = []
            
            # Create nodes for URLs
            for i, url in enumerate(urls):
                result = analysis_results[i] if i < len(analysis_results) else {}
                
                node_features = {
                    'url_length': len(url),
                    'entropy': self._calculate_entropy(url),
                    'suspicious_score': result.get('risk_score', 0.0),
                    'attack_type_encoded': self._encode_attack_type(result.get('attack_type')),
                    'confidence': result.get('confidence', 0.0)
                }
                
                node = GraphNode(
                    node_id=f"url_{i}",
                    node_type="url",
                    features=node_features,
                    metadata={'url': url, 'analysis_result': result},
                    timestamp=datetime.utcnow()
                )
                nodes.append(node)
            
            # Create edges based on similarity
            for i, url1 in enumerate(urls):
                for j, url2 in enumerate(urls[i+1:], i+1):
                    similarity = self._calculate_url_similarity(url1, url2)
                    
                    if similarity > 0.3:  # Threshold for edge creation
                        edge = GraphEdge(
                            source_id=f"url_{i}",
                            target_id=f"url_{j}",
                            edge_type="similarity",
                            weight=similarity,
                            features={'similarity': similarity},
                            timestamp=datetime.utcnow()
                        )
                        edges.append(edge)
            
            return GraphData(
                nodes=nodes,
                edges=edges,
                graph_type=GraphType.URL_RELATIONSHIP,
                metadata={'total_urls': len(urls)}
            )
            
        except Exception as e:
            logger.error("Error building URL relationship graph", error=str(e))
            raise
    
    async def build_ip_relationship_graph(self, ips: List[str], 
                                        analysis_results: List[Dict]) -> GraphData:
        """Build IP relationship graph."""
        try:
            nodes = []
            edges = []
            
            # Create nodes for IPs
            for i, ip in enumerate(ips):
                result = analysis_results[i] if i < len(analysis_results) else {}
                
                node_features = {
                    'ip_encoded': self._encode_ip(ip),
                    'reputation_score': result.get('reputation_score', 0.0),
                    'abuse_score': result.get('abuse_score', 0.0),
                    'attack_type_encoded': self._encode_attack_type(result.get('attack_type')),
                    'confidence': result.get('confidence', 0.0)
                }
                
                node = GraphNode(
                    node_id=f"ip_{i}",
                    node_type="ip",
                    features=node_features,
                    metadata={'ip': ip, 'analysis_result': result},
                    timestamp=datetime.utcnow()
                )
                nodes.append(node)
            
            # Create edges based on network proximity
            for i, ip1 in enumerate(ips):
                for j, ip2 in enumerate(ips[i+1:], i+1):
                    proximity = self._calculate_ip_proximity(ip1, ip2)
                    
                    if proximity > 0.5:  # Threshold for edge creation
                        edge = GraphEdge(
                            source_id=f"ip_{i}",
                            target_id=f"ip_{j}",
                            edge_type="network_proximity",
                            weight=proximity,
                            features={'proximity': proximity},
                            timestamp=datetime.utcnow()
                        )
                        edges.append(edge)
            
            return GraphData(
                nodes=nodes,
                edges=edges,
                graph_type=GraphType.IP_RELATIONSHIP,
                metadata={'total_ips': len(ips)}
            )
            
        except Exception as e:
            logger.error("Error building IP relationship graph", error=str(e))
            raise
    
    async def build_threat_correlation_graph(self, threats: List[Dict]) -> GraphData:
        """Build threat correlation graph."""
        try:
            nodes = []
            edges = []
            
            # Create nodes for threats
            for i, threat in enumerate(threats):
                node_features = {
                    'threat_type_encoded': self._encode_attack_type(threat.get('attack_type')),
                    'severity_encoded': self._encode_severity(threat.get('severity')),
                    'confidence': threat.get('confidence', 0.0),
                    'risk_score': threat.get('risk_score', 0.0),
                    'timestamp_encoded': self._encode_timestamp(threat.get('timestamp'))
                }
                
                node = GraphNode(
                    node_id=f"threat_{i}",
                    node_type="threat",
                    features=node_features,
                    metadata={'threat': threat},
                    timestamp=datetime.utcnow()
                )
                nodes.append(node)
            
            # Create edges based on temporal and semantic correlation
            for i, threat1 in enumerate(threats):
                for j, threat2 in enumerate(threats[i+1:], i+1):
                    correlation = self._calculate_threat_correlation(threat1, threat2)
                    
                    if correlation > 0.4:  # Threshold for edge creation
                        edge = GraphEdge(
                            source_id=f"threat_{i}",
                            target_id=f"threat_{j}",
                            edge_type="correlation",
                            weight=correlation,
                            features={'correlation': correlation},
                            timestamp=datetime.utcnow()
                        )
                        edges.append(edge)
            
            return GraphData(
                nodes=nodes,
                edges=edges,
                graph_type=GraphType.THREAT_CORRELATION,
                metadata={'total_threats': len(threats)}
            )
            
        except Exception as e:
            logger.error("Error building threat correlation graph", error=str(e))
            raise
    
    async def train_graph_model(self, graph_data: GraphData, model_type: str = 'gcn') -> Dict[str, float]:
        """Train a graph neural network model."""
        try:
            # Convert graph data to PyTorch Geometric format
            data = self._convert_to_pyg_data(graph_data)
            
            # Initialize model
            if model_type not in self.models:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model_class = self.models[model_type]
            model = model_class(
                input_dim=data.x.size(1),
                hidden_dim=64,
                output_dim=len(AttackType)
            )
            
            # Training setup
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Generate labels (simplified)
            labels = self._generate_labels(graph_data)
            
            # Training loop
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                
                # Forward pass
                out = model(data.x, data.edge_index)
                
                # Calculate loss
                loss = criterion(out, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Store trained model
            self.trained_models[model_type] = model
            
            # Calculate metrics
            model.eval()
            with torch.no_grad():
                predictions = model(data.x, data.edge_index)
                accuracy = self._calculate_accuracy(predictions, labels)
            
            metrics = {
                'accuracy': accuracy,
                'loss': loss.item(),
                'epochs': 100
            }
            
            logger.info("Graph model training completed", model_type=model_type, metrics=metrics)
            return metrics
            
        except Exception as e:
            logger.error("Error training graph model", error=str(e))
            raise
    
    async def predict_with_graph_model(self, graph_data: GraphData, 
                                     model_type: str = 'gcn') -> Dict[str, Any]:
        """Make predictions using a trained graph model."""
        try:
            if model_type not in self.trained_models:
                raise ValueError(f"Model {model_type} not trained")
            
            model = self.trained_models[model_type]
            data = self._convert_to_pyg_data(graph_data)
            
            model.eval()
            with torch.no_grad():
                predictions = model(data.x, data.edge_index)
                probabilities = F.softmax(predictions, dim=1)
                
                # Get predicted class and confidence
                predicted_class = torch.argmax(predictions, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
            
            # Convert to attack types
            attack_types = [list(AttackType)[pred.item()] for pred in predicted_class]
            confidences = confidence.tolist()
            
            return {
                'attack_types': attack_types,
                'confidences': confidences,
                'probabilities': probabilities.tolist(),
                'model_type': model_type
            }
            
        except Exception as e:
            logger.error("Error making graph model prediction", error=str(e))
            raise
    
    def _convert_to_pyg_data(self, graph_data: GraphData) -> Data:
        """Convert graph data to PyTorch Geometric format."""
        try:
            # Extract node features
            node_features = []
            for node in graph_data.nodes:
                features = list(node.features.values())
                node_features.append(features)
            
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Extract edge indices
            edge_indices = []
            for edge in graph_data.edges:
                source_idx = int(edge.source_id.split('_')[1])
                target_idx = int(edge.target_id.split('_')[1])
                edge_indices.append([source_idx, target_idx])
            
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            else:
                # Create self-loops if no edges
                edge_index = torch.tensor([[i, i] for i in range(len(graph_data.nodes))], dtype=torch.long).t().contiguous()
            
            return Data(x=x, edge_index=edge_index)
            
        except Exception as e:
            logger.error("Error converting to PyTorch Geometric data", error=str(e))
            raise
    
    def _generate_labels(self, graph_data: GraphData) -> torch.Tensor:
        """Generate labels for training."""
        try:
            labels = []
            for node in graph_data.nodes:
                # Extract attack type from metadata
                attack_type = node.metadata.get('analysis_result', {}).get('attack_type', AttackType.UNKNOWN)
                label = list(AttackType).index(attack_type)
                labels.append(label)
            
            return torch.tensor(labels, dtype=torch.long)
            
        except Exception as e:
            logger.error("Error generating labels", error=str(e))
            raise
    
    def _calculate_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate prediction accuracy."""
        try:
            predicted_classes = torch.argmax(predictions, dim=1)
            correct = (predicted_classes == labels).sum().item()
            total = labels.size(0)
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            logger.error("Error calculating accuracy", error=str(e))
            return 0.0
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate entropy of text."""
        if not text:
            return 0.0
        
        char_counts = {}
        for char in text.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0.0
        text_len = len(text)
        
        for count in char_counts.values():
            probability = count / text_len
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_url_similarity(self, url1: str, url2: str) -> float:
        """Calculate similarity between two URLs."""
        try:
            from urllib.parse import urlparse
            
            parsed1 = urlparse(url1)
            parsed2 = urlparse(url2)
            
            # Compare domains
            domain_similarity = 1.0 if parsed1.netloc == parsed2.netloc else 0.0
            
            # Compare paths
            path1_parts = parsed1.path.split('/')
            path2_parts = parsed2.path.split('/')
            path_similarity = len(set(path1_parts) & set(path2_parts)) / max(len(path1_parts), len(path2_parts))
            
            # Combine similarities
            return (domain_similarity * 0.7 + path_similarity * 0.3)
            
        except Exception as e:
            logger.error("Error calculating URL similarity", error=str(e))
            return 0.0
    
    def _calculate_ip_proximity(self, ip1: str, ip2: str) -> float:
        """Calculate network proximity between two IPs."""
        try:
            import ipaddress
            
            ip_obj1 = ipaddress.ip_address(ip1)
            ip_obj2 = ipaddress.ip_address(ip2)
            
            # Calculate network distance
            if ip_obj1.version == ip_obj2.version:
                # Same IP version
                if ip_obj1.version == 4:
                    # IPv4
                    diff = abs(int(ip_obj1) - int(ip_obj2))
                    max_diff = 2**32 - 1
                    return 1.0 - (diff / max_diff)
                else:
                    # IPv6
                    diff = abs(int(ip_obj1) - int(ip_obj2))
                    max_diff = 2**128 - 1
                    return 1.0 - (diff / max_diff)
            else:
                return 0.0
                
        except Exception as e:
            logger.error("Error calculating IP proximity", error=str(e))
            return 0.0
    
    def _calculate_threat_correlation(self, threat1: Dict, threat2: Dict) -> float:
        """Calculate correlation between two threats."""
        try:
            # Temporal correlation
            time1 = threat1.get('timestamp', datetime.utcnow())
            time2 = threat2.get('timestamp', datetime.utcnow())
            
            if isinstance(time1, str):
                time1 = datetime.fromisoformat(time1)
            if isinstance(time2, str):
                time2 = datetime.fromisoformat(time2)
            
            time_diff = abs((time1 - time2).total_seconds())
            temporal_correlation = 1.0 / (1.0 + time_diff / 3600)  # Decay over hours
            
            # Semantic correlation
            attack_type1 = threat1.get('attack_type', AttackType.UNKNOWN)
            attack_type2 = threat2.get('attack_type', AttackType.UNKNOWN)
            semantic_correlation = 1.0 if attack_type1 == attack_type2 else 0.0
            
            # Severity correlation
            severity1 = threat1.get('severity', SeverityLevel.LOW)
            severity2 = threat2.get('severity', SeverityLevel.LOW)
            severity_correlation = 1.0 if severity1 == severity2 else 0.5
            
            # Combine correlations
            return (temporal_correlation * 0.4 + semantic_correlation * 0.4 + severity_correlation * 0.2)
            
        except Exception as e:
            logger.error("Error calculating threat correlation", error=str(e))
            return 0.0
    
    def _encode_attack_type(self, attack_type: AttackType) -> float:
        """Encode attack type as float."""
        if attack_type is None:
            return 0.0
        return float(list(AttackType).index(attack_type)) / len(AttackType)
    
    def _encode_severity(self, severity: SeverityLevel) -> float:
        """Encode severity as float."""
        if severity is None:
            return 0.0
        severity_values = {
            SeverityLevel.LOW: 0.25,
            SeverityLevel.MEDIUM: 0.5,
            SeverityLevel.HIGH: 0.75,
            SeverityLevel.CRITICAL: 1.0
        }
        return severity_values.get(severity, 0.0)
    
    def _encode_ip(self, ip: str) -> float:
        """Encode IP address as float."""
        try:
            import ipaddress
            ip_obj = ipaddress.ip_address(ip)
            return float(int(ip_obj)) / (2**32 - 1) if ip_obj.version == 4 else 0.0
        except:
            return 0.0
    
    def _encode_timestamp(self, timestamp: datetime) -> float:
        """Encode timestamp as float."""
        if timestamp is None:
            return 0.0
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return timestamp.timestamp() / 1e9  # Normalize to reasonable range
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph analysis statistics."""
        try:
            return {
                'trained_models': list(self.trained_models.keys()),
                'cached_graphs': len(self.graph_cache),
                'node_embeddings': len(self.node_embeddings),
                'available_models': list(self.models.keys())
            }
            
        except Exception as e:
            logger.error("Error getting graph statistics", error=str(e))
            return {}


# Global graph neural network engine instance
graph_nn_engine = GraphNeuralNetworkEngine()
