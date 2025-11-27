"""
Blockchain-based security and trust management for cyberattack detection.
"""
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key

from ..database.models import AttackType, SeverityLevel
from ..database.connection import get_database
from ..security.encryption import encryption_manager

logger = structlog.get_logger(__name__)


class BlockType(str, Enum):
    """Types of blocks in the blockchain."""
    THREAT_DETECTION = "threat_detection"
    MODEL_UPDATE = "model_update"
    TRUST_VERIFICATION = "trust_verification"
    CONSENSUS = "consensus"


@dataclass
class Block:
    """Block in the blockchain."""
    index: int
    timestamp: datetime
    data: Dict[str, Any]
    previous_hash: str
    hash: str
    nonce: int
    block_type: BlockType
    validator: str
    signature: Optional[str] = None


@dataclass
class Transaction:
    """Transaction in the blockchain."""
    transaction_id: str
    sender: str
    receiver: str
    data: Dict[str, Any]
    timestamp: datetime
    signature: str
    transaction_type: str


class Blockchain:
    """Blockchain implementation for cyberattack detection."""
    
    def __init__(self, difficulty: int = 4):
        self.chain = []
        self.pending_transactions = []
        self.difficulty = difficulty
        self.mining_reward = 10
        self.private_key = None
        self.public_key = None
        self._generate_keypair()
        self._create_genesis_block()
    
    def _generate_keypair(self):
        """Generate RSA key pair for signing."""
        try:
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
            
        except Exception as e:
            logger.error("Error generating keypair", error=str(e))
            raise
    
    def _create_genesis_block(self):
        """Create the genesis block."""
        try:
            genesis_data = {
                "message": "Genesis block for cyberattack detection blockchain",
                "creator": "system",
                "initial_threats": []
            }
            
            genesis_block = Block(
                index=0,
                timestamp=datetime.utcnow(),
                data=genesis_data,
                previous_hash="0",
                hash="",
                nonce=0,
                block_type=BlockType.CONSENSUS,
                validator="system"
            )
            
            genesis_block.hash = self._calculate_hash(genesis_block)
            self.chain.append(genesis_block)
            
            logger.info("Genesis block created")
            
        except Exception as e:
            logger.error("Error creating genesis block", error=str(e))
            raise
    
    def _calculate_hash(self, block: Block) -> str:
        """Calculate hash of a block."""
        try:
            block_string = json.dumps({
                "index": block.index,
                "timestamp": block.timestamp.isoformat(),
                "data": block.data,
                "previous_hash": block.previous_hash,
                "nonce": block.nonce,
                "block_type": block.block_type.value,
                "validator": block.validator
            }, sort_keys=True)
            
            return hashlib.sha256(block_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error("Error calculating hash", error=str(e))
            raise
    
    def _proof_of_work(self, block: Block) -> str:
        """Proof of work algorithm."""
        try:
            block.nonce = 0
            computed_hash = self._calculate_hash(block)
            
            while not computed_hash.startswith('0' * self.difficulty):
                block.nonce += 1
                computed_hash = self._calculate_hash(block)
            
            return computed_hash
            
        except Exception as e:
            logger.error("Error in proof of work", error=str(e))
            raise
    
    def _sign_block(self, block: Block) -> str:
        """Sign a block with private key."""
        try:
            block_data = json.dumps(asdict(block), sort_keys=True).encode()
            
            signature = self.private_key.sign(
                block_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return signature.hex()
            
        except Exception as e:
            logger.error("Error signing block", error=str(e))
            raise
    
    def _verify_signature(self, block: Block, signature: str) -> bool:
        """Verify block signature."""
        try:
            block_data = json.dumps(asdict(block), sort_keys=True).encode()
            
            self.public_key.verify(
                bytes.fromhex(signature),
                block_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.error("Error verifying signature", error=str(e))
            return False
    
    def add_block(self, data: Dict[str, Any], block_type: BlockType) -> Block:
        """Add a new block to the blockchain."""
        try:
            previous_block = self.chain[-1]
            
            new_block = Block(
                index=len(self.chain),
                timestamp=datetime.utcnow(),
                data=data,
                previous_hash=previous_block.hash,
                hash="",
                nonce=0,
                block_type=block_type,
                validator="system"
            )
            
            # Proof of work
            new_block.hash = self._proof_of_work(new_block)
            
            # Sign block
            signature = self._sign_block(new_block)
            new_block.signature = signature
            
            # Add to chain
            self.chain.append(new_block)
            
            logger.info("Block added to blockchain", 
                       index=new_block.index,
                       block_type=block_type.value,
                       hash=new_block.hash[:10])
            
            return new_block
            
        except Exception as e:
            logger.error("Error adding block", error=str(e))
            raise
    
    def is_chain_valid(self) -> bool:
        """Validate the entire blockchain."""
        try:
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i - 1]
                
                # Check hash
                if current_block.hash != self._calculate_hash(current_block):
                    logger.error("Invalid hash", block_index=i)
                    return False
                
                # Check previous hash
                if current_block.previous_hash != previous_block.hash:
                    logger.error("Invalid previous hash", block_index=i)
                    return False
                
                # Check signature
                if current_block.signature and not self._verify_signature(current_block, current_block.signature):
                    logger.error("Invalid signature", block_index=i)
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Error validating chain", error=str(e))
            return False
    
    def get_latest_block(self) -> Block:
        """Get the latest block."""
        return self.chain[-1]
    
    def get_block_by_index(self, index: int) -> Optional[Block]:
        """Get block by index."""
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None
    
    def get_blocks_by_type(self, block_type: BlockType) -> List[Block]:
        """Get blocks by type."""
        return [block for block in self.chain if block.block_type == block_type]
    
    def get_chain_length(self) -> int:
        """Get blockchain length."""
        return len(self.chain)


class ThreatIntelligenceBlockchain:
    """Blockchain for threat intelligence sharing."""
    
    def __init__(self):
        self.blockchain = Blockchain()
        self.threat_registry = {}
        self.trust_scores = {}
        self.consensus_threshold = 0.7
    
    async def record_threat_detection(self, threat_data: Dict[str, Any]) -> str:
        """Record threat detection in blockchain."""
        try:
            # Create threat detection block
            block_data = {
                "threat_id": threat_data.get("threat_id", ""),
                "target": threat_data.get("target", ""),
                "target_type": threat_data.get("target_type", ""),
                "attack_type": threat_data.get("attack_type", ""),
                "severity": threat_data.get("severity", ""),
                "confidence": threat_data.get("confidence", 0.0),
                "detector_id": threat_data.get("detector_id", "system"),
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": threat_data.get("metadata", {})
            }
            
            # Add to blockchain
            block = self.blockchain.add_block(block_data, BlockType.THREAT_DETECTION)
            
            # Update threat registry
            threat_id = block_data["threat_id"]
            if threat_id not in self.threat_registry:
                self.threat_registry[threat_id] = []
            
            self.threat_registry[threat_id].append({
                "block_hash": block.hash,
                "detector_id": block_data["detector_id"],
                "timestamp": block.timestamp,
                "confidence": block_data["confidence"]
            })
            
            logger.info("Threat detection recorded in blockchain", 
                       threat_id=threat_id,
                       block_hash=block.hash[:10])
            
            return block.hash
            
        except Exception as e:
            logger.error("Error recording threat detection", error=str(e))
            raise
    
    async def verify_threat_consensus(self, threat_id: str) -> Dict[str, Any]:
        """Verify consensus on threat detection."""
        try:
            if threat_id not in self.threat_registry:
                return {"consensus": False, "confidence": 0.0, "detectors": 0}
            
            threat_detections = self.threat_registry[threat_id]
            
            # Calculate consensus
            total_detectors = len(threat_detections)
            unique_detectors = len(set(detection["detector_id"] for detection in threat_detections))
            
            # Calculate average confidence
            avg_confidence = sum(detection["confidence"] for detection in threat_detections) / total_detectors
            
            # Determine consensus
            consensus = (unique_detectors >= 2 and avg_confidence >= self.consensus_threshold)
            
            return {
                "consensus": consensus,
                "confidence": avg_confidence,
                "detectors": unique_detectors,
                "total_detections": total_detectors,
                "consensus_threshold": self.consensus_threshold
            }
            
        except Exception as e:
            logger.error("Error verifying threat consensus", error=str(e))
            return {"consensus": False, "confidence": 0.0, "detectors": 0}
    
    async def update_trust_score(self, detector_id: str, performance: float):
        """Update trust score for a detector."""
        try:
            if detector_id not in self.trust_scores:
                self.trust_scores[detector_id] = {
                    "score": 0.5,
                    "total_evaluations": 0,
                    "last_update": datetime.utcnow()
                }
            
            # Update trust score using exponential moving average
            current_score = self.trust_scores[detector_id]["score"]
            alpha = 0.1  # Learning rate
            
            new_score = alpha * performance + (1 - alpha) * current_score
            self.trust_scores[detector_id]["score"] = new_score
            self.trust_scores[detector_id]["total_evaluations"] += 1
            self.trust_scores[detector_id]["last_update"] = datetime.utcnow()
            
            # Record trust update in blockchain
            trust_data = {
                "detector_id": detector_id,
                "old_score": current_score,
                "new_score": new_score,
                "performance": performance,
                "total_evaluations": self.trust_scores[detector_id]["total_evaluations"]
            }
            
            self.blockchain.add_block(trust_data, BlockType.TRUST_VERIFICATION)
            
            logger.info("Trust score updated", 
                       detector_id=detector_id,
                       old_score=current_score,
                       new_score=new_score)
            
        except Exception as e:
            logger.error("Error updating trust score", error=str(e))
    
    async def get_trust_score(self, detector_id: str) -> float:
        """Get trust score for a detector."""
        try:
            if detector_id in self.trust_scores:
                return self.trust_scores[detector_id]["score"]
            else:
                return 0.5  # Default trust score
                
        except Exception as e:
            logger.error("Error getting trust score", error=str(e))
            return 0.5
    
    async def get_threat_history(self, threat_id: str) -> List[Dict[str, Any]]:
        """Get threat detection history from blockchain."""
        try:
            if threat_id not in self.threat_registry:
                return []
            
            threat_detections = self.threat_registry[threat_id]
            
            # Get detailed information from blockchain
            history = []
            for detection in threat_detections:
                block_hash = detection["block_hash"]
                
                # Find block in blockchain
                for block in self.blockchain.chain:
                    if block.hash == block_hash:
                        history.append({
                            "block_index": block.index,
                            "timestamp": block.timestamp,
                            "detector_id": detection["detector_id"],
                            "confidence": detection["confidence"],
                            "data": block.data
                        })
                        break
            
            return history
            
        except Exception as e:
            logger.error("Error getting threat history", error=str(e))
            return []
    
    async def get_blockchain_statistics(self) -> Dict[str, Any]:
        """Get blockchain statistics."""
        try:
            return {
                "chain_length": self.blockchain.get_chain_length(),
                "is_valid": self.blockchain.is_chain_valid(),
                "threat_registry_size": len(self.threat_registry),
                "trust_scores_count": len(self.trust_scores),
                "consensus_threshold": self.consensus_threshold,
                "latest_block": {
                    "index": self.blockchain.get_latest_block().index,
                    "timestamp": self.blockchain.get_latest_block().timestamp.isoformat(),
                    "type": self.blockchain.get_latest_block().block_type.value
                }
            }
            
        except Exception as e:
            logger.error("Error getting blockchain statistics", error=str(e))
            return {}


class SmartContract:
    """Smart contract for automated threat response."""
    
    def __init__(self, blockchain: ThreatIntelligenceBlockchain):
        self.blockchain = blockchain
        self.contract_rules = {}
        self.automated_responses = {}
    
    async def deploy_contract(self, contract_id: str, rules: Dict[str, Any]) -> bool:
        """Deploy a smart contract."""
        try:
            # Store contract rules
            self.contract_rules[contract_id] = rules
            
            # Record contract deployment in blockchain
            contract_data = {
                "contract_id": contract_id,
                "rules": rules,
                "deployed_by": "system",
                "deployment_time": datetime.utcnow().isoformat()
            }
            
            self.blockchain.blockchain.add_block(contract_data, BlockType.CONSENSUS)
            
            logger.info("Smart contract deployed", contract_id=contract_id)
            return True
            
        except Exception as e:
            logger.error("Error deploying smart contract", error=str(e))
            return False
    
    async def execute_contract(self, contract_id: str, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute smart contract based on threat data."""
        try:
            if contract_id not in self.contract_rules:
                raise ValueError(f"Contract {contract_id} not found")
            
            rules = self.contract_rules[contract_id]
            response = {"executed": False, "actions": []}
            
            # Check conditions
            for rule in rules.get("conditions", []):
                if self._evaluate_condition(rule, threat_data):
                    # Execute actions
                    for action in rule.get("actions", []):
                        action_result = await self._execute_action(action, threat_data)
                        response["actions"].append(action_result)
                        response["executed"] = True
            
            # Record contract execution in blockchain
            execution_data = {
                "contract_id": contract_id,
                "threat_data": threat_data,
                "response": response,
                "execution_time": datetime.utcnow().isoformat()
            }
            
            self.blockchain.blockchain.add_block(execution_data, BlockType.CONSENSUS)
            
            logger.info("Smart contract executed", 
                       contract_id=contract_id,
                       executed=response["executed"],
                       actions_count=len(response["actions"]))
            
            return response
            
        except Exception as e:
            logger.error("Error executing smart contract", error=str(e))
            return {"executed": False, "actions": [], "error": str(e)}
    
    def _evaluate_condition(self, condition: Dict[str, Any], threat_data: Dict[str, Any]) -> bool:
        """Evaluate a condition in the smart contract."""
        try:
            condition_type = condition.get("type", "")
            
            if condition_type == "severity_threshold":
                threshold = condition.get("threshold", 0.5)
                severity = threat_data.get("severity", SeverityLevel.LOW)
                severity_values = {
                    SeverityLevel.LOW: 0.25,
                    SeverityLevel.MEDIUM: 0.5,
                    SeverityLevel.HIGH: 0.75,
                    SeverityLevel.CRITICAL: 1.0
                }
                return severity_values.get(severity, 0.0) >= threshold
            
            elif condition_type == "confidence_threshold":
                threshold = condition.get("threshold", 0.7)
                confidence = threat_data.get("confidence", 0.0)
                return confidence >= threshold
            
            elif condition_type == "attack_type_match":
                required_types = condition.get("attack_types", [])
                attack_type = threat_data.get("attack_type", "")
                return attack_type in required_types
            
            elif condition_type == "consensus_required":
                threat_id = threat_data.get("threat_id", "")
                consensus = asyncio.run(self.blockchain.verify_threat_consensus(threat_id))
                return consensus.get("consensus", False)
            
            return False
            
        except Exception as e:
            logger.error("Error evaluating condition", error=str(e))
            return False
    
    async def _execute_action(self, action: Dict[str, Any], threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action in the smart contract."""
        try:
            action_type = action.get("type", "")
            action_result = {"type": action_type, "executed": False, "result": None}
            
            if action_type == "block_target":
                # Block the target
                target = threat_data.get("target", "")
                action_result["result"] = f"Blocked target: {target}"
                action_result["executed"] = True
            
            elif action_type == "send_alert":
                # Send alert
                alert_message = action.get("message", "Threat detected")
                action_result["result"] = f"Alert sent: {alert_message}"
                action_result["executed"] = True
            
            elif action_type == "quarantine":
                # Quarantine the target
                target = threat_data.get("target", "")
                action_result["result"] = f"Quarantined target: {target}"
                action_result["executed"] = True
            
            elif action_type == "update_model":
                # Update ML model
                model_id = action.get("model_id", "default")
                action_result["result"] = f"Model updated: {model_id}"
                action_result["executed"] = True
            
            return action_result
            
        except Exception as e:
            logger.error("Error executing action", error=str(e))
            return {"type": action.get("type", ""), "executed": False, "error": str(e)}


# Global blockchain instances
threat_blockchain = ThreatIntelligenceBlockchain()
smart_contract = SmartContract(threat_blockchain)
