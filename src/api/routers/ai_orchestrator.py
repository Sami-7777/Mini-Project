"""
AI Orchestrator API endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import structlog

from ...ai.ai_orchestrator import ai_orchestrator
from ...database.models import AttackType, SeverityLevel
from ..dependencies import get_current_user, get_api_key

logger = structlog.get_logger(__name__)
router = APIRouter()


class AIOrchestrationRequest(BaseModel):
    """Request model for AI orchestration."""
    target: str = Field(..., description="URL or IP address to analyze")
    target_type: str = Field(..., description="Type of target: 'url' or 'ip'")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")
    components: Optional[List[str]] = Field(None, description="Specific AI components to use")


class AIOrchestrationResponse(BaseModel):
    """Response model for AI orchestration."""
    target: str
    target_type: str
    final_prediction: AttackType
    final_confidence: float
    component_results: Dict[str, Any]
    consensus_score: float
    explanation: Dict[str, Any]
    recommendations: List[str]
    processing_time_ms: int
    timestamp: str


@router.post("/ai/orchestrate", response_model=AIOrchestrationResponse)
async def orchestrate_ai_analysis(
    request: AIOrchestrationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Orchestrate comprehensive AI analysis."""
    try:
        logger.info("Starting AI orchestration", 
                   target=request.target, 
                   target_type=request.target_type,
                   user=current_user.get('username'))
        
        # Perform AI orchestration
        result = await ai_orchestrator.orchestrate_analysis(
            target=request.target,
            target_type=request.target_type,
            context=request.context
        )
        
        # Convert to response model
        response = AIOrchestrationResponse(
            target=result.target,
            target_type=result.target_type,
            final_prediction=result.final_prediction,
            final_confidence=result.final_confidence,
            component_results=result.component_results,
            consensus_score=result.consensus_score,
            explanation=result.explanation,
            recommendations=result.recommendations,
            processing_time_ms=result.processing_time_ms,
            timestamp=result.timestamp.isoformat()
        )
        
        logger.info("AI orchestration completed", 
                   target=request.target,
                   prediction=result.final_prediction,
                   confidence=result.final_confidence,
                   processing_time_ms=result.processing_time_ms)
        
        return response
        
    except Exception as e:
        logger.error("Error in AI orchestration", 
                    target=request.target, 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"AI orchestration failed: {str(e)}")


@router.get("/ai/components")
async def get_ai_components(
    current_user: dict = Depends(get_current_user)
):
    """Get available AI components."""
    try:
        stats = await ai_orchestrator.get_orchestration_statistics()
        
        return {
            "available_components": stats.get("available_components", []),
            "component_weights": stats.get("component_weights", {}),
            "is_initialized": stats.get("is_initialized", False),
            "total_orchestrations": stats.get("total_orchestrations", 0)
        }
        
    except Exception as e:
        logger.error("Error getting AI components", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving AI components")


@router.get("/ai/statistics")
async def get_ai_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get AI system statistics."""
    try:
        # Get orchestrator statistics
        orchestrator_stats = await ai_orchestrator.get_orchestration_statistics()
        
        # Get component statistics
        component_stats = {}
        
        # Model manager stats
        model_stats = await model_manager.get_model_info()
        component_stats["models"] = model_stats
        
        # Graph NN stats
        graph_stats = await graph_nn_engine.get_graph_statistics()
        component_stats["graph_neural_networks"] = graph_stats
        
        # Quantum ML stats
        quantum_stats = await quantum_ml_engine.get_quantum_statistics()
        component_stats["quantum_ml"] = quantum_stats
        
        # Federated learning stats
        federated_stats = await federated_learning_engine.get_federated_statistics()
        component_stats["federated_learning"] = federated_stats
        
        # Adaptive learning stats
        adaptive_stats = await adaptive_learning_engine.get_adaptation_statistics()
        component_stats["adaptive_learning"] = adaptive_stats
        
        # Explainable AI stats
        explainable_stats = await explainable_ai_engine.get_explanation_statistics()
        component_stats["explainable_ai"] = explainable_stats
        
        # Advanced analytics stats
        analytics_stats = await advanced_analytics_engine.get_analytics_statistics()
        component_stats["advanced_analytics"] = analytics_stats
        
        # Anomaly detection stats
        anomaly_stats = await anomaly_detector.get_anomaly_statistics()
        component_stats["anomaly_detection"] = anomaly_stats
        
        # Novelty detection stats
        novelty_stats = await novelty_detector.get_novelty_statistics()
        component_stats["novelty_detection"] = novelty_stats
        
        return {
            "orchestrator": orchestrator_stats,
            "components": component_stats,
            "system_health": "operational"
        }
        
    except Exception as e:
        logger.error("Error getting AI statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving AI statistics")


@router.post("/ai/train")
async def train_ai_components(
    training_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Train AI components."""
    try:
        logger.info("Starting AI component training", 
                   user=current_user.get('username'))
        
        # Prepare training data
        X_train = training_data.get("features", [])
        y_train = training_data.get("labels", [])
        
        if not X_train or not y_train:
            raise HTTPException(status_code=400, detail="Training data is required")
        
        # Convert to numpy arrays
        import numpy as np
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train components
        training_results = {}
        
        # Train classical models
        model_results = await model_manager.train_all_models(X_train, y_train)
        training_results["classical_ml"] = model_results
        
        # Train quantum models
        try:
            quantum_results = await quantum_ml_engine.train_quantum_model(X_train, y_train, "vqc")
            training_results["quantum_ml"] = quantum_results
        except Exception as e:
            logger.warning("Quantum ML training failed", error=str(e))
            training_results["quantum_ml"] = {"error": str(e)}
        
        # Train graph neural networks
        try:
            # This would require graph data preparation
            training_results["graph_neural_networks"] = {"status": "not_implemented"}
        except Exception as e:
            logger.warning("Graph NN training failed", error=str(e))
            training_results["graph_neural_networks"] = {"error": str(e)}
        
        logger.info("AI component training completed", 
                   results=training_results,
                   user=current_user.get('username'))
        
        return {
            "message": "AI component training completed",
            "results": training_results
        }
        
    except Exception as e:
        logger.error("Error training AI components", error=str(e))
        raise HTTPException(status_code=500, detail=f"AI training failed: {str(e)}")


@router.get("/ai/explain/{target}")
async def explain_ai_prediction(
    target: str,
    target_type: str = "url",
    current_user: dict = Depends(get_current_user)
):
    """Get AI prediction explanation."""
    try:
        # Get recent orchestration result for the target
        recent_result = None
        for result in ai_orchestrator.orchestration_history:
            if result.target == target and result.target_type == target_type:
                recent_result = result
                break
        
        if not recent_result:
            raise HTTPException(status_code=404, detail="No analysis found for target")
        
        # Return explanation
        return {
            "target": target,
            "target_type": target_type,
            "prediction": recent_result.final_prediction,
            "confidence": recent_result.final_confidence,
            "explanation": recent_result.explanation,
            "recommendations": recent_result.recommendations,
            "component_results": recent_result.component_results,
            "consensus_score": recent_result.consensus_score,
            "processing_time_ms": recent_result.processing_time_ms,
            "timestamp": recent_result.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error explaining AI prediction", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving AI explanation")


@router.post("/ai/feedback")
async def submit_ai_feedback(
    feedback_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Submit feedback for AI predictions."""
    try:
        target = feedback_data.get("target")
        feedback = feedback_data.get("feedback")
        actual_result = feedback_data.get("actual_result")
        
        if not target or not feedback:
            raise HTTPException(status_code=400, detail="Target and feedback are required")
        
        # Process feedback with adaptive learning
        from ...ai.adaptive_learning import LearningEpisode
        
        episode = LearningEpisode(
            episode_id=f"feedback_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            input_data={"target": target},
            prediction={"attack_type": "unknown", "confidence": 0.0},
            actual_result=actual_result,
            feedback=feedback,
            performance_metrics={},
            learning_strategy="online_learning",
            model_updates={}
        )
        
        # Process with adaptive learning
        result = await adaptive_learning_engine.process_learning_episode(episode)
        
        logger.info("AI feedback processed", 
                   target=target,
                   feedback=feedback,
                   user=current_user.get('username'))
        
        return {
            "message": "Feedback processed successfully",
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing AI feedback", error=str(e))
        raise HTTPException(status_code=500, detail="Error processing feedback")


@router.get("/ai/analytics")
async def get_ai_analytics(
    time_window: int = 30,
    current_user: dict = Depends(get_current_user)
):
    """Get AI analytics and insights."""
    try:
        # Run comprehensive analytics
        analysis_results = await advanced_analytics_engine.run_comprehensive_analysis(time_window)
        
        # Generate report
        report = await advanced_analytics_engine.generate_analytics_report(analysis_results)
        
        return report
        
    except Exception as e:
        logger.error("Error getting AI analytics", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving AI analytics")


@router.post("/ai/blockchain/record")
async def record_threat_blockchain(
    threat_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Record threat in blockchain."""
    try:
        # Record threat in blockchain
        block_hash = await threat_blockchain.record_threat_detection(threat_data)
        
        logger.info("Threat recorded in blockchain", 
                   block_hash=block_hash,
                   user=current_user.get('username'))
        
        return {
            "message": "Threat recorded in blockchain",
            "block_hash": block_hash
        }
        
    except Exception as e:
        logger.error("Error recording threat in blockchain", error=str(e))
        raise HTTPException(status_code=500, detail="Error recording threat in blockchain")


@router.get("/ai/blockchain/verify/{threat_id}")
async def verify_threat_consensus(
    threat_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Verify threat consensus in blockchain."""
    try:
        # Verify consensus
        consensus = await threat_blockchain.verify_threat_consensus(threat_id)
        
        return {
            "threat_id": threat_id,
            "consensus": consensus
        }
        
    except Exception as e:
        logger.error("Error verifying threat consensus", error=str(e))
        raise HTTPException(status_code=500, detail="Error verifying threat consensus")


@router.get("/ai/blockchain/statistics")
async def get_blockchain_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get blockchain statistics."""
    try:
        stats = await threat_blockchain.get_blockchain_statistics()
        
        return stats
        
    except Exception as e:
        logger.error("Error getting blockchain statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving blockchain statistics")


@router.post("/ai/federated/register")
async def register_federated_client(
    client_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Register a new federated learning client."""
    try:
        client_id = client_data.get("client_id")
        client_info = client_data.get("client_info", {})
        
        if not client_id:
            raise HTTPException(status_code=400, detail="Client ID is required")
        
        # Register client
        success = await federated_learning_engine.register_client(client_id, client_info)
        
        if success:
            return {
                "message": "Client registered successfully",
                "client_id": client_id
            }
        else:
            raise HTTPException(status_code=400, detail="Client registration failed")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error registering federated client", error=str(e))
        raise HTTPException(status_code=500, detail="Error registering federated client")


@router.post("/ai/federated/round")
async def run_federated_round(
    round_config: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Run a federated learning round."""
    try:
        selected_clients = round_config.get("selected_clients")
        aggregation_method = round_config.get("aggregation_method", "fedavg")
        
        # Run federated round
        result = await federated_learning_engine.run_federated_round(
            selected_clients=selected_clients,
            aggregation_method=aggregation_method
        )
        
        logger.info("Federated round completed", 
                   round_id=result.round_id,
                   selected_clients=len(result.selected_clients),
                   user=current_user.get('username'))
        
        return {
            "message": "Federated round completed",
            "round_id": result.round_id,
            "selected_clients": result.selected_clients,
            "performance_improvement": result.performance_improvement,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error("Error running federated round", error=str(e))
        raise HTTPException(status_code=500, detail="Error running federated round")


@router.get("/ai/federated/statistics")
async def get_federated_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get federated learning statistics."""
    try:
        stats = await federated_learning_engine.get_federated_statistics()
        
        return stats
        
    except Exception as e:
        logger.error("Error getting federated statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving federated statistics")
