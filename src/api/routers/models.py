"""
Models API endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
import structlog

from ...database.models import AttackType
from ...models.model_manager import model_manager
from ..dependencies import get_current_user, get_api_key

logger = structlog.get_logger(__name__)
router = APIRouter()


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str
    model_version: str
    is_trained: bool
    training_timestamp: Optional[str] = None
    feature_count: int
    model_size_mb: float


class ModelMetricsResponse(BaseModel):
    """Response model for model metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    training_samples: int
    validation_samples: int
    evaluation_timestamp: str


class TrainingRequest(BaseModel):
    """Request model for model training."""
    model_name: str = Field(..., description="Name of the model to train")
    training_data: Dict[str, Any] = Field(..., description="Training data")
    validation_data: Optional[Dict[str, Any]] = Field(None, description="Validation data")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model parameters")


class PredictionRequest(BaseModel):
    """Request model for model prediction."""
    model_name: str = Field(..., description="Name of the model to use")
    features: Dict[str, Any] = Field(..., description="Features for prediction")


class PredictionResponse(BaseModel):
    """Response model for model prediction."""
    model_name: str
    attack_type: AttackType
    confidence: float
    probabilities: Dict[str, float]
    prediction_timestamp: str


@router.get("/models", response_model=Dict[str, Any])
async def get_models(
    current_user: dict = Depends(get_current_user)
):
    """Get information about all available models."""
    try:
        model_info = await model_manager.get_model_info()
        return model_info
        
    except Exception as e:
        logger.error("Error getting model info", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving model information")


@router.get("/models/{model_name}", response_model=ModelInfoResponse)
async def get_model(
    model_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Get information about a specific model."""
    try:
        if model_name not in model_manager.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = model_manager.models[model_name]
        model_info = model.get_model_info()
        
        return ModelInfoResponse(
            model_name=model_info["model_name"],
            model_version=model_info["model_version"],
            is_trained=model_info["is_trained"],
            training_timestamp=model_info["training_timestamp"].isoformat() if model_info["training_timestamp"] else None,
            feature_count=model_info["feature_count"],
            model_size_mb=model_info["model_size_mb"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting model", model_name=model_name, error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving model information")


@router.get("/models/{model_name}/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(
    model_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Get performance metrics for a specific model."""
    try:
        metrics = await model_manager.get_model_performance(model_name)
        
        if "error" in metrics:
            raise HTTPException(status_code=404, detail=metrics["error"])
        
        return ModelMetricsResponse(
            model_name=metrics["model_name"],
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            auc_roc=metrics.get("auc_roc"),
            training_samples=0,  # Would get from actual metrics
            validation_samples=0,  # Would get from actual metrics
            evaluation_timestamp=metrics["evaluation_timestamp"].isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting model metrics", model_name=model_name, error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving model metrics")


@router.post("/models/{model_name}/train")
async def train_model(
    model_name: str,
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Train a specific model."""
    try:
        # Validate model name
        if model_name not in model_manager.model_configs:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Prepare training data
        training_data = request.training_data
        X_train = np.array(training_data.get("features", []))
        y_train = np.array(training_data.get("labels", []))
        
        X_val = None
        y_val = None
        if request.validation_data:
            X_val = np.array(request.validation_data.get("features", []))
            y_val = np.array(request.validation_data.get("labels", []))
        
        # Train the model
        metrics = await model_manager.train_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        
        logger.info("Model training completed", 
                   model_name=model_name,
                   metrics=metrics,
                   user=current_user.get('username'))
        
        return {
            "message": "Model training completed",
            "model_name": model_name,
            "metrics": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error training model", model_name=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@router.post("/models/train/all")
async def train_all_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Train all available models."""
    try:
        # Prepare training data
        training_data = request.training_data
        X_train = np.array(training_data.get("features", []))
        y_train = np.array(training_data.get("labels", []))
        
        X_val = None
        y_val = None
        if request.validation_data:
            X_val = np.array(request.validation_data.get("features", []))
            y_val = np.array(request.validation_data.get("labels", []))
        
        # Train all models
        results = await model_manager.train_all_models(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        
        logger.info("All models training completed", 
                   results=results,
                   user=current_user.get('username'))
        
        return {
            "message": "All models training completed",
            "results": results
        }
        
    except Exception as e:
        logger.error("Error training all models", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch training failed: {str(e)}")


@router.post("/models/{model_name}/predict", response_model=PredictionResponse)
async def predict_with_model(
    model_name: str,
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Make a prediction using a specific model."""
    try:
        if model_name not in model_manager.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Make prediction
        prediction = await model_manager.predict(
            model_name=model_name,
            features=request.features
        )
        
        return PredictionResponse(
            model_name=prediction["model_name"],
            attack_type=prediction["attack_type"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            prediction_timestamp=prediction["prediction_timestamp"].isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error making prediction", model_name=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/models/ensemble/predict", response_model=PredictionResponse)
async def predict_with_ensemble(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Make a prediction using the ensemble model."""
    try:
        # Make ensemble prediction
        prediction = await model_manager.predict_ensemble(
            ensemble_name="main_ensemble",
            features=request.features
        )
        
        return PredictionResponse(
            model_name=prediction["ensemble_name"],
            attack_type=prediction["attack_type"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            prediction_timestamp=prediction["prediction_timestamp"].isoformat()
        )
        
    except Exception as e:
        logger.error("Error making ensemble prediction", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ensemble prediction failed: {str(e)}")


@router.post("/models/{model_name}/update")
async def update_model(
    model_name: str,
    request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Update a model with new data (online learning)."""
    try:
        if model_name not in model_manager.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Prepare new data
        new_data = request.get("new_data", {})
        X_new = np.array(new_data.get("features", []))
        y_new = np.array(new_data.get("labels", []))
        
        # Update the model
        metrics = await model_manager.update_model(
            model_name=model_name,
            new_data=X_new,
            new_labels=y_new
        )
        
        logger.info("Model updated", 
                   model_name=model_name,
                   metrics=metrics,
                   user=current_user.get('username'))
        
        return {
            "message": "Model updated successfully",
            "model_name": model_name,
            "metrics": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating model", model_name=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Model update failed: {str(e)}")


@router.get("/models/feature-importance/{model_name}")
async def get_feature_importance(
    model_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Get feature importance for a specific model."""
    try:
        if model_name not in model_manager.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = model_manager.models[model_name]
        feature_importance = model.get_feature_importance()
        
        if feature_importance is None:
            raise HTTPException(status_code=404, detail="Feature importance not available")
        
        return {
            "model_name": model_name,
            "feature_importance": feature_importance
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting feature importance", model_name=model_name, error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving feature importance")
