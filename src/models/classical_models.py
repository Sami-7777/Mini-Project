"""
Classical machine learning models for cyberattack detection.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import structlog

from .base_model import BaseModel

logger = structlog.get_logger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest classifier for cyberattack detection."""
    
    def __init__(self, model_name: str = "random_forest", 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 model_version: str = "1.0.0"):
        super().__init__(model_name, model_version)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
    
    def build_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Build Random Forest model."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train Random Forest model."""
        if self.model is None:
            self.build_model(X_train.shape, len(np.unique(y_train)))
        
        # Store feature names if available
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()
        
        # Evaluate on validation set if provided
        metrics = {}
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        
        logger.info("Random Forest model trained successfully", 
                   n_estimators=self.n_estimators,
                   feature_count=len(self.feature_names))
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """XGBoost classifier for cyberattack detection."""
    
    def __init__(self, model_name: str = "xgboost",
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42,
                 model_version: str = "1.0.0"):
        super().__init__(model_name, model_version)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
    
    def build_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Build XGBoost model."""
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train XGBoost model."""
        if self.model is None:
            self.build_model(X_train.shape, len(np.unique(y_train)))
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Prepare training data
        train_data = xgb.DMatrix(X_train, label=y_train)
        
        # Prepare validation data if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            val_data = xgb.DMatrix(X_val, label=y_val)
            eval_set = [(train_data, 'train'), (val_data, 'eval')]
        
        # Train the model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()
        
        # Evaluate on validation set if provided
        metrics = {}
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        
        logger.info("XGBoost model trained successfully", 
                   n_estimators=self.n_estimators,
                   feature_count=len(self.feature_names))
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)


class SVMModel(BaseModel):
    """Support Vector Machine classifier for cyberattack detection."""
    
    def __init__(self, model_name: str = "svm",
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 probability: bool = True,
                 random_state: int = 42,
                 model_version: str = "1.0.0"):
        super().__init__(model_name, model_version)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
    
    def build_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Build SVM model."""
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability,
            random_state=self.random_state,
            class_weight='balanced'
        )
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train SVM model."""
        if self.model is None:
            self.build_model(X_train.shape, len(np.unique(y_train)))
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()
        
        # Evaluate on validation set if provided
        metrics = {}
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        
        logger.info("SVM model trained successfully", 
                   kernel=self.kernel,
                   feature_count=len(self.feature_names))
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier for cyberattack detection."""
    
    def __init__(self, model_name: str = "logistic_regression",
                 C: float = 1.0,
                 penalty: str = 'l2',
                 solver: str = 'liblinear',
                 max_iter: int = 1000,
                 random_state: int = 42,
                 model_version: str = "1.0.0"):
        super().__init__(model_name, model_version)
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
    
    def build_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Build Logistic Regression model."""
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight='balanced'
        )
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train Logistic Regression model."""
        if self.model is None:
            self.build_model(X_train.shape, len(np.unique(y_train)))
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()
        
        # Evaluate on validation set if provided
        metrics = {}
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        
        logger.info("Logistic Regression model trained successfully", 
                   feature_count=len(self.feature_names))
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)


class NaiveBayesModel(BaseModel):
    """Naive Bayes classifier for cyberattack detection."""
    
    def __init__(self, model_name: str = "naive_bayes",
                 model_version: str = "1.0.0"):
        super().__init__(model_name, model_version)
    
    def build_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Build Naive Bayes model."""
        self.model = GaussianNB()
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train Naive Bayes model."""
        if self.model is None:
            self.build_model(X_train.shape, len(np.unique(y_train)))
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()
        
        # Evaluate on validation set if provided
        metrics = {}
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        
        logger.info("Naive Bayes model trained successfully", 
                   feature_count=len(self.feature_names))
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)


class KNNModel(BaseModel):
    """K-Nearest Neighbors classifier for cyberattack detection."""
    
    def __init__(self, model_name: str = "knn",
                 n_neighbors: int = 5,
                 weights: str = 'distance',
                 algorithm: str = 'auto',
                 model_version: str = "1.0.0"):
        super().__init__(model_name, model_version)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
    
    def build_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Build KNN model."""
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            n_jobs=-1
        )
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train KNN model."""
        if self.model is None:
            self.build_model(X_train.shape, len(np.unique(y_train)))
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()
        
        # Evaluate on validation set if provided
        metrics = {}
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        
        logger.info("KNN model trained successfully", 
                   n_neighbors=self.n_neighbors,
                   feature_count=len(self.feature_names))
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)

