"""
Deep learning models for cyberattack detection.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from typing import Dict, List, Optional, Tuple, Any
import structlog
from datetime import datetime

from .base_model import BaseModel

logger = structlog.get_logger(__name__)

# Set TensorFlow to use GPU if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class CNNLSTMModel(BaseModel):
    """CNN-LSTM hybrid model for cyberattack detection."""
    
    def __init__(self, model_name: str = "cnn_lstm",
                 sequence_length: int = 100,
                 embedding_dim: int = 128,
                 cnn_filters: int = 64,
                 cnn_kernel_size: int = 3,
                 lstm_units: int = 64,
                 dropout_rate: float = 0.5,
                 learning_rate: float = 0.001,
                 model_version: str = "1.0.0"):
        super().__init__(model_name, model_version)
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
    
    def build_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Build CNN-LSTM model architecture."""
        # Input layer
        inputs = layers.Input(shape=input_shape, name='input')
        
        # Reshape for CNN (add channel dimension)
        x = layers.Reshape((input_shape[0], 1))(inputs)
        
        # CNN layers for feature extraction
        x = layers.Conv1D(
            filters=self.cnn_filters,
            kernel_size=self.cnn_kernel_size,
            activation='relu',
            padding='same',
            name='conv1d_1'
        )(x)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        x = layers.MaxPooling1D(pool_size=2, name='max_pool_1')(x)
        
        x = layers.Conv1D(
            filters=self.cnn_filters * 2,
            kernel_size=self.cnn_kernel_size,
            activation='relu',
            padding='same',
            name='conv1d_2'
        )(x)
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        x = layers.MaxPooling1D(pool_size=2, name='max_pool_2')(x)
        
        # LSTM layers for sequence modeling
        x = layers.LSTM(
            units=self.lstm_units,
            return_sequences=True,
            dropout=self.dropout_rate,
            name='lstm_1'
        )(x)
        x = layers.LSTM(
            units=self.lstm_units // 2,
            return_sequences=False,
            dropout=self.dropout_rate,
            name='lstm_2'
        )(x)
        
        # Dense layers for classification
        x = layers.Dense(128, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        x = layers.Dense(64, activation='relu', name='dense_2')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Output layer
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
            loss = 'categorical_crossentropy'
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name=self.model_name)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train CNN-LSTM model."""
        if self.model is None:
            self.build_model(X_train.shape[1:], len(np.unique(y_train)))
        
        # Store feature names
        self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Prepare data
        if len(np.unique(y_train)) == 2:
            # Binary classification
            y_train_categorical = y_train
            y_val_categorical = y_val if y_val is not None else None
        else:
            # Multi-class classification
            y_train_categorical = keras.utils.to_categorical(y_train)
            y_val_categorical = keras.utils.to_categorical(y_val) if y_val is not None else None
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train_categorical,
            validation_data=(X_val, y_val_categorical) if X_val is not None else None,
            epochs=100,
            batch_size=32,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()
        
        # Extract metrics from training history
        metrics = {
            'final_loss': float(history.history['loss'][-1]),
            'final_accuracy': float(history.history['accuracy'][-1])
        }
        
        if X_val is not None:
            metrics['val_loss'] = float(history.history['val_loss'][-1])
            metrics['val_accuracy'] = float(history.history['val_accuracy'][-1])
        
        logger.info("CNN-LSTM model trained successfully", 
                   epochs=len(history.history['loss']),
                   feature_count=len(self.feature_names))
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(X)
        
        if predictions.shape[1] == 1:
            # Binary classification
            return (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        probabilities = self.model.predict(X)
        
        if probabilities.shape[1] == 1:
            # Binary classification - return probabilities for both classes
            prob_positive = probabilities.flatten()
            prob_negative = 1 - prob_positive
            return np.column_stack([prob_negative, prob_positive])
        else:
            # Multi-class classification
            return probabilities


class TransformerModel(BaseModel):
    """Transformer model for cyberattack detection."""
    
    def __init__(self, model_name: str = "transformer",
                 sequence_length: int = 100,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dff: int = 512,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001,
                 model_version: str = "1.0.0"):
        super().__init__(model_name, model_version)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
    
    def build_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Build Transformer model architecture."""
        # Input layer
        inputs = layers.Input(shape=input_shape, name='input')
        
        # Positional encoding
        x = layers.Dense(self.d_model, name='input_projection')(inputs)
        x = self._add_positional_encoding(x)
        
        # Transformer blocks
        for i in range(self.num_layers):
            x = self._transformer_block(x, f'block_{i}')
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Classification head
        x = layers.Dense(self.dff, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        x = layers.Dense(self.dff // 2, activation='relu', name='dense_2')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Output layer
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
            loss = 'categorical_crossentropy'
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name=self.model_name)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def _add_positional_encoding(self, x):
        """Add positional encoding to input."""
        seq_len = tf.shape(x)[1]
        d_model = self.d_model
        
        # Create positional encoding matrix
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rads = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        
        # Apply sin to even indices, cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return x + tf.cast(pos_encoding, dtype=tf.float32)
    
    def _transformer_block(self, x, name_prefix):
        """Create a transformer block."""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            name=f'{name_prefix}_attention'
        )(x, x)
        attention_output = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_attention_dropout')(attention_output)
        x = layers.Add(name=f'{name_prefix}_add_1')([x, attention_output])
        x = layers.LayerNormalization(name=f'{name_prefix}_norm_1')(x)
        
        # Feed forward network
        ffn_output = layers.Dense(self.dff, activation='relu', name=f'{name_prefix}_ffn_1')(x)
        ffn_output = layers.Dense(self.d_model, name=f'{name_prefix}_ffn_2')(ffn_output)
        ffn_output = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_ffn_dropout')(ffn_output)
        x = layers.Add(name=f'{name_prefix}_add_2')([x, ffn_output])
        x = layers.LayerNormalization(name=f'{name_prefix}_norm_2')(x)
        
        return x
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train Transformer model."""
        if self.model is None:
            self.build_model(X_train.shape[1:], len(np.unique(y_train)))
        
        # Store feature names
        self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Prepare data
        if len(np.unique(y_train)) == 2:
            # Binary classification
            y_train_categorical = y_train
            y_val_categorical = y_val if y_val is not None else None
        else:
            # Multi-class classification
            y_train_categorical = keras.utils.to_categorical(y_train)
            y_val_categorical = keras.utils.to_categorical(y_val) if y_val is not None else None
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train_categorical,
            validation_data=(X_val, y_val_categorical) if X_val is not None else None,
            epochs=100,
            batch_size=32,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()
        
        # Extract metrics from training history
        metrics = {
            'final_loss': float(history.history['loss'][-1]),
            'final_accuracy': float(history.history['accuracy'][-1])
        }
        
        if X_val is not None:
            metrics['val_loss'] = float(history.history['val_loss'][-1])
            metrics['val_accuracy'] = float(history.history['val_accuracy'][-1])
        
        logger.info("Transformer model trained successfully", 
                   epochs=len(history.history['loss']),
                   feature_count=len(self.feature_names))
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(X)
        
        if predictions.shape[1] == 1:
            # Binary classification
            return (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        probabilities = self.model.predict(X)
        
        if probabilities.shape[1] == 1:
            # Binary classification - return probabilities for both classes
            prob_positive = probabilities.flatten()
            prob_negative = 1 - prob_positive
            return np.column_stack([prob_negative, prob_positive])
        else:
            # Multi-class classification
            return probabilities


class AutoEncoderModel(BaseModel):
    """Autoencoder for anomaly detection in cyberattacks."""
    
    def __init__(self, model_name: str = "autoencoder",
                 encoding_dim: int = 32,
                 hidden_dims: List[int] = [128, 64],
                 learning_rate: float = 0.001,
                 model_version: str = "1.0.0"):
        super().__init__(model_name, model_version)
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.threshold = None
    
    def build_model(self, input_shape: Tuple, num_classes: int = None) -> Any:
        """Build Autoencoder model architecture."""
        input_dim = input_shape[0]
        
        # Input layer
        inputs = layers.Input(shape=(input_dim,), name='input')
        
        # Encoder
        x = inputs
        for i, dim in enumerate(self.hidden_dims):
            x = layers.Dense(dim, activation='relu', name=f'encoder_dense_{i}')(x)
            x = layers.BatchNormalization(name=f'encoder_bn_{i}')(x)
            x = layers.Dropout(0.2, name=f'encoder_dropout_{i}')(x)
        
        # Bottleneck
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoded')(x)
        
        # Decoder
        x = encoded
        for i, dim in enumerate(reversed(self.hidden_dims)):
            x = layers.Dense(dim, activation='relu', name=f'decoder_dense_{i}')(x)
            x = layers.BatchNormalization(name=f'decoder_bn_{i}')(x)
            x = layers.Dropout(0.2, name=f'decoder_dropout_{i}')(x)
        
        # Output layer
        decoded = layers.Dense(input_dim, activation='sigmoid', name='decoded')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=decoded, name=self.model_name)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray = None, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train Autoencoder model."""
        if self.model is None:
            self.build_model(X_train.shape[1:])
        
        # Store feature names
        self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train the model (autoencoder learns to reconstruct input)
        history = self.model.fit(
            X_train, X_train,  # Input and target are the same
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=100,
            batch_size=32,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Calculate reconstruction threshold
        train_reconstructions = self.model.predict(X_train)
        train_errors = np.mean(np.square(X_train - train_reconstructions), axis=1)
        self.threshold = np.percentile(train_errors, 95)  # 95th percentile as threshold
        
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()
        
        # Extract metrics from training history
        metrics = {
            'final_loss': float(history.history['loss'][-1]),
            'final_mae': float(history.history['mae'][-1]),
            'threshold': float(self.threshold)
        }
        
        if X_val is not None:
            metrics['val_loss'] = float(history.history['val_loss'][-1])
            metrics['val_mae'] = float(history.history['val_mae'][-1])
        
        logger.info("Autoencoder model trained successfully", 
                   epochs=len(history.history['loss']),
                   feature_count=len(self.feature_names),
                   threshold=self.threshold)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for anomaly, 0 for normal)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Calculate reconstruction errors
        reconstructions = self.model.predict(X)
        errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Classify as anomaly if error > threshold
        anomalies = (errors > self.threshold).astype(int)
        
        return anomalies
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Calculate reconstruction errors
        reconstructions = self.model.predict(X)
        errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Convert errors to probabilities (normalize by threshold)
        prob_anomaly = np.clip(errors / self.threshold, 0, 1)
        prob_normal = 1 - prob_anomaly
        
        return np.column_stack([prob_normal, prob_anomaly])
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction errors for input data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        reconstructions = self.model.predict(X)
        errors = np.mean(np.square(X - reconstructions), axis=1)
        
        return errors

