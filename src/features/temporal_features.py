"""
Temporal feature extraction for cyberattack detection.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


class TemporalFeatureExtractor:
    """Extract temporal patterns and features for attack detection."""
    
    def __init__(self):
        self.time_windows = {
            'hour': 1,
            'day': 24,
            'week': 168,  # 7 * 24
            'month': 720  # 30 * 24
        }
    
    def extract_temporal_features(self, timestamps: List[datetime], 
                                current_time: Optional[datetime] = None) -> Dict:
        """Extract temporal features from a list of timestamps."""
        if not timestamps:
            return self._get_empty_temporal_features()
        
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Convert to pandas for easier manipulation
        df = pd.DataFrame({'timestamp': timestamps})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        features = {}
        
        # Basic temporal statistics
        features.update(self._extract_basic_temporal_stats(df, current_time))
        
        # Time-based patterns
        features.update(self._extract_time_patterns(df))
        
        # Frequency analysis
        features.update(self._extract_frequency_features(df, current_time))
        
        # Anomaly detection
        features.update(self._extract_temporal_anomalies(df))
        
        # Seasonal patterns
        features.update(self._extract_seasonal_patterns(df))
        
        return features
    
    def _extract_basic_temporal_stats(self, df: pd.DataFrame, current_time: datetime) -> Dict:
        """Extract basic temporal statistics."""
        # Calculate time differences
        df_sorted = df.sort_values('timestamp')
        df_sorted['time_diff'] = df_sorted['timestamp'].diff().dt.total_seconds()
        
        # Remove first row (NaN)
        time_diffs = df_sorted['time_diff'].dropna()
        
        features = {
            'total_events': len(df),
            'time_span_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600,
            'mean_interval_seconds': time_diffs.mean() if len(time_diffs) > 0 else 0,
            'std_interval_seconds': time_diffs.std() if len(time_diffs) > 1 else 0,
            'min_interval_seconds': time_diffs.min() if len(time_diffs) > 0 else 0,
            'max_interval_seconds': time_diffs.max() if len(time_diffs) > 0 else 0,
            'median_interval_seconds': time_diffs.median() if len(time_diffs) > 0 else 0,
        }
        
        # Recent activity
        recent_cutoff = current_time - timedelta(hours=1)
        features['recent_events_1h'] = len(df[df['timestamp'] > recent_cutoff])
        
        recent_cutoff = current_time - timedelta(hours=24)
        features['recent_events_24h'] = len(df[df['timestamp'] > recent_cutoff])
        
        return features
    
    def _extract_time_patterns(self, df: pd.DataFrame) -> Dict:
        """Extract time-based patterns."""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        features = {}
        
        # Hour distribution
        hour_counts = df['hour'].value_counts().sort_index()
        features['peak_hour'] = hour_counts.idxmax() if len(hour_counts) > 0 else 0
        features['hour_entropy'] = self._calculate_entropy(hour_counts.values)
        
        # Day of week distribution
        dow_counts = df['day_of_week'].value_counts().sort_index()
        features['peak_day_of_week'] = dow_counts.idxmax() if len(dow_counts) > 0 else 0
        features['dow_entropy'] = self._calculate_entropy(dow_counts.values)
        
        # Weekend vs weekday activity
        weekend_days = [5, 6]  # Saturday, Sunday
        features['weekend_activity_ratio'] = len(df[df['day_of_week'].isin(weekend_days)]) / len(df)
        
        # Business hours activity (9 AM - 5 PM)
        business_hours = df[(df['hour'] >= 9) & (df['hour'] <= 17)]
        features['business_hours_ratio'] = len(business_hours) / len(df)
        
        return features
    
    def _extract_frequency_features(self, df: pd.DataFrame, current_time: datetime) -> Dict:
        """Extract frequency-based features."""
        features = {}
        
        for window_name, hours in self.time_windows.items():
            cutoff_time = current_time - timedelta(hours=hours)
            window_events = df[df['timestamp'] > cutoff_time]
            
            features[f'events_per_{window_name}'] = len(window_events)
            features[f'events_per_hour_{window_name}'] = len(window_events) / hours if hours > 0 else 0
        
        # Burst detection
        features.update(self._detect_bursts(df))
        
        return features
    
    def _detect_bursts(self, df: pd.DataFrame) -> Dict:
        """Detect burst patterns in activity."""
        features = {}
        
        # Group by hour and count events
        df['hour_bucket'] = df['timestamp'].dt.floor('H')
        hourly_counts = df.groupby('hour_bucket').size()
        
        if len(hourly_counts) > 1:
            # Calculate burst metrics
            mean_events = hourly_counts.mean()
            std_events = hourly_counts.std()
            
            # Burst threshold (mean + 2*std)
            burst_threshold = mean_events + (2 * std_events) if std_events > 0 else mean_events * 2
            
            bursts = hourly_counts[hourly_counts > burst_threshold]
            features['burst_count'] = len(bursts)
            features['max_burst_size'] = hourly_counts.max()
            features['burst_ratio'] = len(bursts) / len(hourly_counts)
        else:
            features['burst_count'] = 0
            features['max_burst_size'] = len(df)
            features['burst_ratio'] = 0
        
        return features
    
    def _extract_temporal_anomalies(self, df: pd.DataFrame) -> Dict:
        """Extract temporal anomaly features."""
        features = {}
        
        if len(df) < 2:
            return {'temporal_anomaly_score': 0.0, 'irregular_intervals': False}
        
        # Calculate intervals
        df_sorted = df.sort_values('timestamp')
        intervals = df_sorted['timestamp'].diff().dt.total_seconds().dropna()
        
        if len(intervals) > 1:
            # Z-score based anomaly detection
            mean_interval = intervals.mean()
            std_interval = intervals.std()
            
            if std_interval > 0:
                z_scores = np.abs((intervals - mean_interval) / std_interval)
                features['temporal_anomaly_score'] = z_scores.max()
                features['irregular_intervals'] = (z_scores > 2).any()
            else:
                features['temporal_anomaly_score'] = 0.0
                features['irregular_intervals'] = False
        else:
            features['temporal_anomaly_score'] = 0.0
            features['irregular_intervals'] = False
        
        return features
    
    def _extract_seasonal_patterns(self, df: pd.DataFrame) -> Dict:
        """Extract seasonal and cyclical patterns."""
        features = {}
        
        if len(df) < 24:  # Need at least 24 hours of data
            return {'seasonal_strength': 0.0, 'cyclical_pattern': False}
        
        # Group by hour to detect daily patterns
        df['hour'] = df['timestamp'].dt.hour
        hourly_counts = df.groupby('hour').size()
        
        # Calculate autocorrelation for cyclical patterns
        if len(hourly_counts) >= 24:
            # Pad to 24 hours if needed
            full_hours = pd.Series(index=range(24), dtype=int).fillna(0)
            full_hours.update(hourly_counts)
            
            # Calculate autocorrelation
            autocorr = full_hours.autocorr(lag=1)
            features['cyclical_pattern'] = autocorr > 0.3 if not pd.isna(autocorr) else False
            features['seasonal_strength'] = abs(autocorr) if not pd.isna(autocorr) else 0.0
        else:
            features['cyclical_pattern'] = False
            features['seasonal_strength'] = 0.0
        
        return features
    
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """Calculate entropy of a distribution."""
        if len(values) == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = values / values.sum()
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return float(entropy)
    
    def _get_empty_temporal_features(self) -> Dict:
        """Return empty temporal features when no data is available."""
        features = {
            'total_events': 0,
            'time_span_hours': 0,
            'mean_interval_seconds': 0,
            'std_interval_seconds': 0,
            'min_interval_seconds': 0,
            'max_interval_seconds': 0,
            'median_interval_seconds': 0,
            'recent_events_1h': 0,
            'recent_events_24h': 0,
            'peak_hour': 0,
            'hour_entropy': 0,
            'peak_day_of_week': 0,
            'dow_entropy': 0,
            'weekend_activity_ratio': 0,
            'business_hours_ratio': 0,
            'events_per_hour': 0,
            'events_per_day': 0,
            'events_per_week': 0,
            'events_per_month': 0,
            'events_per_hour_hour': 0,
            'events_per_hour_day': 0,
            'events_per_hour_week': 0,
            'events_per_hour_month': 0,
            'burst_count': 0,
            'max_burst_size': 0,
            'burst_ratio': 0,
            'temporal_anomaly_score': 0,
            'irregular_intervals': False,
            'seasonal_strength': 0,
            'cyclical_pattern': False
        }
        
        return features


class AccessPatternAnalyzer:
    """Analyze access patterns for attack detection."""
    
    def __init__(self):
        self.normal_patterns = {
            'human_like': {
                'min_interval': 1,  # seconds
                'max_interval': 300,  # 5 minutes
                'burst_threshold': 10
            },
            'bot_like': {
                'min_interval': 0.1,
                'max_interval': 1,
                'burst_threshold': 100
            }
        }
    
    def analyze_access_pattern(self, timestamps: List[datetime]) -> Dict:
        """Analyze access patterns to classify behavior."""
        if not timestamps:
            return {'pattern_type': 'unknown', 'confidence': 0.0}
        
        # Sort timestamps
        sorted_timestamps = sorted(timestamps)
        
        # Calculate intervals
        intervals = []
        for i in range(1, len(sorted_timestamps)):
            interval = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return {'pattern_type': 'single_access', 'confidence': 1.0}
        
        # Analyze intervals
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Classify pattern
        if mean_interval < 1:
            pattern_type = 'bot_like'
            confidence = min(1.0, (1 - mean_interval) * 2)
        elif mean_interval > 300:
            pattern_type = 'sparse'
            confidence = min(1.0, mean_interval / 1000)
        else:
            pattern_type = 'human_like'
            confidence = 0.7
        
        return {
            'pattern_type': pattern_type,
            'confidence': confidence,
            'mean_interval': mean_interval,
            'std_interval': std_interval,
            'total_requests': len(timestamps)
        }


# Global feature extractor instances
temporal_feature_extractor = TemporalFeatureExtractor()
access_pattern_analyzer = AccessPatternAnalyzer()

