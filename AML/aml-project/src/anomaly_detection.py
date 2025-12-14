"""
Anomaly Detection Module (Machine Learning)
============================================
Uses ML algorithms to detect unusual account behavior:

1. Isolation Forest - identifies outliers in multidimensional feature space
2. Local Outlier Factor - density-based anomaly detection
3. Statistical Z-Score - identifies statistical outliers

Features extracted for ML:
- Transaction frequency metrics
- Amount statistics
- Time-based patterns
- Network metrics
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AnomalyAlert:
    """Represents an ML-detected anomaly alert."""
    alert_id: str
    account_id: str
    alert_type: str
    risk_level: str
    anomaly_score: float
    description: str
    features_flagged: List[str]
    feature_values: Dict[str, float]
    percentile_ranks: Dict[str, float]
    timestamp: pd.Timestamp
    details: Dict = field(default_factory=dict)


class AnomalyDetector:
    """
    Machine Learning-based anomaly detection for AML.
    
    Uses multiple algorithms to identify accounts with
    unusual behavior patterns that may indicate money laundering.
    """
    
    def __init__(
        self,
        isolation_contamination: float = 0.05,
        lof_contamination: float = 0.05,
        z_score_threshold: float = 3.0,
        random_state: int = 42
    ):
        self.isolation_contamination = isolation_contamination
        self.lof_contamination = lof_contamination
        self.z_score_threshold = z_score_threshold
        self.random_state = random_state
        
        self.scaler = RobustScaler()
        self.isolation_forest = None
        self.lof = None
        
        self.alerts: List[AnomalyAlert] = []
        self._alert_counter = 0
        
        # Store feature names for interpretability
        self.feature_names: List[str] = []
        self.feature_stats: Dict[str, Dict] = {}
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"ANM-{self._alert_counter:06d}"
    
    def extract_features(
        self, 
        transactions: pd.DataFrame,
        customers: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Extract behavioral features from transaction data.
        
        Creates a feature matrix where each row is an account
        and columns are behavioral metrics.
        """
        print("   ├── Extracting behavioral features...")
        
        # Get all unique accounts
        all_accounts = list(set(transactions['sender_id'].unique()) | 
                           set(transactions['receiver_id'].unique()))
        
        features_list = []
        
        for account_id in all_accounts:
            # Get transactions as sender and receiver
            sent = transactions[transactions['sender_id'] == account_id]
            received = transactions[transactions['receiver_id'] == account_id]
            all_txs = pd.concat([sent, received])
            
            if all_txs.empty:
                continue
            
            # === Transaction Volume Features ===
            total_sent = sent['amount'].sum() if not sent.empty else 0
            total_received = received['amount'].sum() if not received.empty else 0
            
            features = {
                'account_id': account_id,
                
                # Volume metrics
                'tx_count_total': len(all_txs),
                'tx_count_sent': len(sent),
                'tx_count_received': len(received),
                'amount_total': total_sent + total_received,
                'amount_sent': total_sent,
                'amount_received': total_received,
                
                # Balance ratio (mule indicator)
                'balance_ratio': (total_received - total_sent) / max(total_received, 1),
                'flow_asymmetry': abs(total_sent - total_received) / max(total_sent + total_received, 1),
                
                # Amount statistics
                'amount_mean': all_txs['amount'].mean(),
                'amount_std': all_txs['amount'].std() if len(all_txs) > 1 else 0,
                'amount_max': all_txs['amount'].max(),
                'amount_min': all_txs['amount'].min(),
                'amount_median': all_txs['amount'].median(),
                
                # Coefficient of variation (consistency indicator)
                'amount_cv': all_txs['amount'].std() / all_txs['amount'].mean() if all_txs['amount'].mean() > 0 else 0,
                
                # Unique counterparties
                'unique_senders': received['sender_id'].nunique() if not received.empty else 0,
                'unique_receivers': sent['receiver_id'].nunique() if not sent.empty else 0,
                'unique_counterparties': len(set(sent['receiver_id'].unique()) | set(received['sender_id'].unique())),
                
                # Time-based features
                'active_days': all_txs['timestamp'].dt.date.nunique(),
                'tx_per_active_day': len(all_txs) / max(all_txs['timestamp'].dt.date.nunique(), 1),
            }
            
            # === Time pattern features ===
            if len(all_txs) > 1:
                all_txs_sorted = all_txs.sort_values('timestamp')
                time_diffs = all_txs_sorted['timestamp'].diff().dt.total_seconds() / 3600  # hours
                
                features['avg_hours_between_tx'] = time_diffs.mean()
                features['min_hours_between_tx'] = time_diffs.min()
                features['std_hours_between_tx'] = time_diffs.std() if len(time_diffs.dropna()) > 1 else 0
            else:
                features['avg_hours_between_tx'] = 0
                features['min_hours_between_tx'] = 0
                features['std_hours_between_tx'] = 0
            
            # === Transaction type features ===
            if not all_txs.empty:
                tx_type_counts = all_txs['tx_type'].value_counts(normalize=True)
                features['cash_ratio'] = tx_type_counts.get('cash', 0)
                features['wire_ratio'] = tx_type_counts.get('wire', 0)
                features['crypto_ratio'] = tx_type_counts.get('crypto', 0)
                
                # Channel distribution
                channel_counts = all_txs['channel'].value_counts(normalize=True)
                features['online_ratio'] = channel_counts.get('online', 0)
                features['branch_ratio'] = channel_counts.get('branch', 0)
            
            # === Structuring indicators ===
            # Transactions just under $10k
            near_threshold = all_txs[(all_txs['amount'] >= 8000) & (all_txs['amount'] < 10000)]
            features['near_threshold_ratio'] = len(near_threshold) / max(len(all_txs), 1)
            features['near_threshold_count'] = len(near_threshold)
            
            # Round amount ratio
            round_amounts = all_txs[all_txs['amount'] % 1000 < 50]
            features['round_amount_ratio'] = len(round_amounts) / max(len(all_txs), 1)
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Store feature names (excluding account_id)
        self.feature_names = [col for col in features_df.columns if col != 'account_id']
        
        # Calculate feature statistics for percentile ranking
        for feature in self.feature_names:
            self.feature_stats[feature] = {
                'mean': features_df[feature].mean(),
                'std': features_df[feature].std(),
                'min': features_df[feature].min(),
                'max': features_df[feature].max(),
                'median': features_df[feature].median()
            }
        
        print(f"   │   └── Extracted {len(self.feature_names)} features for {len(features_df)} accounts")
        
        return features_df
    
    def _get_percentile_rank(self, value: float, series: pd.Series) -> float:
        """Calculate percentile rank of a value in a series."""
        return (series < value).mean() * 100
    
    def detect_isolation_forest(
        self, 
        features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[AnomalyAlert]]:
        """
        Detect anomalies using Isolation Forest.
        
        Isolation Forest isolates anomalies by randomly selecting features
        and split values. Anomalies require fewer splits to isolate.
        """
        print("   ├── Running Isolation Forest...")
        
        alerts = []
        
        # Prepare feature matrix
        X = features_df[self.feature_names].fillna(0)
        account_ids = features_df['account_id'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.isolation_contamination,
            random_state=self.random_state,
            n_estimators=200,
            max_samples='auto'
        )
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.isolation_forest.fit_predict(X_scaled)
        scores = self.isolation_forest.decision_function(X_scaled)
        
        # Normalize scores to 0-1 (higher = more anomalous)
        anomaly_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        
        # Add results to dataframe
        results = features_df.copy()
        results['if_prediction'] = predictions
        results['if_anomaly_score'] = anomaly_scores
        
        # Generate alerts for anomalies
        anomalies = results[results['if_prediction'] == -1]
        
        for _, row in anomalies.iterrows():
            # Find which features contributed most to anomaly
            feature_z_scores = {}
            for feature in self.feature_names:
                if self.feature_stats[feature]['std'] > 0:
                    z = (row[feature] - self.feature_stats[feature]['mean']) / self.feature_stats[feature]['std']
                    feature_z_scores[feature] = abs(z)
            
            # Top 5 most unusual features
            top_features = sorted(feature_z_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            flagged_features = [f[0] for f in top_features]
            
            # Calculate percentile ranks for flagged features
            percentiles = {}
            feature_values = {}
            for feature in flagged_features:
                feature_values[feature] = row[feature]
                percentiles[feature] = self._get_percentile_rank(
                    row[feature], 
                    features_df[feature]
                )
            
            # Determine risk level based on score
            score = row['if_anomaly_score']
            if score > 0.9:
                risk_level = 'critical'
            elif score > 0.75:
                risk_level = 'high'
            elif score > 0.6:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            alert = AnomalyAlert(
                alert_id=self._generate_alert_id(),
                account_id=row['account_id'],
                alert_type='ISOLATION_FOREST',
                risk_level=risk_level,
                anomaly_score=score,
                description=f"ML anomaly detected: account behavior deviates from normal patterns",
                features_flagged=flagged_features,
                feature_values=feature_values,
                percentile_ranks=percentiles,
                timestamp=pd.Timestamp.now(),
                details={
                    'top_anomalous_features': dict(top_features),
                    'tx_count': row['tx_count_total'],
                    'total_amount': row['amount_total']
                }
            )
            alerts.append(alert)
        
        print(f"   │   └── Found {len(anomalies)} anomalies")
        
        return results, alerts
    
    def detect_lof(
        self, 
        features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[AnomalyAlert]]:
        """
        Detect anomalies using Local Outlier Factor.
        
        LOF measures local density deviation of a data point
        with respect to its neighbors.
        """
        print("   ├── Running Local Outlier Factor...")
        
        alerts = []
        
        # Prepare feature matrix
        X = features_df[self.feature_names].fillna(0)
        account_ids = features_df['account_id'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit LOF
        self.lof = LocalOutlierFactor(
            contamination=self.lof_contamination,
            n_neighbors=20,
            novelty=False
        )
        
        # Predict anomalies
        predictions = self.lof.fit_predict(X_scaled)
        scores = -self.lof.negative_outlier_factor_  # Higher = more anomalous
        
        # Normalize scores to 0-1
        anomaly_scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        # Add results to dataframe
        results = features_df.copy()
        results['lof_prediction'] = predictions
        results['lof_anomaly_score'] = anomaly_scores
        
        # Generate alerts for anomalies
        anomalies = results[results['lof_prediction'] == -1]
        
        for _, row in anomalies.iterrows():
            # Find contributing features
            feature_z_scores = {}
            for feature in self.feature_names:
                if self.feature_stats[feature]['std'] > 0:
                    z = (row[feature] - self.feature_stats[feature]['mean']) / self.feature_stats[feature]['std']
                    feature_z_scores[feature] = abs(z)
            
            top_features = sorted(feature_z_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            flagged_features = [f[0] for f in top_features]
            
            percentiles = {}
            feature_values = {}
            for feature in flagged_features:
                feature_values[feature] = row[feature]
                percentiles[feature] = self._get_percentile_rank(row[feature], features_df[feature])
            
            score = row['lof_anomaly_score']
            if score > 0.9:
                risk_level = 'critical'
            elif score > 0.75:
                risk_level = 'high'
            elif score > 0.6:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            alert = AnomalyAlert(
                alert_id=self._generate_alert_id(),
                account_id=row['account_id'],
                alert_type='LOCAL_OUTLIER_FACTOR',
                risk_level=risk_level,
                anomaly_score=score,
                description=f"Density-based anomaly: account shows unusual local behavior",
                features_flagged=flagged_features,
                feature_values=feature_values,
                percentile_ranks=percentiles,
                timestamp=pd.Timestamp.now(),
                details={
                    'top_anomalous_features': dict(top_features),
                    'tx_count': row['tx_count_total'],
                    'total_amount': row['amount_total']
                }
            )
            alerts.append(alert)
        
        print(f"   │   └── Found {len(anomalies)} anomalies")
        
        return results, alerts
    
    def detect_statistical_outliers(
        self, 
        features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[AnomalyAlert]]:
        """
        Detect outliers using statistical Z-score method.
        
        Flags accounts with multiple features exceeding z-score threshold.
        """
        print("   ├── Running Statistical Outlier Detection...")
        
        alerts = []
        
        # Calculate z-scores for all features
        z_scores_df = features_df[self.feature_names].apply(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        # Count how many features exceed threshold for each account
        outlier_counts = (z_scores_df.abs() > self.z_score_threshold).sum(axis=1)
        
        # Flag accounts with multiple outlier features
        results = features_df.copy()
        results['outlier_feature_count'] = outlier_counts
        results['is_statistical_outlier'] = outlier_counts >= 3
        
        outliers = results[results['is_statistical_outlier']]
        
        for _, row in outliers.iterrows():
            # Find which features are outliers
            account_z_scores = z_scores_df.loc[row.name]
            outlier_features = account_z_scores[account_z_scores.abs() > self.z_score_threshold]
            
            flagged_features = outlier_features.index.tolist()[:5]
            
            percentiles = {}
            feature_values = {}
            for feature in flagged_features:
                feature_values[feature] = row[feature]
                percentiles[feature] = self._get_percentile_rank(row[feature], features_df[feature])
            
            # Risk based on number of outlier features
            n_outliers = row['outlier_feature_count']
            if n_outliers >= 6:
                risk_level = 'critical'
            elif n_outliers >= 5:
                risk_level = 'high'
            elif n_outliers >= 4:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            alert = AnomalyAlert(
                alert_id=self._generate_alert_id(),
                account_id=row['account_id'],
                alert_type='STATISTICAL_OUTLIER',
                risk_level=risk_level,
                anomaly_score=n_outliers / len(self.feature_names),
                description=f"Statistical outlier: {n_outliers} features exceed normal range",
                features_flagged=flagged_features,
                feature_values=feature_values,
                percentile_ranks=percentiles,
                timestamp=pd.Timestamp.now(),
                details={
                    'outlier_feature_count': n_outliers,
                    'z_scores': {f: float(account_z_scores[f]) for f in flagged_features},
                    'tx_count': row['tx_count_total'],
                    'total_amount': row['amount_total']
                }
            )
            alerts.append(alert)
        
        print(f"   │   └── Found {len(outliers)} statistical outliers")
        
        return results, alerts
    
    def run_all_detectors(
        self, 
        transactions: pd.DataFrame,
        customers: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, List[AnomalyAlert], pd.DataFrame]:
        """
        Run all anomaly detection algorithms.
        
        Args:
            transactions: Transaction DataFrame
            customers: Optional customer DataFrame
            
        Returns:
            Tuple of (features_df, alerts list, alerts DataFrame)
        """
        print("🔍 Running ML Anomaly Detection...")
        
        # Extract features
        features_df = self.extract_features(transactions, customers)
        
        # Run detectors
        if_results, if_alerts = self.detect_isolation_forest(features_df)
        lof_results, lof_alerts = self.detect_lof(features_df)
        stat_results, stat_alerts = self.detect_statistical_outliers(features_df)
        
        # Combine results
        all_alerts = if_alerts + lof_alerts + stat_alerts
        self.alerts = all_alerts
        
        # Merge detection results
        features_df['if_anomaly_score'] = if_results['if_anomaly_score']
        features_df['if_prediction'] = if_results['if_prediction']
        features_df['lof_anomaly_score'] = lof_results['lof_anomaly_score']
        features_df['lof_prediction'] = lof_results['lof_prediction']
        features_df['outlier_feature_count'] = stat_results['outlier_feature_count']
        
        # Combined anomaly score (average of methods)
        features_df['combined_anomaly_score'] = (
            features_df['if_anomaly_score'] + 
            features_df['lof_anomaly_score'] +
            features_df['outlier_feature_count'] / len(self.feature_names)
        ) / 3
        
        # Convert alerts to DataFrame
        if all_alerts:
            alerts_df = pd.DataFrame([{
                'alert_id': a.alert_id,
                'account_id': a.account_id,
                'alert_type': a.alert_type,
                'risk_level': a.risk_level,
                'anomaly_score': a.anomaly_score,
                'description': a.description,
                'features_flagged': ', '.join(a.features_flagged),
                'timestamp': a.timestamp
            } for a in all_alerts])
        else:
            alerts_df = pd.DataFrame(columns=[
                'alert_id', 'account_id', 'alert_type', 'risk_level',
                'anomaly_score', 'description', 'features_flagged', 'timestamp'
            ])
        
        print(f"\n✅ ML Anomaly Detection Complete: {len(all_alerts)} alerts generated")
        print(f"   Detection Method Distribution:")
        if not alerts_df.empty:
            for method, count in alerts_df['alert_type'].value_counts().items():
                print(f"   - {method}: {count}")
        
        return features_df, all_alerts, alerts_df


def main():
    """Test anomaly detection on sample data."""
    # Load data
    transactions = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/transactions.csv')
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    
    customers = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/customers.csv')
    
    # Run detection
    detector = AnomalyDetector()
    features_df, alerts, alerts_df = detector.run_all_detectors(transactions, customers)
    
    # Save results
    features_df.to_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/account_features.csv', index=False)
    
    if not alerts_df.empty:
        alerts_df.to_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/anomaly_alerts.csv', index=False)
        print(f"\n📁 Results saved to data/ directory")
    
    return features_df, alerts_df


if __name__ == '__main__':
    main()


