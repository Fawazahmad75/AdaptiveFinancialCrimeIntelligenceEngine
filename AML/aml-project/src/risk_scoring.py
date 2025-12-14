"""
Risk Scoring Module
===================
Calculates composite risk scores for accounts based on:

1. Number and severity of alerts
2. Total amounts involved
3. Transaction frequency patterns
4. Network connections
5. Customer risk factors
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class RiskCategory(Enum):
    """Risk level categories."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


@dataclass
class RiskScore:
    """Represents an account's risk assessment."""
    account_id: str
    overall_score: float  # 0-100
    risk_category: RiskCategory
    alert_score: float
    volume_score: float
    velocity_score: float
    network_score: float
    behavioral_score: float
    contributing_factors: List[str]
    alert_count: int
    total_amount: float
    details: Dict = field(default_factory=dict)


class RiskScorer:
    """
    Calculates comprehensive risk scores for accounts.
    
    Combines multiple risk indicators into a single composite score
    that can be used for prioritizing investigations.
    """
    
    # Risk weights (must sum to 1.0)
    WEIGHTS = {
        'alert_score': 0.35,      # Weight for alert-based risk
        'volume_score': 0.20,     # Weight for transaction volume
        'velocity_score': 0.20,   # Weight for transaction velocity
        'network_score': 0.15,    # Weight for network connections
        'behavioral_score': 0.10  # Weight for behavioral anomalies
    }
    
    # Alert severity multipliers
    ALERT_SEVERITY = {
        'critical': 4.0,
        'high': 3.0,
        'medium': 2.0,
        'low': 1.0
    }
    
    # Alert type weights
    ALERT_TYPE_WEIGHTS = {
        'MULE_PATTERN': 5.0,
        'FRAUD_RING': 5.0,
        'STRUCTURING': 4.0,
        'UNDER_THRESHOLD_DEPOSITS': 4.0,
        'ISOLATION_FOREST': 3.5,
        'LOCAL_OUTLIER_FACTOR': 3.0,
        'ACTIVITY_SPIKE': 3.0,
        'MANY_TO_ONE': 3.0,
        'RAPID_DEPOSITS': 2.5,
        'HIGH_DAILY_FREQUENCY': 2.0,
        'HIGH_HOURLY_FREQUENCY': 2.0,
        'HIGH_DAILY_AMOUNT': 2.0,
        'STATISTICAL_OUTLIER': 1.5,
        'ROUND_AMOUNTS': 1.5
    }
    
    # Risk category thresholds
    THRESHOLDS = {
        'critical': 75,
        'high': 50,
        'medium': 25,
        'low': 0
    }
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        if weights:
            self.weights = weights
        else:
            self.weights = self.WEIGHTS.copy()
        
        if thresholds:
            self.thresholds = thresholds
        else:
            self.thresholds = self.THRESHOLDS.copy()
        
        self.risk_scores: List[RiskScore] = []
    
    def _normalize_score(self, value: float, max_value: float, min_value: float = 0) -> float:
        """Normalize a value to 0-100 scale."""
        if max_value <= min_value:
            return 0
        return min(100, max(0, ((value - min_value) / (max_value - min_value)) * 100))
    
    def _get_risk_category(self, score: float) -> RiskCategory:
        """Determine risk category from score."""
        if score >= self.thresholds['critical']:
            return RiskCategory.CRITICAL
        elif score >= self.thresholds['high']:
            return RiskCategory.HIGH
        elif score >= self.thresholds['medium']:
            return RiskCategory.MEDIUM
        else:
            return RiskCategory.LOW
    
    def calculate_alert_score(
        self, 
        account_id: str, 
        alerts_df: pd.DataFrame
    ) -> Tuple[float, List[str], int]:
        """
        Calculate risk score based on alerts.
        
        Higher score for:
        - More alerts
        - Higher severity alerts
        - More serious alert types
        """
        account_alerts = alerts_df[alerts_df['account_id'] == account_id]
        
        if account_alerts.empty:
            return 0.0, [], 0
        
        # Calculate weighted alert score
        total_weight = 0
        contributing_factors = []
        
        for _, alert in account_alerts.iterrows():
            # Get severity multiplier
            severity = alert.get('risk_level', 'low').lower()
            severity_mult = self.ALERT_SEVERITY.get(severity, 1.0)
            
            # Get alert type weight
            alert_type = alert.get('alert_type', 'UNKNOWN')
            type_weight = self.ALERT_TYPE_WEIGHTS.get(alert_type, 1.0)
            
            alert_weight = severity_mult * type_weight
            total_weight += alert_weight
            
            contributing_factors.append(f"{alert_type} ({severity})")
        
        # Normalize to 0-100 (max expected: 10 critical high-weight alerts)
        max_expected_weight = 10 * 4.0 * 5.0  # 10 alerts * critical * max weight
        score = self._normalize_score(total_weight, max_expected_weight)
        
        return score, list(set(contributing_factors)), len(account_alerts)
    
    def calculate_volume_score(
        self, 
        account_id: str, 
        transactions: pd.DataFrame,
        percentile_threshold: float = 95
    ) -> Tuple[float, float]:
        """
        Calculate risk score based on transaction volume.
        
        Higher score for accounts with volumes above typical thresholds.
        """
        # Get transactions for this account
        account_txs = transactions[
            (transactions['sender_id'] == account_id) |
            (transactions['receiver_id'] == account_id)
        ]
        
        if account_txs.empty:
            return 0.0, 0.0
        
        total_amount = account_txs['amount'].sum()
        
        # Calculate percentile rank against all accounts
        all_amounts = []
        all_accounts = set(transactions['sender_id'].unique()) | set(transactions['receiver_id'].unique())
        
        for acc in all_accounts:
            acc_txs = transactions[
                (transactions['sender_id'] == acc) |
                (transactions['receiver_id'] == acc)
            ]
            if not acc_txs.empty:
                all_amounts.append(acc_txs['amount'].sum())
        
        all_amounts = np.array(all_amounts)
        percentile = (all_amounts < total_amount).mean() * 100
        
        # Score based on percentile (higher percentile = higher risk)
        if percentile >= 99:
            score = 100
        elif percentile >= 95:
            score = 80 + (percentile - 95) * 4
        elif percentile >= 90:
            score = 60 + (percentile - 90) * 4
        elif percentile >= 75:
            score = 30 + (percentile - 75) * 2
        else:
            score = percentile * 0.4
        
        return score, total_amount
    
    def calculate_velocity_score(
        self, 
        account_id: str, 
        transactions: pd.DataFrame
    ) -> float:
        """
        Calculate risk score based on transaction velocity.
        
        Higher score for:
        - High transaction frequency
        - Rapid consecutive transactions
        - Unusual timing patterns
        """
        account_txs = transactions[
            (transactions['sender_id'] == account_id) |
            (transactions['receiver_id'] == account_id)
        ].copy()
        
        if len(account_txs) < 2:
            return 0.0
        
        # Calculate metrics
        account_txs = account_txs.sort_values('timestamp')
        
        # Transactions per day
        active_days = account_txs['timestamp'].dt.date.nunique()
        total_days = (account_txs['timestamp'].max() - account_txs['timestamp'].min()).days + 1
        tx_per_day = len(account_txs) / max(total_days, 1)
        
        # Activity concentration
        activity_concentration = active_days / max(total_days, 1)
        
        # Minimum time between transactions
        time_diffs = account_txs['timestamp'].diff().dt.total_seconds() / 3600  # hours
        min_interval = time_diffs.min()
        avg_interval = time_diffs.mean()
        
        # Calculate score components
        frequency_score = min(100, tx_per_day * 10)  # 10+ tx/day = 100
        
        if min_interval < 1:  # < 1 hour between some transactions
            rapid_score = 100 - min_interval * 100
        else:
            rapid_score = max(0, 50 - min_interval * 2)
        
        concentration_score = activity_concentration * 100
        
        # Combine scores
        velocity_score = (frequency_score * 0.4 + rapid_score * 0.4 + concentration_score * 0.2)
        
        return velocity_score
    
    def calculate_network_score(
        self, 
        account_id: str, 
        transactions: pd.DataFrame
    ) -> float:
        """
        Calculate risk score based on network connections.
        
        Higher score for:
        - Many unique counterparties
        - Connections to flagged accounts
        - Hub-like behavior
        """
        # Get counterparties
        sent_to = transactions[transactions['sender_id'] == account_id]['receiver_id'].unique()
        received_from = transactions[transactions['receiver_id'] == account_id]['sender_id'].unique()
        
        all_counterparties = set(sent_to) | set(received_from)
        n_counterparties = len(all_counterparties)
        
        # Calculate in/out degree ratio (suspicious if very asymmetric)
        in_degree = len(set(received_from))
        out_degree = len(set(sent_to))
        
        if max(in_degree, out_degree) > 0:
            asymmetry = abs(in_degree - out_degree) / max(in_degree, out_degree)
        else:
            asymmetry = 0
        
        # Score based on network metrics
        # Many counterparties could indicate hub account
        counterparty_score = min(100, n_counterparties * 5)  # 20+ = 100
        asymmetry_score = asymmetry * 50  # High asymmetry is suspicious
        
        # Check for circular patterns
        # If account receives from A and sends to B who sends back to original senders
        # This is simplified - real implementation would use graph algorithms
        
        network_score = counterparty_score * 0.7 + asymmetry_score * 0.3
        
        return network_score
    
    def calculate_behavioral_score(
        self, 
        account_id: str, 
        features_df: pd.DataFrame
    ) -> float:
        """
        Calculate risk score based on behavioral anomaly features.
        
        Uses pre-calculated anomaly scores from ML detection.
        """
        if features_df is None or features_df.empty:
            return 0.0
        
        account_features = features_df[features_df['account_id'] == account_id]
        
        if account_features.empty:
            return 0.0
        
        row = account_features.iloc[0]
        
        # Get anomaly scores if available
        if_score = row.get('if_anomaly_score', 0) * 100
        lof_score = row.get('lof_anomaly_score', 0) * 100
        combined_score = row.get('combined_anomaly_score', 0) * 100
        
        # Use combined score if available, otherwise average available scores
        if combined_score > 0:
            return combined_score
        elif if_score > 0 or lof_score > 0:
            return (if_score + lof_score) / 2
        else:
            return 0.0
    
    def calculate_risk_scores(
        self,
        transactions: pd.DataFrame,
        alerts_df: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None,
        customers_df: Optional[pd.DataFrame] = None
    ) -> Tuple[List[RiskScore], pd.DataFrame]:
        """
        Calculate comprehensive risk scores for all accounts.
        
        Args:
            transactions: Transaction DataFrame
            alerts_df: Combined alerts DataFrame
            features_df: Account features DataFrame (from anomaly detection)
            customers_df: Customer information DataFrame
            
        Returns:
            Tuple of (risk scores list, risk scores DataFrame)
        """
        print("🔍 Calculating Risk Scores...")
        
        # Get all unique accounts
        all_accounts = list(set(transactions['sender_id'].unique()) | 
                           set(transactions['receiver_id'].unique()))
        
        risk_scores = []
        
        for account_id in all_accounts:
            # Calculate component scores
            alert_score, contributing_factors, alert_count = self.calculate_alert_score(
                account_id, alerts_df
            )
            volume_score, total_amount = self.calculate_volume_score(account_id, transactions)
            velocity_score = self.calculate_velocity_score(account_id, transactions)
            network_score = self.calculate_network_score(account_id, transactions)
            behavioral_score = self.calculate_behavioral_score(account_id, features_df)
            
            # Calculate weighted overall score
            overall_score = (
                alert_score * self.weights['alert_score'] +
                volume_score * self.weights['volume_score'] +
                velocity_score * self.weights['velocity_score'] +
                network_score * self.weights['network_score'] +
                behavioral_score * self.weights['behavioral_score']
            )
            
            # Get risk category
            risk_category = self._get_risk_category(overall_score)
            
            # Create risk score object
            risk = RiskScore(
                account_id=account_id,
                overall_score=overall_score,
                risk_category=risk_category,
                alert_score=alert_score,
                volume_score=volume_score,
                velocity_score=velocity_score,
                network_score=network_score,
                behavioral_score=behavioral_score,
                contributing_factors=contributing_factors,
                alert_count=alert_count,
                total_amount=total_amount,
                details={
                    'weights': self.weights.copy()
                }
            )
            risk_scores.append(risk)
        
        self.risk_scores = risk_scores
        
        # Convert to DataFrame
        scores_df = pd.DataFrame([{
            'account_id': r.account_id,
            'overall_score': round(r.overall_score, 2),
            'risk_category': r.risk_category.value,
            'alert_score': round(r.alert_score, 2),
            'volume_score': round(r.volume_score, 2),
            'velocity_score': round(r.velocity_score, 2),
            'network_score': round(r.network_score, 2),
            'behavioral_score': round(r.behavioral_score, 2),
            'alert_count': r.alert_count,
            'total_amount': round(r.total_amount, 2),
            'contributing_factors': '; '.join(r.contributing_factors) if r.contributing_factors else ''
        } for r in risk_scores])
        
        # Sort by overall score descending
        scores_df = scores_df.sort_values('overall_score', ascending=False).reset_index(drop=True)
        
        # Print summary
        print(f"\n✅ Risk Scoring Complete: {len(scores_df)} accounts scored")
        print(f"\n   Risk Distribution:")
        for category in ['critical', 'high', 'medium', 'low']:
            count = len(scores_df[scores_df['risk_category'] == category])
            pct = count / len(scores_df) * 100
            print(f"   - {category.upper()}: {count} ({pct:.1f}%)")
        
        print(f"\n   Top 10 Highest Risk Accounts:")
        top_10 = scores_df.head(10)
        for _, row in top_10.iterrows():
            print(f"   - {row['account_id']}: {row['overall_score']:.1f} ({row['risk_category']})")
        
        return risk_scores, scores_df
    
    def get_suspicious_accounts(
        self, 
        scores_df: pd.DataFrame,
        min_risk_level: str = 'medium'
    ) -> pd.DataFrame:
        """
        Get list of suspicious accounts above minimum risk level.
        
        Args:
            scores_df: Risk scores DataFrame
            min_risk_level: Minimum risk level to include
            
        Returns:
            DataFrame of suspicious accounts
        """
        level_order = ['low', 'medium', 'high', 'critical']
        min_idx = level_order.index(min_risk_level)
        
        included_levels = level_order[min_idx:]
        
        suspicious = scores_df[scores_df['risk_category'].isin(included_levels)].copy()
        
        return suspicious


def main():
    """Test risk scoring on sample data."""
    # Load data
    transactions = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/transactions.csv')
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    
    # Load alerts (combine if multiple exist)
    alerts_dfs = []
    
    try:
        structuring = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/structuring_alerts.csv')
        alerts_dfs.append(structuring)
    except FileNotFoundError:
        pass
    
    try:
        velocity = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/velocity_alerts.csv')
        alerts_dfs.append(velocity)
    except FileNotFoundError:
        pass
    
    try:
        anomaly = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/anomaly_alerts.csv')
        alerts_dfs.append(anomaly)
    except FileNotFoundError:
        pass
    
    if alerts_dfs:
        alerts_df = pd.concat(alerts_dfs, ignore_index=True)
    else:
        alerts_df = pd.DataFrame(columns=['account_id', 'alert_type', 'risk_level'])
    
    # Load features if available
    try:
        features_df = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/account_features.csv')
    except FileNotFoundError:
        features_df = None
    
    # Calculate risk scores
    scorer = RiskScorer()
    risk_scores, scores_df = scorer.calculate_risk_scores(
        transactions, alerts_df, features_df
    )
    
    # Save results
    scores_df.to_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/risk_scores.csv', index=False)
    
    # Get suspicious accounts
    suspicious = scorer.get_suspicious_accounts(scores_df, 'medium')
    suspicious.to_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/suspicious_accounts.csv', index=False)
    
    print(f"\n📁 Results saved to data/ directory")
    
    return scores_df


if __name__ == '__main__':
    main()


