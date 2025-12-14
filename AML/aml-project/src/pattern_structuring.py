"""
Structuring / Smurfing Detection Module
=======================================
Detects patterns indicating potential structuring (smurfing) behavior:

1. Multiple deposits just under the $10,000 reporting threshold
2. Multiple senders depositing to the same receiver
3. Rapid-fire deposits within short time windows
4. Round amounts that suggest intentional splitting
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class StructuringAlert:
    """Represents a structuring/smurfing alert."""
    alert_id: str
    account_id: str
    alert_type: str
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    description: str
    transaction_ids: List[str]
    total_amount: float
    transaction_count: int
    time_window_hours: float
    timestamp: pd.Timestamp
    details: Dict = field(default_factory=dict)


class StructuringDetector:
    """
    Detects structuring patterns in transaction data.
    
    Structuring (smurfing) is the practice of breaking up large transactions
    into smaller ones to avoid triggering Currency Transaction Reports (CTRs).
    """
    
    # Detection thresholds
    REPORTING_THRESHOLD = 10000  # CTR threshold
    STRUCTURING_LOWER = 8000    # Lower bound for suspicious range
    STRUCTURING_UPPER = 9999    # Upper bound for suspicious range
    
    MIN_STRUCTURING_COUNT = 3   # Minimum transactions to flag
    TIME_WINDOW_HOURS = 48      # Time window for clustering
    
    ROUND_AMOUNT_TOLERANCE = 50  # Consider amounts within $50 of round numbers
    
    def __init__(
        self,
        reporting_threshold: float = 10000,
        structuring_lower: float = 8000,
        min_count: int = 3,
        time_window_hours: int = 48
    ):
        self.reporting_threshold = reporting_threshold
        self.structuring_lower = structuring_lower
        self.structuring_upper = reporting_threshold - 1
        self.min_count = min_count
        self.time_window_hours = time_window_hours
        self.alerts: List[StructuringAlert] = []
        self._alert_counter = 0
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"STR-{self._alert_counter:06d}"
    
    def detect_under_threshold_deposits(
        self, 
        transactions: pd.DataFrame
    ) -> List[StructuringAlert]:
        """
        Detect multiple deposits just under the reporting threshold.
        
        Red flags:
        - Multiple cash deposits between $8,000-$9,999
        - Same receiver, multiple transactions
        - Within a short time window
        """
        alerts = []
        
        # Filter for incoming deposits in suspicious range
        suspicious = transactions[
            (transactions['amount'] >= self.structuring_lower) &
            (transactions['amount'] < self.reporting_threshold) &
            (transactions['tx_type'].isin(['cash', 'check']))
        ].copy()
        
        if suspicious.empty:
            return alerts
        
        suspicious = suspicious.sort_values('timestamp')
        
        # Group by receiver (potential structurer)
        for receiver_id, group in suspicious.groupby('receiver_id'):
            if len(group) < self.min_count:
                continue
            
            # Use sliding window to find clusters
            group = group.sort_values('timestamp')
            
            for i in range(len(group)):
                window_start = group.iloc[i]['timestamp']
                window_end = window_start + timedelta(hours=self.time_window_hours)
                
                window_txs = group[
                    (group['timestamp'] >= window_start) &
                    (group['timestamp'] <= window_end)
                ]
                
                if len(window_txs) >= self.min_count:
                    total_amount = window_txs['amount'].sum()
                    
                    # Calculate risk level
                    if len(window_txs) >= 5 or total_amount > 50000:
                        risk_level = 'critical'
                    elif len(window_txs) >= 4 or total_amount > 30000:
                        risk_level = 'high'
                    elif total_amount > 20000:
                        risk_level = 'medium'
                    else:
                        risk_level = 'low'
                    
                    alert = StructuringAlert(
                        alert_id=self._generate_alert_id(),
                        account_id=receiver_id,
                        alert_type='UNDER_THRESHOLD_DEPOSITS',
                        risk_level=risk_level,
                        description=f"Multiple deposits just under ${self.reporting_threshold:,} threshold detected",
                        transaction_ids=window_txs['tx_id'].tolist(),
                        total_amount=total_amount,
                        transaction_count=len(window_txs),
                        time_window_hours=(window_txs['timestamp'].max() - window_txs['timestamp'].min()).total_seconds() / 3600,
                        timestamp=window_start,
                        details={
                            'avg_amount': window_txs['amount'].mean(),
                            'unique_senders': window_txs['sender_id'].nunique(),
                            'amounts': window_txs['amount'].tolist()
                        }
                    )
                    
                    # Avoid duplicate alerts for overlapping windows
                    if not any(a.account_id == receiver_id and 
                              set(a.transaction_ids) == set(alert.transaction_ids) 
                              for a in alerts):
                        alerts.append(alert)
        
        return alerts
    
    def detect_many_to_one(
        self, 
        transactions: pd.DataFrame
    ) -> List[StructuringAlert]:
        """
        Detect multiple senders sending to a single receiver.
        
        Red flags:
        - Many different senders → one receiver
        - Amounts just under threshold
        - Short time period
        """
        alerts = []
        
        # Time-based grouping
        transactions = transactions.copy()
        transactions['date'] = transactions['timestamp'].dt.date
        
        for (receiver_id, date), group in transactions.groupby(['receiver_id', 'date']):
            # Count unique senders
            unique_senders = group['sender_id'].nunique()
            
            # Flag if many senders in one day
            if unique_senders >= 4:
                total_amount = group['amount'].sum()
                
                # Check if amounts are in suspicious range
                suspicious_amounts = group[
                    (group['amount'] >= self.structuring_lower) &
                    (group['amount'] < self.reporting_threshold)
                ]
                
                if len(suspicious_amounts) >= 2 or (unique_senders >= 5 and total_amount > 20000):
                    if unique_senders >= 7 or total_amount > 75000:
                        risk_level = 'critical'
                    elif unique_senders >= 5 or total_amount > 50000:
                        risk_level = 'high'
                    elif total_amount > 25000:
                        risk_level = 'medium'
                    else:
                        risk_level = 'low'
                    
                    alert = StructuringAlert(
                        alert_id=self._generate_alert_id(),
                        account_id=receiver_id,
                        alert_type='MANY_TO_ONE',
                        risk_level=risk_level,
                        description=f"Multiple senders ({unique_senders}) depositing to single account",
                        transaction_ids=group['tx_id'].tolist(),
                        total_amount=total_amount,
                        transaction_count=len(group),
                        time_window_hours=24,
                        timestamp=pd.Timestamp(date),
                        details={
                            'unique_senders': unique_senders,
                            'sender_ids': group['sender_id'].unique().tolist(),
                            'suspicious_amount_count': len(suspicious_amounts)
                        }
                    )
                    alerts.append(alert)
        
        return alerts
    
    def detect_rapid_deposits(
        self, 
        transactions: pd.DataFrame,
        max_interval_minutes: int = 60
    ) -> List[StructuringAlert]:
        """
        Detect rapid-fire deposits to same account.
        
        Red flags:
        - Multiple deposits within minutes/hours
        - Same receiver
        - Potentially same sender or coordinated senders
        """
        alerts = []
        
        # Focus on deposits
        deposits = transactions[transactions['amount'] > 0].copy()
        deposits = deposits.sort_values('timestamp')
        
        for receiver_id, group in deposits.groupby('receiver_id'):
            if len(group) < 3:
                continue
            
            group = group.sort_values('timestamp')
            
            # Calculate time differences between consecutive transactions
            time_diffs = group['timestamp'].diff().dt.total_seconds() / 60  # minutes
            
            # Find clusters of rapid transactions
            rapid_mask = time_diffs <= max_interval_minutes
            
            # Group consecutive rapid transactions
            cluster_id = (~rapid_mask).cumsum()
            
            for cid, cluster in group.groupby(cluster_id):
                if len(cluster) >= 3:
                    total_time = (cluster['timestamp'].max() - cluster['timestamp'].min()).total_seconds() / 60
                    total_amount = cluster['amount'].sum()
                    
                    if total_time <= 180:  # Within 3 hours
                        if len(cluster) >= 6 or total_amount > 50000:
                            risk_level = 'critical'
                        elif len(cluster) >= 4 or total_amount > 30000:
                            risk_level = 'high'
                        elif total_amount > 15000:
                            risk_level = 'medium'
                        else:
                            risk_level = 'low'
                        
                        alert = StructuringAlert(
                            alert_id=self._generate_alert_id(),
                            account_id=receiver_id,
                            alert_type='RAPID_DEPOSITS',
                            risk_level=risk_level,
                            description=f"Rapid-fire deposits: {len(cluster)} transactions in {total_time:.0f} minutes",
                            transaction_ids=cluster['tx_id'].tolist(),
                            total_amount=total_amount,
                            transaction_count=len(cluster),
                            time_window_hours=total_time / 60,
                            timestamp=cluster['timestamp'].min(),
                            details={
                                'avg_interval_minutes': time_diffs[cluster.index].mean(),
                                'min_interval_minutes': time_diffs[cluster.index].min(),
                                'tx_types': cluster['tx_type'].value_counts().to_dict()
                            }
                        )
                        alerts.append(alert)
        
        return alerts
    
    def detect_round_amounts(
        self, 
        transactions: pd.DataFrame
    ) -> List[StructuringAlert]:
        """
        Detect patterns of round amounts suggesting intentional splitting.
        
        Red flags:
        - Multiple transactions at exactly $5000, $9000, $9500, etc.
        - Same receiver
        - Amounts that sum to just over reporting threshold
        """
        alerts = []
        
        # Common structuring amounts
        round_amounts = [5000, 7500, 8000, 8500, 9000, 9500, 9900]
        
        transactions = transactions.copy()
        transactions['is_round'] = transactions['amount'].apply(
            lambda x: any(abs(x - ra) <= self.ROUND_AMOUNT_TOLERANCE for ra in round_amounts)
        )
        
        round_txs = transactions[transactions['is_round']]
        
        for receiver_id, group in round_txs.groupby('receiver_id'):
            if len(group) >= 3:
                total_amount = group['amount'].sum()
                
                # Calculate how many multiples of threshold
                threshold_multiples = total_amount / self.reporting_threshold
                
                if threshold_multiples >= 1:
                    if len(group) >= 6 or total_amount > 50000:
                        risk_level = 'high'
                    elif len(group) >= 4:
                        risk_level = 'medium'
                    else:
                        risk_level = 'low'
                    
                    alert = StructuringAlert(
                        alert_id=self._generate_alert_id(),
                        account_id=receiver_id,
                        alert_type='ROUND_AMOUNTS',
                        risk_level=risk_level,
                        description=f"Pattern of round amounts totaling ${total_amount:,.2f}",
                        transaction_ids=group['tx_id'].tolist(),
                        total_amount=total_amount,
                        transaction_count=len(group),
                        time_window_hours=(group['timestamp'].max() - group['timestamp'].min()).total_seconds() / 3600,
                        timestamp=group['timestamp'].min(),
                        details={
                            'amounts': sorted(group['amount'].tolist()),
                            'threshold_multiples': threshold_multiples,
                            'amount_frequency': group['amount'].round(-2).value_counts().to_dict()
                        }
                    )
                    alerts.append(alert)
        
        return alerts
    
    def run_all_detectors(
        self, 
        transactions: pd.DataFrame
    ) -> Tuple[List[StructuringAlert], pd.DataFrame]:
        """
        Run all structuring detection algorithms.
        
        Args:
            transactions: Transaction DataFrame
            
        Returns:
            Tuple of (alerts list, alerts DataFrame)
        """
        print("🔍 Running Structuring Detection...")
        
        # Run all detectors
        all_alerts = []
        
        print("   ├── Checking under-threshold deposits...")
        all_alerts.extend(self.detect_under_threshold_deposits(transactions))
        
        print("   ├── Checking many-to-one patterns...")
        all_alerts.extend(self.detect_many_to_one(transactions))
        
        print("   ├── Checking rapid deposits...")
        all_alerts.extend(self.detect_rapid_deposits(transactions))
        
        print("   └── Checking round amount patterns...")
        all_alerts.extend(self.detect_round_amounts(transactions))
        
        self.alerts = all_alerts
        
        # Convert to DataFrame
        if all_alerts:
            alerts_df = pd.DataFrame([{
                'alert_id': a.alert_id,
                'account_id': a.account_id,
                'alert_type': a.alert_type,
                'risk_level': a.risk_level,
                'description': a.description,
                'total_amount': a.total_amount,
                'transaction_count': a.transaction_count,
                'time_window_hours': a.time_window_hours,
                'timestamp': a.timestamp
            } for a in all_alerts])
        else:
            alerts_df = pd.DataFrame(columns=[
                'alert_id', 'account_id', 'alert_type', 'risk_level',
                'description', 'total_amount', 'transaction_count',
                'time_window_hours', 'timestamp'
            ])
        
        print(f"\n✅ Structuring Detection Complete: {len(all_alerts)} alerts generated")
        print(f"   Risk Level Distribution:")
        if not alerts_df.empty:
            for level, count in alerts_df['risk_level'].value_counts().items():
                print(f"   - {level.upper()}: {count}")
        
        return all_alerts, alerts_df


def main():
    """Test structuring detection on sample data."""
    # Load transactions
    transactions = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/transactions.csv')
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    
    # Run detection
    detector = StructuringDetector()
    alerts, alerts_df = detector.run_all_detectors(transactions)
    
    # Save alerts
    if not alerts_df.empty:
        alerts_df.to_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/structuring_alerts.csv', index=False)
        print(f"\n📁 Alerts saved to data/structuring_alerts.csv")
    
    return alerts_df


if __name__ == '__main__':
    main()


