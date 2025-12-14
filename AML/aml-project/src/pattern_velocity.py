"""
Velocity Rules Detection Module
===============================
Detects suspicious velocity patterns in transaction data:

1. Too many transactions in a short time window
2. Sudden spikes in activity compared to baseline
3. In/out balance patterns indicating mule accounts
4. Unusual transaction frequency changes
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class VelocityAlert:
    """Represents a velocity-based alert."""
    alert_id: str
    account_id: str
    alert_type: str
    risk_level: str
    description: str
    transaction_ids: List[str]
    metric_value: float
    threshold_value: float
    baseline_value: Optional[float]
    time_period: str
    timestamp: pd.Timestamp
    details: Dict = field(default_factory=dict)


class VelocityDetector:
    """
    Detects velocity-based suspicious patterns.
    
    Velocity rules monitor the rate and volume of transactions
    to identify abnormal account behavior.
    """
    
    # Default thresholds
    MAX_DAILY_TRANSACTIONS = 15
    MAX_HOURLY_TRANSACTIONS = 5
    MAX_DAILY_AMOUNT = 50000
    
    SPIKE_MULTIPLIER = 3.0  # Activity 3x above baseline is suspicious
    MULE_BALANCE_THRESHOLD = 0.15  # Mules keep <15% of inflow
    
    def __init__(
        self,
        max_daily_tx: int = 15,
        max_hourly_tx: int = 5,
        max_daily_amount: float = 50000,
        spike_multiplier: float = 3.0,
        mule_balance_threshold: float = 0.15
    ):
        self.max_daily_tx = max_daily_tx
        self.max_hourly_tx = max_hourly_tx
        self.max_daily_amount = max_daily_amount
        self.spike_multiplier = spike_multiplier
        self.mule_balance_threshold = mule_balance_threshold
        self.alerts: List[VelocityAlert] = []
        self._alert_counter = 0
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"VEL-{self._alert_counter:06d}"
    
    def _get_account_transactions(
        self, 
        transactions: pd.DataFrame, 
        account_id: str
    ) -> pd.DataFrame:
        """Get all transactions for an account (as sender or receiver)."""
        return transactions[
            (transactions['sender_id'] == account_id) |
            (transactions['receiver_id'] == account_id)
        ].copy()
    
    def detect_high_frequency(
        self, 
        transactions: pd.DataFrame
    ) -> List[VelocityAlert]:
        """
        Detect accounts with abnormally high transaction frequency.
        
        Checks:
        - Hourly transaction limits
        - Daily transaction limits
        """
        alerts = []
        
        transactions = transactions.copy()
        transactions['date'] = transactions['timestamp'].dt.date
        transactions['hour'] = transactions['timestamp'].dt.floor('H')
        
        # Get all unique accounts
        all_accounts = set(transactions['sender_id'].unique()) | set(transactions['receiver_id'].unique())
        
        for account_id in all_accounts:
            account_txs = self._get_account_transactions(transactions, account_id)
            
            if account_txs.empty:
                continue
            
            # Check daily frequency
            daily_counts = account_txs.groupby('date').size()
            high_days = daily_counts[daily_counts > self.max_daily_tx]
            
            for date, count in high_days.items():
                day_txs = account_txs[account_txs['date'] == date]
                
                if count > self.max_daily_tx * 2:
                    risk_level = 'critical'
                elif count > self.max_daily_tx * 1.5:
                    risk_level = 'high'
                else:
                    risk_level = 'medium'
                
                alert = VelocityAlert(
                    alert_id=self._generate_alert_id(),
                    account_id=account_id,
                    alert_type='HIGH_DAILY_FREQUENCY',
                    risk_level=risk_level,
                    description=f"High daily transaction volume: {count} transactions (limit: {self.max_daily_tx})",
                    transaction_ids=day_txs['tx_id'].tolist(),
                    metric_value=float(count),
                    threshold_value=float(self.max_daily_tx),
                    baseline_value=daily_counts.mean(),
                    time_period='daily',
                    timestamp=pd.Timestamp(date),
                    details={
                        'total_amount': day_txs['amount'].sum(),
                        'tx_types': day_txs['tx_type'].value_counts().to_dict()
                    }
                )
                alerts.append(alert)
            
            # Check hourly frequency
            hourly_counts = account_txs.groupby('hour').size()
            high_hours = hourly_counts[hourly_counts > self.max_hourly_tx]
            
            for hour, count in high_hours.items():
                hour_txs = account_txs[account_txs['hour'] == hour]
                
                if count > self.max_hourly_tx * 3:
                    risk_level = 'critical'
                elif count > self.max_hourly_tx * 2:
                    risk_level = 'high'
                else:
                    risk_level = 'medium'
                
                alert = VelocityAlert(
                    alert_id=self._generate_alert_id(),
                    account_id=account_id,
                    alert_type='HIGH_HOURLY_FREQUENCY',
                    risk_level=risk_level,
                    description=f"High hourly transaction volume: {count} transactions in one hour",
                    transaction_ids=hour_txs['tx_id'].tolist(),
                    metric_value=float(count),
                    threshold_value=float(self.max_hourly_tx),
                    baseline_value=hourly_counts.mean(),
                    time_period='hourly',
                    timestamp=hour,
                    details={
                        'total_amount': hour_txs['amount'].sum(),
                        'avg_interval_minutes': hour_txs['timestamp'].diff().mean().total_seconds() / 60 if len(hour_txs) > 1 else 0
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def detect_activity_spikes(
        self, 
        transactions: pd.DataFrame,
        baseline_days: int = 30
    ) -> List[VelocityAlert]:
        """
        Detect sudden spikes in account activity.
        
        Compares recent activity to historical baseline to identify
        unusual increases in transaction volume or amounts.
        """
        alerts = []
        
        transactions = transactions.copy()
        transactions['date'] = transactions['timestamp'].dt.date
        
        all_accounts = set(transactions['sender_id'].unique()) | set(transactions['receiver_id'].unique())
        
        for account_id in all_accounts:
            account_txs = self._get_account_transactions(transactions, account_id)
            
            if len(account_txs) < 10:  # Need enough history
                continue
            
            # Calculate daily metrics
            daily_stats = account_txs.groupby('date').agg({
                'tx_id': 'count',
                'amount': 'sum'
            }).rename(columns={'tx_id': 'count', 'amount': 'total_amount'})
            
            if len(daily_stats) < baseline_days:
                continue
            
            # Calculate rolling baseline
            daily_stats['count_baseline'] = daily_stats['count'].rolling(
                window=baseline_days, min_periods=7
            ).mean().shift(1)
            
            daily_stats['amount_baseline'] = daily_stats['total_amount'].rolling(
                window=baseline_days, min_periods=7
            ).mean().shift(1)
            
            # Find spikes
            daily_stats['count_ratio'] = daily_stats['count'] / daily_stats['count_baseline'].clip(lower=1)
            daily_stats['amount_ratio'] = daily_stats['total_amount'] / daily_stats['amount_baseline'].clip(lower=1)
            
            # Alert on significant spikes
            spike_days = daily_stats[
                (daily_stats['count_ratio'] > self.spike_multiplier) |
                (daily_stats['amount_ratio'] > self.spike_multiplier)
            ]
            
            for date, row in spike_days.iterrows():
                day_txs = account_txs[account_txs['date'] == date]
                
                max_ratio = max(row['count_ratio'], row['amount_ratio'])
                
                if max_ratio > self.spike_multiplier * 2:
                    risk_level = 'critical'
                elif max_ratio > self.spike_multiplier * 1.5:
                    risk_level = 'high'
                else:
                    risk_level = 'medium'
                
                spike_type = 'volume' if row['count_ratio'] >= row['amount_ratio'] else 'amount'
                
                alert = VelocityAlert(
                    alert_id=self._generate_alert_id(),
                    account_id=account_id,
                    alert_type='ACTIVITY_SPIKE',
                    risk_level=risk_level,
                    description=f"Activity spike detected: {max_ratio:.1f}x baseline ({spike_type})",
                    transaction_ids=day_txs['tx_id'].tolist(),
                    metric_value=row['count'] if spike_type == 'volume' else row['total_amount'],
                    threshold_value=row['count_baseline'] * self.spike_multiplier if spike_type == 'volume' else row['amount_baseline'] * self.spike_multiplier,
                    baseline_value=row['count_baseline'] if spike_type == 'volume' else row['amount_baseline'],
                    time_period='daily',
                    timestamp=pd.Timestamp(date),
                    details={
                        'count_ratio': row['count_ratio'],
                        'amount_ratio': row['amount_ratio'],
                        'count': row['count'],
                        'total_amount': row['total_amount'],
                        'count_baseline': row['count_baseline'],
                        'amount_baseline': row['amount_baseline']
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def detect_mule_patterns(
        self, 
        transactions: pd.DataFrame,
        analysis_window_days: int = 30
    ) -> List[VelocityAlert]:
        """
        Detect money mule account patterns.
        
        Money mules typically:
        - Receive money and quickly send it out
        - Keep very little balance
        - Have high in/out velocity
        - Often use wire transfers or crypto for outflows
        """
        alerts = []
        
        all_accounts = set(transactions['sender_id'].unique()) | set(transactions['receiver_id'].unique())
        
        for account_id in all_accounts:
            # Calculate inflows and outflows
            inflows = transactions[transactions['receiver_id'] == account_id]['amount'].sum()
            outflows = transactions[transactions['sender_id'] == account_id]['amount'].sum()
            
            if inflows < 5000:  # Not enough activity to analyze
                continue
            
            # Calculate retention ratio
            if inflows > 0:
                retention_ratio = max(0, (inflows - outflows)) / inflows
            else:
                continue
            
            # Mule pattern: very low retention
            if retention_ratio < self.mule_balance_threshold and outflows > 0:
                # Additional checks
                account_txs = self._get_account_transactions(transactions, account_id)
                
                # Check for rapid turnaround
                account_txs_sorted = account_txs.sort_values('timestamp')
                
                # Get inflow and outflow transactions
                in_txs = transactions[transactions['receiver_id'] == account_id].sort_values('timestamp')
                out_txs = transactions[transactions['sender_id'] == account_id].sort_values('timestamp')
                
                # Calculate time between receipts and sends
                turnaround_times = []
                for _, out_tx in out_txs.iterrows():
                    recent_ins = in_txs[in_txs['timestamp'] < out_tx['timestamp']]
                    if not recent_ins.empty:
                        last_in = recent_ins.iloc[-1]
                        turnaround = (out_tx['timestamp'] - last_in['timestamp']).total_seconds() / 3600
                        turnaround_times.append(turnaround)
                
                avg_turnaround = np.mean(turnaround_times) if turnaround_times else 999
                
                # High-risk outflow channels
                risky_outflows = out_txs[out_txs['tx_type'].isin(['wire', 'crypto'])]
                risky_ratio = len(risky_outflows) / len(out_txs) if len(out_txs) > 0 else 0
                
                # Determine risk level
                if retention_ratio < 0.05 and avg_turnaround < 24:
                    risk_level = 'critical'
                elif retention_ratio < 0.10 or avg_turnaround < 48:
                    risk_level = 'high'
                elif risky_ratio > 0.5:
                    risk_level = 'high'
                else:
                    risk_level = 'medium'
                
                alert = VelocityAlert(
                    alert_id=self._generate_alert_id(),
                    account_id=account_id,
                    alert_type='MULE_PATTERN',
                    risk_level=risk_level,
                    description=f"Money mule pattern: {retention_ratio*100:.1f}% retention, {avg_turnaround:.1f}h avg turnaround",
                    transaction_ids=account_txs['tx_id'].tolist(),
                    metric_value=retention_ratio,
                    threshold_value=self.mule_balance_threshold,
                    baseline_value=None,
                    time_period=f'{analysis_window_days}_days',
                    timestamp=account_txs['timestamp'].max(),
                    details={
                        'total_inflow': inflows,
                        'total_outflow': outflows,
                        'retention_ratio': retention_ratio,
                        'avg_turnaround_hours': avg_turnaround,
                        'risky_outflow_ratio': risky_ratio,
                        'outflow_types': out_txs['tx_type'].value_counts().to_dict()
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def detect_high_daily_amount(
        self, 
        transactions: pd.DataFrame
    ) -> List[VelocityAlert]:
        """
        Detect accounts exceeding daily transaction amount thresholds.
        """
        alerts = []
        
        transactions = transactions.copy()
        transactions['date'] = transactions['timestamp'].dt.date
        
        all_accounts = set(transactions['sender_id'].unique()) | set(transactions['receiver_id'].unique())
        
        for account_id in all_accounts:
            account_txs = self._get_account_transactions(transactions, account_id)
            
            if account_txs.empty:
                continue
            
            # Daily amount totals
            daily_amounts = account_txs.groupby('date')['amount'].sum()
            high_days = daily_amounts[daily_amounts > self.max_daily_amount]
            
            for date, amount in high_days.items():
                day_txs = account_txs[account_txs['date'] == date]
                
                if amount > self.max_daily_amount * 3:
                    risk_level = 'critical'
                elif amount > self.max_daily_amount * 2:
                    risk_level = 'high'
                else:
                    risk_level = 'medium'
                
                alert = VelocityAlert(
                    alert_id=self._generate_alert_id(),
                    account_id=account_id,
                    alert_type='HIGH_DAILY_AMOUNT',
                    risk_level=risk_level,
                    description=f"High daily amount: ${amount:,.2f} (threshold: ${self.max_daily_amount:,.2f})",
                    transaction_ids=day_txs['tx_id'].tolist(),
                    metric_value=amount,
                    threshold_value=float(self.max_daily_amount),
                    baseline_value=daily_amounts.mean(),
                    time_period='daily',
                    timestamp=pd.Timestamp(date),
                    details={
                        'transaction_count': len(day_txs),
                        'avg_amount': day_txs['amount'].mean(),
                        'max_amount': day_txs['amount'].max()
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def run_all_detectors(
        self, 
        transactions: pd.DataFrame
    ) -> Tuple[List[VelocityAlert], pd.DataFrame]:
        """
        Run all velocity detection algorithms.
        
        Args:
            transactions: Transaction DataFrame
            
        Returns:
            Tuple of (alerts list, alerts DataFrame)
        """
        print("🔍 Running Velocity Detection...")
        
        all_alerts = []
        
        print("   ├── Checking high frequency patterns...")
        all_alerts.extend(self.detect_high_frequency(transactions))
        
        print("   ├── Checking activity spikes...")
        all_alerts.extend(self.detect_activity_spikes(transactions))
        
        print("   ├── Checking mule patterns...")
        all_alerts.extend(self.detect_mule_patterns(transactions))
        
        print("   └── Checking high daily amounts...")
        all_alerts.extend(self.detect_high_daily_amount(transactions))
        
        self.alerts = all_alerts
        
        # Convert to DataFrame
        if all_alerts:
            alerts_df = pd.DataFrame([{
                'alert_id': a.alert_id,
                'account_id': a.account_id,
                'alert_type': a.alert_type,
                'risk_level': a.risk_level,
                'description': a.description,
                'metric_value': a.metric_value,
                'threshold_value': a.threshold_value,
                'baseline_value': a.baseline_value,
                'time_period': a.time_period,
                'timestamp': a.timestamp
            } for a in all_alerts])
        else:
            alerts_df = pd.DataFrame(columns=[
                'alert_id', 'account_id', 'alert_type', 'risk_level',
                'description', 'metric_value', 'threshold_value',
                'baseline_value', 'time_period', 'timestamp'
            ])
        
        print(f"\n✅ Velocity Detection Complete: {len(all_alerts)} alerts generated")
        print(f"   Alert Type Distribution:")
        if not alerts_df.empty:
            for alert_type, count in alerts_df['alert_type'].value_counts().items():
                print(f"   - {alert_type}: {count}")
        
        return all_alerts, alerts_df


def main():
    """Test velocity detection on sample data."""
    # Load transactions
    transactions = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/transactions.csv')
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    
    # Run detection
    detector = VelocityDetector()
    alerts, alerts_df = detector.run_all_detectors(transactions)
    
    # Save alerts
    if not alerts_df.empty:
        alerts_df.to_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/velocity_alerts.csv', index=False)
        print(f"\n📁 Alerts saved to data/velocity_alerts.csv")
    
    return alerts_df


if __name__ == '__main__':
    main()


