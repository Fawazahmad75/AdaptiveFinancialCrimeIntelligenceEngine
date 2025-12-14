"""
AML Data Generator
==================
Generates realistic synthetic transaction and customer data for AML testing.

Includes simulation of:
- Normal customers with regular transaction patterns
- High-risk customers with elevated activity
- Money mules (high in/out volume, short lifespan)
- Fraud rings (interconnected suspicious accounts)
- Structuring/smurfing patterns (deposits just under $10k)
- Layering transactions (rapid movement through multiple accounts)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
from typing import Tuple, List, Dict
import os


class AMLDataGenerator:
    """
    Generates synthetic AML transaction and customer datasets.
    
    Attributes:
        n_customers: Number of customers to generate
        n_transactions: Number of transactions to generate
        start_date: Start date for transaction timeline
        end_date: End date for transaction timeline
        seed: Random seed for reproducibility
    """
    
    # Configuration constants
    REGIONS = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East', 'Africa']
    HIGH_RISK_REGIONS = ['Middle East', 'Latin America', 'Africa']
    
    BUSINESS_TYPES = [
        'Individual', 'Retail', 'Restaurant', 'Real Estate', 
        'Import/Export', 'Cryptocurrency', 'Money Service Business',
        'Casino', 'Jewelry', 'Art Dealer', 'Law Firm', 'Consulting'
    ]
    HIGH_RISK_BUSINESS = ['Money Service Business', 'Casino', 'Cryptocurrency', 'Import/Export', 'Jewelry']
    
    TX_TYPES = ['cash', 'wire', 'e-transfer', 'ACH', 'check', 'crypto']
    CHANNELS = ['branch', 'online', 'mobile', 'ATM', 'third-party']
    
    COUNTRIES = [
        'USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia',
        'Brazil', 'Mexico', 'UAE', 'Singapore', 'Switzerland', 'Cayman Islands',
        'Panama', 'Russia', 'China', 'India', 'Nigeria', 'South Africa'
    ]
    HIGH_RISK_COUNTRIES = ['Cayman Islands', 'Panama', 'Russia', 'Nigeria']
    
    def __init__(
        self,
        n_customers: int = 1000,
        n_transactions: int = 25000,
        start_date: str = '2024-01-01',
        end_date: str = '2024-12-31',
        seed: int = 42
    ):
        self.n_customers = n_customers
        self.n_transactions = n_transactions
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.seed = seed
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Track special account types for transaction generation
        self.mule_accounts: List[str] = []
        self.fraud_ring_accounts: List[str] = []
        self.structuring_accounts: List[str] = []
        self.normal_accounts: List[str] = []
        
    def _generate_account_id(self, prefix: str = 'ACC') -> str:
        """Generate a unique account ID."""
        return f"{prefix}{random.randint(100000, 999999)}"
    
    def _generate_tx_id(self) -> str:
        """Generate a unique transaction ID."""
        chars = string.ascii_uppercase + string.digits
        return 'TX' + ''.join(random.choices(chars, k=10))
    
    def generate_customers(self) -> pd.DataFrame:
        """
        Generate customer dataset with various risk profiles.
        
        Returns:
            DataFrame with customer information
        """
        customers = []
        
        # Distribution of customer types
        n_normal = int(self.n_customers * 0.70)       # 70% normal
        n_high_risk = int(self.n_customers * 0.15)    # 15% high risk
        n_mules = int(self.n_customers * 0.08)        # 8% money mules
        n_fraud_ring = int(self.n_customers * 0.05)   # 5% fraud ring
        n_structuring = self.n_customers - n_normal - n_high_risk - n_mules - n_fraud_ring  # remainder
        
        # Generate normal customers
        for _ in range(n_normal):
            account_id = self._generate_account_id('NRM')
            self.normal_accounts.append(account_id)
            customers.append({
                'account_id': account_id,
                'customer_type': 'normal',
                'customer_age': np.random.randint(22, 75),
                'account_age_days': np.random.randint(365, 3650),  # 1-10 years
                'risk_category': 'low',
                'region': np.random.choice(self.REGIONS, p=[0.35, 0.25, 0.20, 0.10, 0.05, 0.05]),
                'business_type': np.random.choice(['Individual', 'Retail', 'Restaurant', 'Consulting'], 
                                                   p=[0.6, 0.2, 0.1, 0.1]),
                'avg_monthly_transactions': np.random.randint(5, 30),
                'avg_transaction_amount': np.random.uniform(100, 5000)
            })
        
        # Generate high-risk customers (legitimate but elevated risk)
        for _ in range(n_high_risk):
            account_id = self._generate_account_id('HRK')
            self.normal_accounts.append(account_id)  # Still legitimate
            customers.append({
                'account_id': account_id,
                'customer_type': 'high_risk',
                'customer_age': np.random.randint(25, 65),
                'account_age_days': np.random.randint(180, 2000),
                'risk_category': 'high',
                'region': np.random.choice(self.HIGH_RISK_REGIONS),
                'business_type': np.random.choice(self.HIGH_RISK_BUSINESS),
                'avg_monthly_transactions': np.random.randint(20, 100),
                'avg_transaction_amount': np.random.uniform(5000, 50000)
            })
        
        # Generate money mule accounts
        for _ in range(n_mules):
            account_id = self._generate_account_id('MUL')
            self.mule_accounts.append(account_id)
            customers.append({
                'account_id': account_id,
                'customer_type': 'mule',
                'customer_age': np.random.randint(18, 35),  # Often younger
                'account_age_days': np.random.randint(30, 180),  # New accounts
                'risk_category': 'medium',  # Not flagged initially
                'region': np.random.choice(self.REGIONS),
                'business_type': 'Individual',
                'avg_monthly_transactions': np.random.randint(50, 200),  # High volume
                'avg_transaction_amount': np.random.uniform(2000, 15000)
            })
        
        # Generate fraud ring accounts (interconnected)
        ring_base = self._generate_account_id('RNG')[:6]
        for i in range(n_fraud_ring):
            account_id = f"{ring_base}{i:03d}"
            self.fraud_ring_accounts.append(account_id)
            customers.append({
                'account_id': account_id,
                'customer_type': 'fraud_ring',
                'customer_age': np.random.randint(25, 50),
                'account_age_days': np.random.randint(60, 365),
                'risk_category': 'medium',
                'region': np.random.choice(['North America', 'Europe']),  # Same region often
                'business_type': np.random.choice(['Consulting', 'Import/Export', 'Real Estate']),
                'avg_monthly_transactions': np.random.randint(30, 80),
                'avg_transaction_amount': np.random.uniform(10000, 100000)
            })
        
        # Generate structuring accounts
        for _ in range(n_structuring):
            account_id = self._generate_account_id('STR')
            self.structuring_accounts.append(account_id)
            customers.append({
                'account_id': account_id,
                'customer_type': 'structuring',
                'customer_age': np.random.randint(30, 60),
                'account_age_days': np.random.randint(180, 1500),
                'risk_category': 'low',  # Tries to appear normal
                'region': np.random.choice(['North America', 'Europe']),
                'business_type': np.random.choice(['Individual', 'Retail', 'Restaurant']),
                'avg_monthly_transactions': np.random.randint(15, 50),
                'avg_transaction_amount': np.random.uniform(5000, 9500)  # Just under threshold
            })
        
        return pd.DataFrame(customers)
    
    def _generate_timestamp(self, cluster: bool = False, base_time: datetime = None) -> datetime:
        """Generate a random timestamp, optionally clustered around a base time."""
        if cluster and base_time:
            # Cluster within 1-4 hours
            delta = timedelta(minutes=np.random.randint(5, 240))
            return base_time + delta
        else:
            total_days = (self.end_date - self.start_date).days
            random_days = np.random.randint(0, total_days)
            random_seconds = np.random.randint(0, 86400)
            return self.start_date + timedelta(days=random_days, seconds=random_seconds)
    
    def generate_transactions(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate transaction dataset with various patterns.
        
        Args:
            customers_df: Customer DataFrame from generate_customers()
            
        Returns:
            DataFrame with transaction data
        """
        transactions = []
        all_accounts = customers_df['account_id'].tolist()
        
        # Calculate transaction distribution
        n_normal_tx = int(self.n_transactions * 0.60)
        n_structuring_tx = int(self.n_transactions * 0.12)
        n_mule_tx = int(self.n_transactions * 0.12)
        n_ring_tx = int(self.n_transactions * 0.10)
        n_layering_tx = self.n_transactions - n_normal_tx - n_structuring_tx - n_mule_tx - n_ring_tx
        
        # 1. Generate normal transactions
        for _ in range(n_normal_tx):
            sender = np.random.choice(self.normal_accounts) if self.normal_accounts else np.random.choice(all_accounts)
            receiver = np.random.choice(all_accounts)
            while receiver == sender:
                receiver = np.random.choice(all_accounts)
            
            transactions.append({
                'tx_id': self._generate_tx_id(),
                'timestamp': self._generate_timestamp(),
                'sender_id': sender,
                'receiver_id': receiver,
                'amount': round(np.random.lognormal(7, 1.5), 2),  # Log-normal distribution
                'tx_type': np.random.choice(self.TX_TYPES, p=[0.15, 0.25, 0.30, 0.20, 0.08, 0.02]),
                'country': np.random.choice(self.COUNTRIES, p=self._get_country_weights()),
                'channel': np.random.choice(self.CHANNELS, p=[0.15, 0.35, 0.30, 0.15, 0.05]),
                'pattern_type': 'normal'
            })
        
        # 2. Generate structuring transactions (just under $10k)
        for account in self.structuring_accounts:
            n_tx_per_account = n_structuring_tx // max(len(self.structuring_accounts), 1)
            base_time = self._generate_timestamp()
            
            for i in range(n_tx_per_account):
                # Multiple deposits just under $10k
                amount = round(np.random.uniform(8500, 9900), 2)
                
                transactions.append({
                    'tx_id': self._generate_tx_id(),
                    'timestamp': self._generate_timestamp(cluster=True, base_time=base_time) if i > 0 else base_time,
                    'sender_id': np.random.choice([a for a in all_accounts if a != account]),
                    'receiver_id': account,
                    'amount': amount,
                    'tx_type': 'cash',  # Structuring often involves cash
                    'country': 'USA',
                    'channel': np.random.choice(['branch', 'ATM']),
                    'pattern_type': 'structuring'
                })
        
        # 3. Generate mule transactions (rapid in/out)
        for account in self.mule_accounts:
            n_tx_per_account = n_mule_tx // max(len(self.mule_accounts), 1)
            
            for _ in range(n_tx_per_account // 2):
                base_time = self._generate_timestamp()
                amount = round(np.random.uniform(3000, 20000), 2)
                
                # Money in
                transactions.append({
                    'tx_id': self._generate_tx_id(),
                    'timestamp': base_time,
                    'sender_id': np.random.choice([a for a in all_accounts if a != account]),
                    'receiver_id': account,
                    'amount': amount,
                    'tx_type': np.random.choice(['wire', 'e-transfer']),
                    'country': np.random.choice(self.COUNTRIES),
                    'channel': 'online',
                    'pattern_type': 'mule_in'
                })
                
                # Quick money out (within hours)
                out_time = base_time + timedelta(hours=np.random.randint(1, 24))
                transactions.append({
                    'tx_id': self._generate_tx_id(),
                    'timestamp': out_time,
                    'sender_id': account,
                    'receiver_id': np.random.choice([a for a in all_accounts if a != account]),
                    'amount': round(amount * np.random.uniform(0.85, 0.98), 2),  # Slight reduction
                    'tx_type': np.random.choice(['wire', 'crypto']),
                    'country': np.random.choice(self.HIGH_RISK_COUNTRIES + ['USA']),
                    'channel': 'online',
                    'pattern_type': 'mule_out'
                })
        
        # 4. Generate fraud ring transactions (circular patterns)
        if len(self.fraud_ring_accounts) > 2:
            for _ in range(n_ring_tx):
                # Pick sequential accounts in the ring
                idx = np.random.randint(0, len(self.fraud_ring_accounts))
                sender = self.fraud_ring_accounts[idx]
                receiver = self.fraud_ring_accounts[(idx + 1) % len(self.fraud_ring_accounts)]
                
                transactions.append({
                    'tx_id': self._generate_tx_id(),
                    'timestamp': self._generate_timestamp(),
                    'sender_id': sender,
                    'receiver_id': receiver,
                    'amount': round(np.random.uniform(15000, 150000), 2),
                    'tx_type': np.random.choice(['wire', 'ACH']),
                    'country': np.random.choice(['USA', 'Canada', 'UK', 'Cayman Islands']),
                    'channel': 'online',
                    'pattern_type': 'fraud_ring'
                })
        
        # 5. Generate layering transactions (rapid movement through accounts)
        for _ in range(n_layering_tx):
            # Pick 3-5 accounts for layering chain
            chain_length = np.random.randint(3, 6)
            chain = random.sample(all_accounts, min(chain_length, len(all_accounts)))
            base_time = self._generate_timestamp()
            base_amount = round(np.random.uniform(25000, 200000), 2)
            
            for i in range(len(chain) - 1):
                transactions.append({
                    'tx_id': self._generate_tx_id(),
                    'timestamp': base_time + timedelta(hours=i * np.random.randint(1, 6)),
                    'sender_id': chain[i],
                    'receiver_id': chain[i + 1],
                    'amount': round(base_amount * (1 - 0.02 * i), 2),  # Small decrease each hop
                    'tx_type': 'wire',
                    'country': np.random.choice(self.COUNTRIES),
                    'channel': 'online',
                    'pattern_type': 'layering'
                })
        
        # Create DataFrame and sort by timestamp
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def _get_country_weights(self) -> List[float]:
        """Generate probability weights for countries."""
        n_countries = len(self.COUNTRIES)
        weights = np.ones(n_countries)
        # Higher weight for common countries
        for i, country in enumerate(self.COUNTRIES):
            if country in ['USA', 'Canada', 'UK', 'Germany']:
                weights[i] = 5
            elif country in self.HIGH_RISK_COUNTRIES:
                weights[i] = 0.5
        return (weights / weights.sum()).tolist()
    
    def generate_and_save(self, output_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate both datasets and save to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
            
        Returns:
            Tuple of (customers_df, transactions_df)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("🏦 Generating customer data...")
        customers_df = self.generate_customers()
        
        print("💸 Generating transaction data...")
        transactions_df = self.generate_transactions(customers_df)
        
        # Save to CSV
        customers_path = os.path.join(output_dir, 'customers.csv')
        transactions_path = os.path.join(output_dir, 'transactions.csv')
        
        customers_df.to_csv(customers_path, index=False)
        transactions_df.to_csv(transactions_path, index=False)
        
        print(f"\n✅ Data generation complete!")
        print(f"   📊 Customers: {len(customers_df):,} records → {customers_path}")
        print(f"   📊 Transactions: {len(transactions_df):,} records → {transactions_path}")
        
        # Print summary statistics
        print(f"\n📈 Customer Distribution:")
        print(customers_df['customer_type'].value_counts().to_string())
        
        print(f"\n📈 Transaction Pattern Distribution:")
        print(transactions_df['pattern_type'].value_counts().to_string())
        
        return customers_df, transactions_df


def main():
    """Generate AML datasets."""
    generator = AMLDataGenerator(
        n_customers=1000,
        n_transactions=25000,
        seed=42
    )
    
    customers, transactions = generator.generate_and_save(
        output_dir='/Users/fawazahmad/Desktop/AML/aml-project/data'
    )
    
    return customers, transactions


if __name__ == '__main__':
    main()


