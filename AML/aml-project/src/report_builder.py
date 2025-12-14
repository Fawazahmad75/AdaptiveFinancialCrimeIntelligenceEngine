"""
Report Builder Module
=====================
Generates comprehensive AML reports and dashboards:

1. Executive Summary Report
2. Detailed Alert Reports
3. Account Investigation Reports
4. Statistical Summaries
5. Export to CSV/HTML formats
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os


@dataclass
class ReportSummary:
    """Summary statistics for a report."""
    total_transactions: int
    total_amount: float
    total_accounts: int
    suspicious_accounts: int
    total_alerts: int
    critical_alerts: int
    high_alerts: int
    medium_alerts: int
    low_alerts: int
    top_risks: List[Dict]
    alert_breakdown: Dict[str, int]


class AMLReportBuilder:
    """
    Generates AML investigation reports and summaries.
    
    Creates professional reports suitable for compliance
    documentation and investigator review.
    """
    
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_summary(
        self,
        transactions: pd.DataFrame,
        alerts_df: pd.DataFrame,
        risk_scores: pd.DataFrame
    ) -> ReportSummary:
        """
        Generate summary statistics for the report.
        """
        # Transaction stats
        total_transactions = len(transactions)
        total_amount = transactions['amount'].sum()
        
        # Account stats
        all_accounts = set(transactions['sender_id'].unique()) | set(transactions['receiver_id'].unique())
        total_accounts = len(all_accounts)
        
        # Alert stats
        total_alerts = len(alerts_df) if not alerts_df.empty else 0
        
        if not alerts_df.empty:
            critical_alerts = len(alerts_df[alerts_df['risk_level'] == 'critical'])
            high_alerts = len(alerts_df[alerts_df['risk_level'] == 'high'])
            medium_alerts = len(alerts_df[alerts_df['risk_level'] == 'medium'])
            low_alerts = len(alerts_df[alerts_df['risk_level'] == 'low'])
            alert_breakdown = alerts_df['alert_type'].value_counts().to_dict()
        else:
            critical_alerts = high_alerts = medium_alerts = low_alerts = 0
            alert_breakdown = {}
        
        # Risk stats
        if not risk_scores.empty:
            suspicious_accounts = len(risk_scores[risk_scores['risk_category'].isin(['critical', 'high', 'medium'])])
            top_risks = risk_scores.nlargest(10, 'overall_score')[
                ['account_id', 'overall_score', 'risk_category', 'alert_count']
            ].to_dict('records')
        else:
            suspicious_accounts = 0
            top_risks = []
        
        return ReportSummary(
            total_transactions=total_transactions,
            total_amount=total_amount,
            total_accounts=total_accounts,
            suspicious_accounts=suspicious_accounts,
            total_alerts=total_alerts,
            critical_alerts=critical_alerts,
            high_alerts=high_alerts,
            medium_alerts=medium_alerts,
            low_alerts=low_alerts,
            top_risks=top_risks,
            alert_breakdown=alert_breakdown
        )
    
    def generate_executive_summary(
        self,
        summary: ReportSummary,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate executive summary report.
        """
        if output_file is None:
            output_file = f"{self.output_dir}/executive_summary_{self.report_id}.txt"
        
        report = []
        report.append("=" * 80)
        report.append("AML SUSPICIOUS ACTIVITY DETECTION - EXECUTIVE SUMMARY")
        report.append("=" * 80)
        report.append(f"\nReport Generated: {self.report_timestamp}")
        report.append(f"Report ID: {self.report_id}")
        report.append("\n" + "-" * 80)
        
        # Overview Section
        report.append("\n1. OVERVIEW")
        report.append("-" * 40)
        report.append(f"   Total Transactions Analyzed:     {summary.total_transactions:,}")
        report.append(f"   Total Transaction Volume:        ${summary.total_amount:,.2f}")
        report.append(f"   Total Accounts:                  {summary.total_accounts:,}")
        report.append(f"   Suspicious Accounts Identified:  {summary.suspicious_accounts:,}")
        report.append(f"   Suspicion Rate:                  {summary.suspicious_accounts/summary.total_accounts*100:.1f}%")
        
        # Alert Summary
        report.append("\n\n2. ALERT SUMMARY")
        report.append("-" * 40)
        report.append(f"   Total Alerts Generated:          {summary.total_alerts:,}")
        report.append(f"   Critical Alerts:                 {summary.critical_alerts:,}")
        report.append(f"   High Priority Alerts:            {summary.high_alerts:,}")
        report.append(f"   Medium Priority Alerts:          {summary.medium_alerts:,}")
        report.append(f"   Low Priority Alerts:             {summary.low_alerts:,}")
        
        # Alert Breakdown by Type
        report.append("\n\n3. ALERTS BY DETECTION TYPE")
        report.append("-" * 40)
        for alert_type, count in sorted(summary.alert_breakdown.items(), key=lambda x: x[1], reverse=True):
            report.append(f"   {alert_type:35} {count:,}")
        
        # Top Risk Accounts
        report.append("\n\n4. TOP 10 HIGHEST RISK ACCOUNTS")
        report.append("-" * 40)
        report.append(f"   {'Account ID':<20} {'Score':>10} {'Category':>12} {'Alerts':>8}")
        report.append("   " + "-" * 50)
        
        for risk in summary.top_risks:
            report.append(f"   {risk['account_id']:<20} {risk['overall_score']:>10.1f} "
                         f"{risk['risk_category']:>12} {risk['alert_count']:>8}")
        
        # Recommendations
        report.append("\n\n5. RECOMMENDED ACTIONS")
        report.append("-" * 40)
        
        if summary.critical_alerts > 0:
            report.append(f"   ⚠️  URGENT: {summary.critical_alerts} critical alerts require immediate investigation")
        
        if summary.suspicious_accounts > summary.total_accounts * 0.1:
            report.append("   ⚠️  HIGH RISK: Unusual number of suspicious accounts detected")
        
        report.append("   • Review all critical and high-priority alerts within 24 hours")
        report.append("   • Generate SARs for accounts with scores above 75")
        report.append("   • Schedule enhanced due diligence for medium-risk accounts")
        report.append("   • Update customer risk ratings based on findings")
        
        report.append("\n" + "=" * 80)
        report.append("END OF EXECUTIVE SUMMARY")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"📄 Executive summary saved: {output_file}")
        
        return report_text
    
    def generate_html_dashboard(
        self,
        summary: ReportSummary,
        transactions: pd.DataFrame,
        alerts_df: pd.DataFrame,
        risk_scores: pd.DataFrame,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate an HTML dashboard report.
        """
        if output_file is None:
            output_file = f"{self.output_dir}/aml_dashboard_{self.report_id}.html"
        
        # Calculate additional stats for charts
        if not transactions.empty:
            daily_tx = transactions.groupby(transactions['timestamp'].dt.date).agg({
                'amount': 'sum',
                'tx_id': 'count'
            }).reset_index()
            daily_tx.columns = ['date', 'amount', 'count']
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AML Detection Dashboard - {self.report_id}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0f0f23;
            --bg-secondary: #1a1a2e;
            --bg-card: #16213e;
            --text-primary: #e8e8e8;
            --text-secondary: #a0a0a0;
            --accent-red: #e53935;
            --accent-orange: #ff7043;
            --accent-amber: #ffc107;
            --accent-green: #4caf50;
            --accent-blue: #2196f3;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-card) 100%);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #fff, var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        header p {{
            color: var(--text-secondary);
            font-size: 1.1rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: var(--bg-card);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        }}
        
        .metric-card h3 {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .metric-card .value {{
            font-size: 2.2rem;
            font-weight: 700;
        }}
        
        .metric-card.critical .value {{ color: var(--accent-red); }}
        .metric-card.high .value {{ color: var(--accent-orange); }}
        .metric-card.medium .value {{ color: var(--accent-amber); }}
        .metric-card.low .value {{ color: var(--accent-green); }}
        .metric-card.info .value {{ color: var(--accent-blue); }}
        
        .section {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        
        .section h2 {{
            font-size: 1.4rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--bg-card);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--bg-card);
        }}
        
        th {{
            background: var(--bg-card);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
        }}
        
        tr:hover {{
            background: rgba(255,255,255,0.02);
        }}
        
        .risk-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .risk-critical {{ background: rgba(229,57,53,0.2); color: var(--accent-red); }}
        .risk-high {{ background: rgba(255,112,67,0.2); color: var(--accent-orange); }}
        .risk-medium {{ background: rgba(255,193,7,0.2); color: var(--accent-amber); }}
        .risk-low {{ background: rgba(76,175,80,0.2); color: var(--accent-green); }}
        
        .chart-container {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        
        footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 AML Detection Dashboard</h1>
            <p>Suspicious Transaction Analysis Report • Generated: {self.report_timestamp}</p>
        </header>
        
        <div class="metrics-grid">
            <div class="metric-card info">
                <h3>Total Transactions</h3>
                <div class="value">{summary.total_transactions:,}</div>
            </div>
            <div class="metric-card info">
                <h3>Transaction Volume</h3>
                <div class="value">${summary.total_amount/1000000:.1f}M</div>
            </div>
            <div class="metric-card critical">
                <h3>Critical Alerts</h3>
                <div class="value">{summary.critical_alerts}</div>
            </div>
            <div class="metric-card high">
                <h3>High Alerts</h3>
                <div class="value">{summary.high_alerts}</div>
            </div>
            <div class="metric-card medium">
                <h3>Suspicious Accounts</h3>
                <div class="value">{summary.suspicious_accounts}</div>
            </div>
            <div class="metric-card low">
                <h3>Total Accounts</h3>
                <div class="value">{summary.total_accounts:,}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Alert Distribution</h2>
            <div class="grid-2">
                <div class="chart-container" id="alertPieChart"></div>
                <div class="chart-container" id="alertTypeChart"></div>
            </div>
        </div>
        
        <div class="section">
            <h2>⚠️ Top 10 Highest Risk Accounts</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Account ID</th>
                        <th>Risk Score</th>
                        <th>Category</th>
                        <th>Alert Count</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''
                    <tr>
                        <td>{i+1}</td>
                        <td><strong>{risk['account_id']}</strong></td>
                        <td>{risk['overall_score']:.1f}</td>
                        <td><span class="risk-badge risk-{risk['risk_category']}">{risk['risk_category']}</span></td>
                        <td>{risk['alert_count']}</td>
                    </tr>
                    ''' for i, risk in enumerate(summary.top_risks))}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>📈 Alert Types Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Alert Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''
                    <tr>
                        <td>{alert_type}</td>
                        <td>{count}</td>
                        <td>{count/summary.total_alerts*100:.1f}%</td>
                    </tr>
                    ''' for alert_type, count in sorted(summary.alert_breakdown.items(), key=lambda x: x[1], reverse=True))}
                </tbody>
            </table>
        </div>
        
        <footer>
            <p>AML Suspicious Transaction Detector • Report ID: {self.report_id}</p>
            <p>This report is for compliance and investigation purposes only.</p>
        </footer>
    </div>
    
    <script>
        // Risk distribution pie chart
        var alertData = [{{
            values: [{summary.critical_alerts}, {summary.high_alerts}, {summary.medium_alerts}, {summary.low_alerts}],
            labels: ['Critical', 'High', 'Medium', 'Low'],
            type: 'pie',
            marker: {{
                colors: ['#e53935', '#ff7043', '#ffc107', '#4caf50']
            }},
            textinfo: 'label+percent',
            textfont: {{ color: 'white' }},
            hole: 0.4
        }}];
        
        var alertLayout = {{
            title: {{ text: 'Alerts by Risk Level', font: {{ color: '#e8e8e8' }} }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#a0a0a0' }},
            showlegend: true,
            legend: {{ font: {{ color: '#e8e8e8' }} }}
        }};
        
        Plotly.newPlot('alertPieChart', alertData, alertLayout);
        
        // Alert types bar chart
        var alertTypes = {list(summary.alert_breakdown.keys())};
        var alertCounts = {list(summary.alert_breakdown.values())};
        
        var typeData = [{{
            x: alertCounts,
            y: alertTypes,
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: '#2196f3',
                opacity: 0.8
            }}
        }}];
        
        var typeLayout = {{
            title: {{ text: 'Alerts by Type', font: {{ color: '#e8e8e8' }} }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#a0a0a0' }},
            xaxis: {{ title: 'Count', gridcolor: 'rgba(255,255,255,0.1)' }},
            yaxis: {{ automargin: true }},
            margin: {{ l: 200 }}
        }};
        
        Plotly.newPlot('alertTypeChart', typeData, typeLayout);
    </script>
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"📊 HTML dashboard saved: {output_file}")
        
        return output_file
    
    def generate_suspicious_accounts_report(
        self,
        risk_scores: pd.DataFrame,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate detailed report of suspicious accounts.
        """
        if output_file is None:
            output_file = f"{self.output_dir}/suspicious_accounts_{self.report_id}.csv"
        
        # Filter to suspicious accounts
        suspicious = risk_scores[
            risk_scores['risk_category'].isin(['critical', 'high', 'medium'])
        ].copy()
        
        suspicious = suspicious.sort_values('overall_score', ascending=False)
        
        suspicious.to_csv(output_file, index=False)
        
        print(f"📋 Suspicious accounts report saved: {output_file}")
        print(f"   Total suspicious accounts: {len(suspicious)}")
        
        return output_file
    
    def generate_full_report(
        self,
        transactions: pd.DataFrame,
        alerts_df: pd.DataFrame,
        risk_scores: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Generate all reports.
        
        Returns dict of report paths.
        """
        print("\n📝 Generating AML Reports...")
        print("=" * 50)
        
        # Generate summary
        summary = self.generate_summary(transactions, alerts_df, risk_scores)
        
        # Generate all reports
        reports = {}
        
        reports['executive_summary'] = self.generate_executive_summary(summary)
        reports['html_dashboard'] = self.generate_html_dashboard(
            summary, transactions, alerts_df, risk_scores
        )
        reports['suspicious_accounts'] = self.generate_suspicious_accounts_report(risk_scores)
        
        print("\n✅ All reports generated successfully!")
        print(f"   Output directory: {self.output_dir}")
        
        return reports


def main():
    """Generate AML reports from analysis results."""
    # Load data
    transactions = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/transactions.csv')
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    
    # Load alerts
    alerts_dfs = []
    alert_files = ['structuring_alerts.csv', 'velocity_alerts.csv', 'anomaly_alerts.csv']
    
    for file in alert_files:
        try:
            df = pd.read_csv(f'/Users/fawazahmad/Desktop/AML/aml-project/data/{file}')
            alerts_dfs.append(df)
        except FileNotFoundError:
            pass
    
    if alerts_dfs:
        alerts_df = pd.concat(alerts_dfs, ignore_index=True)
    else:
        alerts_df = pd.DataFrame()
    
    # Load risk scores
    try:
        risk_scores = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/risk_scores.csv')
    except FileNotFoundError:
        risk_scores = pd.DataFrame()
    
    # Generate reports
    reporter = AMLReportBuilder(output_dir='/Users/fawazahmad/Desktop/AML/aml-project/reports')
    reports = reporter.generate_full_report(transactions, alerts_df, risk_scores)
    
    return reports


if __name__ == '__main__':
    main()


