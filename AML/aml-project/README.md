# 🔍 AML Suspicious Transaction Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Network](https://img.shields.io/badge/Network-NetworkX-red.svg)

**A comprehensive Anti-Money Laundering (AML) detection system using rule-based pattern detection and machine learning.**

[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Architecture](#-architecture) •
[Documentation](#-documentation)

</div>

---

## 🎯 Overview

This project implements a **production-grade AML detection pipeline** that mirrors real-world financial crime detection systems like Verafin, Actimize, and SAS AML. It demonstrates:

- **Rule-based detection** for known suspicious patterns
- **Machine learning** for anomaly detection
- **Network analysis** to identify fraud rings
- **Risk scoring** for investigator prioritization
- **Interactive dashboards** for data exploration

Perfect for **portfolio demonstrations** or as a **learning resource** for AML/compliance technology.

---

## 📸 Screenshots

### Network Visualization
*Transaction network graph with risk-colored nodes and suspicious flow highlights*

![Network Graph](data/network_graph.png)

### Dashboard
*Interactive Jupyter dashboard with real-time metrics*

---

## ✨ Features

### 🔴 Pattern Detection Modules

| Module | Description | Red Flags Detected |
|--------|-------------|-------------------|
| **Structuring Detector** | Identifies smurfing patterns | Deposits under $10k, many-to-one flows, rapid deposits |
| **Velocity Rules** | Monitors transaction speed | High frequency, activity spikes, mule patterns |
| **Anomaly Detection** | ML-based outlier detection | Behavioral deviations, unusual metrics |
| **Risk Scoring** | Composite risk calculation | Multi-factor risk assessment |

### 🕸️ Network Graph Analysis

- **Transaction flow visualization** with NetworkX
- **Community detection** (Louvain algorithm)
- **PageRank centrality** for hub identification
- **Suspicious path detection** for layering schemes
- **GEXF export** for Gephi visualization

### 📊 Reporting & Dashboards

- **Executive summary** reports
- **Interactive HTML dashboard**
- **Suspicious accounts list**
- **Jupyter notebook** with full analysis pipeline

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/aml-project.git
cd aml-project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### Option 1: Run the Jupyter Dashboard

```bash
cd notebooks
jupyter notebook aml_dashboard.ipynb
```

Run all cells to execute the complete AML analysis pipeline.

### Option 2: Run Individual Modules

```bash
# Generate synthetic data
python src/data_generator.py

# Run detection modules
python src/pattern_structuring.py
python src/pattern_velocity.py
python src/anomaly_detection.py

# Calculate risk scores
python src/risk_scoring.py

# Build network graph
python src/graph_builder.py

# Generate reports
python src/report_builder.py
```

---

## 🏗️ Architecture

```
aml-project/
│
├── data/                          # Generated data and outputs
│   ├── customers.csv              # Customer information
│   ├── transactions.csv           # Transaction records
│   ├── risk_scores.csv            # Account risk scores
│   ├── suspicious_accounts.csv    # Flagged accounts
│   └── network_graph.png          # Network visualization
│
├── src/                           # Source code modules
│   ├── data_generator.py          # Synthetic data generation
│   ├── pattern_structuring.py     # Structuring/smurfing detection
│   ├── pattern_velocity.py        # Velocity rules engine
│   ├── anomaly_detection.py       # ML anomaly detection
│   ├── risk_scoring.py            # Composite risk scoring
│   ├── graph_builder.py           # NetworkX graph analysis
│   └── report_builder.py          # Report generation
│
├── notebooks/                     # Jupyter notebooks
│   └── aml_dashboard.ipynb        # Interactive dashboard
│
├── reports/                       # Generated reports
│   ├── executive_summary_*.txt    # Text summaries
│   └── aml_dashboard_*.html       # HTML dashboards
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 📖 Documentation

### Data Generation

The `AMLDataGenerator` creates realistic synthetic data with various customer profiles:

| Profile | Percentage | Characteristics |
|---------|------------|-----------------|
| Normal | 70% | Regular transaction patterns |
| High-Risk | 15% | Elevated activity, risky regions |
| Money Mules | 8% | High in/out velocity, low retention |
| Fraud Rings | 5% | Interconnected accounts |
| Structuring | 2% | Deposits under $10k threshold |

### Detection Algorithms

#### 1. Structuring Detection
- Identifies deposits between $8,000-$9,999
- Detects many-to-one deposit patterns
- Flags rapid consecutive deposits
- Analyzes round amount patterns

#### 2. Velocity Analysis
- Monitors hourly/daily transaction limits
- Detects activity spikes vs. baseline
- Identifies money mule behavior (low retention)
- Tracks high-value transaction thresholds

#### 3. ML Anomaly Detection
- **Isolation Forest**: Tree-based outlier detection
- **Local Outlier Factor**: Density-based anomalies
- **Statistical Z-Score**: Multi-feature outliers

### Risk Scoring Formula

```
Risk Score = (Alert Score × 0.35) + 
             (Volume Score × 0.20) + 
             (Velocity Score × 0.20) + 
             (Network Score × 0.15) + 
             (Behavioral Score × 0.10)
```

Risk Categories:
- 🔴 **Critical**: Score ≥ 75
- 🟠 **High**: Score ≥ 50
- 🟡 **Medium**: Score ≥ 25
- 🟢 **Low**: Score < 25

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (IsolationForest, LOF) |
| **Graph Analysis** | NetworkX |
| **Visualization** | Matplotlib, Plotly, Seaborn |
| **Notebooks** | Jupyter |
| **Optional** | Streamlit (web dashboard) |

---

## 📈 Sample Output

### Alert Summary
```
📊 Transaction Analysis:
   • Transactions analyzed: 25,000
   • Total volume: $45,000,000+
   • Accounts monitored: 1,000

🚨 Alert Summary:
   • Total alerts: 500+
   • Critical: 45
   • High: 120
   • Medium: 200
   • Low: 135

⚠️ Suspicious Accounts: 180
   Suspicion Rate: 18%
```

### Top Risk Accounts
| Account | Score | Category | Alerts |
|---------|-------|----------|--------|
| MUL456789 | 92.4 | Critical | 12 |
| STR234567 | 87.2 | Critical | 9 |
| RNG000001 | 81.5 | Critical | 8 |

---

## 🔮 Future Enhancements

- [ ] **Streamlit web dashboard** for real-time monitoring
- [ ] **Deep learning models** (Autoencoders, LSTM)
- [ ] **Real-time streaming** with Apache Kafka
- [ ] **API endpoints** for integration
- [ ] **SAR auto-generation** module
- [ ] **Case management** integration

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📬 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/aml-project](https://github.com/yourusername/aml-project)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ for the AML/Compliance community

</div>


