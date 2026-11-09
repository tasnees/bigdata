# Scalable Investment Risk & Market Volatility Engine
### Distributed Big Data Lakehouse (PySpark + Delta Lake) & Deep Learning Analytics

[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.6-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Delta Lake](https://img.shields.io/badge/Delta%20Lake-3.3.2-00ADD8?logo=delta&logoColor=white)](https://delta.io/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Java](https://img.shields.io/badge/Java%20OpenJDK-17-ED8B00?logo=openjdk&logoColor=white)](https://learn.microsoft.com/en-us/java/openjdk/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)

---

## 1. Executive Summary & Architecture Overview

This project implements an enterprise-grade, distributed **Lakehouse pipeline** and **Deep Learning inference framework** designed for quantitative financial risk management, historical daily bar ingestion, market feature engineering, and volatility forecasting.

The solution bridges high-throughput batch distributed data engineering (**PySpark** + **Delta Lake ACID storage**) with modern deep-learning architectures (**PyTorch**, **Optuna**, **SHAP**, and high-performance **Vectorized PySpark Pandas UDF** batch scoring).

```
                                 DATA INGESTION & STORAGE LAYER
                     ┌────────────────────────────────────────────────────────┐
                     │ Yahoo Finance API (Stocks, ETFs, Benchmark Indices)    │
                     └──────────────────────────┬─────────────────────────────┘
                                                │  (Batched & Rate-Limited API Pull)
                                                ▼
                     ┌────────────────────────────────────────────────────────┐
                     │ Raw Layer: Multi-Index Cleaned Parquet Partitions       │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
                                  ACID DELTA LAKEHOUSE (MEDALLION)
                     ┌────────────────────────────────────────────────────────┐
                     │ Bronze Table: Standardized Schema, Deduplicated        │
                     └──────────────────────────┬─────────────────────────────┘
                                                │  (Quality Audit & Outlier Quarantine)
                                                ▼
                     ┌────────────────────────────────────────────────────────┐
                     │ Silver Table: Validated Canonical Market Bars (OHLCV)  │
                     └─────────────┬────────────────────────────┬─────────────┘
                                   │                            │
                     ┌─────────────▼─────────────┐ ┌────────────▼─────────────┐
                     │ Gold: Classical Risk      │ │ Deep Learning Engine     │
                     │ - 20d/60d Rolling Vol     │ │ - Dilated Causal TCN     │
                     │ - 20d/60d Drawdown        │ │ - Transformer Encoder    │
                     │ - Benchmark Pivot & Beta  │ │ - Multi-Asset Embedding  │
                     └───────────────────────────┘ └────────────┬─────────────┘
                                                                │  (Spark Pandas UDF)
                                                   ┌────────────▼─────────────┐
                                                   │ Gold: dl_risk_scores     │
                                                   │ - Predicted Volatility   │
                                                   │ - Model Audit & Features │
                                                   └──────────────────────────┘
```

---

## 2. Technical Stack & Environment Configuration

| Layer | Component | Version / Specification | Technical Role |
| :--- | :--- | :--- | :--- |
| **Compute Core** | Apache Spark | `3.5.6` | Distributed data processing, windowing, partition management |
| **Storage Engine**| Delta Lake | `3.3.2` | Parquet + ACID Transaction logs, time travel, partition pruning |
| **Runtime** | Java OpenJDK | `Microsoft OpenJDK 17.0.21` | JVM runtime required for Spark driver and cluster worker nodes |
| **Windows Native**| Hadoop Binaries| `Hadoop 3.3.x winutils.exe` | POSIX file system emulation for Windows platforms |
| **ML Engine** | PyTorch | `2.4.0` | Deep neural network training, CUDA/CPU tensor computations |
| **HPO & MLOps** | Optuna + MLflow| Latest | Bayesian hyperparameter optimization & artifact experiment tracking |
| **Explainability**| SHAP | KernelExplainer | Game-theoretic feature attribution & local risk interpretability |
| **Serving API** | Flask + Spark UDF| REST daemon + PyArrow | Vectorized distributed model execution across Spark executors |

---

## 3. Directory & Repository Structure

```text
bigdata/
├── .github/                               # CI/CD automation & pipeline checks
├── .gitignore                             # Ignore rules for local JVMs, virtual environments, large data
├── README.md                              # Technical project documentation
├── apple_data.csv                         # Offline sample benchmark data
│
├── dl_advanced_risk_model.ipynb           # Advanced Deep Learning pipeline:
│                                          #   - Dilated TCN, Transformer, Multi-Asset Embedding
│                                          #   - Technical indicators (RSI, MACD, Bollinger Bands)
│                                          #   - Optuna HPO, MLflow artifact logging, SHAP explainability
│                                          #   - Flask microservice & distributed Spark Pandas UDF
│
├── yahoo_finance_dl_risk.ipynb            # Lightweight DL integration prototype (LSTM + Silver scoring)
│
├── yahoo_finance_only_investment_risk_pyspark.ipynb
│                                          # Core Data Lakehouse ETL:
│                                          #   - Batched yfinance pull with exponential backoff
│                                          #   - Medallion Architecture (Bronze -> Silver -> Gold)
│                                          #   - Quality assertions & Quarantine routing
│                                          #   - Financial windowing (Beta, Drawdowns, Annualized Vol)
│
├── data/                                  # Local Data Lakehouse storage root (Git-ignored)
│   ├── raw/yahoo_finance/                 # Intermediate batched Parquet files
│   ├── lakehouse/
│   │   ├── bronze/market_bars/            # Deduplicated raw state (Year/Month partition)
│   │   ├── silver/market_bars/            # Validated market bars (Cleaned schema)
│   │   ├── quarantine/market_bars/        # Failed data quality check records
│   │   └── gold/
│   │       ├── investment_risk_features/  # Classical risk metrics & benchmark betas
│   │       └── dl_risk_scores/            # Deep-learning predicted volatility scores
│   └── reports/                           # Audit CSVs & SHAP interpretation figures
│
├── models/                                # Persisted model checkpoints (.pt)
│   ├── vol_tcn.pt                         # Trained Temporal Convolutional Network
│   ├── vol_transformer.pt                 # Trained Transformer Encoder
│   └── vol_multi_asset_lstm.pt            # Trained Multi-Asset Joint Embedding Network
│
├── hadoop/                                # Windows Hadoop POSIX compatibility binaries
│   └── bin/winutils.exe                   # Windows native execution adapter
│
└── spark_temp/                            # Spark local shuffle & scratch workspace
```

---

## 4. Deep Dive: Medallion Architecture & Data Engineering

### Ingestion & Resiliency
- Ingests cross-asset universes (Equities: `AAPL`, `MSFT`, `NVDA`, `TSLA`; ETFs: `SPY`, `QQQ`, `GLD`, `TLT`; Indices: `^GSPC`, `^IXIC`, `^DJI`).
- Resilient multi-tier downloads with exponential backoff retry algorithms to handle Yahoo Finance API rate limits and socket drops.
- Column normalization that automatically flattens varying MultiIndex schemas output by batch scrapers.

### Data Quality & Quarantine Policy
Data integrity is enforced at the entry to the Silver layer. Records failing any of the following constraints are branched into a dedicated **Quarantine Delta table**:
1. `close <= 0` or `null` closing prices.
2. `volume < 0`.
3. Broken candle physics: `high < low`, `high < max(open, close)`, or `low > min(open, close)`.
4. Duplicate candidate natural keys `(symbol, observation_date)`.

### Partitioning & Delta Optimizations
- Tables are partitioned along temporal boundaries `(year, month)` to facilitate partition pruning on downstream range scans.
- Deduplication leverages Spark SQL analytical window specifications `row_number() over (partition by symbol, observation_date order by ingestion_timestamp desc)`.

---

## 5. Deep Learning & Quantitative Risk Engine

### Advanced Model Architectures

#### 1. Temporal Convolutional Network (TCN)
- Utilizes 1D dilated convolutions with causal padding to ensure no future information leakage.
- Exponential dilation progression ($d = 1, 2, 4$) provides receptive fields covering multi-week market dynamics while avoiding the vanishing/exploding gradient pitfalls of standard recurrent networks.
- Identity/1x1 projection residual paths allow deep representations to converge smoothly.

#### 2. Transformer Risk Encoder
- Multi-Head Self-Attention (`nn.TransformerEncoderLayer`) mapping cross-time dependencies directly without iterative recurrent recurrence.
- Models long-range volatility clustering and macroeconomic shocks.

#### 3. Multi-Asset Joint Embedding Network
- Addresses asset-level cold starts and cross-ticker correlation using a trainable entity embedding layer `nn.Embedding(num_symbols, embed_dim=8)`.
- Learns dense continuous representations for correlated market symbols, enabling statistical strength sharing between high-liquidity large caps and volatile small caps.

---

### Technical Indicators & Feature Engineering
Calculated independently per ticker partition to guarantee zero cross-asset bleed:
- **Log Returns:** $r_t = \ln(P_t / P_{t-1})$
- **Realized Volatility:** 10-day rolling sample standard deviation annualized.
- **RSI (14-day):** Exponential moving average momentum balance:
  $$\text{RSI} = 100 - \frac{100}{1 + \frac{\text{EMA}(\text{Gain}, 14)}{\text{EMA}(\text{Loss}, 14)}}$$
- **MACD Signal Line:** Difference between 12-day and 26-day fast/slow EMAs smoothed by a 9-day signal EMA.
- **Bollinger Band Width:** Volatility expansion/contraction measure $\frac{2 \times \sigma_{20}}{\text{SMA}_{20}}$.

---

### MLOps: Optuna Tuning, MLflow & SHAP

- **Optuna Hyperparameter Search:** Multi-trial Bayesian optimization searching over hidden units, learning rates, layer depth, and dropout.
- **MLflow Tracking:** Logs parameters, epoch trajectories, validation RMSE, and model binaries to an organized experiment store.
- **SHAP (Shapley Additive Explanations):** Generates local feature-importance distributions using `KernelExplainer`, producing interpretability artifacts for financial compliance and model risk management.

---

## 6. Distributed Model Serving via Spark Pandas UDF

For scalable batch inference without leaving the Spark engine, models are operationalized through two interchangeable avenues:
1. **Direct In-Executor Tensor Evaluation:** Loading model weights once per Python executor daemon via a PySpark Scalar Pandas UDF (`@pandas_udf("double")`).
2. **Microservice REST Endpoint:** A standalone Flask endpoint serving the model at `/predict` queried in vector batches by Spark worker tasks through Apache Arrow serialization, decoupling compute worker nodes from the model inference lifecycle.

---

## 7. Windows Setup & Reproducibility Guide

### 1. Prerequisites
- **Python:** 3.10 or 3.11 installed.
- **Java:** Microsoft OpenJDK 17 installed.

### 2. Configure Windows Hadoop Environment
Set your system environment variables (PowerShell):
```powershell
[Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Microsoft\jdk-17.0.21.21-hotspot", "User")
[Environment]::SetEnvironmentVariable("HADOOP_HOME", "$PWD\hadoop", "User")
```

### 3. PySpark Driver Hook (Crucial for Windows JVM)
To prevent `Py4JJavaError: Shell.checkHadoopHomeInner` initialization errors, pass the Hadoop path to the driver before the JVM boots:
```python
import os
hadoop_fwd = os.environ["HADOOP_HOME"].replace("\\", "/")
os.environ["PYSPARK_SUBMIT_ARGS"] = f'--driver-java-options "-Dhadoop.home.dir={hadoop_fwd}" pyspark-shell'
```

### 4. Running the Pipelines
1. Execute `yahoo_finance_only_investment_risk_pyspark.ipynb` to ingest market data and populate the **Bronze**, **Silver**, and **Gold** Delta lakes.
2. Execute `dl_advanced_risk_model.ipynb` to run technical feature engineering, train the deep learning architectures, perform Optuna HPO, evaluate SHAP interpretability, and output deep-learning volatility forecasts to `data/lakehouse/gold/dl_risk_scores`.
