# Nifty50 Portfolio Optimizer

AI-powered Decision Support System (DSS) for constructing optimized Nifty 50 portfolios using LSTM price forecasts and sentiment signals (FinBERT + momentum fallback). Provides an interactive Streamlit dashboard to explore forecasts, optimized allocations and investment plans.

---

## Key features
- Per-ticker LSTM forecasting using multivariate inputs (price, volume, technicals, sentiment).
- Sentiment pipeline with FinBERT (news) and a momentum-based fallback when news/FinBERT unavailable.
- Covariance matrix calculation and constraint-aware portfolio optimization (Sharpe ratio maximization).
- Interactive Streamlit UI: refresh data, set constraints (num stocks, target return, max risk), visualize allocations and per-stock charts.
- Export investment plan as CSV.

---

## Architecture (high level)
- Data ingestion: yfinance (historical OHLCV) and optional news.
- Feature engine: returns, volatility, moving averages, RSI and sentiment alignment.
- Modeling: per-ticker LSTM models + scalers, multi-day rolling forecasts.
- Risk engine: historical returns → covariance matrix.
- Optimizer: Monte‑Carlo / constrained search → best Sharpe ratio portfolio.
- Presentation: Streamlit app reads serialized artifact (`portfolio_model_data.pkl`) and displays results.

Simple diagram (paste into mermaid.live to render)
```mermaid
flowchart TD
  Data[Price & News Data (yfinance)]
  ETL[ETL / Preprocessing]
  Sentiment[Sentiment (FinBERT ⇄ Momentum)]
  LSTM[LSTM Trainer & Predictor]
  Risk[Covariance Matrix]
  Optim[Optimizer (Sharpe Max)]
  UI[Streamlit Dashboard]
  Data --> ETL --> Sentiment --> LSTM --> Risk --> Optim --> UI
  UI -->|user control| ETL
```

---

## Quick start

1. Clone repository
   ```bash
   git clone https://github.com/your-org/nifty50-portfolio-optimizer.git
   cd nifty50-portfolio-optimizer
   ```

2. Create Python environment (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

   Important packages: streamlit, pandas, numpy, yfinance, plotly, scikit-learn, tensorflow, transformers, torch (optional if FinBERT used).

3. Prepare data / initial artifact
   - If `portfolio_model_data.pkl` is not present, either:
     - Run the provided notebook / pipeline to generate it, or
     - Launch the app and click "Refresh Market Data" (note: full refresh trains LSTMs and can be compute intensive)

4. Run the app
   ```bash
   streamlit run app.py
   ```

---

## Usage notes
- "Refresh Market Data" runs a full ETL → sentiment → train → predict → optimize pipeline. This is CPU/GPU and time intensive.
- FinBERT loading requires network access and PyTorch; the app falls back to technical/momentum-based sentiment if unavailable.
- Models and scalers are saved inside `portfolio_model_data.pkl`. For production, replace with a model registry (MLflow / S3).

---

## Files & important modules
- app.py — Streamlit UI & orchestration
- portfolio_functions.py — optimization utilities and portfolio metric calculations
- portfolio_model_data.pkl — serialized artifact (created by refresh or notebook)
- notebook/ — optional training / data-prep notebooks (if present)
- README.md — this document

---

## Evaluation & experimentation
- Evaluate LSTM forecasts with MSE/MAE on holdout windows.
- Backtest optimized portfolios vs benchmark (Nifty 50) for realized returns, Sharpe ratio, max drawdown.
- Ablation experiments: FinBERT vs momentum-only vs no-sentiment features.

---

## Development notes
- For faster iteration, reduce number of tickers or training epochs.
- To disable FinBERT entirely, run `refresh_data` with the FinBERT-loading block commented or ensure HuggingFace snapshot is unavailable.
- Consider decoupling heavy tasks into background jobs (Airflow / Kubernetes) for production.

---

## License & authors
Authors: Aritra Sarkar, Satish Prem Anand, Dhiranjit Daimary  
License: MIT (add LICENSE file if desired)

---

## Contact
For issues or contributions, open an issue or PR on the project repository.
