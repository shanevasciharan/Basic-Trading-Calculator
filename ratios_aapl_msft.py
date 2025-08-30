
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Please install yfinance first: pip install yfinance")

# ---------------------------
# Config 
# ---------------------------
TICKERS = ["AAPL", "MSFT"]
START = "2015-01-01"
END = None  # None = up to today
RF_ANNUAL = 0.00        # simple assumption (0%)
PERIODS_PER_YEAR = 252  # daily

# Helpers
def download_prices(tickers, start, end=None) -> pd.DataFrame:
    """Download adjusted close prices for tickers into a DataFrame of shape [date x tickers]."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance returns multi-index columns if multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data["Close"].to_frame(tickers[0])
    close = close.dropna(how="all")
    return close

def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Percent change returns, drop first NaN row."""
    rets = prices.pct_change()
    return rets.dropna(how="all")

def cagr_from_returns(returns: pd.Series, periods_per_year=252) -> float:
    """Compound Annual Growth Rate from a series of periodic returns."""
    if len(returns) == 0:
        return np.nan
    total_return = (1 + returns).prod()
    years = len(returns) / periods_per_year
    if years <= 0:
        return np.nan
    return total_return ** (1 / years) - 1

def max_drawdown_from_returns(returns: pd.Series) -> float:
    """Max drawdown from a series of periodic returns (using cumulative equity)."""
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return drawdown.min() if len(drawdown) else np.nan

def sharpe_ratio(returns: pd.Series, rf_annual=0.0, periods_per_year=252) -> float:
    """Annualized Sharpe with a constant annual rf (assumes rf spread evenly)."""
    if len(returns) == 0:
        return np.nan
    rf_period = rf_annual / periods_per_year
    excess = returns - rf_period
    mu = excess.mean() * periods_per_year
    sigma = excess.std(ddof=0) * np.sqrt(periods_per_year)
    return mu / sigma if sigma != 0 else np.nan

def sortino_ratio(returns: pd.Series, rf_annual=0.0, periods_per_year=252, mar_annual=0.0) -> float:
    """
    Annualized Sortino ratio with constant annual rf & MAR.
    - MAR (minimum acceptable return) defaults to 0 annualized.
    - Downside deviation uses returns below MAR/period.
    """
    if len(returns) == 0:
        return np.nan
    rf_period = rf_annual / periods_per_year
    mar_period = mar_annual / periods_per_year
    excess = returns - rf_period
    downside = np.where(returns < mar_period, returns - mar_period, 0.0)
    downside_std = np.sqrt(np.mean(downside**2))
    # Annualize
    mu = excess.mean() * periods_per_year
    dd_annual = downside_std * np.sqrt(periods_per_year)
    return mu / dd_annual if dd_annual != 0 else np.nan

def calmar_ratio(returns: pd.Series, periods_per_year=252) -> float:
    """Calmar = CAGR / |Max Drawdown|."""
    cagr = cagr_from_returns(returns, periods_per_year)
    mdd = max_drawdown_from_returns(returns)
    if pd.isna(cagr) or pd.isna(mdd) or mdd == 0:
        return np.nan
    return cagr / abs(mdd)

def compute_metrics(returns: pd.Series, name: str) -> dict:
    """Compute all metrics for a single series of returns."""
    return {
        "name": name,
        "cagr": cagr_from_returns(returns, PERIODS_PER_YEAR),
        "max_drawdown": max_drawdown_from_returns(returns),
        "sharpe": sharpe_ratio(returns, RF_ANNUAL, PERIODS_PER_YEAR),
        "sortino": sortino_ratio(returns, RF_ANNUAL, PERIODS_PER_YEAR, mar_annual=0.0),
        "calmar": calmar_ratio(returns, PERIODS_PER_YEAR),
        "n_periods": int(len(returns))
    }

def plot_equity(prices: pd.DataFrame, out_path="equity_curves.png"):
    """Plot growth of $1 for each column in `prices` (default matplotlib style)."""
    equity = (prices / prices.iloc[0])
    plt.figure()
    for col in equity.columns:
        plt.plot(equity.index, equity[col], label=str(col))
    plt.title("Growth of $1 (AAPL & MSFT)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

def main():
    prices = download_prices(TICKERS, START, END)
    rets = simple_returns(prices)

    # Compute metrics per ticker
    metrics = []
    for col in rets.columns:
        metrics.append(compute_metrics(rets[col].dropna(), name=str(col)))

    # Optional: equal-weight portfolio of AAPL & MSFT (still only these two assets)
    if len(rets.columns) == 2:
        ew = rets.mean(axis=1)
        metrics.append(compute_metrics(ew, name="EQUAL_WEIGHT_50_50"))

    # Save metrics
    met_df = pd.DataFrame(metrics)
    met_df = met_df[["name", "cagr", "max_drawdown", "sharpe", "sortino", "calmar", "n_periods"]]
    met_df.to_csv("metrics.csv", index=False)
    print("Saved: metrics.csv")
    print(met_df)

    # Plot equity
    plot_equity(prices, out_path="equity_curves.png")

if __name__ == "__main__":
    main()
