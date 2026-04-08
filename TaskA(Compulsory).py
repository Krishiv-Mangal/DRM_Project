import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt



# ======================================================
# 1. BINOMIAL PRICER
# ======================================================

def binomial_option_price(S0, K, T, r, sigma, N=30, option_type="call"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    if p < 0.0 or p > 1.0:
        return 0.0

    stock_T = np.array([S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)])

    if option_type == "call":
        option_vals = np.maximum(stock_T - K, 0.0)
    else:
        option_vals = np.maximum(K - stock_T, 0.0)

    for _ in range(N - 1, -1, -1):
        option_vals = disc * (p * option_vals[1:] + (1 - p) * option_vals[:-1])

    return float(option_vals[0])


# ======================================================
# 2. SYNTHETIC CALL PAYOFF
# ======================================================

def synthetic_call_payoff(S0, ST, K):
    # synthetic = long stock + long put
    return (ST - S0) + max(K - ST, 0.0)


# ======================================================
# 3. MAIN FUNCTION – pick 3 valid months
# ======================================================

def synthetic_vs_actual_call():
    data = yf.download(
        "RELIANCE.NS",
        start="2023-01-01",
        auto_adjust=False,
    )

    # Flatten MultiIndex columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Keep only Close
    data = data[["Close"]].dropna()

    # Daily returns & rolling 60-day annual vol
    returns = data["Close"].pct_change()
    sigma_annual = returns.rolling(60).std() * np.sqrt(252)
    data["sigma"] = sigma_annual

    # Drop rows without sigma
    data = data.dropna(subset=["sigma"])
    if len(data) == 0:
        raise ValueError("No rows with valid sigma; check data range.")

    all_dates = data.index
    all_months = all_dates.to_period("M")
    unique_months = pd.unique(all_months)

    # Look back over the last 6 months (or fewer if not available),
    # starting from the most recent month, and pick the first 3 that work.
    candidate_months = list(unique_months[-6:])[::-1]  # reverse: newest → oldest

    r = 0.06
    horizon_days = 30
    T = horizon_days / 252.0
    steps = 30

    results = []

    for m in candidate_months:
        # trading days in this month
        month_mask = all_months == m
        month_dates = all_dates[month_mask]
        if len(month_dates) == 0:
            continue

        # first trading day of that month (with valid sigma)
        start_date = month_dates[0]

        S0 = float(data.loc[start_date, "Close"])
        sigma = float(data.loc[start_date, "sigma"])

        # ATM strike ≈ nearest 50
        K = round(S0 / 50.0) * 50.0

        # price call & put with binomial
        call_price = binomial_option_price(S0, K, T, r, sigma,
                                           N=steps, option_type="call")
        put_price = binomial_option_price(S0, K, T, r, sigma,
                                          N=steps, option_type="put")

        # expiry ≈ 30 calendar days later → nearest trading day
        target = start_date + pd.Timedelta(days=horizon_days)
        end_pos = all_dates.searchsorted(target)
        if end_pos >= len(all_dates):
            # no future data for this month → skip it
            continue
        end_date = all_dates[end_pos]
        ST = float(data.loc[end_date, "Close"])

        actual_call_payoff = max(ST - K, 0.0)
        synthetic_payoff = synthetic_call_payoff(S0, ST, K)

        results.append({
            "Start Date": start_date.date(),
            "End Date": end_date.date(),
            "S0": S0,
            "ST": ST,
            "Strike": K,
            "Sigma": sigma,
            "Call Premium": call_price,
            "Put Premium": put_price,
            "Actual Call Cost": call_price,
            "Synthetic Cost": S0 + put_price,
            "Actual Call Payoff": actual_call_payoff,
            "Synthetic Payoff": synthetic_payoff,
            "Actual Call P&L": actual_call_payoff - call_price,
            "Synthetic P&L": synthetic_payoff - (S0 + put_price),
        })

        # stop once we have 3 valid months
        if len(results) == 3:
            break

    return pd.DataFrame(results)


# ======================================================
# 4. RUN
# ======================================================

if __name__ == "__main__":
    df = synthetic_vs_actual_call()
    pd.set_option("display.float_format", lambda x: f"{x:0.2f}")
    print(df.to_string())


    # ---------------------------
    # Payoff graph for last row
    # ---------------------------
    # Pick the most recent month (row 0 if you like, or -1 for last)
    row = df.iloc[0]   # use 0, 1, or 2 depending which month you want

    S0 = row["S0"]
    K = row["Strike"]

    # Range of possible expiry prices around strike
    S_range = np.linspace(0.6 * K, 1.4 * K, 200)

    # Payoff per unit at expiry
    call_payoff = np.maximum(S_range - K, 0.0)
    synthetic_payoff = (S_range - S0) + np.maximum(K - S_range, 0.0)

    plt.figure(figsize=(8, 5))
    plt.axhline(0, linewidth=1)
    plt.plot(S_range, call_payoff, label="Actual Call Payoff")
    plt.plot(S_range, synthetic_payoff, linestyle="--",
             label="Synthetic Call Payoff (Stock + Put)")

    plt.xlabel("Stock Price at Expiry $S_T$")
    plt.ylabel("Payoff per unit")
    plt.title(f"Payoff Comparison at Expiry\nStart {row['Start Date']}, Strike K = {K:.0f}")
    plt.legend()
    plt.grid(True)

    # If you want to save instead of just show:
    # plt.savefig("synthetic_vs_call_payoff.png", dpi=300, bbox_inches="tight")

    plt.show()
