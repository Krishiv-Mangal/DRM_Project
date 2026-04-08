import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt



# Calculate the past 1-year dividend yield using yfinance
def get_dividend_yield(ticker="RELIANCE.NS"):
    """
    Trailing 12-month dividend yield:
        dividend_yield = (total dividends paid in last 12 months) / (current stock price)
    """

    stock = yf.Ticker(ticker)

    # Get all historical dividends
    dividends = stock.dividends

    # If company has no dividend data at all
    if dividends.empty:
        return 0.0

    # Sometimes yfinance gives timezone-aware dates -> convert to normal dates
    if dividends.index.tz is not None:
        dividends.index = dividends.index.tz_convert(None)

    # We only want dividends from the last 12 months
    one_year_ago = pd.Timestamp.today() - pd.Timedelta(days=365)
    last_year_dividends = dividends[dividends.index >= one_year_ago]

    # If no dividends were paid in the last year
    if last_year_dividends.empty:
        return 0.0

    # Total dividends paid in the past 12 months
    total_div = last_year_dividends.sum()

    # Get the latest closing price to divide by
    history = stock.history(period="1d")
    if history.empty:
        return 0.0

    current_price = history["Close"].iloc[-1]

    # Guard: avoid division by zero
    if current_price == 0:
        return 0.0

    # Final yield
    return float(total_div / current_price)



def binomial_option_price(S0, K, T, r, sigma, q, N=30, option_type="call"):     # Funtion to find the binomial option price with N=30
    
    dt = T / N
    # up/down factors driven by volatility
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)    # Discounting rate
    
    # risk-neutral probability adjusted for dividend yield q
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if p < 0.0 or p > 1.0:
        return 0.0

    stock_expiry = [] 
    
    for j in range(N + 1):
        final_price = S0 * (u ** j) * (d ** (N - j))           # Finding all combinations for the final price of stock
        stock_expiry.append(final_price)                       # Appending the values into the list of final stock prices

    stock_T = np.array(stock_expiry)                           # Converting the list into an array


    # terminal option payoff
    if option_type == "call":                                  # If loop executes if call option otherwise executes the put option
        opt = np.maximum(stock_T - K, 0.0)
    else:
        opt = np.maximum(K - stock_T, 0.0)

    # backward induction
    for _ in range(N-1, -1, -1):                                    
        opt = disc * (p * opt[1:] + (1-p) * opt[:-1])        # Value of the option is discounted backwards 

    return float(opt[0])            # Returning final option value




def get_price_data(ticker, start, end):             # Defining a funtion to find the stock market values and rolling volatility 
    data = yf.download(ticker, start=start, end=end, auto_adjust=False) # Auto adjusting false so we get the raw stock prices

    # flatten MultiIndex columns if present, (eg CLOSE,OPEN values would be in one column and the Ticker in another causing issue in finding values)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Close"]].dropna() 

    ret = np.log(data["Close"] / data["Close"].shift(1))   # log returns
    sigma_annual = ret.rolling(60).std() * np.sqrt(252)    # fird volatility on basis of the previous 60 days and annualize it 

    data["sigma"] = sigma_annual
    return data



def build_periods(dates, days_horizon=30):    # Used for defining months (30 day) in periods 
    
    periods = []          
    i = 0

    while i < len(dates) - 1:
        start_date = dates[i]
        end_target = start_date + pd.Timedelta(days=days_horizon)               # Add 30 days to the first date to find the index values 

        j = i
        while j < len(dates) and dates[j] < end_target:
            j += 1

        if j >= len(dates):
            break

        periods.append((i, j))              # Append the index pairs into the periods list 
        i = j

    return periods


def month_returns(S0, ST, sigma, q, r=0.06,                    # Risk free rate has been found out and approximated ( 91 day treasury bill )
                  horizon_days=30,
                  capital=100_000,
                  steps=30):
    
    T = horizon_days / 252.0

    if np.isnan(sigma) or sigma <= 0:
        return None, None, None

    # Covered Call
    K_cc = 1.05 * S0  # 5% OTM call
    C_cc = binomial_option_price(S0, K_cc, T, r, sigma, q,
                                 N=steps, option_type="call")

    # Effective Initial Cost of the Covered Call = S0 - C_cc (we receive premium)
    Initial_Cost = S0 - C_cc
    ret_cc = None
    if Initial_Cost > 0:
        units_cc = int(capital // Initial_Cost)      # Total units from Covered Call
        if units_cc > 0:
            payoff_call_short = max(ST - K_cc, 0.0)  # Payoff due to the short call
            final_per_unit = ST - payoff_call_short  # Final stock price subtracted by the payoff
            init_total = units_cc * Initial_Cost
            final_total = units_cc * final_per_unit
            ret_cc = (final_total - capital) / capital # Final return from the strategy

    # Protective Put
    K_pp = S0  # ATM put
    P_pp = binomial_option_price(S0, K_pp, T, r, sigma, q,
                                 N=steps, option_type="put")
    Initial_Put = S0 + P_pp
    ret_pp = None
    if Initial_Put > 0:
        units_pp = int(capital // Initial_Put)        # Total units from Protective Put
        if units_pp > 0:
            payoff_put_long = max(K_pp - ST, 0.0)     # Payoff due to the long put 
            final_per_unit = ST + payoff_put_long     # Final stock price and payoff from put 
            init_total = units_pp * Initial_Put
            final_total = units_pp * final_per_unit
            ret_pp = (final_total - capital) / capital  # Final return from the strategy

    # Long Straddle 
    K_ls = S0  # ATM call & put
    C_ls = binomial_option_price(S0, K_ls, T, r, sigma, q,
                                 N=steps, option_type="call")
    P_ls = binomial_option_price(S0, K_ls, T, r, sigma, q,
                                 N=steps, option_type="put")
    init_ls_per_unit = C_ls + P_ls      # Total cost of premium 
    ret_ls = None
    if init_ls_per_unit > 0:
        units_ls = int(capital // init_ls_per_unit)      # No of units from the strategy 
        if units_ls > 0:
            payoff_call = max(ST - K_ls, 0.0)            # Payoff from call
            payoff_put = max(K_ls - ST, 0.0)             # Payoff from put
            final_per_unit = payoff_call + payoff_put    # Total payoff
            init_total = units_ls * init_ls_per_unit
            final_total = units_ls * final_per_unit
            ret_ls = (final_total - capital) / capital   # Return from the strategy

    return ret_cc, ret_pp, ret_ls



def backtest_options_strategies(
    ticker="RELIANCE.NS",
    backtest_start="2022-01-01",
    backtest_end="2024-01-01"
):
    
    # get extra history before backtest for volatility
    data_start = "2020-01-01"

    data = get_price_data(ticker, data_start, backtest_end)
    # keep only rows inside backtest window
    data_bt = data.loc[backtest_start:backtest_end].dropna(subset=["sigma"])

    dates_bt = data_bt.index
    periods = build_periods(dates_bt, days_horizon=30)

    # get trailing 12M dividend yield for the ticker
    q = get_dividend_yield(ticker)
    print("Dividend Yield used:", q)

    rows = []

    for start, end in periods:
        start_date = dates_bt[start]
        end_date = dates_bt[end]

        S0 = float(data_bt.iloc[start]["Close"])
        ST = float(data_bt.iloc[end]["Close"])
        sigma = float(data_bt.iloc[start]["sigma"])

        ret_cc, ret_pp, ret_ls = month_returns(S0, ST, sigma, q)

        nifty_ret = (ST - S0) / S0  # underlying monthly return

        rows.append({                                           # Appending required values for the monthly log
            "Month_Start": start_date.date(),
            "Month_End": end_date.date(),
            "S0": S0,
            "ST": ST,
            "Sigma": sigma,
            "Covered_Call_Return": ret_cc,
            "Protective_Put_Return": ret_pp,
            "Straddle_Return": ret_ls,
            "Underlying_Return": nifty_ret,
        })

    df = pd.DataFrame(rows)
    return df


def strategy_metrics(monthly_returns: pd.Series):   #  Take a Series of monthly returns (e.g. 0.03 for +3%)
    
    # 1. Clean the data: remove any NaN values
    r = monthly_returns.dropna()
    if r.empty:
        # If there are no valid returns, we can't compute any metrics
        return None

    # 2. Cumulative growth of ₹1 invested
    
    cum_wealth = (1 + r).cumprod()

    # Total return = final wealth - 1
    total_return = cum_wealth.iloc[-1] - 1

    # Win rate = fraction of months where return > 0
    win_rate = (r > 0).mean()

    # Drawdown: how far we fall from previous peaks
    #    peak_wealth tracks the running maximum of cum_wealth
    peak_wealth = cum_wealth.cummax()

    # Drawdown at each month = (current - peak) / peak  (will be <= 0)
    drawdown = (cum_wealth - peak_wealth) / peak_wealth

    # Max drawdown is the worst (most negative) drawdown
    max_drawdown = drawdown.min()

    # 4. Best and worst single-month returns
    best_month = r.max()
    worst_month = r.min()

    
    vol = r.std() * np.sqrt(12)

    # 6. Package everything 
    return {
        "Total Return": total_return,
        "Win Rate": win_rate,
        "Max Drawdown": max_drawdown,
        "Best Month": best_month,
        "Worst Month": worst_month,
        "Volatility": vol,
    }

#  Run everything


if __name__ == "__main__":
    # Run the backtest and get the monthly returns table
    df = backtest_options_strategies()

    # When we print numbers, show 4 decimal places
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")

    CAPITAL = 100_000   # we pretend we put 1 lakh into each strategy

    # Turn % returns into monthly P&L (in ₹) 
    df["CC_PnL_Rupees"]         = df["Covered_Call_Return"] * CAPITAL
    df["PP_PnL_Rupees"]         = df["Protective_Put_Return"] * CAPITAL
    df["LS_PnL_Rupees"]         = df["Straddle_Return"] * CAPITAL
    df["Underlying_PnL_Rupees"] = df["Underlying_Return"] * CAPITAL

    # Build cumulative wealth paths 
    # Start from 1,00,000 and multiply by (1 + monthly return) each month
    df["CC_Wealth"]         = CAPITAL * (1 + df["Covered_Call_Return"].fillna(0)).cumprod()
    df["PP_Wealth"]         = CAPITAL * (1 + df["Protective_Put_Return"].fillna(0)).cumprod()
    df["LS_Wealth"]         = CAPITAL * (1 + df["Straddle_Return"].fillna(0)).cumprod()
    df["Underlying_Wealth"] = CAPITAL * (1 + df["Underlying_Return"].fillna(0)).cumprod()

    # Dump the raw monthly log (can comment this out if it’s too long) 
    print("\n=== Monthly Log ===\n")
    print(df.to_string(index=False))

    # Final wealth after the whole 2-year period 
    print("\n===== Terminal Wealth (₹) =====")
    print(f"Covered Call:       {df['CC_Wealth'].iloc[-1]:,.2f}")
    print(f"Protective Put:     {df['PP_Wealth'].iloc[-1]:,.2f}")
    print(f"Long Straddle:      {df['LS_Wealth'].iloc[-1]:,.2f}")
    print(f"Underlying (Stock): {df['Underlying_Wealth'].iloc[-1]:,.2f}")

    #  Plot 1: Wealth vs time 
    dates_end   = list(df["Month_End"])
    start_date0 = df["Month_Start"].iloc[0]

    # X-axis = start date + all month-end dates
    dates_plot = [start_date0] + dates_end

    # Each line starts at 1,00,000 and then follows the wealth column
    cc_plot = [CAPITAL] + list(df["CC_Wealth"])
    pp_plot = [CAPITAL] + list(df["PP_Wealth"])
    ls_plot = [CAPITAL] + list(df["LS_Wealth"])
    ul_plot = [CAPITAL] + list(df["Underlying_Wealth"])

    plt.figure(figsize=(9, 5))
    plt.plot(dates_plot, cc_plot, label="Covered Call")
    plt.plot(dates_plot, pp_plot, label="Protective Put")
    plt.plot(dates_plot, ls_plot, label="Long Straddle")
    plt.plot(dates_plot, ul_plot, label="Underlying Stock")

    plt.xlabel("Date")
    plt.ylabel("Wealth (₹)")
    plt.title("Cumulative Wealth (₹100,000 initial)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 2: Drawdown (how much we fall from the peak) 
    drawdowns = pd.DataFrame(index=df.index)

    for name, col in [
        ("Covered Call",   "CC_Wealth"),
        ("Protective Put", "PP_Wealth"),
        ("Long Straddle",  "LS_Wealth"),
        ("Underlying",     "Underlying_Wealth"),
    ]:
        wealth = df[col]
        peak   = wealth.cummax()               # running max so far
        drawdowns[name] = (wealth - peak) / peak   # 0 at new highs, negative in drawdown

    plt.figure(figsize=(9, 5))
    for name in drawdowns.columns:
        plt.plot(dates_end, drawdowns[name], label=name)

    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("Drawdown Comparison")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 3: Distribution of monthly returns 
    strategies = {
        "Covered Call":     "Covered_Call_Return",
        "Protective Put":   "Protective_Put_Return",
        "Long Straddle":    "Straddle_Return",
        "Underlying Stock": "Underlying_Return",
    }

    for name, col in strategies.items():
        plt.figure(figsize=(6, 4))
        plt.hist(df[col].dropna(), bins=15)
        plt.xlabel("Monthly Return")
        plt.ylabel("Frequency (number of months)")
        plt.title(f"Histogram of Monthly Returns – {name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
