import yfinance as yf         #  Importing all the required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = yf.download("RELIANCE.NS", start="2023-11-01", end="2025-11-01")      # Downloading the stock data from 2023-11-01 to 2025-11-01

# Flatten MultiIndex if present (keep only first level: Close, Open, and remove the ticker)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print(data)   # Printing the data 

data["log_ret"] = np.log(data["Close"] / data["Close"].shift(1))          # Finding LOG returns
data = data.dropna()

sigma_daily = data["log_ret"].std()
sigma_annual = sigma_daily * np.sqrt(252)

print("Volatility =", sigma_annual)



S0 = data["Close"].iloc[-1]   # current stock price from last row
print("Current spot price S0 =", S0)

# User inputs
K = float(input("Enter Strike Price K: "))
T = float(input("Enter Time to Maturity T (in years, e.g. 0.5 for 6 months): "))
r = float(input("Enter annual risk-free rate r (in decimal, e.g. 0.07 for 7%): "))
N = int(input("Enter number of steps in binomial tree (>= 50): "))

sigma = sigma_annual         # use the annual volatility you just computed
print("Using annual volatility σ =", sigma)


def euro_option_binomial(S0, K, T, r, sigma, N, option_type="call"):
    
    dt = T / N
    q = 0.0035 # Dividend Yield of Reliance ( Found from task A )
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)   # risk-neutral probability and assuming dividend yield is 0.0035

    

    # Stock prices at maturity (time N): S0 * u^j * d^(N-j) for j = 0..N
    j = np.arange(N + 1)
    ST = S0 * (u ** j) * (d ** (N - j))

    if option_type.lower() == "call":
        payoff = np.maximum(ST - K, 0.0)                                    # Payoff from call option
    elif option_type.lower() == "put":
        payoff = np.maximum(K - ST, 0.0)                                    # Payoff from put option
    else:
        raise ValueError("option_type must be 'call' or 'put'")


    # Backward induction
    for _ in range(N):                                                       # Used for valuing the option at time 0
        payoff = disc * (p * payoff[1:] + (1 - p) * payoff[:-1])

    # First element is the option value at t=0
    return float(payoff[0])




call_price = euro_option_binomial(S0, K, T, r, sigma, N, option_type="call")        # Call price 
put_price  = euro_option_binomial(S0, K, T, r, sigma, N, option_type="put")         # Put price

print(f"\nBinomial model ({N} steps):")
print("European Call Premium =", call_price)
print("European Put Premium  =", put_price)

