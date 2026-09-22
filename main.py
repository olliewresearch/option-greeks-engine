import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def d1_d2(S, K, T, r, sigma):
    """Calculate the d1 and d2 terms used throughout the Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes price of a European call option."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes price of a European put option."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def call_delta(S, K, T, r, sigma):
    """Delta of a call option: how much the call price moves per $1 move in S."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)


def put_delta(S, K, T, r, sigma):
    """Delta of a put option: how much the put price moves per $1 move in S."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    """Gamma: how much Delta itself changes per $1 move in S. Same formula for calls and puts."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def plot_delta_vs_stock_price(K, T, r, sigma):
    """Plot call and put Delta across a range of stock prices, strike held fixed."""
    S_range = np.linspace(1, 2 * K, 200)
    call_deltas = call_delta(S_range, K, T, r, sigma)
    put_deltas = put_delta(S_range, K, T, r, sigma)

    plt.plot(S_range, call_deltas, label="Call Delta")
    plt.plot(S_range, put_deltas, label="Put Delta")
    plt.axvline(K, color="grey", linestyle="--", label=f"Strike (K={K})")
    plt.xlabel("Stock Price (S)")
    plt.ylabel("Delta")
    plt.title("Delta vs Stock Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_delta_vs_strike(S, T, r, sigma):
    """Plot call and put Delta across a range of strike prices, stock price held fixed."""
    K_range = np.linspace(1, 2 * S, 200)
    call_deltas = call_delta(S, K_range, T, r, sigma)
    put_deltas = put_delta(S, K_range, T, r, sigma)

    plt.plot(K_range, call_deltas, label="Call Delta")
    plt.plot(K_range, put_deltas, label="Put Delta")
    plt.axvline(S, color="grey", linestyle="--", label=f"Stock Price (S={S})")
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Delta")
    plt.title("Delta vs Strike Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_delta_vs_expiration(S, K, r, sigma):
    """Plot call and put Delta across a range of times to expiration."""
    T_range = np.linspace(0.01, 2, 200)
    call_deltas = call_delta(S, K, T_range, r, sigma)
    put_deltas = put_delta(S, K, T_range, r, sigma)

    plt.plot(T_range, call_deltas, label="Call Delta")
    plt.plot(T_range, put_deltas, label="Put Delta")
    plt.xlabel("Time to Expiration (years)")
    plt.ylabel("Delta")
    plt.title("Delta vs Time to Expiration")
    plt.legend()
    plt.grid(True)
    plt.show()


def delta_gamma_approximation(S, K, T, r, sigma, dS):
    """
    Estimate the new call price after the stock moves by dS, using a
    Delta + Gamma (second-order Taylor) approximation, then compare it
    to the actual recalculated Black-Scholes price.
    """
    original_price = black_scholes_call(S, K, T, r, sigma)
    delta = call_delta(S, K, T, r, sigma)
    g = gamma(S, K, T, r, sigma)

    approx_price = original_price + delta * dS + 0.5 * g * dS ** 2
    actual_price = black_scholes_call(S + dS, K, T, r, sigma)

    print(f"Original price: {original_price:.4f}")
    print(f"Approximated new price (Delta+Gamma): {approx_price:.4f}")
    print(f"Actual new price (full Black-Scholes): {actual_price:.4f}")
    print(f"Approximation error: {abs(actual_price - approx_price):.4f}")


if __name__ == "__main__":
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2

    print("Call price:", black_scholes_call(S, K, T, r, sigma))
    print("Put price:", black_scholes_put(S, K, T, r, sigma))
    print("Call Delta:", call_delta(S, K, T, r, sigma))
    print("Put Delta:", put_delta(S, K, T, r, sigma))
    print("Gamma:", gamma(S, K, T, r, sigma))

    plot_delta_vs_stock_price(K, T, r, sigma)
    plot_delta_vs_strike(S, T, r, sigma)
    plot_delta_vs_expiration(S, K, r, sigma)

    delta_gamma_approximation(S, K, T, r, sigma, dS=5)