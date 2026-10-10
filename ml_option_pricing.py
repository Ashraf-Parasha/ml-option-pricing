"""
European call pricing three ways: Black-Scholes, Monte Carlo, neural network.

1. Black-Scholes closed form (reference price).
2. Monte Carlo under a geometric Brownian motion, with its convergence rate.
3. A neural network trained to learn the pricing function
       (S/K, T, sigma, r)  ->  price / K
   then compared with the two others on accuracy and speed.

Usage:  python ml_option_pricing.py
"""
import time

import numpy as np
from scipy.stats import norm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_TRAIN, N_TEST = 100_000, 20_000


# ------------------------------------------------------------ Black-Scholes
def black_scholes_call(s, k, t, sigma, r):
    """Black-Scholes price of a European call (no dividend). Vectorised."""
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


# -------------------------------------------------------------- Monte Carlo
def monte_carlo_call(s, k, t, sigma, r, n_paths, rng):
    """Monte Carlo price of a European call. Returns (price, std error)."""
    z = rng.standard_normal(n_paths)
    s_t = s * np.exp((r - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)
    payoff = np.exp(-r * t) * np.maximum(s_t - k, 0.0)
    return payoff.mean(), payoff.std(ddof=1) / np.sqrt(n_paths)


# ------------------------------------------------------------ ML data set
def sample_options(n, rng):
    """Random option parameters. Features: moneyness S/K, maturity, vol, rate.

    Prices are homogeneous in (S, K), so the network learns price / K
    as a function of S / K: one input fewer and no scale problem.
    """
    x = np.column_stack([
        rng.uniform(0.6, 1.4, n),      # moneyness S / K
        rng.uniform(0.1, 2.0, n),      # maturity in years
        rng.uniform(0.10, 0.60, n),    # volatility
        rng.uniform(0.00, 0.06, n),    # risk-free rate
    ])
    y = black_scholes_call(x[:, 0], 1.0, x[:, 1], x[:, 2], x[:, 3])
    return x, y


def main():
    rng = np.random.default_rng(SEED)

    # --- 1. Black-Scholes vs Monte Carlo on one option
    s, k, t, sigma, r = 100.0, 100.0, 1.0, 0.20, 0.03
    bs = black_scholes_call(s, k, t, sigma, r)
    print(f"Reference option: S = K = 100, T = 1y, sigma = 20%, r = 3%")
    print(f"Black-Scholes price = {bs:.4f}\n")
    print("--- Monte Carlo convergence ---")
    path_counts = [1_000, 10_000, 100_000, 1_000_000]
    mc_errors, mc_std = [], []
    for n in path_counts:
        price, se = monte_carlo_call(s, k, t, sigma, r, n, rng)
        mc_errors.append(abs(price - bs))
        mc_std.append(se)
        print(f"N = {n:>9,}   price = {price:8.4f}   std error = {se:.4f}   |error| = {abs(price - bs):.4f}")

    # --- 2. Neural network
    x_train, y_train = sample_options(N_TRAIN, rng)
    x_test, y_test = sample_options(N_TEST, rng)
    scaler = StandardScaler().fit(x_train)
    model = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", max_iter=200,
                         early_stopping=True, random_state=SEED)
    start = time.perf_counter()
    model.fit(scaler.transform(x_train), y_train)
    train_time = time.perf_counter() - start

    pred = model.predict(scaler.transform(x_test))
    err = pred - y_test
    liquid = y_test > 0.01                      # options worth more than 1% of the strike
    print(f"\n--- Neural network (2 x 64 neurons, {N_TRAIN:,} training prices) ---")
    print(f"Training time            = {train_time:.1f} s  ({model.n_iter_} epochs)")
    print(f"Test R^2                 = {1 - np.sum(err**2) / np.sum((y_test - y_test.mean())**2):.5f}")
    print(f"Mean abs error           = {np.mean(np.abs(err)) * 100:.3f}% of strike")
    print(f"RMSE                     = {np.sqrt(np.mean(err**2)) * 100:.3f}% of strike")
    print(f"99th percentile |error|  = {np.percentile(np.abs(err), 99) * 100:.3f}% of strike")
    print(f"Mean relative error      = {np.mean(np.abs(err[liquid]) / y_test[liquid]):.2%}"
          f"  (options with price > 1% of strike, {liquid.mean():.0%} of the test set)")

    # --- 3. Speed: time to price the 20,000 test options
    start = time.perf_counter()
    black_scholes_call(x_test[:, 0], 1.0, x_test[:, 1], x_test[:, 2], x_test[:, 3])
    t_bs = time.perf_counter() - start

    start = time.perf_counter()
    model.predict(scaler.transform(x_test))
    t_nn = time.perf_counter() - start

    n_timed, n_paths = 200, 100_000
    start = time.perf_counter()
    for row in x_test[:n_timed]:
        monte_carlo_call(row[0], 1.0, row[1], row[2], row[3], n_paths, rng)
    t_mc = (time.perf_counter() - start) / n_timed * N_TEST    # extrapolated to 20,000 options

    print(f"\n--- Time to price {N_TEST:,} options ---")
    print(f"Black-Scholes (closed form)       = {t_bs * 1e3:9.1f} ms")
    print(f"Neural network                    = {t_nn * 1e3:9.1f} ms")
    print(f"Monte Carlo ({n_paths:,} paths each) = {t_mc * 1e3:9.0f} ms  (extrapolated from {n_timed} options)")
    print(f"Neural network vs Monte Carlo: {t_mc / t_nn:,.0f}x faster")

    # --- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.loglog(path_counts, mc_std, "o-", label="Monte Carlo std error")
    ax1.loglog(path_counts, mc_std[0] * np.sqrt(path_counts[0] / np.array(path_counts)),
               "--", color="grey", label=r"$1/\sqrt{N}$")
    ax1.set_xlabel("Number of paths N")
    ax1.set_ylabel("Standard error")
    ax1.set_title("Monte Carlo convergence")
    ax1.legend()
    ax2.hist(err * 100, bins=80, color="tab:blue")
    ax2.set_xlabel("Neural network error (% of strike)")
    ax2.set_ylabel("Number of test options")
    ax2.set_title("Neural network pricing error")
    fig.tight_layout()
    fig.savefig("ml_pricing.png", dpi=150)
    print("\nSaved ml_pricing.png")


if __name__ == "__main__":
    main()
