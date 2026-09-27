import numpy as np

"""
DESCRIPTION
Use your favorite programming language to simulate Arithmetic Brownian Motion, 
Geometric Brownian Motion, Ornstein-Uhlenbeck, and square-root processes.
For each process, use the same set of parameters to simulate it 500 times and compute their mean
and variance at t = 1. The parameters to be used are: κ = 0.5, μ = 0.7, σ = 0.2, X0 = 0.1. Let
t ∈ [0, 1] with intervals ∆t = 0.01
"""

"""
COMMENTS
- simulate 500 independent paths
- each path starts at t = 0, stops at t = 1, steps ∆t = 0.01; meaning 100 time steps per path
- from each path take the final value X1
- compute mean and variance of those 500 terminal values
"""

MEAN_REVERSION_SPEED = 0.5 # kappa
DRIFT = 0.7 # mu
VOLATILITY = 0.2 # sigma
STEP = 0.01
INITIAL_VALUE = 0.1 # x0

NUM_ITERATIONS = 500

def simulate_arithmetic_brownian_motion(mu: float, sigma: float, x_n: float):
    Z_n = np.random.normal(0,1)
    delta_W = np.sqrt(STEP) * Z_n
    return x_n + mu * STEP + sigma * delta_W

# return: x_n+1
def simulate_geometric_brownian_motion(mu: float, sigma: float, x_n: float) -> float:
    Z_n = np.random.normal(0,1)
    delta_W = np.sqrt(STEP) * Z_n
    return x_n + mu * x_n * STEP + sigma * x_n * delta_W

def simulate_ornstein_uhlenbeck(kappa: float, mu: float, sigma: float, x_n: float):
    Z_n = np.random.normal(0,1)
    delta_W = np.sqrt(STEP) * Z_n
    return x_n + kappa * (mu - x_n) * STEP + sigma * delta_W

def simulate_square_root(kappa: float, mu: float, sigma: float, x_n: float):
    Z_n = np.random.normal(0,1)
    delta_W = np.sqrt(STEP) * Z_n
    # Euler-Maruyama here requires x_n > 0 so we must clamp it to 0 at the lowest
    return x_n + kappa * (mu - x_n) * STEP + sigma * np.sqrt(x_n) * delta_W

def calculate_mean(nums: list):
    return sum(nums) / len(nums)

def calculate_variance(nums: list):
    return np.var(nums)

if __name__ == "__main__":
    n_steps = 1 / STEP

    # Arithmetic Brownian Motion Simulation
    abm_outputs = []
    
    for i in range(NUM_ITERATIONS):
        value = INITIAL_VALUE

        for _ in range(int(n_steps)):
            value = simulate_arithmetic_brownian_motion(DRIFT, VOLATILITY, value)

        abm_outputs.append(value)

    abm_mean = calculate_mean(abm_outputs)
    abm_variance = calculate_variance(abm_outputs)
    
    # Geometric Brownian Motion Simulation
    gbm_outputs = []
    
    for i in range(NUM_ITERATIONS):
        value = INITIAL_VALUE

        for _ in range(int(n_steps)):
            value = simulate_geometric_brownian_motion(DRIFT, VOLATILITY, value)

        gbm_outputs.append(value)

    gbm_mean = calculate_mean(gbm_outputs)
    gbm_variance = calculate_variance(gbm_outputs)
    
    # Ornstein Uhlenbeck Motion Simulation
    ou_outputs = []
    
    for i in range(NUM_ITERATIONS):
        value = INITIAL_VALUE

        for _ in range(int(n_steps)):
            value = simulate_ornstein_uhlenbeck(MEAN_REVERSION_SPEED, DRIFT, VOLATILITY, value)

        ou_outputs.append(value)

    ou_mean = calculate_mean(ou_outputs)
    ou_variance = calculate_variance(ou_outputs)

    # Square Root Simulation
    sqrt_outputs = []
    
    for i in range(NUM_ITERATIONS):
        value = INITIAL_VALUE

        for _ in range(int(n_steps)):
            value = simulate_square_root(MEAN_REVERSION_SPEED, DRIFT, VOLATILITY, value)

        sqrt_outputs.append(value)

    sqrt_mean = calculate_mean(sqrt_outputs)
    sqrt_variance = calculate_variance(sqrt_outputs)

    print("Arithmetic Brownian Motion Results: ", "\nMean: ", abm_mean, "\nVariance ", abm_variance)
    print("\nGeometric Brownian Motion Results: ", "\nMean: ", gbm_mean, "\nVariance ", gbm_variance)
    print("\nOrnstein Uhlenbeck Results: ", "\nMean: ", ou_mean, "\nVariance ", ou_variance )
    print("\nSquare Root Results: ", "\nMean: ", sqrt_mean, "\nVariance ", sqrt_variance)