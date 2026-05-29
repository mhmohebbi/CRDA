import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def generate_anm_data(n_samples=1000, 
                      x_dist="uniform", 
                      noise_dist="gaussian", 
                      function="nonlinear", 
                      seed=None):
    """
    Generate data from an Additive Noise Model (ANM): Y = f(X) + N
    where X is 3-dimensional (X1, X2, X3)

    Parameters:
        n_samples (int): Number of data points
        x_dist (str): Distribution of X ("uniform" or "gaussian")
        noise_dist (str): Distribution of noise ("gaussian" or "uniform")
        function (str): Function type ("linear", "nonlinear", or custom function)
        seed (int): Random seed for reproducibility

    Returns:
        X (np.ndarray): Shape (n_samples, 3) - three features
        Y (np.ndarray): Shape (n_samples,) - target variable
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Step 1: Sample X (3 features)
    if x_dist == "uniform":
        X = np.random.uniform(-2, 2, size=(n_samples, 3))
    elif x_dist == "gaussian":
        X = np.random.normal(0, 1, size=(n_samples, 3))
    else:
        raise ValueError("Unknown x_dist")

    # Step 2: Define f(X) for 3-dimensional input
    if function == "linear":
        # Linear combination of the 3 features
        f_X = 2 * X[:, 0] + 1.5 * X[:, 1] - 0.8 * X[:, 2] + 1
    elif function == "nonlinear":
        # Nonlinear function of the 3 features
        f_X = X[:, 0]**2 + X[:, 1] * X[:, 2]
    elif callable(function):
        f_X = function(X)
    else:
        raise ValueError("Unknown function")

    # Step 3: Sample noise N
    if noise_dist == "gaussian":
        noise = np.random.normal(0, 0.5, size=n_samples)
    elif noise_dist == "uniform":
        noise = np.random.uniform(-0.5, 0.5, size=n_samples)
    else:
        raise ValueError("Unknown noise_dist")

    # Step 4: Generate Y
    Y = f_X + noise

    return X, Y

# Example usage
X, Y = generate_anm_data(n_samples=50000, function="nonlinear", seed=42)
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"First 5 X samples:\n{X[:5]}")
print(f"First 5 Y samples: {Y[:5]}")

# save the data to a csv file
df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
df["target"] = Y
df.to_csv("./data/anm_data.csv", index=False)

# Optional: Plot (showing X1 vs Y as an example)
import matplotlib.pyplot as plt
plt.scatter(X[:, 2], Y, alpha=0.5, s=10)
plt.title("Additive Noise Model Data (Y = f(X1, X2, X3) + N)")
plt.xlabel("X1")
plt.ylabel("Y")
plt.grid(True)
plt.show()
