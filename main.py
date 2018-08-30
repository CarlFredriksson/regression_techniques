import os
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import linear_regression_1D
from nn_regression import nn_regression_1D

def create_output_dir():
    """Create output dir if it does not exist."""
    cwd = os.getcwd()
    output_dir_path = os.path.join(cwd, "output")
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

def generate_random_data(a=1, b=2, non_linear=False, noise_stdev=1):
    """Generate data with random noise."""
    X = np.expand_dims(np.linspace(-5, 5, num=50), axis=1)
    Y = a * X + b
    if non_linear:
        Y = a * X**2 + b
    noise = np.random.normal(0, noise_stdev, size=X.shape)
    Y += noise
    Y = Y.astype("float32")

    return X, Y

def plot_data(X, Y, plot_name):
    """Save generated data."""
    plt.scatter(X, Y, color="blue")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("output/" + plot_name, bbox_inches="tight")
    plt.clf()

def plot_results(X, Y, y_estimate, plot_name):
    """Save resulting plot."""
    plt.scatter(X, Y, color="blue")
    plt.plot(X, y_estimate, color="red")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("output/" + plot_name, bbox_inches="tight")
    plt.clf()

LEARNING_RATE = 0.0001
NUM_ITERATIONS = 50000

create_output_dir()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Linear data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
X_train, Y_train = generate_random_data()
X_test, Y_test = generate_random_data()
plot_data(X_train, Y_train, "data_linear_train.png")
plot_data(X_test, Y_test, "data_linear_test.png")

# Linear regression
y_estimate, J_train, J_test = linear_regression_1D(X_train, Y_train, X_test, Y_test, LEARNING_RATE, NUM_ITERATIONS)
plot_results(X_train, Y_train, y_estimate, "linear_regression_train_1.png")
plot_results(X_test, Y_test, y_estimate, "linear_regression_test_1.png")
print("Linear regression on linear data - J_train: " + str(J_train) + ", J_test: " + str(J_test))

# NN regression
y_estimate, J_train, J_test = nn_regression_1D(X_train, Y_train, X_test, Y_test, LEARNING_RATE, NUM_ITERATIONS)
plot_results(X_train, Y_train, y_estimate, "nn_regression_train_1.png")
plot_results(X_test, Y_test, y_estimate, "nn_regression_test_1.png")
print("NN regression on linear data - J_train: " + str(J_train) + ", J_test: " + str(J_test))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Non-linear data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
X_train, Y_train = generate_random_data(non_linear=True)
X_test, Y_test = generate_random_data(non_linear=True)
plot_data(X_train, Y_train, "data_non_linear_train.png")
plot_data(X_test, Y_test, "data_non_linear_test.png")

# Linear regression
y_estimate, J_train, J_test = linear_regression_1D(X_train, Y_train, X_test, Y_test, LEARNING_RATE, NUM_ITERATIONS)
plot_results(X_train, Y_train, y_estimate, "linear_regression_train_2.png")
plot_results(X_test, Y_test, y_estimate, "linear_regression_test_2.png")
print("Linear regression on non-linear data - J_train: " + str(J_train) + ", J_test: " + str(J_test))

# NN regression
y_estimate, J_train, J_test = nn_regression_1D(X_train, Y_train, X_test, Y_test, LEARNING_RATE, NUM_ITERATIONS)
plot_results(X_train, Y_train, y_estimate, "nn_regression_train_2.png")
plot_results(X_test, Y_test, y_estimate, "nn_regression_test_2.png")
print("NN regression on non-linear data - J_train: " + str(J_train) + ", J_test: " + str(J_test))
