import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        """
        Computes the derivative of the loss function with respect to a specific weight.
        
        Args:
        model_prediction (NDArray[np.float64]): Model predictions of shape (n,).
        ground_truth (NDArray[np.float64]): True values of shape (n,).
        N (int): Number of samples.
        X (NDArray[np.float64]): Input dataset of shape (n, 3).
        desired_weight (int): Index of the weight to compute derivative for.
        
        Returns:
        float: The computed derivative.
        """
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the linear regression model's predictions based on input dataset X and weights.
        
        Args:
        X (NDArray[np.float64]): Input dataset of shape (n, 3).
        weights (NDArray[np.float64]): Model weights of shape (3,).
        
        Returns:
        NDArray[np.float64]: Predicted values of shape (n,).
        """
        return np.matmul(X, weights)  # Matrix multiplication for prediction

    def train_model(self, X: NDArray[np.float64], Y: NDArray[np.float64], num_iterations: int, initial_weights: NDArray[np.float64], learning_rate: float = 0.01) -> NDArray[np.float64]:
        """
        Trains the linear regression model using gradient descent.

        Args:
        X (NDArray[np.float64]): Input dataset of shape (n, 3).
        Y (NDArray[np.float64]): True values of shape (n,).
        num_iterations (int): Number of iterations for training.
        initial_weights (NDArray[np.float64]): Initial weight values of shape (3,).
        learning_rate (float): Step size for gradient descent (default=0.01).

        Returns:
        NDArray[np.float64]: Final weights after training, rounded to 5 decimal places.
        """
        weights = initial_weights.copy()  # Avoid modifying the original weights
        N = len(X)  # Number of samples

        for _ in range(num_iterations):
            predictions = self.get_model_prediction(X, weights)

            # Compute gradient for each weight and update using gradient descent
            for i in range(len(weights)):
                gradient = self.get_derivative(predictions, Y, N, X, i)
                weights[i] -= learning_rate * gradient

        return np.round(weights, 5)  # Return weights rounded to 5 decimal places

# Example usage
solution = Solution()

X = np.array([[1, 2, 3], [1, 1, 1]])  # Input dataset
Y = np.array([6, 3])  # Ground truth values
num_iterations = 10  # Number of training iterations
initial_weights = np.array([0.2, 0.1, 0.6])  # Initial weight values

final_weights = solution.train_model(X, Y, num_iterations, initial_weights)

print("Final Weights:", final_weights)
