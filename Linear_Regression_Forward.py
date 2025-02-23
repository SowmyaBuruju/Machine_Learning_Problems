import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_model_prediction(self, X: NDArray, weights: NDArray) -> NDArray:
        """
        Calculates the model's predictions based on input dataset X and weights.
        
        Args:
        X (NDArray): Input dataset of shape (n, 3).
        weights (NDArray): Model weights of shape (3,).
        
        Returns:
        NDArray: Predicted values of shape (n,).
        """
        return np.matmul(X, weights)  # Matrix multiplication of X and weights

    def get_error(self, model_prediction: NDArray, ground_truth: NDArray) -> float:
        """
        Computes the Mean Squared Error (MSE) between model predictions and ground truth.
        
        Args:
        model_prediction (NDArray): Predicted values of shape (n,).
        ground_truth (NDArray): True values of shape (n,).
        
        Returns:
        float: Mean squared error.
        """
        return np.mean(np.square(model_prediction - ground_truth))  # MSE formula

# Example usage
solution = Solution()

# Example dataset (n=3, features=3)
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Example weights
weights = np.array([0.1, 0.2, 0.3])  

# Example ground truth values
ground_truth = np.array([1, 2, 3])  

# Get predictions
predictions = solution.get_model_prediction(X, weights)

# Calculate mean squared error
error = solution.get_error(predictions, ground_truth)

# Print outputs
print("Predictions:", predictions)
print("Mean Squared Error:", error)
