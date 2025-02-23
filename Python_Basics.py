import torch
import torch.nn.functional as F

class Solution:
    def reshape_tensor(self, to_reshape: torch.Tensor) -> torch.Tensor:
        """
        Reshapes an MxN tensor into a ((M*N)//2)x2 tensor.
        """
        M, N = to_reshape.shape
        return to_reshape.reshape((M * N) // 2, 2)

    def average_columns(self, to_avg: torch.Tensor) -> torch.Tensor:
        """
        Computes the average of every column in a tensor.
        """
        return torch.mean(to_avg, dim=0)

    def concatenate_tensors(self, cat_one: torch.Tensor, cat_two: torch.Tensor) -> torch.Tensor:
        """
        Combines an MxN tensor and an MxM tensor into an Mx(M+N) tensor.
        """
        return torch.cat((cat_one, cat_two), dim=1)

    def compute_mse_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculates the mean squared error (MSE) loss between a prediction and target tensor.
        """
        return F.mse_loss(prediction, target).item()

# Example usage
solution = Solution()

# Example 1: Reshape
to_reshape = torch.tensor([
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0]
])
reshaped = solution.reshape_tensor(to_reshape)

# Example 2: Average Columns
to_avg = torch.tensor([
    [0.8088, 1.2614, -1.4371],
    [-0.0056, -0.2050, -0.7201]
])
column_avg = solution.average_columns(to_avg)

# Example 3: Concatenate
cat_one = torch.tensor([
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0]
])
cat_two = torch.tensor([
    [1.0, 1.0],
    [1.0, 1.0]
])
concatenated = solution.concatenate_tensors(cat_one, cat_two)

# Example 4: Compute MSE Loss
prediction = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
mse_loss = solution.compute_mse_loss(prediction, target)

# Print Outputs
print("Reshaped Tensor:\n", reshaped)
print("Column Averages:\n", column_avg)
print("Concatenated Tensor:\n", concatenated)
print("Mean Squared Error Loss:", mse_loss)
