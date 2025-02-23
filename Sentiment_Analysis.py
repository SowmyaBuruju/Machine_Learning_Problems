import torch
import torch.nn as nn

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)  # Ensuring reproducibility
        
        # Define the architecture
        self.embedding = nn.Embedding(vocabulary_size, 16)  # Embedding layer (vocab_size → 16)
        self.fc = nn.Linear(16, 1)  # Single neuron output layer (16 → 1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for probability output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1), with values between 0 and 1.
        """
        x = self.embedding(x.long())  # Convert input indices to embeddings
        x = torch.mean(x, dim=1)  # Compute the average embeddings (Bag of Words approach)
        x = self.fc(x)  # Apply the linear layer
        x = self.sigmoid(x)  # Apply sigmoid activation
        
        return x

# Example usage
vocabulary_size = 170000  # Number of unique words in vocabulary
model = Solution(vocabulary_size)

# Example input (word indices representing tokenized sentences)
x = torch.tensor([
    [2, 7, 14, 8, 0, 0, 0, 0, 0, 0, 0, 0],  # "The movie was okay"
    [1, 4, 12, 3, 10, 5, 15, 11, 6, 9, 13, 7]  # "I don't think anyone should ever waste their money on this movie"
])

# Get model prediction
output = model(x)

# Print output
print("Model Prediction:\n", output)
