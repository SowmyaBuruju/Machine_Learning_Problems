import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)  # Set seed for reproducibility
        
        # Define the architecture
        self.fc1 = nn.Linear(28 * 28, 512)  # First Linear layer (input -> 512 neurons)
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(p=0.2)  # Dropout layer with probability p=0.2
        self.fc2 = nn.Linear(512, 10)  # Output layer (512 -> 10 neurons)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
        images (torch.Tensor): Input tensor of shape (batch_size, 28*28).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, 10) with probabilities for each digit.
        """
        x = self.fc1(images)  # Linear transformation
        x = self.relu(x)  # ReLU activation
        x = self.dropout(x)  # Dropout layer
        x = self.fc2(x)  # Second Linear transformation
        x = self.sigmoid(x)  # Sigmoid activation
        return x

# Example usage
model = DigitClassifier()

# Example input: A single flattened 28x28 image (batch_size=1)
example_input = torch.rand((1, 28 * 28))  # Random values to simulate an image

# Get model prediction
output = model(example_input)

# Print output
print("Model Prediction:", output)
