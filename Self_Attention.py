import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        """
        Implements self-attention mechanism for transformers.

        Args:
        embedding_dim (int): The input embedding size.
        attention_dim (int): The attention head size.
        """
        super().__init__()
        torch.manual_seed(0)  # Ensuring reproducibility
        
        # Linear layers for Key, Query, and Value (bias=False as per Transformer convention)
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the self-attention mechanism.

        Args:
        embedded (torch.Tensor): Input tensor of shape (batch_size, context_length, embedding_dim).

        Returns:
        torch.Tensor: Attention-weighted values of shape (batch_size, context_length, attention_dim).
        """
        # Extract dimensions
        batch_size, context_length, _ = embedded.shape

        # Compute Key, Query, and Value matrices
        K = self.key(embedded)  # (B, T, A)
        Q = self.query(embedded)  # (B, T, A)
        V = self.value(embedded)  # (B, T, A)

        # Compute attention scores (scaled dot-product attention)
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # (B, T, T)

        # Scale scores by sqrt(attention_dim) for stability
        attention_scores = attention_scores / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, T, T)

        # Multiply attention weights with the Value matrix
        attended_values = torch.bmm(attention_weights, V)  # (B, T, A)

        return attended_values

# Example usage
embedding_dim = 2
attention_dim = 3
model = SelfAttention(embedding_dim, attention_dim)

# Example input tensor
embedded = torch.tensor([
    [[-1.4381, 0.1232], [-0.1080, 0.3458]],
    [[0.1929, -0.8567], [-0.1160, 1.2547]]
], dtype=torch.float32)

# Get model prediction
output = model(embedded)

# Print output
print("Self-Attention Output:\n", output)
