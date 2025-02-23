import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        """
        Implements a single-head self-attention mechanism.

        Args:
        embedding_dim (int): The input embedding size.
        attention_dim (int): The attention head size.
        """
        super().__init__()
        torch.manual_seed(0)  # Ensuring reproducibility
        
        # Key, Query, and Value projection layers (bias=False per Transformer convention)
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of self-attention.

        Args:
        embedded (torch.Tensor): Input tensor of shape (batch_size, context_length, embedding_dim).

        Returns:
        torch.Tensor: Attention-weighted values of shape (batch_size, context_length, attention_dim).
        """
        batch_size, context_length, _ = embedded.shape

        # Compute Key, Query, and Value matrices
        K = self.key(embedded)  # (B, T, A)
        Q = self.query(embedded)  # (B, T, A)
        V = self.value(embedded)  # (B, T, A)

        # Compute attention scores (scaled dot-product attention)
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # (B, T, T)

        # Scale scores by sqrt(attention_dim)
        scale_factor = torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32))
        attention_scores = attention_scores / scale_factor

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, T, T)

        # Multiply attention weights with the Value matrix
        attended_values = torch.bmm(attention_weights, V)  # (B, T, A)

        return attended_values


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        """
        Implements Multi-Headed Self-Attention using multiple SingleHeadAttention modules.

        Args:
        embedding_dim (int): The input embedding size.
        attention_dim (int): The output attention dimension.
        num_heads (int): Number of attention heads (attention_dim must be divisible by num_heads).
        """
        super().__init__()
        torch.manual_seed(0)

        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads  # Size per head

        # Create multiple attention heads
        self.attention_heads = nn.ModuleList([
            SingleHeadAttention(embedding_dim, self.head_dim) for _ in range(num_heads)
        ])

        # Final linear layer to project concatenated heads into attention_dim
        self.fc_out = nn.Linear(attention_dim, attention_dim)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-headed self-attention.

        Args:
        embedded (torch.Tensor): Input tensor of shape (batch_size, context_length, embedding_dim).

        Returns:
        torch.Tensor: Multi-head attention output of shape (batch_size, context_length, attention_dim).
        """
        # Compute attention for each head
        attention_outputs = [head(embedded) for head in self.attention_heads]

        # Concatenate along the last dimension
        concatenated = torch.cat(attention_outputs, dim=-1)  # (B, T, attention_dim)

        # Project concatenated output back to the attention dimension
        output = self.fc_out(concatenated)

        return torch.round(output, decimals=4)  # Round output to 4 decimal places

# Example usage
embedding_dim = 2
attention_dim = 3
num_heads = 1
model = MultiHeadedSelfAttention(embedding_dim, attention_dim, num_heads)

# Example input tensor
embedded = torch.tensor([
    [[-1.4381, 0.1232], [-0.1080, 0.3458]],
    [[0.1929, -0.8567], [-0.1160, 1.2547]]
], dtype=torch.float32)

# Get model prediction
output = model(embedded)

# Print output
print("Multi-Headed Self-Attention Output:\n", output)
