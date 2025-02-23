import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        """
        Implements Multi-Headed Self-Attention using multiple attention heads.

        Args:
        model_dim (int): The input embedding size and attention output size.
        num_heads (int): Number of attention heads (model_dim must be divisible by num_heads).
        """
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads  # Dimension per head

        # Projection layers for Q, K, V (bias=False as per Transformer convention)
        self.qkv_proj = nn.Linear(model_dim, model_dim * 3, bias=False)
        self.fc_out = nn.Linear(model_dim, model_dim)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-headed self-attention.

        Args:
        embedded (torch.Tensor): Input tensor of shape (batch_size, context_length, model_dim).

        Returns:
        torch.Tensor: Attention output of shape (batch_size, context_length, model_dim).
        """
        batch_size, context_length, _ = embedded.shape

        # Compute Query, Key, and Value projections
        qkv = self.qkv_proj(embedded)  # (B, T, 3*model_dim)
        Q, K, V = torch.chunk(qkv, 3, dim=-1)  # Split into Q, K, V

        # Reshape for multi-heads
        Q = Q.view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        K = K.view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        V = V.view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)

        # Compute attention scores (scaled dot-product attention)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, H, T, T)

        # Multiply attention weights with the Value matrix
        attended_values = torch.matmul(attention_weights, V)  # (B, H, T, head_dim)

        # Reshape output back to (batch_size, context_length, model_dim)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, context_length, self.num_heads * self.head_dim)

        # Final linear projection
        output = self.fc_out(attended_values)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        """
        Implements a Transformer block containing:
        - Layer Normalization
        - Multi-Headed Self-Attention
        - Feedforward Neural Network (MLP)
        - Residual Connections

        Args:
        model_dim (int): The model embedding and attention dimension.
        num_heads (int): Number of self-attention heads.
        """
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.model_dim = model_dim
        self.num_heads = num_heads

        # Layer Normalization before Attention and before Feedforward
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        # Multi-Headed Self-Attention
        self.attention = MultiHeadedSelfAttention(model_dim, num_heads)

        # Feedforward Network (MLP)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),  # Expand model_dim by a factor of 4
            nn.ReLU(),
            nn.Linear(4 * model_dim, model_dim)  # Project back to model_dim
        )

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer block.

        Args:
        embedded (torch.Tensor): Input tensor of shape (batch_size, context_length, model_dim).

        Returns:
        torch.Tensor: Transformer block output of shape (batch_size, context_length, model_dim).
        """
        # Layer Norm -> Multi-Headed Attention -> Residual Connection
        normed_embedded = self.norm1(embedded)
        attention_out = self.attention(normed_embedded)
        residual1 = embedded + attention_out  # Skip connection

        # Layer Norm -> Feedforward Network -> Residual Connection
        normed_residual = self.norm2(residual1)
        feedforward_out = self.feedforward(normed_residual)
        residual2 = residual1 + feedforward_out  # Skip connection

        return torch.round(residual2, decimals=4)  # Round output to 4 decimal places


# Example usage
model_dim = 4
num_heads = 2
model = TransformerBlock(model_dim, num_heads)

# Example input tensor
embedded = torch.tensor([
    [[-0.6775, 1.4919, 0.8760, 0.9440],
     [0.4388, 0.5290, -0.2510, -1.2941]],
    [[2.0576, 0.6107, -0.7395, -0.2010],
     [0.4728, 1.0233, -0.9400, 2.0409]]
], dtype=torch.float32)

# Get model prediction
output = model(embedded)

# Print output
print("Transformer Block Output:\n", output)
