import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)

        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_embedding = nn.Embedding(context_length, model_dim)

        self.transformer_blocks = nn.Sequential(
            *[self.TransformerBlock(model_dim, num_heads) for _ in range(num_blocks)]
        )

        self.layer_norm = nn.LayerNorm(model_dim)
        self.fc_out = nn.Linear(model_dim, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(0)

        batch_size, context_length = context.shape
        token_embedded = self.token_embedding(context)

        positions = torch.arange(context_length, device=context.device).expand(batch_size, context_length)
        positional_embedded = self.positional_embedding(positions)

        embedded = token_embedded + positional_embedded
        transformer_out = self.transformer_blocks(embedded)
        normed_out = self.layer_norm(transformer_out)
        logits = self.fc_out(normed_out)

        return torch.round(F.softmax(logits, dim=-1), decimals=4)

    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    torch.manual_seed(0)
                    self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.value_gen = nn.Linear(model_dim, head_size, bias=False)
                
                def forward(self, embedded: torch.Tensor) -> torch.Tensor:
                    k = self.key_gen(embedded)
                    q = self.query_gen(embedded)
                    v = self.value_gen(embedded)

                    scores = q @ torch.transpose(k, 1, 2) 
                    attention_dim = k.shape[-1]
                    scores = scores / (attention_dim ** 0.5)

                    lower_triangular = torch.tril(torch.ones_like(scores))
                    mask = lower_triangular == 0
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim=2)

                    return scores @ v
                
            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                torch.manual_seed(0)
                self.att_heads = nn.ModuleList([
                    self.SingleHeadAttention(model_dim, model_dim // num_heads) for _ in range(num_heads)
                ])

            def forward(self, embedded: torch.Tensor) -> torch.Tensor:
                head_outputs = [head(embedded) for head in self.att_heads]
                return torch.cat(head_outputs, dim=2) 
        
        class VanillaNeuralNetwork(nn.Module):
            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                torch.manual_seed(0)
                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.linear_network = self.VanillaNeuralNetwork(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: torch.Tensor) -> torch.Tensor:
            torch.manual_seed(0)
            embedded = embedded + self.attention(self.first_norm(embedded)) 
            embedded = embedded + self.linear_network(self.second_norm(embedded))  
            return embedded


# Example usage
vocab_size = 5
context_length = 5
model_dim = 16
num_blocks = 4
num_heads = 4
model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)

# Example tokenized input (context of words converted to indices)
context = torch.tensor([[0, 1, 2, 3, 1]])  # 'With', 'great', 'power', 'comes', 'great'

# Get model prediction
output = model(context)

# Print output
print("GPT Model Output:\n", output)
