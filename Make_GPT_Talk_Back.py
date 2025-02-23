import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Define GPT Model
class GPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)

        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_embedding = nn.Embedding(context_length, model_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(model_dim, num_heads) for _ in range(num_blocks)]
        )
        self.layer_norm = nn.LayerNorm(model_dim)
        self.fc_out = nn.Linear(model_dim, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch_size, context_length = context.shape
        token_embedded = self.token_embedding(context)
        positions = torch.arange(context_length, device=context.device).expand(batch_size, context_length)
        positional_embedded = self.positional_embedding(positions)

        embedded = token_embedded + positional_embedded
        transformer_out = self.transformer_blocks(embedded)
        normed_out = self.layer_norm(transformer_out)
        logits = self.fc_out(normed_out)

        return logits

# ✅ Define Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.attention = MultiHeadedSelfAttention(model_dim, num_heads)
        self.linear_network = VanillaNeuralNetwork(model_dim)
        self.first_norm = nn.LayerNorm(model_dim)
        self.second_norm = nn.LayerNorm(model_dim)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        embedded = embedded + self.attention(self.first_norm(embedded))  # Skip Connection
        embedded = embedded + self.linear_network(self.second_norm(embedded))  # Skip Connection
        return embedded

# ✅ Define Multi-Headed Self Attention
class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.att_heads = nn.ModuleList([
            SingleHeadAttention(model_dim, model_dim // num_heads) for _ in range(num_heads)
        ])

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(embedded) for head in self.att_heads]
        return torch.cat(head_outputs, dim=2) 

# ✅ Define Single Head Attention
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

        # Causal Masking
        lower_triangular = torch.tril(torch.ones_like(scores))
        mask = lower_triangular == 0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = F.softmax(scores, dim=2)

        return scores @ v  

# ✅ Define Feedforward Network
class VanillaNeuralNetwork(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.up_projection = nn.Linear(model_dim, model_dim * 4)
        self.relu = nn.ReLU()
        self.down_projection = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(0.2) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

# ✅ Define Solution Class for Text Generation
class Solution:
    def generate(self, model, new_chars: int, context: torch.Tensor, context_length: int, int_to_char: dict) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        generated_tokens = context.tolist()[0]

        for _ in range(new_chars):
            context_tensor = torch.tensor([generated_tokens[-context_length:]], dtype=torch.long)
            logits = model(context_tensor)[:, -1, :]  # Get logits for last token
            probabilities = torch.softmax(logits, dim=-1)
            generator.set_state(initial_state)
            next_token = torch.multinomial(probabilities, 1, generator=generator).item()
            generated_tokens.append(next_token)

        return ''.join(int_to_char.get(token, '') for token in generated_tokens)

# ✅ Initialize GPT Model
model = GPT(104, 128, 252, 6, 6)

# ✅ Define Character Mapping
int_to_char = {
    0: '\n', 1: ' ', 2: '!', 3: '"', 5: '%', 6: '&', 7: "'", 8: '(', 9: ')', 10: '*', 11: '+',
    12: ',', 13: '-', 14: '.', 15: '/', 16: '0', 17: '1', 18: '2', 19: '3', 20: '4', 21: '5',
    22: '6', 23: '7', 24: '8', 25: '9', 26: ':', 27: ';', 28: '?', 29: 'A', 30: 'B', 31: 'C',
    32: 'D', 33: 'E', 34: 'F', 35: 'G', 36: 'H', 37: 'I', 38: 'J', 39: 'K', 40: 'L', 41: 'M',
    42: 'N', 43: 'O', 44: 'P', 45: 'Q', 46: 'R', 47: 'S', 48: 'T', 49: 'U', 50: 'V', 51: 'W',
    52: 'X', 53: 'Y', 54: 'Z', 55: '[', 56: ']', 57: '_', 58: 'a', 59: 'b', 60: 'c', 61: 'd',
    62: 'e', 63: 'f', 64: 'g', 65: 'h', 66: 'i', 67: 'j', 68: 'k', 69: 'l', 70: 'm', 71: 'n',
    72: 'o', 73: 'p', 74: 'q', 75: 'r', 76: 's', 77: 't', 78: 'u', 79: 'v', 80: 'w', 81: 'x',
    82: 'y', 83: 'z', 84: '{', 85: '|', 86: '}', 87: 'à', 88: 'á', 89: 'è', 90: 'é', 91: 'ë',
    92: 'ñ', 93: 'ó', 94: 'ú', 95: '\u2005', 96: '–', 97: '—', 98: '‘', 99: '’', 100: '“',
    101: '”', 102: '…', 103: '\u205f'
}

# ✅ Run Test Case
def test_generate_one_character():
    context = torch.zeros((1, 1), dtype=torch.long)
    new_chars = 1
    solution = Solution()
    result = solution.generate(model, new_chars, context, 128, int_to_char)
    print("✅ Test Passed: Generate One Character ->", result)

test_generate_one_character()
