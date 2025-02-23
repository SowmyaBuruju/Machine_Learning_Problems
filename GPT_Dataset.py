import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Generates batches of training data from a raw text dataset for a GPT-like model.

        Args:
        raw_dataset (str): The text dataset to generate examples from.
        context_length (int): The number of tokens used as input context.
        batch_size (int): The number of sequences to generate.

        Returns:
        Tuple[List[List[str]], List[List[str]]]: X (input sequences), Y (target sequences)
        """
        torch.manual_seed(0)  # Set seed for reproducibility
        
        # Tokenize the text into words
        tokens = raw_dataset.split()
        dataset_size = len(tokens)

        # Ensure the dataset is large enough
        if dataset_size <= context_length:
            raise ValueError("The raw dataset must be longer than the context length.")

        # Generate `batch_size` different random starting indices
        start_indices = torch.randint(0, dataset_size - context_length, (batch_size,)).tolist()

        # Prepare input (X) and target (Y) sequences
        X = []
        Y = []

        for start_idx in start_indices:
            input_seq = tokens[start_idx : start_idx + context_length]
            target_seq = tokens[start_idx + 1 : start_idx + context_length + 1]
            
            X.append(input_seq)
            Y.append(target_seq)

        return X, Y

# Example usage
solution = Solution()

raw_dataset = "Hello darkness my old friend"
context_length = 3
batch_size = 2

X, Y = solution.batch_loader(raw_dataset, context_length, batch_size)

# Print the results
print("Input Sequences (X):", X)
print("Target Sequences (Y):", Y)
