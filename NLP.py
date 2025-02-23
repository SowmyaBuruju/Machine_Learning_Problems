import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> torch.Tensor:
        """
        Encodes a dataset of strings into a (2N × T) integer tensor.

        Args:
        positive (List[str]): List of positive sentiment sentences.
        negative (List[str]): List of negative sentiment sentences.

        Returns:
        torch.Tensor: Encoded integer tensor representation of the dataset.
        """
        # Combine all sentences and tokenize into unique words
        all_sentences = positive + negative
        words = sorted(set(word for sentence in all_sentences for word in sentence.split()))
        
        # Create a dictionary mapping words to indices (lexicographical order starting from 1)
        word_to_index = {word: idx + 1 for idx, word in enumerate(words)}

        # Function to encode sentences
        def encode_sentence(sentence: str):
            return torch.tensor([word_to_index[word] for word in sentence.split()], dtype=torch.float32)

        # Encode sentences
        encoded_positive = [encode_sentence(sentence) for sentence in positive]
        encoded_negative = [encode_sentence(sentence) for sentence in negative]

        # Combine and pad sequences (ensuring batch_first=True)
        all_encoded = encoded_positive + encoded_negative
        padded_tensor = pad_sequence(all_encoded, batch_first=True, padding_value=0.0)

        return padded_tensor

# Example usage
solution = Solution()

positive = ["Dogecoin to the moon"]
negative = ["I will short Tesla today"]

encoded_dataset = solution.get_dataset(positive, negative)

# Display the encoded tensor
print("Encoded Tensor:\n", encoded_dataset)
