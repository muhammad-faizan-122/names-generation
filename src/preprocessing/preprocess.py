import torch
import random
from typing import List, Dict, Tuple


class Preprocessor:
    """Handles character encoding and decoding."""

    def __init__(self, words: List[str]):
        self.stoi = self._create_encoding(words)
        self.itos = self._create_decoding()

    def _create_encoding(self, words: List[str]) -> Dict[str, int]:
        """Creates a character-to-index mapping."""
        chars = sorted(set("".join(words)))
        stoi = {s: i + 1 for i, s in enumerate(chars)}
        stoi["."] = 0  # End-of-word token
        return stoi

    def _create_decoding(self) -> Dict[int, str]:
        """Creates a index-to-character mapping."""
        itos = {i: s for s, i in self.stoi.items()}
        return itos

    def encode(self, word: str) -> List[int]:
        """Encodes a word into a list of integers."""
        return [self.stoi[ch] for ch in word]

    def decode(self, indices: List[int]) -> str:
        """Decodes a list of integers into a word."""
        return "".join([self.itos[i] for i in indices])


class DatasetProcessor:
    """Processes and splits the dataset into training, validation, and test sets."""

    def __init__(self, words: List[str], preprocessor: Preprocessor, block_size: int):
        self.words = words
        self.preprocessor = preprocessor
        self.block_size = block_size

    def _build_dataset(self, words: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes words into tensor format for training."""
        X, Y = [], []

        for word in words:
            context = [0] * self.block_size  # Start with padding
            for ch in word + ".":
                ix = self.preprocessor.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]  # Shift context

        X_tensor, Y_tensor = torch.tensor(X, dtype=torch.long), torch.tensor(
            Y, dtype=torch.long
        )
        print(f"Dataset shape: {X_tensor.shape}, {Y_tensor.shape}")
        return X_tensor, Y_tensor

    def split_dataset(self, train_ratio: float, test_ratio: float) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Splits data into training, validation, and test sets."""
        random.shuffle(self.words)

        n_train = int(train_ratio * len(self.words))
        n_test = int((train_ratio + test_ratio) * len(self.words))

        X_train, Y_train = self._build_dataset(self.words[:n_train])
        X_dev, Y_dev = self._build_dataset(self.words[n_train:n_test])
        X_test, Y_test = self._build_dataset(self.words[n_test:])

        return X_train, Y_train, X_dev, Y_dev, X_test, Y_test
