import torch
import os
import requests
import random
from typing import List, Tuple

random.seed(42)


# 1. Dataset Management
def download_dataset(url: str, path: str, max_retries: int = 3) -> None:
    """Downloads a dataset if it's not already present locally."""
    if os.path.exists(path):
        print(f"File already exists: {path}")
        return

    print(f"Downloading dataset from {url}...")
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()  # Raises HTTPError for bad responses

            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)

            print(f"Download complete: {path}")
            return
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")

    print("Download failed after multiple attempts.")


def read_names(file_path: str) -> List[str]:
    """Reads names from the dataset file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            names = [line.strip() for line in file]
        print(f"Total names loaded: {len(names)}")
        return names
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []


# 2. Character Encoding
def encode_chars(words: List[str]) -> dict:
    """Creates a character-to-index mapping."""
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0  # End-of-word token
    return stoi


def decode_chars(stoi: dict) -> dict:
    """Creates an index-to-character mapping."""
    return {i: s for s, i in stoi.items()}


# 3. Dataset Preparation
def build_dataset(
    words: List[str], stoi: dict, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encodes words into tensor format for training."""
    X, Y = [], []

    for word in words:
        context = [0] * block_size  # Start with padding
        for ch in word + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # Shift context

    X_tensor, Y_tensor = torch.tensor(X, dtype=torch.long), torch.tensor(
        Y, dtype=torch.long
    )
    print(f"Dataset shape: {X_tensor.shape}, {Y_tensor.shape}")
    return X_tensor, Y_tensor


def split_dataset(
    words: List[str], stoi: dict, block_size: int, train_ratio: float, test_ratio: float
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Splits data into training, validation, and test sets."""
    random.shuffle(words)

    n_train = int(train_ratio * len(words))
    n_test = int((train_ratio + test_ratio) * len(words))

    X_train, Y_train = build_dataset(words[:n_train], stoi, block_size)
    X_dev, Y_dev = build_dataset(words[n_train:n_test], stoi, block_size)
    X_test, Y_test = build_dataset(words[n_test:], stoi, block_size)

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test


def get_dataset(
    path: str = "dataset/names.txt",
    block_size: int = 3,
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    dataset_url: str = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Downloads, loads, encodes, and splits the dataset into training, validation, and test sets.

    Returns:
        X_train, Y_train: Training set
        X_dev, Y_dev: Validation set
        X_test, Y_test: Test set
    """
    # Download dataset if not present
    download_dataset(dataset_url, path)

    # Load names from file
    names = read_names(path)
    if not names:
        raise RuntimeError("Failed to load dataset. Please check the file path.")

    # Character encoding
    stoi = encode_chars(names)

    # Split dataset
    return split_dataset(names, stoi, block_size, train_ratio, test_ratio)
