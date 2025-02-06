import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import requests
import random

random.seed(42)


def download_dataset(
    url: str = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
    path: str = "dataset/names.txt",
    max_retries: int = 3,
) -> None:
    """Download dataset if not present locally."""
    if os.path.exists(path):
        print(f"File already exists: {path}")
        return

    print(f"Dataset is downloading...")
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)
                print(f"Download complete: {path}")
                return
            else:
                print(f"Failed to download. HTTP Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")

    print("Download failed after multiple attempts.")


def read_names(fpath: str) -> list[str]:
    """Read names from the dataset file."""
    with open(fpath, "r") as file:
        names = [line.strip() for line in file]
    print(f"Total names: {len(names)}")
    return names


def encode_chars(words):
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    return stoi


def decode_chars(stoi):
    itos = {i: s for s, i in stoi.items()}
    return itos


def build_dataset(words, stoi, block_size):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y


def get_encoding_splitted_dataset(
    words,
    stoi,
    block_size,
    train_ratio,
    test_ratio,
):
    random.shuffle(words)

    n1 = int(train_ratio * len(words))
    n2 = int((train_ratio + test_ratio) * len(words))

    Xtr, Ytr = build_dataset(words[:n1], stoi, block_size=block_size)  # 80%
    Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size=block_size)  # 10%
    Xte, Yte = build_dataset(words[n2:], stoi, block_size=block_size)  # 10

    return Xtr, Ytr, Xdev, Ydev, Xte, Yte


def get_dataset(path, block_size=3, train_ratio=0.8, test_ratio=0.10):
    """return train/test/validation tensor dataset"""
    # download dataset if not exist locally
    download_dataset()
    # read names text files
    names = read_names(path)
    # character encoding (char 2 integer)
    stoi = encode_chars(names)
    # get the tensor train/test/validation splitted dataset
    Xtr, Ytr, Xdev, Ydev, Xte, Yte = get_encoding_splitted_dataset(
        names, stoi, block_size, train_ratio, test_ratio
    )

    return Xtr, Ytr, Xdev, Ydev, Xte, Yte
