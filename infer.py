import torch
import torch.nn.functional as F
from preprocess import Preprocessor, DatasetProcessor
from data_loader import DatasetLoader


# Dataset Configuration
DATASET_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
LOCAL_PATH = "dataset/names.txt"
BLOCK_SIZE = 3
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1

# Load dataset
dataset_loader = DatasetLoader(DATASET_URL, LOCAL_PATH)
dataset_loader.download_dataset()
words = dataset_loader.read_dataset()

if not words:
    raise RuntimeError("Failed to load dataset. Please check the file path.")

# Preprocess dataset
preprocessor = Preprocessor(words)
dataset_processor = DatasetProcessor(words, preprocessor, BLOCK_SIZE)

# Split dataset
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = dataset_processor.split_dataset(
    TRAIN_RATIO, TEST_RATIO
)


@torch.no_grad()  # this decorator disables gradient tracking
def split_loss(split):
    x, y = {
        "train": (X_train, Y_train),
        "val": (X_dev, Y_dev),
        "test": (X_test, Y_test),
    }[split]
    emb = C[x]  # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1)  # concat into (N, block_size * n_embd)
    hpreact = embcat @ W1  # + b1
    # hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)  # (N, n_hidden)
    logits = h @ W2 + b2  # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())


split_loss("train")
split_loss("val")
