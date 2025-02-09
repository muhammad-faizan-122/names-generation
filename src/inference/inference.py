from src.preprocessing.preprocess import Preprocessor
from src.training.model import MLPModel
import torch
import torch.nn.functional as F


def compute_loss(model: MLPModel, X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute loss for a given dataset split."""
    with torch.no_grad():
        emb = model.C[X]  # (N, block_size, n_embd)
        embcat = emb.view(emb.shape[0], -1)  # Flatten embeddings
        hpreact = embcat @ model.W1
        hpreact = (
            model.bngain * (hpreact - model.bnmean_running) / model.bnstd_running
        ) + model.bnbias
        h = torch.tanh(hpreact)
        logits = h @ model.W2 + model.b2
        loss = F.cross_entropy(logits, Y)
        return loss.item()


def generate_names(
    model: MLPModel,
    preprocessor: Preprocessor,
    num_samples: int = 20,
    block_size: int = 3,
) -> None:
    """Generate new words using the trained model."""
    g = torch.Generator().manual_seed(2147483647 + 10)  # Set seed for reproducibility

    for _ in range(num_samples):
        out = []
        context = [0] * block_size  # Initialize context window

        while True:
            emb = model.C[torch.tensor(context).view(1, -1)]
            embcat = emb.view(emb.shape[0], -1)
            hpreact = embcat @ model.W1
            hpreact = (
                model.bngain * (hpreact - model.bnmean_running) / model.bnstd_running
            ) + model.bnbias
            h = torch.tanh(hpreact)
            logits = h @ model.W2 + model.b2
            probs = F.softmax(logits, dim=1)

            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)

            if ix == 0:  # Stop at the special '.' token
                break

        print(preprocessor.decode(out))  # Decode and print generated word
