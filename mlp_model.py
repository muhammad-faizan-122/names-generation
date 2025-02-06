import torch
import torch.nn.functional as F


class MLPModel:
    """Multi-Layer Perceptron Model for training"""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int = 10,
        n_hidden: int = 200,
        seed: int = 2147483647,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_hidden = n_hidden

        self.g = torch.Generator().manual_seed(seed)  # for reproducibility

        # Initialize parameters
        self.C = torch.randn((vocab_size, n_embd), generator=self.g)
        self.W1 = (
            torch.randn(
                (n_embd * block_size, n_hidden),
                generator=self.g,
            )
            * (5 / 3)
            / ((n_embd * block_size) ** 0.5)
        )
        self.W2 = torch.randn((n_hidden, vocab_size), generator=self.g) * 0.01
        self.b2 = torch.zeros(vocab_size)

        # BatchNorm parameters
        self.bngain = torch.ones((1, n_hidden))
        self.bnbias = torch.zeros((1, n_hidden))
        self.bnmean_running = torch.zeros((1, n_hidden))
        self.bnstd_running = torch.ones((1, n_hidden))

        self.parameters = [self.C, self.W1, self.W2, self.b2, self.bngain, self.bnbias]
        for p in self.parameters:
            p.requires_grad = True

        print("Total trainable parameters:", sum(p.nelement() for p in self.parameters))

    def forward(self, X: torch.Tensor):
        """Performs a forward pass on the input tensor."""
        emb = self.C[X]  # Embed characters
        embcat = emb.view(emb.shape[0], -1)  # Flatten embeddings

        # Linear transformation
        hpreact = embcat @ self.W1

        # Batch normalization
        bnmeani = hpreact.mean(0, keepdim=True)
        bnstdi = hpreact.std(0, keepdim=True)
        hpreact = self.bngain * (hpreact - bnmeani) / bnstdi + self.bnbias

        # Running statistics update
        with torch.no_grad():
            self.bnmean_running = 0.999 * self.bnmean_running + 0.001 * bnmeani
            self.bnstd_running = 0.999 * self.bnstd_running + 0.001 * bnstdi

        # Non-linearity
        h = torch.tanh(hpreact)

        # Output layer
        logits = h @ self.W2 + self.b2
        return logits
