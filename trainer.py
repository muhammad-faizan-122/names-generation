import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Trainer:
    """Handles training of the model"""

    def __init__(
        self, model, X_train, Y_train, max_steps=200000, batch_size=32, seed=2147483647
    ):
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.losses = []
        # For reproducibility
        self.generator = torch.Generator().manual_seed(seed)

    def train(self):
        """Trains the model"""
        print("Starting training...")

        for i in range(self.max_steps):
            # Minibatch construction
            ix = torch.randint(
                0, self.X_train.shape[0], (self.batch_size,), generator=self.generator
            )
            Xb, Yb = self.X_train[ix], self.Y_train[ix]  # Get batch

            # Forward pass
            logits = self.model.forward(Xb)
            loss = F.cross_entropy(logits, Yb)

            # Backward pass
            for p in self.model.parameters:
                p.grad = None
            loss.backward()

            # Learning rate decay
            lr = 0.1 if i < 100000 else 0.01
            for p in self.model.parameters:
                p.data += -lr * p.grad

            # Track loss
            if i % 10000 == 0:
                print(f"{i:7d}/{self.max_steps:7d}: {loss.item():.4f}")
            self.losses.append(loss.log10().item())

        print("Training complete!")
        self.plot_loss()
        return self.model.parameters

    def plot_loss(self):
        """Plots training loss curve"""
        plt.plot(self.losses)
        plt.xlabel("Iterations")
        plt.ylabel("Log Loss")
        plt.title("Training Loss Curve")
        plt.show()
