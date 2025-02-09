import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.training.model import MLPModel
from src.inference.inference import compute_loss


class Trainer:
    """Handles training of the model"""

    def __init__(
        self,
        model,
        X_train,
        Y_train,
        epochs=200000,
        batch_size=32,
        seed=2147483647,
        model_path="trained_model/model.pth",
    ):
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.epochs = epochs
        self.batch_size = batch_size
        self.losses = []
        # For reproducibility
        self.generator = torch.Generator().manual_seed(seed)
        self.model_path = model_path

    def train(self):
        """Trains the model"""
        print("Starting training...")

        for i in range(self.epochs):
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
                print(f"{i:7d}/{self.epochs:7d}: {loss.item():.4f}")

            self.losses.append(loss.log10().item())

        print("Training complete!")
        self.save_model()
        print(f"Model saved at {self.model_path}")
        self.plot_loss()

    def save_model(self):
        """Saves model parameters to a file"""
        torch.save(
            {
                "C": self.model.C,
                "W1": self.model.W1,
                "W2": self.model.W2,
                "b2": self.model.b2,
                "bngain": self.model.bngain,
                "bnbias": self.model.bnbias,
                "bnmean_running": self.model.bnmean_running,
                "bnstd_running": self.model.bnstd_running,
            },
            self.model_path,
        )
        print(f"Model saved to {self.model_path}")

    def evaluate_model(self, X_train, Y_train, X_dev, Y_dev, X_test, Y_test) -> None:
        """Evaluate model on training, validation, and test sets."""
        print("Evaluating Model...")
        for split, (X, Y) in {
            "train": (X_train, Y_train),
            "val": (X_dev, Y_dev),
            "test": (X_test, Y_test),
        }.items():
            loss = compute_loss(self.model, X, Y)
            print(f"{split.capitalize()} Loss: {loss:.4f}")

    def plot_loss(self):
        """Plots training loss curve"""
        plt.plot(self.losses)
        plt.xlabel("Iterations")
        plt.ylabel("Log Loss")
        plt.title("Training Loss Curve")
        plt.show()
