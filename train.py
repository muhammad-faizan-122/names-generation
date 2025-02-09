from src.data.loader import DatasetLoader
from src.preprocessing.preprocess import Preprocessor, DatasetProcessor
from src.training.model import MLPModel
from src.training.trainer import Trainer
from src.utils import load_config

# Load Configurations
configs = load_config()

# Load dataset
dataset_loader = DatasetLoader(
    configs["dataset"]["url"], configs["dataset"]["local_path"]
)
dataset_loader.download_dataset()
words = dataset_loader.read_dataset()

if not words:
    raise RuntimeError("Failed to load dataset. Please check the file path.")

# Preprocess dataset
preprocessor = Preprocessor(words)
dataset_processor = DatasetProcessor(
    words, preprocessor, configs["training"]["block_size"]
)

# Split dataset
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = dataset_processor.split_dataset(
    configs["training"]["train_ratio"], configs["training"]["test_ratio"]
)

print("Dataset processing complete!")

# Initialize Model
vocab_size = len(preprocessor.stoi)
model = MLPModel(vocab_size, configs["training"]["block_size"])

# Train Model
trainer = Trainer(
    model,
    X_train,
    Y_train,
    epochs=configs["training"]["epochs"],
    batch_size=configs["training"]["batch_size"],
    model_path=configs["model"]["path"],
)
print("Training Model...")
trained_params = trainer.train()
print("Model training complete!")

# Evaluate Model
print("\n\nEvaluating model... ")
trainer.evaluate_model(X_train, Y_train, X_dev, Y_dev, X_test, Y_test)
