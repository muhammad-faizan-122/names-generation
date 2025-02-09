from src.preprocessing.preprocess import Preprocessor, DatasetProcessor
from src.data.loader import DatasetLoader
from src.training.model import MLPModel
from src.inference.inference import evaluate_model, generate_names
from src.utils import load_config

configs = load_config()

print("Loading dataset...")
words = DatasetLoader(
    configs["dataset"]["url"], configs["dataset"]["local_path"]
).read_dataset()

if not words:
    raise RuntimeError("Failed to load dataset. Please check the file path.")

print("Preprocessing dataset...")
preprocessor = Preprocessor(words)
dataset_processor = DatasetProcessor(
    words, preprocessor, configs["training"]["block_size"]
)

# Split dataset
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = dataset_processor.split_dataset(
    configs["training"]["train_ratio"], configs["training"]["test_ratio"]
)

print("Loading trained model...")
model = MLPModel(
    vocab_size=configs["training"]["vocab_size"],
    block_size=configs["training"]["block_size"],
    n_embd=configs["training"]["n_embd"],
    n_hidden=configs["training"]["n_hidden"],
)
model.load_parameters(configs["model"]["path"])


print("\nGenerating Sample Words:")
generate_names(
    model,
    preprocessor,
    configs["sampling"]["num_samples"],
    block_size=configs["training"]["block_size"],
)
